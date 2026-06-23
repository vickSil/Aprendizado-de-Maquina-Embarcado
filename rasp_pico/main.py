# main.py
# RFID + Deep Learning – Raspberry Pi Pico (RP2040)
# Substitui RPi.GPIO por machine, usa LED RGB e UART

from machine import Pin, SPI, PWM, UART
import time
import json

# Módulos locais
from rfid_rc522 import read_uid
from rgb_led import set_color, rgb_off
from buzzer import beep_normal, beep_suspicious, beep_blocked, beep_alert
from uart_terminal import uart_print
from model import classify, LABELS

# ── Constantes ──────────────────────────────────────────────
DB_FILE = "rfid_db.json"
WINDOW = 8          # janela temporal para a LSTM
BTN_PIN = 15        # botão de cadastro (segurar 3s)
BTN_HOLD_MS = 3000

# ── Inicialização dos periféricos ──────────────────────────
# RFID RC522 – SPI0
spi = SPI(0, baudrate=1_000_000, polarity=0, phase=0,
          sck=Pin(18), mosi=Pin(19), miso=Pin(16))
# O driver rfid_rc522.py usará esses pinos (SDA=GP17, RST=GP20)
# Mas a função read_uid já lida com isso.

# Botão
btn = Pin(BTN_PIN, Pin.IN, Pin.PULL_UP)

# ── Banco de dados de UIDs ──────────────────────────────────
def load_db():
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

db = load_db()

# ── Histórico de acessos por cartão ────────────────────────
history = {}   # uid_hex -> lista de [hora, dia, delta, uid_hash]

def update_history(uid_hex, uid_hash):
    now = time.localtime()
    hora = now[3] * 3600 + now[4] * 60 + now[5]   # segundos do dia
    dia = now[6]   # 0=segunda, 6=domingo

    if uid_hex not in history:
        history[uid_hex] = []

    hist = history[uid_hex]
    if hist:
        last_hora = hist[-1][2] * 86400   # o delta armazenado é normalizado
        # Na verdade, armazenamos o delta_t, então precisamos do timestamp absoluto.
        # Vamos guardar o horário real do último acesso em um dicionário separado.
        # Mas para simplificar, usaremos o delta normalizado.
        # Ajuste: guardamos o delta_t como (hora_atual - hora_anterior)/86400
        # Precisamos da hora anterior. Vamos guardar a hora do último acesso.
        # Vou modificar: armazenar no histórico a lista [hora, dia, delta, uid_hash],
        # onde delta é (hora - ultima_hora)/86400. Para o primeiro, delta=0.5.
        last_hora = hist[-1][0] * 86400   # a hora normalizada estava em [0,1]
        delta = max(0, (hora - last_hora) / 86400)
    else:
        delta = 0.5   # primeiro acesso

    feat = [hora / 86400, dia / 7, delta, uid_hash / 255]
    hist.append(feat)
    if len(hist) > WINDOW:
        hist.pop(0)
    while len(hist) < WINDOW:
        hist.insert(0, feat)   # padding com o primeiro valor
    return hist[-WINDOW:]

# ── Função para ler UID do módulo RFID ─────────────────────
def get_uid():
    """Retorna (uid_hex, uid_hash) ou (None, None) se não houver cartão."""
    # A função read_uid do rfid_rc522.py deve retornar uma lista de bytes ou None.
    # Vamos adaptar o rfid_rc522.py para ter uma função read_uid().
    # Por enquanto, assumimos que read_uid() retorna bytes ou None.
    raw = read_uid()
    if raw is None:
        return None, None
    uid_hex = "".join(f"{b:02X}" for b in raw)
    uid_hash = sum(raw) % 256
    return uid_hex, uid_hash

# ── Modo cadastro ───────────────────────────────────────────
def cadastro_mode():
    uart_print("MODO CADASTRO – Aproxime o cartão")
    set_color("blue")
    beep_normal()

    deadline = time.ticks_add(time.ticks_ms(), 10000)  # 10s de timeout
    while time.ticks_diff(deadline, time.ticks_ms()) > 0:
        uid_hex, uid_hash = get_uid()
        if uid_hex is not None:
            if uid_hex not in db:
                db[uid_hex] = {"nome": f"User_{uid_hex[:4]}", "hash": uid_hash}
                save_db(db)
                uart_print(f"CADASTRADO! UID={uid_hex}")
                beep_normal(ms=300)
            else:
                uart_print(f"JÁ CADASTRADO: {db[uid_hex]['nome']}")
                beep_suspicious()
            time.sleep(2)
            return
        time.sleep_ms(100)

    uart_print("Timeout – cadastro cancelado")
    time.sleep(1)

# ── Loop principal ──────────────────────────────────────────
def main():
    uart_print("RFID+Deep Learning – Pico RP2040")
    uart_print("Aguardando cartão...")
    rgb_off()

    btn_press_start = None

    while True:
        # Verifica botão (segurar 3s para cadastro)
        if not btn.value():
            if btn_press_start is None:
                btn_press_start = time.ticks_ms()
            elif time.ticks_diff(time.ticks_ms(), btn_press_start) >= BTN_HOLD_MS:
                cadastro_mode()
                btn_press_start = None
                uart_print("Aguardando...")
                continue
        else:
            btn_press_start = None

        # Tenta ler RFID
        uid_hex, uid_hash = get_uid()
        if uid_hex is None:
            time.sleep_ms(100)
            continue

        t_start = time.ticks_ms()

        # Verifica se UID está cadastrado
        if uid_hex not in db:
            set_color("red")
            uart_print(f"ACESSO NEGADO – UID {uid_hex[:8]} (desconhecido)")
            beep_alert()
            time.sleep(2)
            uart_print("Aguardando...")
            rgb_off()
            continue

        # Atualiza histórico e faz inferência
        features = update_history(uid_hex, uid_hash)
        label, conf = classify(features)
        cls_idx = LABELS.index(label)
        t_ms = time.ticks_diff(time.ticks_ms(), t_start)

        nome = db[uid_hex]["nome"][:7]

        # Ações conforme classe
        if cls_idx == 0:   # NORMAL
            set_color("green")
            uart_print(f"LIBERADO ({conf}%) – {nome}")
            beep_normal()
        elif cls_idx == 1: # SUSPEITO
            set_color("yellow")
            uart_print(f"ATIPICO ({conf}%) – UID {uid_hex[:8]}")
            beep_suspicious()
        else:              # BLOQUEADO
            set_color("red")
            uart_print(f"BLOQUEADO ({conf}%) – UID {uid_hex[:8]}")
            beep_blocked()

        # Log com timestamp
        lt = time.localtime()
        ts = f"{lt[3]:02d}:{lt[4]:02d}:{lt[5]:02d}"
        uart_print(f"[{ts}] UID={uid_hex}  {label:10s} conf={conf}%  T={t_ms}ms")

        time.sleep(2)
        uart_print("Aguardando...")
        rgb_off()

# ── Execução ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()