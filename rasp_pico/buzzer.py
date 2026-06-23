# buzzer.py
# Buzzer ativo via PWM

from machine import PWM, Pin
import time

BUZZER_PIN = 5
buzzer = PWM(Pin(BUZZER_PIN))

def beep(freq=1500, ms=150):
    buzzer.freq(freq)
    buzzer.duty_u16(32768)
    time.sleep_ms(ms)
    buzzer.duty_u16(0)

def beep_normal(freq=1200, ms=100):
    beep(freq, ms)

def beep_suspicious(freq=800, ms=300):
    beep(freq, ms)

def beep_blocked(freq=2000, ms=200):
    for _ in range(3):
        beep(freq, 100)
        time.sleep_ms(50)

def beep_alert():
    for f in [2000, 1500, 2000]:
        beep(f, 120)
        time.sleep_ms(60)