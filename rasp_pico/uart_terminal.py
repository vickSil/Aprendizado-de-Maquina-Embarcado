# uart_terminal.py
# Envia mensagens pela UART (GP0=TX, GP1=RX) e também pelo REPL (print)

from machine import UART, Pin
import sys

# Inicializa UART0 com baudrate 115200
uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1), timeout=1000)

def uart_print(msg):
    """Envia mensagem pela UART e também imprime no REPL."""
    uart.write(msg + "\r\n")
    print(msg)   # para ver no console USB também