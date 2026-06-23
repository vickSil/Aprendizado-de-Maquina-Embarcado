# rgb_led.py
# LED RGB com cátodo comum – níveis altos acendem

from machine import Pin

# Defina os pinos conforme sua montagem
PIN_R = 2
PIN_G = 3
PIN_B = 4

led_r = Pin(PIN_R, Pin.OUT)
led_g = Pin(PIN_G, Pin.OUT)
led_b = Pin(PIN_B, Pin.OUT)

def set_color(color):
    """Define a cor do LED RGB.
       Cores suportadas: 'off', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white'
    """
    # Desliga todos
    led_r.value(0); led_g.value(0); led_b.value(0)
    
    if color == "red":
        led_r.value(1)
    elif color == "green":
        led_g.value(1)
    elif color == "blue":
        led_b.value(1)
    elif color == "yellow":
        led_r.value(1); led_g.value(1)
    elif color == "cyan":
        led_g.value(1); led_b.value(1)
    elif color == "magenta":
        led_r.value(1); led_b.value(1)
    elif color == "white":
        led_r.value(1); led_g.value(1); led_b.value(1)
    # 'off' já está desligado

def rgb_off():
    led_r.value(0); led_g.value(0); led_b.value(0)