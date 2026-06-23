"""
Demo de testes sintéticos RFID
Raspberry Pi Pico 2W Deep Learning
"""

from model import classify
import random


random.seed(10)


# ======================================================
# TESTE 1 - ACESSO NORMAL
# ======================================================

normal = [

    [0.33,0.14,0.50,0.45],
    [0.36,0.28,0.60,0.45],
    [0.40,0.42,0.70,0.45],
    [0.45,0.57,0.80,0.45],
    [0.50,0.42,0.55,0.45],
    [0.55,0.28,0.90,0.45],
    [0.60,0.57,0.65,0.45],
    [0.62,0.42,0.70,0.45]

]


# ======================================================
# TESTE 2 - COMPORTAMENTO SUSPEITO
# ======================================================

suspeito = [

    [0.35,0.14,0.50,0.50],
    [0.38,0.28,0.60,0.50],
    [0.42,0.42,0.70,0.50],
    [0.55,0.57,0.80,0.50],

    [0.03,0.85,0.20,0.50],
    [0.05,0.00,0.15,0.50],
    [0.02,0.85,0.10,0.50],
    [0.04,0.00,0.12,0.50]

]


# ======================================================
# TESTE 3 - ATAQUE / BLOQUEADO
# ======================================================

bloqueado = [

    [0.90,0.85,0.0001,0.60],
    [0.90,0.85,0.0002,0.60],
    [0.91,0.85,0.0001,0.60],
    [0.90,0.85,0.0003,0.60],
    [0.92,0.85,0.0001,0.60],
    [0.91,0.85,0.0002,0.60],
    [0.90,0.85,0.0001,0.60],
    [0.91,0.85,0.0002,0.60]

]


testes = [

    ("ACESSO NORMAL", normal),

    ("MUDANCA DE PADRAO SUSPEITA", suspeito),

    ("ATAQUE RFID BLOQUEADO", bloqueado)

]


print("="*60)
print(" RFID DEEP LEARNING - TESTE")
print("="*60)


for nome,entrada in testes:


    resultado, confianca = classify(entrada)


    print()

    print("-"*60)

    print("Entrada:")
    print(nome)

    print()

    print("Resultado do modelo:")

    print(
        "Classe:",
        resultado
    )

    print(
        "Confianca:",
        str(confianca)+"%"
    )


print()
print("="*60)
print("TESTE FINALIZADO")
print("="*60)