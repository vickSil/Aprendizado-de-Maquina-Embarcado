# Aprendizado-de-Maquina-Embarcado
# 🛡️ RFID Deep Learning - Raspberry Pi Pico 2W

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
<img src="https://img.shields.io/badge/Deep%20Learning-LSTM-orange">
<img src="https://img.shields.io/badge/Hardware-Raspberry%20Pi%20Pico%202W-green?logo=raspberrypi">
<img src="https://img.shields.io/badge/Model-INT8%20Quantized-red">
<img src="https://img.shields.io/badge/MicroPython-Compatible-purple">

</p>

---

## 📌 Sobre o Projeto

Sistema inteligente de análise de acessos RFID utilizando **Deep Learning embarcado** em uma **Raspberry Pi Pico 2W**.

O sistema utiliza uma rede neural temporal para identificar padrões de comportamento de usuários através do histórico de acessos RFID.

O modelo classifica cada sequência de acessos em três categorias:

| Classe       | Descrição                                        |
| ------------ | ------------------------------------------------ |
| 🟢 NORMAL    | Comportamento esperado                           |
| 🟡 SUSPEITO  | Alteração de padrão de acesso                    |
| 🔴 BLOQUEADO | Possível ataque, clonagem ou tentativa excessiva |

---
<img width="1268" height="700" alt="image" src="https://github.com/user-attachments/assets/6c47ad89-ee73-4697-b8ef-c220686ac400" />

# 🧠 Arquitetura da Rede Neural

Modelo utilizado:

```
Entrada RFID
     |
     |
LSTM (16 neurônios)
     |
     |
LSTM (16 neurônios)
     |
     |
Dense (8 neurônios)
     |
     |
Softmax (3 classes)
     |
     |
NORMAL / SUSPEITO / BLOQUEADO
```

---

# 📥 Entrada do Modelo

Cada acesso RFID é convertido em um vetor:

```
[
 hora_do_acesso,
 dia_da_semana,
 intervalo_entre_acessos,
 UID_hash
]
```

Cada amostra contém uma sequência temporal:

```
8 acessos consecutivos
```

Formato:

```
(8,4)
```

---

# 📊 Dataset Sintético

Como ainda não existe uma base real de acessos RFID disponível, foi desenvolvido um gerador de dados sintéticos baseado em comportamentos reais.

## 🟢 NORMAL

Características:

* Horário comercial
* Dias úteis
* Intervalos regulares

Exemplo:

```
08:00
10:30
13:00
17:00
```

Classe:

```
0 - NORMAL
```

---

## 🟡 SUSPEITO

Simula mudança de comportamento.

Padrão:

```
Acessos normais
        ↓
Acessos em horários incomuns
```

Exemplo:

```
09h
10h
16h
02h
03h
01h
```

Classe:

```
1 - SUSPEITO
```

---

## 🔴 BLOQUEADO

Simula ataque ou clonagem RFID.

Características:

* Muitas tentativas
* Intervalo menor que 30 segundos

Exemplo:

```
Tentativa
Tentativa
Tentativa
Tentativa
```

Classe:

```
2 - BLOQUEADO
```

---

# ⚙️ Treinamento

Ambiente:

```
Python + NumPy
```

Dataset utilizado:

| Tipo        | Quantidade      |
| ----------- | --------------- |
| Treinamento | 6000 sequências |
| Validação   | 1500 sequências |

Formato:

```
X_train = (6000,8,4)

X_val = (1500,8,4)
```

---

# 📈 Resultados do Treinamento

Arquitetura final:

```
LSTM1:
64 x 20

LSTM2:
64 x 32

Dense:
16 x 8

Output:
8 x 3
```

Resultado:

| Métrica            | Resultado   |
| ------------------ | ----------- |
| Acurácia treino    | ~75%        |
| Acurácia validação | ~75%        |
| Memória Float32    | 14476 bytes |
| Memória INT8       | 3619 bytes  |

---

# ✂️ Compressão do Modelo

Foi aplicada poda de pesos:

```
|peso| < 0.03 → removido
```

Resultado:

Antes:

```
3619 parâmetros
```

Depois:

```
417 pesos removidos
```

A acurácia permaneceu praticamente igual:

```
Antes: 75.4%

Depois: 75.5%
```

---

# 🚀 Quantização INT8

Para permitir execução em microcontroladores:

```
FLOAT32 → INT8
```

Comparação:

| Modelo  | Memória     |
| ------- | ----------- |
| Float32 | 14476 bytes |
| INT8    | 3619 bytes  |

Redução:

```
≈ 4 vezes menor
```

---

# 🛠️ Correções Implementadas

Durante a validação do modelo embarcado foi encontrado um problema no exportador.

O treinamento no PC apresentava:

```
Acurácia ≈ 75%
```

Porém o teste do modelo exportado apresentava:

```
33.3%
```

## Problema encontrado

As matrizes Dense estavam sendo lidas com índices invertidos.

Treinamento:

```
Dense:

(16,8)

Output:

(8,3)
```

Exportação incorreta:

```
Dense:

(8,16)

Output:

(3,8)
```

Isso fazia o modelo embarcado utilizar pesos errados.

---

## Correção aplicada

Dense corrigido:

```python
WD[j*DH+k]
```

Saída corrigida:

```python
WO[j*NC+k]
```

Após a correção:

* O modelo exportado passou a utilizar os mesmos pesos treinados.
* A inferência MicroPython ficou equivalente ao modelo original.

---

# 🧪 Testes Sintéticos

Arquivo:

```
test_model.py
```

O teste gera novos acessos simulados e valida a classificação.

Exemplo:

Entrada:

```
Horários comerciais
Intervalos normais
```

Resultado:

```
NORMAL
```

---

Entrada:

```
Mudança para madrugada
```

Resultado:

```
SUSPEITO
```

---

Entrada:

```
Tentativas consecutivas <30s
```

Resultado:

```
BLOQUEADO
```

---

# 📂 Estrutura do Projeto

```
RFID-DeepLearning/
│
├── train_and_export.py
│
├── model.py
│
├── test_model.py
│
├── README.md
│
└── Pico2W/
    └── firmware MicroPython
```

---

# ▶️ Como Executar

## Instalar dependência

```bash
pip install numpy
```

---

## Treinar modelo

```bash
python train_and_export.py
```

Será gerado:

```
model.py
```

---

## Testar modelo

```bash
python test_model.py
```

---

## Enviar para Raspberry Pi Pico 2W

```bash
mpremote cp model.py :model.py
```

---

# 🔧 Tecnologias

<p align="center">

<img src="https://img.shields.io/badge/Python-Programming-blue?logo=python">

<img src="https://img.shields.io/badge/NumPy-Matrix%20Operations-green">

<img src="https://img.shields.io/badge/LSTM-Neural%20Network-orange">

<img src="https://img.shields.io/badge/MicroPython-Embedded-purple">

<img src="https://img.shields.io/badge/RFID-Security-red">

</p>

---

# ✅ Conclusão

O projeto demonstrou que é possível executar uma rede neural temporal compacta em um microcontrolador.

O pipeline completo desenvolvido foi:

```
Dataset Sintético
        |
        ↓
Treinamento Deep Learning
        |
        ↓
Poda de Pesos
        |
        ↓
Quantização INT8
        |
        ↓
Exportação MicroPython
        |
        ↓
Inferência Embarcada
```

O modelo final apresenta:

✅ baixo consumo de memória
✅ execução em Raspberry Pi Pico 2W
✅ classificação temporal RFID
✅ modelo compatível com MicroPython

---

# 👨‍💻 Projeto Acadêmico

Sistema desenvolvido para estudo de:

* Inteligência Artificial embarcada
* Segurança RFID
* Redes neurais compactas
* Edge AI
* Microcontroladores
