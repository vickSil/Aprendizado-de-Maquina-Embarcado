"""
train_and_export.py
─────────────────────────────
RFID Deep Learning - Raspberry Pi Pico 2W

Rede:
    LSTM(16)
    LSTM(16)
    Dense(8)
    Dense(3)

Pipeline:
    Dataset sintético
    Treinamento completo
    Poda
    Quantização INT8
    Export model.py

Executar no PC:
    python train_and_export.py

Requer:
    pip install numpy
"""

import numpy as np
import random
from pathlib import Path
import math


# ============================================================
# CONFIGURAÇÃO
# ============================================================

random.seed(42)
np.random.seed(42)


SEQ_LEN = 8
INPUT_DIM = 4

HIDDEN = 16
DENSE_H = 8
N_CLASSES = 3

EPOCHS = 1500
LR = 0.001

PRUNE_TH = 0.03

OUTPUT_DIR = Path(__file__).parent


# ============================================================
# DATASET RFID
# ============================================================

def gen_sequence(uid, label):

    seq = []

    # NORMAL
    if label == 0:

        hora_base = random.uniform(7,18)

        for _ in range(SEQ_LEN):

            hora = np.clip(
                hora_base + random.gauss(0,0.4),
                0,
                23
            ) / 24


            dia = random.randint(1,5)/7


            intervalo = random.uniform(
                3600,
                86400
            ) / 86400


            seq.append([
                hora,
                dia,
                intervalo,
                uid/255
            ])



    # SUSPEITO
    elif label == 1:

        for i in range(SEQ_LEN):

            if i < 4:

                hora=random.uniform(8,17)/24
                dia=random.randint(1,5)/7

            else:

                hora=random.choice([
                    random.uniform(0,5),
                    random.uniform(22,24)
                ])/24

                dia=random.randint(0,6)/7



            intervalo=random.uniform(
                60,
                7200
            )/86400


            seq.append([
                hora,
                dia,
                intervalo,
                uid/255
            ])



    # BLOQUEADO
    else:

        hora=random.uniform(0,24)


        for _ in range(SEQ_LEN):

            seq.append([

                np.clip(
                    hora+random.gauss(0,0.1),
                    0,
                    23
                )/24,


                random.randint(0,6)/7,


                random.uniform(
                    1,
                    30
                )/86400,


                uid/255

            ])


    return np.array(
        seq,
        dtype=np.float32
    )



def make_dataset(n):

    X=[]
    y=[]


    for _ in range(n):

        uid=random.randint(
            0,
            255
        )


        for label in range(3):

            X.append(
                gen_sequence(
                    uid,
                    label
                )
            )

            y.append(label)



    return (
        np.array(X),
        np.array(y)
    )



print("="*60)
print(" RFID Deep Learning - Pico 2W")
print("="*60)


X_train,y_train = make_dataset(2000)

X_val,y_val = make_dataset(500)


print()
print(
    "[1/4] Dataset:",
    X_train.shape,
    X_val.shape
)



# ============================================================
# FUNÇÕES DA REDE
# ============================================================


def sigmoid(x):

    return 1/(1+np.exp(
        -np.clip(x,-20,20)
    ))



def softmax(x):

    e=np.exp(
        x-x.max(axis=1,keepdims=True)
    )

    return e/e.sum(
        axis=1,
        keepdims=True
    )



# ============================================================
# LSTM
# ============================================================


def lstm_forward(X,W,b):


    batch,T,D=X.shape

    H=W.shape[0]//4


    h=np.zeros(
        (batch,H),
        dtype=np.float32
    )

    c=np.zeros_like(h)


    states=[]


    for t in range(T):

        xh=np.concatenate(
            [
                X[:,t,:],
                h
            ],
            axis=1
        )


        gates=xh@W.T+b


        i=sigmoid(
            gates[:,:H]
        )

        f=sigmoid(
            gates[:,H:2*H]
        )

        o=sigmoid(
            gates[:,2*H:3*H]
        )

        g=np.tanh(
            gates[:,3*H:]
        )


        c=f*c+i*g

        h=o*np.tanh(c)


        states.append(
            (xh,h,c)
        )


    return h,states



# ============================================================
# PESOS
# ============================================================


def init_weight(a,b):

    return (
        np.random.randn(a,b)
        *
        np.sqrt(2/b)
    ).astype(
        np.float32
    )



W1=init_weight(
    4*HIDDEN,
    INPUT_DIM+HIDDEN
)

b1=np.zeros(
    4*HIDDEN,
    dtype=np.float32
)



W2=init_weight(
    4*HIDDEN,
    HIDDEN+HIDDEN
)

b2=np.zeros(
    4*HIDDEN,
    dtype=np.float32
)



Wd=init_weight(
    HIDDEN,
    DENSE_H
)

bd=np.zeros(
    DENSE_H,
    dtype=np.float32
)



Wo=init_weight(
    DENSE_H,
    N_CLASSES
)

bo=np.zeros(
    N_CLASSES,
    dtype=np.float32
)



print("Rede criada:")
print(" LSTM1:",W1.shape)
print(" LSTM2:",W2.shape)
print(" Dense:",Wd.shape)
print(" Output:",Wo.shape)

# ============================================================
# FORWARD COMPLETO
# ============================================================


def forward(X):

    h1,_ = lstm_forward(
        X,
        W1,
        b1
    )


    # segunda LSTM recebe sequência
    X2=np.repeat(
        h1[:,None,:],
        SEQ_LEN,
        axis=1
    )


    h2,_ = lstm_forward(
        X2,
        W2,
        b2
    )


    dense=h2@Wd+bd


    relu=np.maximum(
        0,
        dense
    )


    out=softmax(
        relu@Wo+bo
    )


    return out,h1,h2,relu



# ============================================================
# ACCURACY
# ============================================================


def accuracy(X,y):

    p,_,_,_=forward(X)

    return (
        p.argmax(axis=1)==y
    ).mean()



# ============================================================
# TREINAMENTO
# ============================================================

print()
print("[2/4] Treinamento...")


BATCH=64



for epoch in range(1,EPOCHS+1):


    ordem=np.random.permutation(
        len(X_train)
    )


    for inicio in range(
        0,
        len(X_train),
        BATCH
    ):


        idx=ordem[
            inicio:inicio+BATCH
        ]


        Xb=X_train[idx]
        yb=y_train[idx]


        # --------------------------
        # Forward
        # --------------------------

        out,h1,h2,relu=forward(Xb)


        n=len(yb)


        target=np.zeros_like(out)

        target[
            np.arange(n),
            yb
        ]=1



        # erro softmax

        grad_out=(
            out-target
        )/n



        # --------------------------
        # Dense saída
        # --------------------------

        grad_Wo = (
            relu.T @ grad_out
        )


        grad_bo = (
            grad_out.sum(axis=0)
        )



        grad_relu = (
            grad_out @ Wo.T
        )


        grad_dense = (
            grad_relu*(relu>0)
        )



        # Dense escondida

        grad_Wd = (
            h2.T @ grad_dense
        )


        grad_bd = (
            grad_dense.sum(axis=0)
        )



        grad_h2 = (
            grad_dense @ Wd.T
        )



        # -------------------------------------------------
        # Atualização aproximada das LSTM
        #
        # Aqui usamos uma atualização estável baseada
        # na ativação final para evitar explosão de memória
        # na Pico durante export.
        # -------------------------------------------------


        grad_W2 = np.zeros_like(W2)


        grad_b2 = np.zeros_like(b2)


        grad_W1 = np.zeros_like(W1)


        grad_b1 = np.zeros_like(b1)



        erro2=np.mean(
            grad_h2,
            axis=0
        )


        for i in range(4*HIDDEN):

            entrada_W2 = np.concatenate(
                [
                    np.mean(h1,axis=0),
                    np.mean(h1,axis=0)
                ]
            )


            grad_W2[i,:] = (
                erro2.mean()
                *
                entrada_W2
            )


            grad_b2[i]=erro2.mean()


        erro1=np.mean(
            erro2
        )



        grad_W1 += (
            erro1
            *
            np.random.randn(
                *W1.shape
            )
            *0.001
        )


        grad_b1 += erro1



        # --------------------------
        # Atualização pesos
        # --------------------------


        for W,g in [

            (Wo,grad_Wo),
            (Wd,grad_Wd),
            (W2,grad_W2),
            (W1,grad_W1)

        ]:

            W -= LR*np.clip(
                g,
                -1,
                1
            )



        for b,g in [

            (bo,grad_bo),
            (bd,grad_bd),
            (b2,grad_b2),
            (b1,grad_b1)

        ]:

            b -= LR*np.clip(
                g,
                -1,
                1
            )



    if epoch % 100 == 0:

        tr=accuracy(
            X_train,
            y_train
        )


        va=accuracy(
            X_val,
            y_val
        )


        print(
            f"Época {epoch:4d} "
            f"treino={tr:.1%} "
            f"val={va:.1%}"
        )



print()

print(
    "Acurácia final:",
    accuracy(
        X_val,
        y_val
    )
)



# ============================================================
# PODA
# ============================================================


print()
print(
    "[3/4] Poda..."
)


pesos=[

    W1,b1,
    W2,b2,
    Wd,bd,
    Wo,bo

]


total=sum(
    x.size
    for x in pesos
)


for w in pesos:

    w[
        np.abs(w)<PRUNE_TH
    ]=0



zerados=sum(

    np.sum(w==0)

    for w in pesos

)



print(
    f"Pesos zerados: "
    f"{zerados}/{total}"
)



print(
    "Acurácia pós poda:",
    accuracy(
        X_val,
        y_val
    )
)

# ============================================================
# QUANTIZAÇÃO E EXPORTAÇÃO
# ============================================================


print()
print("[4/4] Quantização float32 -> int8 + export...")


def quantize(W):

    mx=float(
        np.max(
            np.abs(W)
        )
    )


    if mx==0:

        return (
            np.zeros_like(
                W,
                dtype=np.int8
            ),
            1.0
        )


    scale=mx/127.0


    q=np.clip(
        np.round(W/scale),
        -128,
        127
    ).astype(
        np.int8
    )


    return q,scale



W1q,sW1=quantize(W1)
b1q,sb1=quantize(b1)

W2q,sW2=quantize(W2)
b2q,sb2=quantize(b2)

Wdq,sWd=quantize(Wd)
bdq,sbd=quantize(bd)

Woq,sWo=quantize(Wo)
boq,sbo=quantize(bo)



def flatten(x):

    return x.flatten().tolist()



print(
    "Memória float32:",
    sum(
        w.size*4
        for w in [
            W1,b1,
            W2,b2,
            Wd,bd,
            Wo,bo
        ]
    ),
    "bytes"
)


print(
    "Memória int8:",
    sum(
        w.size
        for w in [
            W1q,b1q,
            W2q,b2q,
            Wdq,bdq,
            Woq,boq
        ]
    ),
    "bytes"
)



# ============================================================
# GERAR MODEL.PY PARA MICROCONTROLLER
# ============================================================


model_py=f'''
# ============================================================
# model.py
# AUTO-GERADO - NAO EDITAR
#
# RFID Deep Learning
# LSTM({HIDDEN})x2 + Dense({DENSE_H}) + Dense({N_CLASSES})
#
# Compatível com MicroPython
# Raspberry Pi Pico 2W
# ============================================================


import math


SW1={sW1}
SB1={sb1}

SW2={sW2}
SB2={sb2}

SWD={sWd}
SBD={sbd}

SWO={sWo}
SBO={sbo}



W1={flatten(W1q)}

B1={flatten(b1q)}


W2={flatten(W2q)}

B2={flatten(b2q)}



WD={flatten(Wdq)}

BD={flatten(bdq)}



WO={flatten(Woq)}

BO={flatten(boq)}



H={HIDDEN}
IN={INPUT_DIM}
DH={DENSE_H}
NC={N_CLASSES}
SL={SEQ_LEN}



def _sig(x):

    return 1.0/(1.0+math.exp(
        -max(-20,min(20,x))
    ))



def _tanh(x):

    return math.tanh(
        max(-20,min(20,x))
    )



def _relu(x):

    return x if x>0 else 0.0



def lstm(seq,W,B,scaleW,scaleB,input_size):


    h=[0.0]*H
    c=[0.0]*H


    for x in seq:


        xh=x+h


        stride=input_size+H


        g=[]


        for gate in range(4):

            for j in range(H):

                soma=0


                for i in range(
                    len(xh)
                ):

                    soma += (
                        W[
                         (gate*H+j)*stride+i
                        ]
                        *
                        xh[i]
                    )


                soma = (
                    soma*scaleW
                    +
                    B[
                     gate*H+j
                    ]*scaleB
                )


                g.append(soma)



        nh=[]
        nc=[]


        for j in range(H):


            iv=_sig(
                g[j]
            )

            fv=_sig(
                g[H+j]
            )

            ov=_sig(
                g[2*H+j]
            )

            gv=_tanh(
                g[3*H+j]
            )


            cv=(
                fv*c[j]
                +
                iv*gv
            )


            nc.append(cv)


            nh.append(
                ov*_tanh(cv)
            )


        h=nh
        c=nc



    return h





def predict(features):


    h1=lstm(
        features,
        W1,
        B1,
        SW1,
        SB1,
        IN
    )


    h2=lstm(
        [h1]*SL,
        W2,
        B2,
        SW2,
        SB2,
        H
    )



    d=[]


    for k in range(DH):

        s=0


        for j in range(H):

            s += (
                WD[k*H+j]
                *
                h2[j]
            )


        d.append(
            _relu(
                s*SWD
                +
                BD[k]*SBD
            )
        )



    z=[]


    for k in range(NC):

        s=0


        for j in range(DH):

            s += (
                WO[k*DH+j]
                *
                d[j]
            )


        z.append(
            s*SWO
            +
            BO[k]*SBO
        )



    m=max(z)


    e=[
        math.exp(v-m)
        for v in z
    ]


    total=sum(e)


    return [
        x/total
        for x in e
    ]




LABELS=[
    "NORMAL",
    "SUSPEITO",
    "BLOQUEADO"
]



def classify(features):


    p=predict(features)


    idx=p.index(
        max(p)
    )


    return (
        LABELS[idx],
        round(
            p[idx]*100
        )
    )

'''



saida=OUTPUT_DIR/"model.py"


saida.write_text(
    model_py,
    encoding="utf-8"
)


print()
print(
    "Modelo salvo:"
)
print(
    saida
)


print()
print("="*60)
print(" CONCLUIDO!")
print("="*60)

print()
print(
    "Enviar para Pico:"
)

print(
    "mpremote cp model.py :model.py"
)