import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import math

#Carrego el model "HW_trained_model.py"
fitxer_json = open('model.json', 'r')
lectura_fitxer_json = fitxer_json.read()
fitxer_json.close()
model_entrenat = model_from_json(lectura_fitxer_json)

#Carrego els weights
model_entrenat.load_weights("model.h5")
print("El model s'ha carregat correctament\n")

print(model_entrenat.summary())

compilar = False
mida = 15
salt_de_text = 25
imatge = np.zeros((28 * mida, 28 * mida, 1))

def dibuixar(event, x, y, flag, params):
    global compilar,imatge,salt_de_text
    if event == cv2.EVENT_LBUTTONDOWN:
        compilar = True
    elif event == cv2.EVENT_MOUSEMOVE:  
        if compilar == True:
            cv2.circle(imatge, (x,y), 2 * (mida - 5), (255,255,255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        compilar = False
        cv2.circle(imatge, (x,y), 2 * (mida - 5), (255,255,255), -1)
        escala_de_grisos = cv2.resize(imatge, (28, 28))
        escala_de_grisos = escala_de_grisos.reshape(1, 784)
        output = np.argmax(model_entrenat.predict(escala_de_grisos))
        output = 'Nombre predit : {}'.format(output)
        cv2.putText(imatge, org=(25,salt_de_text), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.75, text= output, color=(255,0,0), thickness=2)
        salt_de_text += 25

    elif event == cv2.EVENT_RBUTTONDOWN:
        imatge = np.zeros((28 * mida, 28 * mida, 1))
        salt_de_text = 25


cv2.namedWindow('imatge input')
cv2.setMouseCallback('imatge input', dibuixar)


while True:    
    cv2.imshow("imatge input", imatge)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()