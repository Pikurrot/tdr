import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dades = pd.read_excel(r"data/input_data.xlsx", sheet_name = "Hoja1")
inputs_entrenament = []
nombre_d_outputs = 1
nombre_d_inputs = len(dades.columns.ravel()) - nombre_d_outputs
for i in range(nombre_d_inputs):
    inputs_entrenament.append(dades["input_" + str(i+1)].tolist())
inputs_entrenament = np.array(inputs_entrenament).T.astype(float)
outputs_entrenament = np.array([dades["output"].tolist()]).T

tf.random.set_random_seed(1)
n_neurones = 2
taxa = 0.3
epoques = 5000
x_grafic, y_grafic = [], []
pas = 50

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

pesos_capa_1 = tf.Variable(tf.random_uniform([nombre_d_inputs, n_neurones], -1.0, 1.0))
pesos_capa_2 = tf.Variable(tf.random_uniform([n_neurones, nombre_d_outputs], -1.0, 1.0))

bias_capa_1 = tf.Variable(tf.zeros([n_neurones]))
bias_capa_2 = tf.Variable(tf.zeros([nombre_d_outputs]))

output_capa_1 = tf.sigmoid(tf.matmul(X, pesos_capa_1) + bias_capa_1)
output_final = tf.sigmoid(tf.matmul(output_capa_1, pesos_capa_2) + bias_capa_2)

error = tf.reduce_mean(-Y*tf.log(output_final) - (1-Y) * tf.log(1-output_final))

optimitzador = tf.train.GradientDescentOptimizer(taxa).minimize(error)

inicialitzacio = tf.global_variables_initializer()

with tf.Session() as sessio:
    sessio.run(inicialitzacio)

    for epoca in range(epoques + 1):
        _, valor_error = sessio.run([optimitzador, error], 
            feed_dict = {X: inputs_entrenament, Y: outputs_entrenament})  

        if epoca % pas == 0:
            x_grafic.append(epoca)
            y_grafic.append(valor_error)

    valor_pesos = (sessio.run(pesos_capa_1),sessio.run(pesos_capa_2))
    valor_bias = (sessio.run(bias_capa_1),sessio.run(bias_capa_2))

    output_arrodonit = tf.equal(tf.floor(output_final + 0.1), Y)
    precisió = tf.reduce_mean(tf.cast(output_arrodonit, "float"))

    output = sessio.run([output_final], feed_dict = {X: inputs_entrenament})

    np.set_printoptions(suppress=True)
    print("\nOutputs:")
    for i in output:
        print(i)
    print("\nPrecisió: ", precisió.eval({X: inputs_entrenament, Y: outputs_entrenament}) * 100, "%")
    print('\npesos_capa_1:\n', valor_pesos[0])
    print('\npesos_capa_2:\n', valor_pesos[1])
    print('\nbias_capa_2:\n', valor_bias[0])
    print('\nbias_capa_2:\n', valor_bias[1])

    pesos = valor_pesos
    bias = valor_bias
    figura, grafics = plt.subplots(1, 2, figsize=(12,5),squeeze=False)
    grafic = grafics[0][0]
    grafic.plot(x_grafic, y_grafic)
    grafic.axis([0, epoques, 0, max(y_grafic)])
    grafic.set_xlabel("època")
    grafic.set_ylabel("error")
    grafic.set_title(f'Gràfic error-època')
    grafic = grafics[0][1]
    for i,(x,y) in enumerate(zip([0,0,1,1],[0,1,0,1])):
        if outputs_entrenament[i] == 1:
            grafic.scatter(x,y,c='b',s=100)
        else:
            grafic.scatter(x,y,c='r',s=100)
    a = (1-bias[0])/pesos[0][0]
    b = (1-bias[0])/pesos[0][1]
    pendent = -b/a
    X = grafic.axis()
    Y = [pendent*x+b for x in X]
    grafic.plot(X,Y,c='black')
    grafic.set_xlabel("input 1")
    grafic.set_ylabel("input 2")
    grafic.set_title('Gràfic de classificació')
    plt.show()


