import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import math

mnist = keras.datasets.mnist
(inputs_entrenament, outputs_entrenament), (_, _) = mnist.load_data()
inputs_entrenament = inputs_entrenament / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy")

model.fit(inputs_entrenament, outputs_entrenament, epochs=3)

model_json = model.to_json()
with open("model.json", "w") as fitxer_json:
    fitxer_json.write(model_json)
model.save_weights("model.h5")
print("Model guardat")

pesos = []
bias = []
for i in range(len(model_entrenat.layers)):
    try:
        p = model_entrenat.layers[i].get_weights()[0]
        b = model_entrenat.layers[i].get_weights()[1]
        pesos.append(p)
        bias.append(b)
    except:
        continue
imatges = []
capa_1 = (pesos[0]+bias[0]).T
capa_2 = (pesos[1]+bias[1]).T
capes = np.dot(capa_2,capa_1)
figura = plt.figure(figsize=(9,9))
mida = math.sqrt(len(capes))+1
for i in range(len(capes)):
    capes[i] *= 255.0
    capes[i] = capes[i].astype(int)
    imatges.append([])
    for j in range(0,784,28):
        imatges[i].append(np.array(capes[i][j:j+28]))
    f = figura.add_subplot(mida,mida,i+1)
    plt.imshow(imatges[i], interpolation='nearest')
    color_map = plt.imshow(imatges[i])
    color_map.set_cmap("Greys_r")
    plt.tight_layout()
    plt.axis("off")
plt.tight_layout()
plt.show()


