import numpy as np
import matplotlib.pyplot as plt

class XarxaNeuronal():

	def __init__(self):
		np.random.seed(1)
		self.pesos = np.random.random((2, 1))
		self.bias = np.random.random((1,1))
		self.error = np.array([])
		self.epoques = 300
		self.taxa = 0.05
		self.pas = 10

	def sigmoid(self, x):	
		return 1 / (1 + np.exp(-x))

	def predir(self, inputs):
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.pesos) + self.bias)
		output = np.around(output, decimals = 3)
		output.shape = (4,1)
		return output

	def entrenar(self, inputs_entrenament, outputs_entrenament):
		for n in range(self.epoques + 1):

			output = self.predir(inputs_entrenament)
			error = outputs_entrenament - output
			ajustos = np.dot(inputs_entrenament.T, error)
			self.pesos += ajustos
			self.bias += np.mean(error)

			if n % self.pas == 0:
				self.error = np.append(self.error, np.mean(error**2))

			if n in [0, 500, self.epoques]:
				print(f"""
----------Època {n}:
Actuals pesos: 
{self.pesos}
Actual bias:
{self.bias}
Actual output: 
{output}""")


XN = XarxaNeuronal()
inputs_entrenament = np.array([	[0, 0],
								[0, 1],
								[1, 0],
								[1, 1]])
outputs_entrenament = np.array([[0, 0, 0, 1]]).T

XN.entrenar(inputs_entrenament, outputs_entrenament)

plt.plot(np.array(range(int(XN.epoques / XN.pas)+1))*XN.pas, XN.error)
plt.axis([0, XN.epoques, 0, max(XN.error)])
plt.xlabel("època")
plt.ylabel("error")
plt.show()

pesos = XN.pesos
bias = XN.bias
fig, ax = plt.subplots(1, 1, figsize=(8,5))
for i,(a,b) in enumerate(zip([0,0,1,1],[0,1,0,1])):
	if outputs_entrenament[i] == 1:
		plt.scatter(a,b,c='b',s=100)
	else:
		plt.scatter(a,b,c='r',s=100)
x = (1-bias)/pesos[0]
y = (1-bias)/pesos[1]
m = -y/x
X = plt.xlim()
Y = [y+m*i for i in X]
ax.plot(X,Y,c='black')
plt.show()


