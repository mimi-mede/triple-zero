import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# 1 Inicializamos el modelo
m = 0
c = 0
L = 0.01  #  learning Rate
epochs = 1000  # Numero de epocas o iteraciones de nuestro gd

n = float(len(X))

# Gradient Descent 
for i in range(epochs): 
  Y_pred = m*X + c 
  D_m = (-2/n) * sum(X * (Y - Y_pred))  # 2 Calcular Derivada m
  D_c = (-2/n) * sum(Y - Y_pred)  # Derivada c
  m = m - L * D_m  # 3 Actualizamos valores
  c = c - L * D_c  # 
    
print (m, c)

model_y= [m*i + c for i in train_x]
plt.plot(train_x, model_y, color='#bcbdf6')
plt.scatter(X, Y,color='#e0d6ff')
plt.show()