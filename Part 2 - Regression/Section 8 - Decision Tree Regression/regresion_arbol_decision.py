# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:41:43 2021

@author: Claudia
"""

#Regresion con arboles de decisión
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values #para convertir el vector a matriz
y = dataset.iloc[:, 2].values #y es un vector

#Dividir el data set en conjunto de entrenamiento y conjunto de testing
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
'''

#Escalado de variables
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)


#Prediccion de nuestros modelos
y_pred = regression.predict([[6.5]])

#Visualización de resultados
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title("Modelo de regresión")
plt.xlabel('--------------')
plt.ylabel('-------------------')
plt.show()

