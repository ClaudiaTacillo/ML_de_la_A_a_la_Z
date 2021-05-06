# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:53:26 2021

@author: Claudia
"""

#Regresión polinómica 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values #para convertir el vector a matriz
y = dataset.iloc[:, 2].values #y es un vector

#Ajustar la regresión lineal con el dataset 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualización de resultados modelo lineal
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Modelo de regresión lineal")
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo anual (en $)')
plt.show()

#Visualización de resultados modelo polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Modelo de regresión polinómica")
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo anual (en $)')
plt.show()

#Prediccion de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))