# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:30:14 2021

@author: Claudia
"""

#Plantilla de preprocesado

#Como importoar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar dataset

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Solo cuadno existen datos faltantes
#Tratamiento de los NANs
"""from sklearn.impute import SimpleImputer
simp = SimpleImputer(missing_values = "np.nan", strategy = "mean")
simp = SimpleImputer().fit(X[:, 1:3])
X[:, 1:3] = simp.transform(X[:, 1:3])"""

#Solo cuando se tienen datos categóricos
#Codificar datos categóricos
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])], remainder= "passthrough")
X = np.array(transformer.fit_transform(X), dtype = np.float)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""

#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
