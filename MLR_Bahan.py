#import the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the dataset
dataset = pd.read_csv('kekuatan_bahan_dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#data preprocesing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [0])], remainder='passthrough')
x = ct.fit_transform(x)

#split the data to 80% train and 20% test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

#train the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict the accuration of new data
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#r2 accuracy
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"Model R^2 Score : {r2:.2f}")

#get into the new function
def prediksi_kekuatan(Material, Density, Hardness):
  input_data = [[Material, Density, Hardness]]
  input_encoder = ct.transform(input_data)
  prediction = regressor.predict(input_encoder)[0]
  return prediction, r2*100

prediksi_kekuatan("Cast Iron", 5.75, 200)
