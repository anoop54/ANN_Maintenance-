# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:49:45 2018

@author: Anoop
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

#Get data and put into arrays for training 
colums  = np.array(range(64))
colums = np.append(colums,'y')
df = pd.read_csv('load.txt',names = colums)

X = df.iloc[:,0:64]
Y = df.iloc[:,64:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)




sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(output_dim = 62, init = 'uniform',activation = 'relu',input_dim = 64))
classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid'))
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])


classifier.fit(X_train,Y_train, batch_size = 2, nb_epoch = 4)


model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")


x = np.array([[5.53562420e+09,2.45044493e+02,3.58006752e+03,9.01347242e+03,1.44138858e+03,4.60437573e+01,6.40261939e+02,2.74622521e+03,1.23139912e+04,1.70660452e+04,2.01572806e+02,1.83371129e+03,6.75398927e+03,2.86597518e+02,1.08859482e+03,4.36263307e+03,3.62835325e+03,2.30562554e+05,4.13553679e+02,1.76739627e+03,4.38149009e+03,6.78511196e+02,6.61703574e+02,2.20714034e+01,1.10511422e+03,1.53519827e+03,2.36191083e+03,3.44767342e+03,2.37805835e+03,2.18093159e+02,8.64054215e+00,9.34565860e+02,4.24000000e+02,8.71500921e+02,9.26122178e+01,5.52568009e+02,1.23274984e+03,2.90663443e+03,1.50013380e+03,1.04782440e+02,1.43684131e+04,7.69191697e+02,1.06807728e+03,7.38261649e+02,1.75491962e+02,1.31764207e+03,1.68625401e+02,3.86021653e+02,4.03564675e+03,3.48836579e+02,1.40416904e+03,3.11106559e+02,7.63023617e+03,7.65662711e+03,2.63904020e+04,3.39684829e+03,3.08481497e+02,5.33157960e+03,2.70389657e+02,2.01733320e+03,2.98259573e+03,1.05250472e+03,1.49328484e+03,5.76724273e+02]])

X = sc.transform(x)

classifier.predict(X)