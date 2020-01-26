import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

data  =  pd.read_csv('Train_v2.csv')
data = data.drop(columns="uniqueid")

#separate bank_account
data_bank  = data['bank_account']
data = data.drop(columns="bank_account")

#converting to numpy
X  = data.loc[:,:].values
Y = data_bank.loc[:].values

#encoding the data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X[:,0] = label.fit_transform(X[:,0])
X[:,1] = label.fit_transform(X[:,1])
X[:,2] = label.fit_transform(X[:,2])
X[:,3] = label.fit_transform(X[:,3])
X[:,4] = label.fit_transform(X[:,4])
X[:,5] = label.fit_transform(X[:,5])
X[:,6] = label.fit_transform(X[:,6])
X[:,7] = label.fit_transform(X[:,7])
X[:,8] = label.fit_transform(X[:,8])
X[:,9] = label.fit_transform(X[:,9])
X[:,10] = label.fit_transform(X[:,10])

Y = label.fit_transform(Y)




#Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(2)
new_X = pca.fit_transform(X)
ratio = pca.explained_variance_ratio_


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train, y_train)
print(regressor.score(X_test,y_test))
