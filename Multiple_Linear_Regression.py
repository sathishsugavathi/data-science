# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:55:32 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score
data=pd.read_csv("E:/project/Linear Regression/50_Startups.csv")
data.head()
data.tail()
data.describe()
print(data.shape)
data.head(10)
# changing columns using .columns() 
data.columns = ['RandD_Spend','Administration','Marketing_Spend','State','Profit']
data.head(10) 
data1= data[['RandD_Spend','Administration','Marketing_Spend','Profit']]
data1.head(5)
#feature selection
viz = data1[['RandD_Spend','Administration','Marketing_Spend','Profit']]
viz.hist()
plt.show()

plt.scatter(data1.RandD_Spend, data1.Profit, color='blue')
plt.xlabel("R&D_Spend")
plt.ylabel("Profit")
plt.show()

plt.scatter(data1.Administration, data1.Profit, color='blue')
plt.xlabel("Administration")
plt.ylabel("Profit")
plt.show()

plt.scatter(data1.Marketing_Spend, data1.Profit, color='blue')
plt.xlabel("Administration")
plt.ylabel("Profit")
plt.show()


X = data1[['RandD_Spend','Administration','Marketing_Spend',]] .values  
X[0:10]
print(X.shape)
y=data1[['Profit']]
y[0:10]
print(y.shape)
#split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, test_size=0.2, random_state=3)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn import linear_model
multi_regr = linear_model.LinearRegression()
multi_regr.fit(X_train,y_train)
#predict the values
y_pred=multi_regr.predict(X_test)
print(y_pred)
print(X_train)
# The coefficients
print ('Coefficients: ', multi_regr.coef_)
print ('Intercept: ',multi_regr.intercept_)
#Model Evalution
from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred -y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred , y_test) ) 
a=[[100000,123334,303319]]
Z_test=np.array(a)
type(Z_test)
y_rd = multi_regr.predict(Z_test)
print(y_rd)


