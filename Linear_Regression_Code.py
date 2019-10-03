# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:12:35 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 02:19:45 2019

@author: user
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Load the data set
df = pd.read_csv("E:/project/Linear Regression/FuelConsumption.csv")
# take a look at the dataset
df.head()
df.head(20)
# summarize the data
df.describe()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
#Visualize the data and Feature selection
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()
#show best feature
X=cdf[['ENGINESIZE']]
y=cdf[['CO2EMISSIONS']]
plt.scatter(X, y,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
#split the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
print(X_train[0:10])
print(y_train[0:10])
print(X_test[0:10])
print(y_test[0:10])
#Model
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
#predict the values
y_pred=regr.predict(X_test)
print(y_pred)
#Linear model visualization
x = np.asanyarray(X_train[['ENGINESIZE']])
plt.scatter(X, y,  color='blue')
plt.plot(x, regr.coef_[0][0]*x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#Model Evalution
from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred -y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred , y_test) ) 
#predict the new data 
a=[[3.6],[1.8],[1.5],[5.0]]
Z_test=np.array(a)
type(Z_test)
y_rd = regr.predict(Z_test)
#visualize the predict data
print(y_rd)
plt.scatter(a,y_rd,color="red")
plt.xlabel("Engine size")
plt.ylable("predict value")
plt.show()


