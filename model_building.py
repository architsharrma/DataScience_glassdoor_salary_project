# -*- coding: utf-8 -*-
"""
Created on Fri May 28 00:24:21 2021

@author: Archit Sharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# let's start model building
df=pd.read_csv('EDA_DATA.csv')

# we will be implementing multiple ML Algorithm to checl which model will give good accuracy.
df.columns
model_data=df[['avg_salary','Headquarters','Rating','Size','Type of ownership','Industry','Sector','hourly','employer_provided','job_state','same_state','age','python_yn','spark','aws','excel','job_title','seniority_level','total_competitor']]

# let's convert all the categorical data into numerical data for model building

df_dummies=pd.get_dummies(model_data)

from sklearn.model_selection import train_test_split

X = df_dummies.drop('avg_salary', axis =1)
y = df_dummies.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm=LinearRegression()

lm.fit(X_train,y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

#Lasso Regression
lr=Lasso(alpha=0.05)
lr.fit(X_train,y_train)
np.mean(cross_val_score(lr,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    LR = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(LR,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

#random forest
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
np.mean(cross_val_score(rf,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

#Hyperparamaeter optimization
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

GS=GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv= 3)
GS.fit(X_train,y_train)
GS.best_score_
GS.best_estimator_

# test data 
tpred_lm = lm.predict(X_test)
tpred_lml = lr.predict(X_test)
tpred_rf = GS.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)
import pickle
picle={'model':GS.best_estimator_}
pickle.dump(picle, open('model_file' + ".p","wb"))
file_name="model_file.p"
with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]


list(X_test.iloc[1,:])

