# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:54:49 2022

@author: apmel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:55:59 2022

@author: apmel
"""

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

#Loading the dataframe
df = pd.read_csv(r'C:\Users\apmel\OneDrive\IDEs\Anaconda-Files-Python\myPractice\housing_dataset\housing.csv')
df.head()
#Shuffling the dataset
df = df.sample(n=len(df), random_state=1)
df.head()

#Initial data visualization
df.shape
df.info()
df.isnull().sum()
df.describe()
#df.columns.values

#Plotting histograms for all numerical values
df.hist(bins=50, figsize=(18, 9))

#Checking for value_counts()
#df['median_house_value'].value_counts() #Not necessary, since it is not a classification problem
df['ocean_proximity'].value_counts()
df['ocean_proximity'].value_counts().plot(kind='barh', figsize=(10,7))

#Separating the features from the target
X = df.drop('median_house_value', axis=1).copy()
y = df['median_house_value'].copy()


#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train
y_train

#Looking for correlations usign the standard correlation coefficient that goes from -1 to 1
corr_matrix = df.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

###Preparing the data for machine learning###

#Separating numerical features from categorical features
X_train_num = X_train.drop('ocean_proximity', axis=1)
X_train_cat = X_train[['ocean_proximity']]
X_train_num.head()
X_train_cat.head()

#Feature scaling and transformations pipelines
#Applying a pipeline for numerical and categorical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

#Now the full pipeline adding categorical dta with the numerical pipeline
num_attibutes = list(X_train_num)
cat_attributes = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attibutes),
    ('cat', OneHotEncoder(), cat_attributes)
    ])

#Fitting the trainning data into the pipeline
X_train_prepared = full_pipeline.fit_transform(X_train)

#Building a Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

#Evaluating the linear regression model with RMSE
y_train_lin_pred = lin_reg.predict(X_train_prepared)
lin_mse = mean_squared_error(y_train, y_train_lin_pred)
lin_mse = np.sqrt(lin_mse)
lin_mse

#Building a DecisionTreeRegression model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_prepared, y_train)

#Evaluating the DecisionTreeRegression model
y_train_tree_pred = tree_reg.predict(X_train_prepared)
tree_mse = mean_squared_error(y_train, y_train_tree_pred)
tree_mse = np.sqrt(tree_mse)
tree_mse














#Numerical features: Filling the missing values with the median
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train_num)
array = imputer.transform(X_train_num)
X_train_num_tr = pd.DataFrame(array, columns=X_train_num.columns)
X_train_num_tr.head()

X_train_num_tr.isnull().sum()


X_test_num = X_test.drop('ocean_proximity', axis=1)
imputer.fit(X_test_num)
array = imputer.transform(X_test_num)
X_test_num_tr = pd.DataFrame(array, columns=X_test_num.columns)
X_test_num_tr.head()



X_test_num.isnull().sum() ####### Go back on this!!!
X_test_num_tr.isnull().sum()





#Categorical features: Applying OneHotEncored
cat_encoder = OneHotEncoder()

X_train_cat_1hot = cat_encoder.fit_transform(X_train_cat)
type(X_train_cat_1hot) #SciPy sparse matrix type
X_train_cat_1hot.toarray() #converting it to a Numpy array










#Creating a pipeline to handle numerical values filling the missing values with the median and applying the StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])




















