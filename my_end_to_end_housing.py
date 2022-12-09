# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:58:55 2022

@author: apmel
"""

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

#Loading the dataframe
df = pd.read_csv(r'C:\Users\apmel\OneDrive\IDEs\Anaconda-Files-Python\myPractice\housing_dataset\housing.csv')
df.head()

### Initial data visualization ###
df.shape
df.info()
df.isnull().sum()
df.describe()
#df.columns.values

#Plotting histograms for all numerical values
df.hist(bins=50, figsize=(18, 9))

#Median income should be multiplied by 10,000 -> eg. 3 means $30,000
fig = plt.figure(figsize=(15, 5))
sns.histplot(data=df, x='median_income')
# df['median_income'].hist(bins=50)

#Checking for value_counts() on the categorical feature
#df['median_house_value'].value_counts() #Not necessary, since it is not a classification problem
df['ocean_proximity'].value_counts()
fig = plt.figure(figsize=(15, 5))
df['ocean_proximity'].value_counts().plot(kind='barh', figsize=(10,7))

#Looking for correlations usign the standard correlation coefficient that goes from -1 to 1
corr_matrix = df.corr() #Creating a dataframe with the correlations
corr_matrix['median_house_value'].sort_values(ascending=False)

#Now looking for correlations usign scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(df[attributes], figsize=(12, 8))

#Looks like the median_income is a promissing attribute to predict the median_house_value
#Plotting only the scatter_matrix with the median_income
fig = plt.figure(figsize=(12, 8))
sns.scatterplot(df, x='median_income', y='median_house_value')
#df.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

#Stratifying the data considering that the median_income has a great impact on the target (median_house_income)
#Creating categories. First from 0 to 1.5, second from 1.5 to 3 and so one.
df['income_cat'] = pd.cut(df['median_income'],
                          bins=[0, 1.5, 3, 4.5, 6, np.inf],
                          labels=[1, 2, 3 ,4, 5])
df['income_cat'].value_counts()

fig = plt.figure(figsize=(10, 7))
sns.histplot(df['income_cat'])
#df['income_cat'].hist()

#Separating features from the target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

### Cleaning the data ###

#Creating a pipeline for dealing the numerical data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

#Separating numerical features from categorical features and storing in a list
num_attribs = X.select_dtypes(exclude='object').columns
cat_attribs = X.select_dtypes(include='object').columns

#Handling numerical and categorical data together in another pipeline with Columtransformer
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
    ])

#Fitting the pipeline
X_prepared = full_pipeline.fit_transform(X)

#Converting it again to a DataFrame
X_prepared = pd.DataFrame(X_prepared, columns=X.columns)

#Stratifying the data considering that the median_income has a great impact on the target (median_house_income)
#Creating categories. First from 0 to 1.5, second from 1.5 to 3 and so one.
X_prepared['income_cat'] = pd.cut(X_prepared['median_income'],
                          bins=[0, 1.5, 3, 4.5, 6, np.inf],
                          labels=[1, 2, 3 ,4, 5])
X_prepared['income_cat'].value_counts()

#Using the train_test_split to split the data and stratify it using income_cat feature
X_train_prepared, X_test_prepared, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42, stratify=df['income_cat'])

#Cheking the result comparing with the main dataset
X_train_prepared["income_cat"].value_counts() / len(X_train_prepared)
df["income_cat"].value_counts() / len(df)








X_train_prepared, X_test_prepared, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42, shuffle=True)


strat_train_set, strat_test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['income_cat'])

X_train_prepared, X_test_prepared, y_train, y_test
































