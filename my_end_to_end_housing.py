# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:55:59 2022

@author: apmel
"""


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

#Loading the dataframe
df = pd.read_csv(r'C:\Users\apmel\OneDrive\IDEs\Anaconda-Files-Python\myPractice\housing_dataset\housing.csv')
df.head()

#Shuffling the dataframe
df = df.sample(frac=1)
#df = df.sample(n=len(df), random_state=1)
df.head()

#Initial data visualization
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

# fig = plt.figure(figsize=(15, 5))
# df['median_income'].hist(bins=50)

#Checking for value_counts()
#df['median_house_value'].value_counts() #Not necessary, since it is not a classification problem
df['ocean_proximity'].value_counts()
df['ocean_proximity'].value_counts().plot(kind='barh', figsize=(10,7))

#Looking for correlations usign the standard correlation coefficient that goes from -1 to 1
corr_matrix = df.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

#Now looking for correlations usign scatter_matrix
from pandas.plotting import scatter_matrix

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
sns.histplot(df['income_cat'])
#df['income_cat'].hist()

#Using the train_test_split to split the data and stratify it
strat_train_set, strat_test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['income_cat'])

#Cheking the result comparing with the main dataset
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
df["income_cat"].value_counts() / len(df)

#Now removing the 'income_cat' column from strat_test_set and strat_train_set
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


#More data visualization, now with the strat_train_set dataframe
housing = strat_train_set.copy()


fig = plt.figure(figsize=(10, 7))
sns.scatterplot(housing, x='longitude', y='latitude', alpha=0.1)
#housing.plot(kind='scatter', x='longitude', y='latitude')

#Plotting where s represents the district's population and c represents the price. 
#The yellow represents a higher price and light purple a low price
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value")

#Combining attributes
housing['room_per_household'] = housing['total_rooms'] / housing['households']
housing['bedroom_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

#Now let's look for the correlation matrix again:
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

############################################
#continue from here#
############################################

##Preparing for ML##
#Separating the features from the target
X = df.drop('median_house_value', axis=1).copy()
y = df['median_house_value'].copy()

#Separating numerical features from categorical features
X_num = X.drop('ocean_proximity', axis=1)
X_cat = X['ocean_proximity']

num_features_list = X.select_dtypes(exclude='object').columns
cat_features_list = X.select_dtypes(include='object').columns

#Feature scaling and transformations pipelines - numerical and categorical features
#Applying a pipeline for numerical features:
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

#Now adding the categorical features
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features_list),
    ('cat', OneHotEncoder(), cat_features_list)
    ])

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train
y_train

#Fitting the trainning data into the pipeline
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train
X_train_prepared

#Fitting the test data into the pipeline
X_test_prepared = full_pipeline.fit_transform(X_test)
X_test
X_test_prepared

#Building a Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

#Better evaluating using cross-validation
scores = cross_val_score(lin_reg, X_train_prepared, y_train, cv=5)
mean_scores = scores.mean()
scores
mean_scores

##Evaluating the linear regression model with RMSE
#Evaluating the train data with Root Mean Squared Error
y_train_prepared_lin_pred = lin_reg.predict(X_train_prepared)
lin_rmse_train = mean_squared_error(y_train, y_train_prepared_lin_pred, squared=False)
lin_rmse_train

#Evaluating the test data with Root Mean Squared Error
y_test_lin_pred = lin_reg.predict(X_test_prepared)
lin_rmse_test = mean_squared_error(y_test, y_test_lin_pred, squared=False)
lin_rmse_test


#Function to compare the predicted values with the real ones
def compare(model, test_data, y_true):
    y_true_5 = y_true[:5]
    y_pred = model.predict(test_data[:5])
    comparision_dataframe = pd.DataFrame(data={'True values': y_true_5, 'Predicted values': y_pred})
    comparision_dataframe['Difference'] = comparision_dataframe['True values'] - comparision_dataframe['Predicted values']
    print(comparision_dataframe)    
    
compare(lin_reg, X_test_prepared, y)

#Building a DecisionTreeRegression model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_prepared, y_train)

#Better evaluating using cross-validation
scores = cross_val_score(tree_reg, X_train_prepared, y_train, cv=5)
mean_scores = scores.mean()
scores
mean_scores

#Evaluating the DecisionTreeRegression model
#Evaluating the train data with Root Mean Squared Error
y_train_prepared_tree_pred = tree_reg.predict(X_train_prepared)
tree_rmse_train = mean_squared_error(y_train, y_train_prepared_tree_pred, squared=False)
tree_rmse_train #Overfitting the data

#Evaluating the test data with Root Mean Squared Error
y_test_prepared_tree_pred = tree_reg.predict(X_test_prepared)
tree_rmse_test = mean_squared_error(y_test, y_test_prepared_tree_pred, squared=False)
tree_rmse_test

#Building a RandomForestRegressor model
rnd_forest = RandomForestRegressor(max_depth=10)
rnd_forest.fit(X_train_prepared, y_train)

#Better evaluating using cross-validation
scores = cross_val_score(rnd_forest, X_train_prepared, y_train, cv=5)
mean_scores = scores.mean()
scores
mean_scores

#Evaluating the train data with Root Mean Squared Error
y_train_prepared_rnd_pred = rnd_forest.predict(X_train_prepared)
rnd_rmse_train = mean_squared_error(y_train, y_train_prepared_rnd_pred, squared=False)
rnd_rmse_train

#Evaluating the test data with Root Mean Squared Error
y_test_prepared_rnd_pred = rnd_forest.predict(X_test_prepared)
rnd_rmse_test = mean_squared_error(y_test, y_test_prepared_rnd_pred, squared=False)
rnd_rmse_test

#Building a GradientBoostingRegressor model
gbr_model = GradientBoostingRegressor(n_estimators=250)
gbr_model.fit(X_train_prepared, y_train)

#Better evaluating using cross-validation
scores = cross_val_score(gbr_model, X_train_prepared, y_train, cv=5)
mean_scores = scores.mean()
scores
mean_scores

#Evaluating the train data with Root Mean Squared Error
y_train_prepared_gbr_pred = gbr_model.predict(X_train_prepared)
gbr_rmse_train = mean_squared_error(y_train, y_train_prepared_gbr_pred, squared=False)
gbr_rmse_train

#Evaluating the test data with Root Mean Squared Error
y_test_prepared_gbr_pred = gbr_model.predict(X_test_prepared)
gbr_rmse_test = mean_squared_error(y_test, y_test_prepared_gbr_pred, squared=False)
gbr_rmse_test

#Fine tunning the model
#Using the RandomForestRegressor model 
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8], 'max_depth': [1, 10, 20]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]


rnd_forest = RandomForestRegressor()
grid_search = GridSearchCV(rnd_forest, param_grid, cv=5)
grid_search.fit(X_train_prepared, y_train)

grid_search.best_params_
grid_search.best_estimator_

best_RandomForestRegressor_model = grid_search.best_estimator_

final_RandomForestRegressor_predictions = best_RandomForestRegressor_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_RandomForestRegressor_predictions, squared=False)
final_mse



#Using the GradientBoostingRegressor model 
#get back on this...
param_grid = {
    "n_estimators": [1, 100, 250, 500],
    "max_leaf_nodes": [2, 10, 100],
    "learning_rate": [0.01, 0.1, 1],
    }

gbr_model = GradientBoostingRegressor()
grid_search = GridSearchCV(gbr_model, param_grid, cv=5)
grid_search.fit(X_train_prepared, y_train)

grid_search.best_params_
grid_search.best_estimator_

best_GradientBoostingRegressor_model = grid_search.best_estimator_

best_GradientBoostingRegressor_model = best_RandomForestRegressor_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, best_GradientBoostingRegressor_model, squared=False)
final_mse

