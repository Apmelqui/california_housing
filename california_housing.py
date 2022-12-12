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


#### 1 - Loading the dataframe and getting a big picture of it ####
df = pd.read_csv(r'C:\Users\apmel\OneDrive\IDEs\Anaconda-Files-Python\myPractice\housing_dataset\housing.csv')
df.head()

#Shuffling the dataframe
#df = df.sample(frac=1)
#df = df.sample(n=len(df), random_state=1)

#Initial data visualization
df.shape
df.info()
df.isnull().sum()
df.describe()

#Plotting histograms for all numerical values
df.hist(bins=50, figsize=(18, 9))

#Median income should be multiplied by 10,000 -> eg. 3 means $30,000
fig = plt.figure(figsize=(15, 5))
sns.histplot(data=df, x='median_income')
#df['median_income'].hist(bins=50)

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

#### 2 - Spliting the dataframe into train and test set for deeper analysys (on the train data) ####

#Using the train_test_split to split the data and stratify it
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['income_cat'])

#Confirming the proportions of the income_cat in both train_set and main dataset after the split
test_set["income_cat"].value_counts() / len(test_set)
df["income_cat"].value_counts() / len(df) #confirmed

#Now removing the created 'income_cat' column from test_set and train_set
for column in (train_set, test_set):
    column.drop('income_cat', axis=1, inplace=True)

#More data visualization, now with the train_set 
fig = plt.figure(figsize=(10, 7))
sns.scatterplot(train_set, x='longitude', y='latitude', alpha=0.1)
#train_set.plot(kind='scatter', x='longitude', y='latitude')

#Plotting where s represents the district's population and c represents the price. 
#The yellow represents a higher price and light purple a low price
train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=train_set["population"]/100, label="population", figsize=(10,7),
             c="median_house_value")

#Creating a copy of the train_set to test combined features
train_set_comb_feat = train_set.copy()

train_set_comb_feat['room_per_household'] = train_set_comb_feat['total_rooms'] / train_set_comb_feat['households']
train_set_comb_feat['bedroom_per_room'] = train_set_comb_feat['total_bedrooms'] / train_set_comb_feat['total_rooms']
train_set_comb_feat['population_per_household'] = train_set_comb_feat['population'] / train_set_comb_feat['households']

#Now let's look for the correlation matrix again:
corr_matrix = train_set_comb_feat.corr(numeric_only=True)
corr_matrix['median_house_value'].sort_values(ascending=False)

#Inserting 'room_per_household' and 'population_per_household' in both train and test data. Discarding bedroom_per_room
for set in train_set, test_set:
    set['room_per_household'] = set['total_rooms'] / set['households']
    set['population_per_household'] = set['population'] / set['households']

#Splitting the target from the features in the train set
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']

####3 - Data cleaning ####

#Create a pipeline for dealing numerical data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

#Separating numerical and categorical features and storing them in a list
num_features = X_train.select_dtypes(exclude='object').columns
cat_features = X_train.select_dtypes(include='object').columns
#cat_attribs = ['ocean_proximity']

#Applying ColumnTransformer with the previous pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', OneHotEncoder(), cat_features)
    ])

X_train_prepared = full_pipeline.fit_transform(X_train)


#### 4 -Training the models ####

#Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

#Function to compare the predicted values with the real ones
def compare(model, X_data, y_data):
    y_true_5 = y_data.iloc[:5]
    y_pred = model.predict(X_data[:5])
    comparision_dataframe = pd.DataFrame(data={'True values': y_true_5.round(2), 'Predicted values': y_pred.round(2)})
    comparision_dataframe['Difference'] = comparision_dataframe['True values'] - comparision_dataframe['Predicted values']
    
    print(comparision_dataframe) 
    
compare(lin_reg, X_train_prepared, y_train)

#Applying RMSE (Root Mean Squared Error) to evaluate the linear regression model on the train data
y_train_prepared_lin_reg_pred = lin_reg.predict(X_train_prepared)
lin_rmse_train = mean_squared_error(y_train, y_train_prepared_lin_reg_pred, squared=False)
lin_rmse_train #68,911
#print("{:,.2f}".format(lin_rmse_train)) #68,911

#Better evaluating using cross-validation
scores = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)
lin_reg_scores = np.sqrt(-scores)
lin_reg_scores
mean_lin_reg_scores = lin_reg_scores.mean()
mean_lin_reg_scores #69,153
std_deviation_lin_reg = lin_reg_scores.std()
std_deviation_lin_reg

#DecisionTreeRegressor model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_prepared, y_train)

compare(tree_reg, X_train_prepared, y_train) # Predicted the values 100% correct???

#Applying RMSE (Root Mean Squared Error) to evaluate the DecisionTreeRegressor model on the train data
y_train_prepared_tree_reg_pred = tree_reg.predict(X_train_prepared)
tree_rmse_train = mean_squared_error(y_train, y_train_prepared_tree_reg_pred, squared=False)
tree_rmse_train #0 - Overfitting!!

#Better evaluating using cross-validation
scores = cross_val_score(tree_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)
tree_reg_scores = np.sqrt(-scores)
tree_reg_scores
mean_tree_reg_scores = tree_reg_scores.mean()
mean_tree_reg_scores #70,731 - Performing worse than linear regression
std_deviation_tree_reg = tree_reg_scores.std()
std_deviation_tree_reg

#RandomForestRegressor model
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_prepared, y_train)

compare(forest_reg, X_train_prepared, y_train)

#Applying RMSE (Root Mean Squared Error) to evaluate the DecisionTreeRegressor model on the train data
y_train_prepared_forest_reg_pred = forest_reg.predict(X_train_prepared)
forest_rmse_train = mean_squared_error(y_train, y_train_prepared_forest_reg_pred, squared=False)
forest_rmse_train #18,686

#Better evaluating using cross-validation
scores = cross_val_score(forest_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)
forest_reg_scores = np.sqrt(-scores)
forest_reg_scores
mean_forest_reg_scores = forest_reg_scores.mean()
mean_forest_reg_scores #50,329 - Better performance till now
std_deviation_forest_reg = forest_reg_scores.std()
std_deviation_forest_reg

#### 5 - Fine tunning the RandomForestRegressor model ####
param_grid = [
    {'n_estimators': [3, 10, 30, 40, 50], 'max_features': [2, 4, 6, 8, 10]}, # 5 x 5 = 25 combinations
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]} # 2 x 3 = 6 combinations
    ] #total of 25 + 6 = 31 combinations

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, 
                           cv=5, 
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)

grid_search.fit(X_train_prepared, y_train)

grid_search.best_params_ #{'max_features': 6, 'n_estimators': 40}
grid_search.best_estimator_
grid_search.cv_results_

#Evaluating scores for each iteration
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(f'{np.sqrt(-mean_score):.3f} for {params}') #49,435 - The result was improved!

## Evaluating the system on the test set ##
#Creating the best model with the best estimators from the GridSearch
final_model = grid_search.best_estimator_

#Spliting the features from the target
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value'].copy()

#Transforming the test data with the pipeline
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 
final_rmse #47,044

compare(final_model, X_test_prepared, y_test)

#### 6 - Saving the best model ####
joblib.dump(final_model, 'final_model.pkl')
#forest_reg_model_loaded = joblib.load('final_model.pkl')




