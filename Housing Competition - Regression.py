from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

#Import Library
#Import other necessary libraries like pandas, numpy
from sklearn import linear_model

#Load Train and Test datasets
iowa_file_path = r'C:\Users\Mike\Anaconda3\Kaggle\Housing Competition\train.csv'
train_data = pd.read_csv(iowa_file_path)

test_data_path = r'C:\Users\Mike\Anaconda3\Kaggle\Housing Competition\test.csv'
test_data = pd.read_csv(test_data_path)



#Identify feature and response variable(s) and values must be numeric and numpy arrays

# remove string objects
for col in train_data:
    if train_data[col].dtype == object:
        train_data.drop([col],axis=1, inplace=True)
    elif train_data[col].dtype == np.float64:
            train_data[col].astype = np.int64
            
for col in test_data:
    if test_data[col].dtype == object:
        test_data.drop([col],axis=1, inplace=True)
    elif test_data[col].dtype == np.float64:
            test_data[col].astype = np.int64

# check for NaN
for col in train_data:
    for index, row in train_data.iterrows():
        if np.isnan(row[col]) == True:
            train_data.drop([col], axis=1, inplace=True)
            test_data.drop([col], axis=1, inplace=True)
            break

# check for NaN
for col in test_data:
    for index, row in test_data.iterrows():
        if pd.isnull(row[col]) == True:
            test_data.drop([col], axis=1, inplace=True)
            train_data.drop([col], axis=1, inplace=True)
            break            
            
for col in test_data:
    for index, row in test_data.iterrows():
        if pd.isnull(row[col]) == True:
            print(col,row,index)
            test_data.drop([col], axis=1, inplace=True)
            break
#test_data = test_data.dropna()
        
print(test_data.dtypes)
print(train_data.dtypes)

y_train=train_data.SalePrice
x_train = train_data
features = x_train
features = features.drop(['SalePrice'], axis=1, inplace=True)
x_train = sm.add_constant(x_train) ## let's add an intercept (beta_0) to our model

x_test=test_data

x_test = sm.add_constant(x_test) ## let's add an intercept (beta_0) to our model

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
score = linear.score(x_train, y_train)
print('R^2: \n', score)

#Equation coefficient and Intercept
print('Coefficient: \n',linear.coef_)

print('Intercept: \n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)
print(predicted)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predicted})
output.to_csv(r'C:\Users\Mike\Anaconda3\Kaggle\Housing Competition\submission_reg2.csv', index=False)