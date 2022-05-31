import pandas
import numpy
import seaborn

import my_functions

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from collections import OrderedDict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Loading data

ecg = my_functions.loadData()
print(ecg.shape)
# Make a version with no NaNs because Seaborn doesn't like them
ecgNoMissing = ecg.fillna(0)
seaborn.pairplot(ecgNoMissing)

# print(ecg.isnull().sum())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Replace missing values
for columnName in ecg.keys():
  # print(columnName)
  median = ecg[columnName].median()
  ecg[columnName] = ecg[columnName].fillna(median)

# print(ecg.isnull().sum())
# print(ecg.shape)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Inputs and outputs, feature normalization

# Make a copy of the data
features = ecg.copy()
# pop off the regression target
target = features.pop('survival')
# Change from months to years
target = target 

# print(features.shape)
# print(target.shape)

# Normalize features by mean and stadard deviation
for columnName in features.keys():
  mean = features[columnName].mean()
  std = features[columnName].std()
  features[columnName] = (features[columnName] - mean) / std
  
seaborn.pairplot(features)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Train/test split

# Split off 20% of data for testing later

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.20)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Model tranning 

# Linear Regression

lr = LinearRegression()

my_functions.trainShow(lr, X_train, Y_train)

my_functions.testShow("Linear Regression", lr, X_test, Y_test)

# Polynomial Regression

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

param_grid = {
    'polynomialfeatures__degree': numpy.arange(10), 
    'linearregression__fit_intercept': [True, False], 
    'linearregression__normalize': [True, False]}

pr = GridSearchCV(PolynomialRegression(), param_grid, cv=10, scoring='neg_mean_squared_error')

my_functions.trainShow(pr, X_train, Y_train)

my_functions.testShow("Polynomial Regression", pr, X_test, Y_test)

# Decision Three Regression

parameters={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            "max_features":["auto","log2","sqrt"],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }

dtr = GridSearchCV(DecisionTreeRegressor(), param_grid=parameters, scoring='neg_mean_squared_error', cv=3, verbose=3)

my_functions.trainShow(dtr, X_train, Y_train)

my_functions.testShow("Decision Tree Regressor", dtr, X_test, Y_test)

# Random Forest Regression

rfr = GridSearchCV(
                    RandomForestRegressor(), cv=5, error_score=numpy.nan,
                    # dictionaries containing values to try for the parameters
                    param_grid={
                        'max_depth'   : [ 2,  5,  7, 10],
                        'n_estimators': [20, 30, 50, 75]
                        }
                    )

my_functions.trainShow(rfr, X_train, Y_train)

my_functions.testShow("Random Forest Regressor", rfr, X_test, Y_test)

Support Vector Regression

svr = GridSearchCV(
                    SVR(),
                    cv=5,
                    param_grid={
                      "kernel":["rbf", "linear"],
                      "C": [0.2, 0.5, 1],
                      "gamma": [0.1, 0.3, 0.5],
                      "epsilon": numpy.logspace(-6, 0, 10)
                      }
                    )
    
my_functions.trainShow(svr, X_train, Y_train)

my_functions.testShow("Support Vector Regression ", svr, X_test, Y_test)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Test MSE
# MNE - svako odsupanje na kvadrat, zbog toga radim koren da dobijem apsolutno prosecno odstupanje modela

models = OrderedDict()
models["lr"] = lr
models["pr"] = pr
models["dtr"] = dtr
models["rfr"] = rfr
models["svr"] = svr

for name, model in models.items():
  print("mse: %s" % (my_functions.mse(model, X_test, Y_test)))