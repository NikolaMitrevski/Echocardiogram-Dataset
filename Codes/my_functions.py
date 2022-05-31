import pandas

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt

dataFile = "echocardiogram_data.txt"
nameOfColumns = ['survival', 'still_alive', 'age_at_heart_attack', 'pericardial_effusion', 'fractional_shortening', 'epss', 'lvdd', 'wall_motion_score', 'wall_motion_index', 'mult', 'name', 'group', 'alive_at_1']

# Function for loading dataset

def loadData():
    # Loading dataset from file
    ecg = pandas.read_csv(dataFile, na_values=["?"], error_bad_lines=False, names=nameOfColumns)
    
    # Rows elimination  
    ecg = ecg[ecg.survival.notnull()]
    ecg = ecg[ecg.still_alive.notnull()]
    ecg = ecg[ecg.still_alive == 0]
    
    # Delete columns
    del ecg['name']
    del ecg['group']
    del ecg['mult']
    del ecg['still_alive']
    del ecg['alive_at_1']
    return ecg

# Function for traning model

def trainShow(model, trainX, trainY):
  # Fit the model.  Everything from scikit implements fit 
  model.fit(trainX, trainY)
  
  try:
      # In the case of GridSearchCV, we can show the best parameters
      print("Best Parameters: ", model.best_params_)
      print("Best Score: ", model.best_score_)
  except:
      print("Best params not implemented for model: %s" % type(model))
  
# Function for test model

def testShow(name, model, testX, testY):
    predY = model.predict(testX)
    
    # plt.title(name)
    # plt.scatter(testY, predY)
    # plt.xlabel("True Survival")                          
    # plt.ylabel("Predicted Survival")
    # plt.show()
    
    # print(len(testY))
    # print(len(predY))
    
    plt.title(name)
    plt.scatter(range(len(testY)), testY, color='blue', label="True Survival")
    plt.scatter(range(len(predY)), predY, color='red', label="Predicted Survival")
    plt.show()
    
    print('Predicted Value: ', predY[3])
    print('Actual Value: ', testY.values[3])
          
# Function for calculating MNE

def mse(model, testX, testY):
    predY = model.predict(testX)
    
    r2 = r2_score(testY, predY) # koliko izlaz zavisi od ulaza
    print("r2_score: ", r2)
    
    return mean_squared_error(testY, predY)
