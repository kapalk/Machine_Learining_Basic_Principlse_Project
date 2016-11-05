import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as mse

import featureSelection
import multivariateRidgeRegressor


#load data
trainingDF = pd.read_csv('/Users/kasperipalkama/Downloads/regression_dataset_training.csv')
testingDF = pd.read_csv('/Users/kasperipalkama/Downloads/regression_dataset_testing.csv')
solutionDF = pd.read_csv('/Users/kasperipalkama/Downloads/regression_dataset_testing_solution.csv')

# transform dataframes to numpy arrays, get rid of index column and
# separates features and targets to different arrays
X_training = np.asarray(trainingDF)[:,1:51]
y_training = np.asarray(trainingDF)[:,51]
X_testing = np.asarray(testingDF)[:,1:]
y_testing = np.asarray(solutionDF)[:,1]

validated_error = np.empty(50)
for k in range(1,51):
    # feauture selelection: how many features taken into account gives the smallest validation error
    bestFeatureIdxs = featureSelection.forRegression(X_training,y_training,k)
    X_training_best_selected_features = X_training[:,bestFeatureIdxs]
    #adds bias term
    ones = np.ones(len(X_training_best_selected_features))
    X_training_best_selected_features = np.column_stack((ones,X_training_best_selected_features))
    
    error = 0
    #cross validation
    foldCount = 10
    kf = KFold(len(X_training_best_selected_features), n_folds = foldCount)
    for train_cv, validation_cv in kf:
        X_train, X_validation = X_training_best_selected_features[train_cv,:], X_training_best_selected_features[validation_cv,:]
        y_train, y_validate = y_training[train_cv], y_training[validation_cv]
        # gets optimized theta
        theta_opt= multivariateRidgeRegressor.fit(X_train,y_train,1)
        # calculates validation error on each k
        error += mse(y_validate,X_validation.dot(theta_opt))
    validated_error[k-1] = error/foldCount

k_optimal = 1 + np.argmin(validated_error)
optimalFeatures = featureSelection.forRegression(X_training,y_training,k_optimal)

#final training data
X_training_final = X_training[:,optimalFeatures]
#add bias term
ones = np.ones(len(X_training_final))
X_training_final = np.column_stack((ones,X_training_final))
# final predictor
optimal_alpha = 1
finalPredictor = multivariateRidgeRegressor.fit(X_training_final,y_training,optimal_alpha)
#test error
#add bias term
X_testing = X_testing[:,optimalFeatures]
ones = np.ones(len(X_testing))
X_testing = np.column_stack((ones,X_testing))
testError = mse(y_testing,X_testing.dot(finalPredictor))
print(testError)
 
