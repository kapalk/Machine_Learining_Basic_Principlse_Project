import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Ridge

import featureSelection
import multivariateRidgeRegressor as RR


def getData():
    trainData = np.loadtxt('data/regression_dataset_training.csv',
                           delimiter = ',',skiprows = 1)[:,1:]
    testData = np.loadtxt('data/regression_dataset_testing.csv',
                          delimiter = ',',skiprows = 1)[:,1:]
    solutionData = np.loadtxt('data/regression_dataset_'\
                              'testing_solution.csv',delimiter = ',',
                              skiprows = 1)[:,1:]
    return trainData, testData, solutionData

def getFeaturesAndTargets(M):
    '''
    separates features and targets from numpy array
    '''
    return M[:,0:M.shape[1]-1], M[:,-1]

def kFoldsValidationForRR(featureArr,targetArr,foldCount,alpha=1):
    """does kfolds cross validation for ridge regression.

    Args:
        featureArr: A numpy arrays containing features.
        targetArr: A numpy array containing targets.
        foldCount: A number of folds
    Returns:
        validation error
    """
    error = 0
    kf = KFold(n_splits = foldCount)
    for train_cv, valid_cv in kf.split(featureArr):
        featTrain,  = featureArr[train_cv,:],
        featValid = featureArr[valid_cv,:]
        targTrain, targValid = targetArr[train_cv], targetArr[valid_cv]
        theta_opt= RR.fit(featTrain,targTrain,alpha)
        prediction = RR.predict(featValid,theta_opt)
        error += mse(targValid,prediction)
    return error/foldCount

def getOptimalFeatures(FeatureArr,TargetArr):
    """validates features used for learning.

    Args:
        FeatureArr: A numpy arrays containing features.
        TargetArr: A numpy array containing targets.
    Returns:
        optimalFeatures: A numpy array of selected features

    """
    shape = FeatureArr.shape
    errorMat = np.empty(shape[1])
    for idx,featureCount in enumerate(range(1,shape[1]+1)):
        bestFeatureIdxs = featureSelection.forRegression(FeatureArr,
                                                         TargetArr,
                                                         featureCount)
        selectedFeatures = FeatureArr[:,bestFeatureIdxs]
        errorMat[idx] = kFoldsValidationForRR(selectedFeatures,
                                        TargetArr,
                                        foldCount = 5)        
    optimalFeatCount = np.argmin(errorMat) + 1
    optimalFeaturesIdxs = featureSelection.forRegression(FeatureArr,
                                                     TargetArr,
                                                     optimalFeatCount)
    return optimalFeaturesIdxs

def getOptimalAlpha(featureArr,targetArr,alphaRange):
    """validates optimal scaling coefficinet for regulation.

    Args:
        featureArr: A numpy arrays containing features.
        targetArr: A numpy array containing targets.
    Returns:
        OptimalAlpha: An optimal scaling coefficent for regulation

    """
    alphas = np.arange(alphaRange[0],alphaRange[1],0.01)
    errorMat = np.empty(len(alphas))
    for idx,alpha in enumerate(alphas):
        errorMat[idx] = kFoldsValidationForRR(featureArr,
                                        targetArr,
                                        foldCount=5,
                                        alpha=alpha)
        
    optimalAlpha = alphas[np.argmin(errorMat)]
    return optimalAlpha

if __name__ == '__main__':
    trainData, testData, solutionData = getData()
    trainFeatures, trainTargets = getFeaturesAndTargets(trainData)
    optimalFeaturesIdxs = getOptimalFeatures(trainFeatures,trainTargets)
    alpha = getOptimalAlpha(trainFeatures[optimalFeaturesIdxs],
                            trainTargets,alphaRange=[0,5])
    theta = RR.fit(trainFeatures[:,optimalFeaturesIdxs],trainTargets,alpha)
    prediction = RR.predict(testData[:,optimalFeaturesIdxs],theta)
    testError = mse(solutionData,prediction)
    print(testError)

