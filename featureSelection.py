from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
import numpy as np

def forRegression(X,y,k):
    '''
    -calculates cross correlataion between each feature (X) and target (y)
    
    -converts correlation to F score and then to p-value
    
    -selects k features with lowest p-values and return their corresponging indices
    '''
    F, p_values = f_regression(X,y)
    bestFeaturesIdxs = p_values.argsort()[0:k]
    return bestFeaturesIdxs


def forClassification(X,y,k):
    '''
    -computes ANOVA to each feauture (X) and target (y)
    
    -returns k indices of features with lowest p-values
    '''
    F, p_values = f_classif(X,y)
    bestFeaturesIdxs = p_values.argsort()[0:k]
    return bestFeaturesIdxs


    


