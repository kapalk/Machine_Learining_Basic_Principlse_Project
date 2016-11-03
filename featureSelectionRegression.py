from sklearn.feature_selection import f_regression
import numpy as np

def featureSelectionRegression(X,y,k):
    '''
    -calculates cross correlataion between each feature (X) and target (y)
    
    -converts correlation to F score and then to p-value
    
    -selects k features with highest p-value and return their correspongin indices
    '''
    F, p_values = f_regression(X,y)
    bestFeaturesIdxs = p_values.argsort()[-k:][::-1]
    return bestFeaturesIdxs


    


