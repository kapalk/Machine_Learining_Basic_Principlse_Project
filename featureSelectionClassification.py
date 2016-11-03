from sklearn.feature_selection import f_classif
import numpy as np

def featureSelectionClassification(X,y,k):
    '''
    -computes ANOVA to each feauture (X) and target (y)
    
    -returns k indices of features of highest p-values
    '''
    F, p_values = f_classif(X,y)
    bestFeaturesIdxs = p_values.argsort()[-k:][::-1]
    return bestFeaturesIdxs

