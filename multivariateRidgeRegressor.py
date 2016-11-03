import numpy as np

def multivariateRidgeRegression(X,y,alpha):
    '''
    - NOTICE! remember to add bias term to X
    
    - X = features, y = targets, alpha = regulation scaling coefficient
    
    - calculates vector which minimizes the error function of multivariate regression
    
    - regulates complexity using L_2 norma as regulation parameter
    
    - returns optimal regression vector
    
    '''
    Xt = np.transpose(X)
    lambdaIdentity = lam*np.identity(len(Xt))
    inverse = np.linalg.inv(Xt.dot(X)+lambdaIdentity)
    w_opt = np.dot(np.dot(inverse, Xt), y)
    return w_opt



