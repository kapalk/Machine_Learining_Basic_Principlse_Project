import numpy as np

def fit(X,y,alpha):
    '''returns weigth array which miminizes ridge regression error
    
    Args:
        - X: A numpy array of features
        - Y: A numpy array of targets
        - alpha: A scalar scaling coefficient
    Returns:
        - theta_opt: A numpy array which minimizes regression error
    '''
    ones = np.ones(len(X))
    XandBias = np.column_stack((ones,X))
    Xt = np.transpose(XandBias)
    lambdaIdentity = alpha*np.identity(len(Xt))
    inverse = np.linalg.pinv(Xt.dot(XandBias)+lambdaIdentity)
    theta_opt = np.dot(np.dot(inverse, Xt), y)
    return theta_opt

def predict(X,theta):
    '''return prediction
    
    Args:
        - X: A numpy array of features
        - theta: optimal weigth vector
    Returns:
        - prediction: A numpy array of prediction
    '''
    ones = np.ones(len(X))
    XandBias = np.column_stack((ones,X))
    return XandBias.dot(theta)
