import numpy as np
import pandas as pd
from scipy.optimize import minimize, fmin_bfgs
from sklearn.datasets import load_iris

class LogisticRegression():

    def __init__(self):
        self.fitted = False
        self.predicted = False
        self.theta = None

    def fit(self, X, y):
        theta = np.zeros((X.shape[1] + 1, 1))
        features = np.ones((X.shape[0], X.shape[1] + 1))
        features[:,1:] = X
        labels = np.squeeze(y)
        self.theta = fmin_bfgs(self.cost, theta, fprime=self.gradient, args=(features, labels))
        self.fitted = True
        print('Prediction Model Created.')
        return self.theta

    def predict(self, X):
        if self.fitted:
            data = np.ones((X.shape[0], X.shape[1] + 1))
            data[:,1:] = X
            prob = self.sigmoid(self.theta, data)
            pred = np.where(prob >= 0.5, 1, 0)
            self.prediction = pred
            self.predicted = True
            return self.prediction
        else:
            print('Model has not been fitted. Cannot predict. Aborting.')

    def score(self):
        """Estimated performance score for Learner.""" ##only returns how many it got right...
        if self.predicted:
            print('Correctly predicted items: %d/%d.' % (np.sum(self.labels == self.prediction), len(self.labels)))
        else:
            print('No prediction. Cannot score. Aborting.')

    def sigmoid(self, theta, X):
        """Sigmoid function for Logistic Regression"""
        return 1.0/ (1.0 + np.exp(-X.dot(theta)))

    def cost(self, theta, X, y):
        """Cost function for Logistic Regression"""
        p = self.sigmoid(theta, X)
        return ((-y) * np.log(p) - (1 - y) * np.log(1 - p)).mean()

    def gradient(self, theta, X, y):
        error = self.sigmoid(theta, X) - y
        return error.T.dot(X) / y.size

