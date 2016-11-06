import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import featureSelection
from LogisticRegression import LogisticRegression

trainSet = pd.read_csv('../LogisticRegression/classification_dataset_training.csv')
testSet = pd.read_csv('../LogisticRegression/classification_dataset_testing.csv')
solutionSet = pd.read_csv('../LogisticRegression/classification_dataset_testing_solution.csv')

X_train = np.array(trainSet.ix[:,1:51])
y_train = np.array(trainSet.ix[:,51])
X_test = np.array(testSet.ix[:,1:])
y_test = np.array(solutionSet.ix[:,1])

logreg = LogisticRegression()
errorlist = np.empty(50)
for k in range(1,51):
    featureMask = featureSelection.forClassification(X_train, y_train, k)
    X = X_train[:,featureMask]

    error = 0
    folds = 10
    kf = KFold(n_splits=folds)
    for traincv, validationcv in kf.split(X):
        X_training, X_validation = X[traincv,:], X[validationcv,:]
        y_training, y_validation = y_train[traincv], y_train[validationcv]

        #logreg = LogisticRegression()
        theta = logreg.fit(X_training, y_training)
        error += mse(y_validation, logreg.predict(X_validation))
    errorlist[k-1] = error/folds

k_opt = 1 + np.argmin(errorlist)
print('Optimal feature number is: %d.' % int(k_opt))
featureMask = featureSelection.forClassification(X_train, y_train, k_opt)

logreg.fit(X_train[:,featureMask], y_train)
prediction =logreg.predict(X_test[:,featureMask])
testError = mse(y_test, prediction)
print(testError)
print(np.sum(y_test == prediction))

# testing with sklearn logreg

from sklearn.linear_model import LogisticRegression

logarega = LogisticRegression()
logarega.fit(X_train[:,featureMask], y_train)
predator = logarega.predict(X_test[:,featureMask])
print(np.sum(y_test == predator))


