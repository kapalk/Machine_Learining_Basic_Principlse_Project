import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.naive_bayes import MultinomialNB
import sklearn.linear_model as linear_model
import featureSelection
from LogisticRegression import LogisticRegression

trainSet = pd.read_csv('data/classification_dataset_training.csv')
testSet = pd.read_csv('data/classification_dataset_testing.csv')
solutionSet = pd.read_csv('data/classification_dataset_testing_solution.csv')

X_train = np.array(trainSet.ix[:,1:51])
y_train = np.array(trainSet.ix[:,51])
X_test = np.array(testSet.ix[:,1:])
y_test = np.array(solutionSet.ix[:,1])

logarega = linear_model.LogisticRegression()
naiveB = MultinomialNB()
logreg = LogisticRegression()
errorlist = np.empty(50)
errorlistNB = np.empty(50)
errorlistLR = np.empty(50)
for k in range(1,51):
    featureMask = featureSelection.forClassification(X_train, y_train, k)
    X = X_train[:,featureMask]

    error = 0
    errorNB = 0
    errorLR = 0
    folds = 10
    kf = KFold(n_splits=folds)
    for traincv, validationcv in kf.split(X):
        X_training, X_validation = X[traincv,:], X[validationcv,:]
        y_training, y_validation = y_train[traincv], y_train[validationcv]

        #logreg = LogisticRegression()
        logreg.fit(X_training, y_training)
        naiveB.fit(X_training, y_training)
        logarega.fit(X_training, y_training)
        errorNB += mse(y_validation, naiveB.predict(X_validation))
        error += mse(y_validation, logreg.predict(X_validation))
        errorLR += mse(y_validation, logarega.predict(X_validation))
    errorlist[k-1] = error/folds
    errorlistNB[k-1] = errorNB/folds
    errorlistLR[k-1] = errorLR/folds

k_opt = 1 + np.argmin(errorlist)
k_optNB = 1 + np.argmin(errorlistNB)
k_optLR = 1 + np.argmin(errorlistLR)
print('Optimal feature number is: %d.' % int(k_opt))
print('Optimal features for multiNB is: %d.' % int(k_optNB))
print('Optimal features for LR is: %d.' % int(k_optLR))
featureMask = featureSelection.forClassification(X_train, y_train, k_opt)
featureMaskNB = featureSelection.forClassification(X_train, y_train, k_optNB)
featureMaskLR = featureSelection.forClassification(X_train, y_train, k_optLR)

logreg.fit(X_train[:,featureMask], y_train)
prediction =logreg.predict(X_test[:,featureMask])
testError = mse(y_test, prediction)
print(testError)
print(np.sum(y_test == prediction))

naiveB.fit(X_train[:,featureMaskNB], y_train)
print('Naive bayes test error is: %.5f.' % mse(y_test, naiveB.predict(X_test[:,featureMaskNB])))
print(np.sum(y_test == naiveB.predict(X_test[:,featureMaskNB])))
# testing with sklearn logreg

#from sklearn.linear_model import LogisticRegression

#logarega = LogisticRegression()
logarega.fit(X_train[:,featureMaskLR], y_train)
predator = logarega.predict(X_test[:,featureMaskLR])
print(np.sum(y_test == predator))

print(mse(y_test, predator))
