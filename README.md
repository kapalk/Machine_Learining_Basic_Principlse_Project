# Prediction on the Yelp dataset
Supervised learning project on Yelp dataset. Goal was to build a regressor and a classifier which would predict the number of useful votes on a review and classify if the review received more than 3 stars. The data consisted of 6000 Yelp user reviews: 50 features which correspond to the word counts for 50 manually selected words that appeared most frequently (bag-of-words model).


Under `data/`:

- all data for training and testing

# Scripts
- `featureSelection.py`: feature selection for both claasification and regression. For regression, features are selected using cross-correlation. For classification feature selection was performed by implementing the Analysis of Variance to get the proportion of variance explained by the feature to the total variance in the data
- `multivariateRidgeRegressor.py`: multivariate ridge regression from scratch
- `regressor_optimization.py`: optimizing predictor and testing it
- `LogisticRegression.py`: logistic regression from scrach
- `classification_optimization.py`: optimizing classifier and testing it
