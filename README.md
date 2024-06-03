
# RegressSuite: Comprehensive Regression Models Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

Welcome to RegressSuite! This repository contains a suite of regression models, including Linear Regression, Multilinear Regression, Polynomial Regression, Support Vector Regression, Decision Tree Regression, and Random Forest Regression. It also includes a data preprocessing template to prepare your data for analysis. Each model is implemented in Python using popular libraries such as scikit-learn, pandas, and numpy.



## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preprocessing](#data-preprocessing)
3. [Linear Regression](#linear-regression)
4. [Multilinear Regression](#multilinear-regression)
5. [Polynomial Regression](#polynomial-regression)
6. [Support Vector Regression](#support-vector-regression)
7. [Decision Tree Regression](#decision-tree-regression)
8. [Random Forest Regression](#random-forest-regression)
9. [References](#references)
10. [License](#license)

## Getting Started

To get started, clone the repository to your local machine:

```sh
git clone https://github.com/chowdhary19/RegressionSuite.git
cd RegressionSuite
```

Install the necessary dependencies:

```sh
pip install -r requirements.txt
```

## Data Preprocessing

Before applying any regression models, it is crucial to preprocess the data. The data preprocessing script includes steps for handling missing values, encoding categorical data, feature scaling, and splitting the dataset into training and test sets.

```python
# datapreprocessingtemplate(ysc).py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load dataset
dataset = pd.read_csv('data.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
dataset.iloc[:, :] = imputer.fit_transform(dataset)

# Encode categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
dataset = np.array(ct.fit_transform(dataset))

# Feature scaling
sc = StandardScaler()
dataset = sc.fit_transform(dataset)

# Split the dataset
X = dataset[:, :-1]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```



## Linear Regression

Linear regression is the simplest form of regression that establishes a relationship between the dependent variable (y) and one independent variable (X) using a straight line.

```python
# simple_regression.py
from sklearn.linear_model import LinearRegression

# Create and train the model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = linear_regressor.predict(X_test)
```



## Multilinear Regression

Multilinear regression is an extension of linear regression that uses multiple independent variables to predict the dependent variable.

```python
# multilinear_regression.py
from sklearn.linear_model import LinearRegression

# Create and train the model
multilinear_regressor = LinearRegression()
multilinear_regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = multilinear_regressor.predict(X_test)
```



## Polynomial Regression

Polynomial regression is a form of regression analysis in which the relationship between the independent variable (X) and the dependent variable (y) is modeled as an nth degree polynomial.

```python
# polynomial_regression.py
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Transforming the data to include polynomial features
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_train)

# Predicting the test results
y_pred = poly_regressor.predict(poly_reg.transform(X_test))
```


## Support Vector Regression

Support Vector Regression (SVR) uses the same principles as the SVM for classification, with a few minor differences to make it suitable for regression.

```python
# supportvector_regression.py
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# Create and train the model
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X_train, y_train.ravel())

# Predicting the test results
y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))
```



## Decision Tree Regression

Decision Tree Regression models observe features of an object and train a model in the form of a tree to predict data in the future.

```python
# decisiontree_regression.py
from sklearn.tree import DecisionTreeRegressor

# Create and train the model
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = tree_regressor.predict(X_test)
```



## Random Forest Regression

Random Forest Regression is an ensemble method that uses multiple decision trees to improve the accuracy and control over-fitting.

```python
# randomforest_regression.py
from sklearn.ensemble import RandomForestRegressor

# Create and train the model
forest_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
forest_regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = forest_regressor.predict(X_test)
```



## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Numpy Documentation](https://numpy.org/doc/stable/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

