# -*- coding: utf-8 -*-
"""Regression(Simple Regression).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vEgQ7R460tluzthlJuvluiHQmf8Tpd_u

# **Simple Linear Regression**

## **Importing Libraries**

---
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""## **Importing Dataset**

---


"""

dataset= pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"""## **Splitting into Test and Train**

---



"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print(y_train)

"""## **Training Simple Regression Model on Training Data**

---


"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

"""## **Predicting the Test Set Result**"""

y_pred=regressor.predict(x_test)

"""## **Visualising The Training Set Data**"""

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train))
plt.title('SALARY V/S EXPERIENCE (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""## **Visualising the Test Set Data**"""

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train))
plt.title('Salary V/S Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()