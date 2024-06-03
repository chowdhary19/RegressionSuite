import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(x, y)

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_Reg2 = LinearRegression()
lin_Reg2.fit(x_poly, y)

plt.scatter(x, y, color='Red')
plt.plot(x, lin_reg.predict(x), color='Blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

plt.scatter(x, y, color='Red')
plt.plot(x, lin_Reg2.predict(poly_reg.fit_transform(x)), color='Blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_Reg2.predict(poly_reg.fit_transform([[6.5]])))
