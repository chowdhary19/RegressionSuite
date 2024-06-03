import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='Red')
plt.plot(sc_x.transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='Blue')
plt.title('Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()
