import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('dataset.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
print('Accuracy : '),regressor.score(x,y)*100
y_pred=regressor.predict([[7]])
print(y_pred)
hours=int(input('Enter hours:'))
eq=regressor.coef_*hours+regressor.intercept_
regressor.coef_
regressor.intercept_
eq
y_pred=regressor.predict([[10]])
print(y_pred)
plt.plot(x,y,'o')
plt.plot(x,regressor.predict(x))
plt.show()
