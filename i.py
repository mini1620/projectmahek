import pandas 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
df = pandas.read_csv('Lr1.csv')

x = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x, y)
test_y = regr.predict(X)

predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)