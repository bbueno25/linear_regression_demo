"""
DOCSTRING
"""
import matplotlib.pyplot as pyplot
import pandas
import sklearn.linear_model as linear_model

dataframe = pandas.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)
pyplot.scatter(x_values, y_values)
pyplot.plot(x_values, body_reg.predict(x_values))
pyplot.show()
