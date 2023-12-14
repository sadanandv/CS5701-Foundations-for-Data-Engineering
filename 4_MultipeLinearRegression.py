'''4. Perform Multiple Linear Regression on the FuelConsumption dataset. Express CO2EMISSIONS as a linear function of 'ENGINESIZE', 'CYLINDERS', and 'FUELCONSUMPTION_COMB' using sklearn: linear_model.LinearRegression().'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"C:\Users\sadur\OneDrive - SSN Trust\Semester 1\Foundations for Data Engineering\Lab\EndSemSolutions\FuelConsumption.csv")

# Selecting features and target
X = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
Y = data['CO2EMISSIONS']

# Create the model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Make predictions
Y_pred = model.predict(X)

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['ENGINESIZE'], data['CYLINDERS'], data['FUELCONSUMPTION_COMB'], c='blue', marker='o', alpha=0.5)
ax.plot_trisurf(data['ENGINESIZE'], data['CYLINDERS'], Y_pred, color='red', alpha=0.5)
ax.set_xlabel('Engine Size')
ax.set_ylabel('Cylinders')
ax.set_zlabel('Fuel Consumption Comb')
plt.show()

# Compute total squared error
tse = mean_squared_error(Y, Y_pred) * len(Y)
print("Total Squared Error:", tse)
