'''3. Perform Simple Linear Regression on the given dataset.
Write Python code to implement Simple Linear Regression on the provided dataset.
Plot the regression line on the scatter plot of the data.
Calculate and report the mean squared error of your model.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming 'data' is the given dataset with columns 'X' and 'Y'
# Replace this with the actual dataset
data = pd.read_csv(r"C:\Users\sadur\OneDrive - SSN Trust\Semester 1\Foundations for Data Engineering\Lab\EndSemSolutions\company_sales_data.csv")
X = data['facecream'].values.reshape(-1, 1)  # Feature
Y = data['total_profit'].values  # Target

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Predictions
Y_pred = model.predict(X)

# Plotting
plt.scatter(X, Y, color='blue')  # Actual points
plt.plot(X, Y_pred, color='red')  # Regression line
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Mean Squared Error
mse = mean_squared_error(Y, Y_pred)
print("Mean Squared Error:", mse)
