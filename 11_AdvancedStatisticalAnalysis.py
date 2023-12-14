import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\sadur\OneDrive - SSN Trust\Semester 1\Foundations for Data Engineering\Lab\EndSemSolutions\FuelConsumption.csv")
data = data.drop(['MAKE', 'MODEL','VEHICLECLASS','ENGINESIZE', 'CYLINDERS', 'TRANSMISSION', 'FUELTYPE'], axis=1)
# Basic Statistical Analysis
print("Basic Statistical Analysis:")
print(data.describe())

# Variance
print("\nVariance of each feature:")
print(data.var())

# Standard Deviation
print("\nStandard Deviation of each feature:")
print(data.std())

# Plotting distributions
for col in data.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Covariance and Correlation between two features
# Replace 'feature1' and 'feature2' with actual feature names from your dataset
feature1 = 'FUELCONSUMPTION_CITY'
feature2 = 'FUELCONSUMPTION_HWY'

covariance = data[[feature1, feature2]].cov().iloc[0, 1]
correlation = data[[feature1, feature2]].corr().iloc[0, 1]

print(f"\nCovariance between {feature1} and {feature2}: {covariance}")
print(f"Correlation coefficient between {feature1} and {feature2}: {correlation}")

# Function to calculate mean
def mean(values):
    return sum(values) / len(values)

# Function to calculate variance
def variance(values, mean_value):
    return sum((x - mean_value) ** 2 for x in values) / len(values)

# Function to calculate standard deviation
def std_dev(values, mean_value):
    return np.sqrt(variance(values, mean_value))

# Plotting distributions
for col in data.columns:
    plt.figure(figsize=(8, 4))
    plt.hist(data[col], bins=20, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.show()

# Manually calculate statistical measures
for col in data.columns:
    col_mean = mean(data[col])
    col_variance = variance(data[col], col_mean)
    col_std_dev = std_dev(data[col], col_mean)

    print(f"\n{col} - Mean: {col_mean}, Variance: {col_variance}, Standard Deviation: {col_std_dev}")

# Covariance and Correlation between two features
# Replace 'feature1' and 'feature2' with actual feature names from your dataset
feature1 = 'FUELCONSUMPTION_CITY'
feature2 = 'FUELCONSUMPTION_HWY'

def covariance(feature1, feature2, mean1, mean2):
    covar = sum((feature1[i] - mean1) * (feature2[i] - mean2) for i in range(len(feature1))) / len(feature1)
    return covar

mean1, mean2 = mean(data[feature1]), mean(data[feature2])
covar = covariance(data[feature1], data[feature2], mean1, mean2)
corr = covar / (std_dev(data[feature1], mean1) * std_dev(data[feature2], mean2))

print(f"\nCovariance between {feature1} and {feature2}: {covar}")
print(f"Correlation coefficient between {feature1} and {feature2}: {corr}")