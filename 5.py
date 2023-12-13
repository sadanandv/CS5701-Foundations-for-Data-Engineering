import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\sadur\OneDrive - SSN Trust\Semester 1\Foundations for Data Engineering\Lab\EndSemSolutions\FuelConsumption.csv")

# a. Provide Statistical Information
print("Statistical Information:\n", data.describe())

# b. Analyze Distributions
# Assuming the dataset has numerical columns. Adjust according to your dataset.
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.distplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()

# c. Create Box Plots
# Adjust according to the categorical and numerical columns of your dataset
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col])
    plt.title(f'Box Plot of {col}')
    plt.show()
