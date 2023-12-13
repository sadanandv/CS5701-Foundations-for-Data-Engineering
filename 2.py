'''1.	Perform the following using python
a.	Define and plot the function for a sigmoid, let that be f1(z).
b.	Derive and plot the derivative of the sigmoid function and let that be f2(z).
c.	Plot f3(z) = f1(z) * (1-f1(z))
d.	Plot f4(z) = f2(z) = f3(z).
'''

import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the derivative of the sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define f3(z)
def f3(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define f4(z)
def f4(z):
    return sigmoid_derivative(z) - f3(z)

# Plotting the functions
z = np.linspace(-10, 10, 100)

plt.figure(figsize=(12, 8))

# Plotting f1(z)
plt.subplot(2, 2, 1)
plt.plot(z, sigmoid(z), label="f1(z) = Sigmoid(z)")
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("f1(z)")
plt.grid(True)
plt.legend()

# Plotting f2(z)
plt.subplot(2, 2, 2)
plt.plot(z, sigmoid_derivative(z), label="f2(z) = Sigmoid Derivative")
plt.title("Sigmoid Derivative")
plt.xlabel("z")
plt.ylabel("f2(z)")
plt.grid(True)
plt.legend()

# Plotting f3(z)
plt.subplot(2, 2, 3)
plt.plot(z, f3(z), label="f3(z) = f1(z) * (1 - f1(z))")
plt.title("f3(z)")
plt.xlabel("z")
plt.ylabel("f3(z)")
plt.grid(True)
plt.legend()

# Plotting f4(z)
plt.subplot(2, 2, 4)
plt.plot(z, f4(z), label="f4(z) = f2(z) - f3(z)")
plt.title("f4(z)")
plt.xlabel("z")
plt.ylabel("f4(z)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
