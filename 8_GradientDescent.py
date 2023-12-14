import numpy as np
import matplotlib.pyplot as plt

# Function to minimize
def function_to_minimize(x, y):
    return x**2 + y**2

# Gradient of the function
def gradient(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Gradient Descent
def gradient_descent(learning_rate, num_iterations):
    # Initial values
    x = 5.0
    y = 5.0

    # Lists to store values for plotting
    x_values = [x]
    y_values = [y]
    z_values = [function_to_minimize(x, y)]

    # Gradient Descent iterations
    for _ in range(num_iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]

        # Store values for plotting
        x_values.append(x)
        y_values.append(y)
        z_values.append(function_to_minimize(x, y))

    return x_values, y_values, z_values

# Surface plot
def plot_surface():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = function_to_minimize(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z )#, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F(x, y)')
    ax.set_title('Surface Plot')

    plt.show()

# Contour plot
def plot_contour():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = function_to_minimize(X, Y)

    plt.contour(X, Y, Z)# levels=20, cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot')
    plt.show()

# Gradient Descent parameters
learning_rate = 0.1
num_iterations = 50

# Perform Gradient Descent
x_values, y_values, z_values = gradient_descent(learning_rate, num_iterations)

# Plot results
plot_surface()
plot_contour()

# Plotting the path of gradient descent on the contour plot
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = function_to_minimize(X, Y)

plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot')
plt.plot(x_values, y_values, 'ro-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot with Gradient Descent Path')
plt.show()
