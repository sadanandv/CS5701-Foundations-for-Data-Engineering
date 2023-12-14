'''8. Given  f(x,y)=3x 2 +2y 2 - 4x+6y+7 , find the minimum value of the function. Write Python code to find the minimum value of the function using gradient descent. Obtain contour plots of the derived function.'''
  
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return 3*x**2 + 2*y**2 - 4*x + 6*y + 7

# Define the gradient of the function
def grad_f(x, y):
    df_dx = 6*x - 4
    df_dy = 4*y + 6
    return np.array([df_dx, df_dy])

# Gradient Descent
def gradient_descent(start_x, start_y, learning_rate, num_iterations):
    x, y = start_x, start_y
    for _ in range(num_iterations):
        grad_x, grad_y = grad_f(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
    return x, y

# Perform Gradient Descent
min_x, min_y = gradient_descent(start_x=0.5, start_y=0.5, learning_rate=0.1, num_iterations=100)

# Print the minimum value
print(f"Minimum at: ({min_x}, {min_y})")
print(f"Minimum value of the function: {f(min_x, min_y)}")

# Contour plot
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=50)
plt.plot(min_x, min_y, 'r*', markersize=10)
plt.title('Contour plot of and Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

