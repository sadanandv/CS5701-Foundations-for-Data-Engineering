import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    predictions = np.round(y_pred)
    correct = np.sum(predictions == y_true)
    return correct / len(y_true)

# Neural network with one hidden layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward propagation
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)
        output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return output

    def backprop(self, X, y, output, learning_rate):
        # Backpropagation
        d_output_error = 2 * (y - output) * sigmoid_derivative(output)
        d_hidden_layer_error = np.dot(d_output_error, self.weights2.T) * sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.weights2 += np.dot(self.hidden.T, d_output_error) * learning_rate
        self.bias2 += np.sum(d_output_error, axis=0, keepdims=True) * learning_rate
        self.weights1 += np.dot(X.T, d_hidden_layer_error) * learning_rate
        self.bias1 += np.sum(d_hidden_layer_error, axis=0, keepdims=True) * learning_rate

# Example usage
input_size = 4  # Adjust as needed
hidden_size = 2  # Can be changed
output_size = 1  # Adjust based on the problem

# Initialize the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Example data (X: input data, y: target output)
X = np.array([[0, 0, 1, 1],
              [1, 1, 1, 0],
              [1, 0, 0, 1],
              [0, 1, 0, 0]])
y = np.array([[0], [1], [1], [0]])

# Training loop
epochs = 1000  # Can be adjusted
learning_rate = 0.1  # Can be adjusted

for epoch in range(epochs):
    output = nn.forward(X)
    nn.backprop(X, y, output, learning_rate)

    # Calculate accuracy
    accuracy = calculate_accuracy(y, output)
    print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}")

# Test the network
print("Output after training:")
print(nn.forward(X))