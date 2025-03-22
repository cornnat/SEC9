import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, layer_sizes, output_size):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        layer_dimensions = [self.input_size] + self.layer_sizes + [self.output_size]
        
        # Random initialization of weights and biases
        for i in range(len(layer_dimensions) - 1):
            self.weights.append(np.random.randn(layer_dimensions[i], layer_dimensions[i+1]))
            self.biases.append(np.zeros((layer_dimensions[i+1],)))
    
    def relu_activation(self, z):
        return np.maximum(0, z)
    
    def sigmoid_activation(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward_pass(self, input_data):
        self.layer_cache = {}  # Cache for backward pass
        self.layer_cache['A0'] = input_data  # Store input as A0
        activation = input_data
        
        for i in range(len(self.weights) - 1):
            linear_transform = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.relu_activation(linear_transform)  # Using ReLU activation
            self.layer_cache[f'Z{i+1}'] = linear_transform
            self.layer_cache[f'A{i+1}'] = activation
        
        # Output layer with sigmoid activation
        linear_transform = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid_activation(linear_transform)  # Sigmoid activation
        self.layer_cache[f'Z{len(self.weights)}'] = linear_transform
        self.layer_cache[f'A{len(self.weights)}'] = output
        return output

    def backward_pass(self, input_data, target_data, learning_rate=0.01):
        num_examples = input_data.shape[0]  # Number of examples
        gradients = {}
        
        # Output layer gradient calculation
        A_output = self.layer_cache[f'A{len(self.weights)}']
        output_error = A_output - target_data
        delta_output = output_error * A_output * (1 - A_output)  # Derivative of sigmoid
        
        gradients[f'dW{len(self.weights)}'] = np.dot(self.layer_cache[f'A{len(self.weights)-1}'].T, delta_output) / num_examples
        gradients[f'db{len(self.weights)}'] = np.sum(delta_output, axis=0, keepdims=True) / num_examples

        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            delta_hidden = np.dot(delta_output, self.weights[i+1].T)
            delta_output = delta_hidden * (self.layer_cache[f'A{i+1}'] > 0)  # Derivative of ReLU
            gradients[f'dW{i+1}'] = np.dot(self.layer_cache[f'A{i}'].T, delta_output) / num_examples
            gradients[f'db{i+1}'] = np.sum(delta_output, axis=0, keepdims=True) / num_examples
        
        # Update parameters (Weights and Biases)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[f'dW{i+1}']
            self.biases[i] -= learning_rate * gradients[f'db{i+1}'].flatten()
        
        return gradients

# Example of how to initialize and use the NeuralNetwork class:
input_size = 6
layer_sizes = [4, 3, 2]
output_size = 1

# Create the neural network model
nn_model = NeuralNetwork(input_size, layer_sizes, output_size)

# Example forward pass
X_example = np.random.randn(5, input_size)  # 5 examples with 6 features each
y_example = np.random.randn(5, output_size)  # 5 examples of output

# Perform forward pass
predictions = nn_model.forward_pass(X_example)

# Perform backward pass
gradients = nn_model.backward_pass(X_example, y_example)

print("Predictions:", predictions)
print("Gradients:", gradients)
