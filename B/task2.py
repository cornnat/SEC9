# ----------------------------  
# Part a
# ----------------------------

import matplotlib.pyplot as plt
import networkx as nx

def visualize_cnn_structure():
    # Create a directed graph for CNN layers
    graph = nx.DiGraph()

    # Define the layers and operations as nodes
    graph.add_nodes_from([
        'Input', 'Conv Layer 1', 'Activation Layer 1', 'Pooling Layer 1', 
        'Conv Layer 2', 'Activation Layer 2', 'Pooling Layer 2', 'Fully Connected', 'Output'
    ])
    
    # Define the flow of data as edges between layers
    graph.add_edges_from([
        ('Input', 'Conv Layer 1'),
        ('Conv Layer 1', 'Activation Layer 1'),
        ('Activation Layer 1', 'Pooling Layer 1'),
        ('Pooling Layer 1', 'Conv Layer 2'),
        ('Conv Layer 2', 'Activation Layer 2'),
        ('Activation Layer 2', 'Pooling Layer 2'),
        ('Pooling Layer 2', 'Fully Connected'),
        ('Fully Connected', 'Output')
    ])

    # Specify the layout for visualization
    positions = {
        'Input': (0, 3),
        'Conv Layer 1': (1, 3),
        'Activation Layer 1': (2, 3),
        'Pooling Layer 1': (3, 3),
        'Conv Layer 2': (1, 2),
        'Activation Layer 2': (2, 2),
        'Pooling Layer 2': (3, 2),
        'Fully Connected': (2, 1),
        'Output': (3, 1)
    }
    
    # Plot the network structure
    plt.figure(figsize=(10, 7))
    nx.draw(graph, positions, with_labels=True, node_size=3000, node_color='lightcoral', font_size=10, font_weight='bold', arrows=True)
    plt.title("Forward Computational Flow in a 3-Layer CNN")
    plt.savefig('plots/cnn_structure.png', dpi=300)
    plt.show()

# Call the function to visualize the CNN structure
visualize_cnn_structure()

# ----------------------------  
# Part b - CNN Components and Backpropagation
# ----------------------------

import numpy as np

# Define ReLU activation function and its derivative
def relu_activation(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define Sigmoid activation function and its derivative
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid_value = sigmoid_activation(x)
    return sigmoid_value * (1 - sigmoid_value)

# Define the Convolutional Layer class
class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Initialize filters with small random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def iterate_over_regions(self, image):
        height, width = image.shape
        for i in range(height - self.filter_size + 1):
            for j in range(width - self.filter_size + 1):
                image_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield image_region, i, j

    def forward_pass(self, input):
        self.last_input = input
        height, width = input.shape
        output = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for image_region, i, j in self.iterate_over_regions(input):
            output[i, j] = np.sum(image_region * self.filters, axis=(1, 2))
        self.output = output  # Store the output here
        return output

    def backward_pass(self, d_L_d_output, learning_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for image_region, i, j in self.iterate_over_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_output[i, j, f] * image_region
        self.filters -= learning_rate * d_L_d_filters
        return None  # No need to return anything for this simple example

# Define the Fully Connected Layer class
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)

    def forward_pass(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        return np.dot(input, self.weights) + self.biases

    def backward_pass(self, d_L_d_output, learning_rate):
        d_L_d_input = np.dot(d_L_d_output, self.weights.T)
        d_L_d_weights = np.dot(self.last_input[:, np.newaxis], d_L_d_output[np.newaxis, :])
        d_L_d_biases = d_L_d_output

        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases
        return d_L_d_input.reshape(self.last_input_shape)

# Define the CNN class
class ConvolutionalNeuralNetwork:
    def __init__(self, num_filters, filter_size, input_shape, num_classes, activation='relu'):
        self.conv_layer = ConvolutionalLayer(num_filters, filter_size)
        self.fc_layer = FullyConnectedLayer((input_shape[0] - filter_size + 1) * (input_shape[1] - filter_size + 1) * num_filters, num_classes)
        self.activation_function = relu_activation if activation == 'relu' else sigmoid_activation
        self.activation_derivative_function = relu_derivative if activation == 'relu' else sigmoid_derivative

    def forward_pass(self, input):
        self.last_input = input
        conv_output = self.conv_layer.forward_pass(input)
        activated_output = self.activation_function(conv_output)
        fc_output = self.fc_layer.forward_pass(activated_output)
        return fc_output

    def backward_pass(self, d_L_d_output, learning_rate):
        d_L_d_fc = self.fc_layer.backward_pass(d_L_d_output, learning_rate)
        d_L_d_activated = d_L_d_fc.reshape(self.conv_layer.output.shape) * self.activation_derivative_function(self.conv_layer.output)
        self.conv_layer.backward_pass(d_L_d_activated, learning_rate)

# Example usage:
cnn = ConvolutionalNeuralNetwork(num_filters=8, filter_size=3, input_shape=(28, 28), num_classes=10, activation='relu')
output = cnn.forward_pass(np.random.randn(28, 28))
cnn.backward_pass(np.random.randn(10), learning_rate=0.005)

# Visualization Functions
def display_filters(conv_layer):
    filters = conv_layer.filters
    num_filters = filters.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    
    for i in range(num_filters):
        axes[i].imshow(filters[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    plt.suptitle('Convolutional Filters')
    plt.savefig('plots/convolutional_filters.png', dpi=300)
    plt.show()

# Visualize the convolutional filters
display_filters(cnn.conv_layer)

def display_feature_maps(feature_maps):
    num_feature_maps = feature_maps.shape[2]
    fig, axes = plt.subplots(1, num_feature_maps, figsize=(15, 5))
    
    for i in range(num_feature_maps):
        axes[i].imshow(feature_maps[:, :, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Feature Map {i+1}')
    
    plt.suptitle('Feature Maps')
    plt.savefig('plots/feature_maps.png', dpi=300)
    plt.show()

# Get feature maps and display
input_image = np.random.randn(28, 28)
conv_output = cnn.conv_layer.forward_pass(input_image)

# Display feature maps
display_feature_maps(conv_output)

def display_activated_feature_maps(activated_feature_maps):
    num_feature_maps = activated_feature_maps.shape[2]
    fig, axes = plt.subplots(1, num_feature_maps, figsize=(15, 5))
    
    for i in range(num_feature_maps):
        axes[i].imshow(activated_feature_maps[:, :, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Activated Feature Map {i+1}')
    
    plt.suptitle('Activated Feature Maps')
    plt.savefig('plots/activated_feature_maps.png', dpi=300)
    plt.show()

# Display activated feature maps
activated_output = cnn.activation_function(conv_output)
display_activated_feature_maps(activated_output)

def display_fc_output(fc_output):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(fc_output)), fc_output)
    plt.title('Fully Connected Layer Output')
    plt.xlabel('Class')
    plt.ylabel('Output Value')
    plt.savefig('plots/fully_connected_output.png', dpi=300)
    plt.show()

# Get and display fully connected layer output
fc_output = cnn.fc_layer.forward_pass(activated_output)
display_fc_output(fc_output)

def display_gradients(gradients):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(gradients)), gradients)
    plt.title('Gradients during Backpropagation')
    plt.xlabel('Parameter')
    plt.ylabel('Gradient Value')
    plt.savefig('plots/gradients_during_backprop.png', dpi=300)
    plt.show()

# Get gradients and display
d_L_d_output = np.random.randn(10)
cnn.backward_pass(d_L_d_output, learning_rate=0.005)

# Display gradients
display_gradients(cnn.fc_layer.weights.flatten())
