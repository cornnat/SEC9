import numpy as np

class MLP:
    def __init__(self, input_dim=6, hidden_dims=[4, 3, 2]):
        """Initialize weights and biases for a 4-layer MLP."""
        layer_dims = [input_dim] + hidden_dims
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_dims) - 1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_dims[i + 1])))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of Sigmoid function."""
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, X):
        """Forward pass through the network."""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        # Output layer with sigmoid activation
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward pass using backpropagation with np.einsum for gradient calculations."""
        m = X.shape[0]  # Number of samples
        
        # Compute the loss gradient with respect to the output
        dz = self.activations[-1] - y
        
        weight_gradients = []
        bias_gradients = []
        
        for i in reversed(range(len(self.weights))):
            # Compute weight gradient using np.einsum for outer product
            dw = np.einsum('bi,bj->ij', self.activations[i], dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dw)
            bias_gradients.insert(0, db)
            
            # Backpropagate error to previous layer
            if i > 0:
                dz = dz @ self.weights[i].T * self.relu_derivative(self.z_values[i-1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the MLP using gradient descent."""
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean((self.activations[-1] - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Example usage with random data
np.random.seed(42)
X_train = np.random.rand(100, 6)  # 100 samples, 6 features
y_train = np.random.randint(0, 2, (100, 2))  # Binary classification (one-hot encoded)

mlp = MLP()
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01)