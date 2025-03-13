import numpy as np

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Convolution operation (valid padding)
def conv2d(X, W):
    """Performs 2D convolution (single channel, valid padding)."""
    X_h, X_w = X.shape
    W_h, W_w = W.shape
    output_h = X_h - W_h + 1
    output_w = X_w - W_w + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[i, j] = np.sum(X[i:i+W_h, j:j+W_w] * W)
    return output

# Backpropagation through convolution (computing gradient wrt filter)
def conv2d_backprop_filter(X, d_out, W_shape):
    """Computes gradient of convolution filter using outer product."""
    W_h, W_w = W_shape
    dW = np.zeros(W_shape)

    for i in range(d_out.shape[0]):
        for j in range(d_out.shape[1]):
            dW += np.einsum('ij,->ij', X[i:i+W_h, j:j+W_w], d_out[i, j])

    return dW

# Backpropagation to compute gradient w.r.t. input
def conv2d_backprop_input(d_out, W):
    """Computes gradient w.r.t input by convolving back the error."""
    W_h, W_w = W.shape
    dX = np.zeros((d_out.shape[0] + W_h - 1, d_out.shape[1] + W_w - 1))

    for i in range(d_out.shape[0]):
        for j in range(d_out.shape[1]):
            dX[i:i+W_h, j:j+W_w] += d_out[i, j] * W

    return dX

# Fully Connected Layer (Reused from MLP)
class FullyConnected:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))

    def forward(self, X):
        self.X = X  # Store for backprop
        return np.dot(self.W, X) + self.b

    def backward(self, d_out, learning_rate=0.01):
        dW = np.dot(d_out, self.X.T)
        db = np.sum(d_out, axis=1, keepdims=True)
        dX = np.dot(self.W.T, d_out)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dX

# CNN Class
class CNN:
    def __init__(self):
        self.conv1_W = np.random.randn(3, 3) * 0.01  # 3x3 filter
        self.conv2_W = np.random.randn(3, 3) * 0.01  # Another 3x3 filter
        self.fc = FullyConnected(16, 2)  # Assume input size reduces to 16

    def forward(self, X):
        """Forward pass through CNN layers."""
        self.X = X
        self.conv1_out = relu(conv2d(X, self.conv1_W))
        self.conv2_out = relu(conv2d(self.conv1_out, self.conv2_W))
        self.fc_in = self.conv2_out.flatten().reshape(-1, 1)  # Flatten for FC layer
        self.fc_out = sigmoid(self.fc.forward(self.fc_in))
        return self.fc_out

    def backward(self, d_loss, learning_rate=0.01):
        """Backward pass for CNN."""
        d_fc = d_loss * sigmoid_derivative(self.fc_out)
        d_fc_in = self.fc.backward(d_fc, learning_rate)

        # Reshape d_fc_in to match conv2_out shape
        d_conv2 = d_fc_in.reshape(self.conv2_out.shape) * relu_derivative(self.conv2_out)

        # Compute gradients for conv2 filter and input
        d_conv2_W = conv2d_backprop_filter(self.conv1_out, d_conv2, self.conv2_W.shape)
        self.conv2_W -= learning_rate * d_conv2_W  # Update filter weights

        d_conv1 = conv2d_backprop_input(d_conv2, self.conv2_W) * relu_derivative(self.conv1_out)
        d_conv1_W = conv2d_backprop_filter(self.X, d_conv1, self.conv1_W.shape)
        self.conv1_W -= learning_rate * d_conv1_W  # Update filter weights

# Example usage
cnn = CNN()
X_sample = np.random.randn(8, 8)  # Example 8x8 input
output = cnn.forward(X_sample)
print("CNN Output:", output)

# Simulated loss gradient for testing backpropagation
d_loss = np.random.randn(2, 1)
cnn.backward(d_loss)
