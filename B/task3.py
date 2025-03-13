import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from tensorflow.keras.datasets import mnist  # To load MNIST dataset

# Import MLP and CNN from previous tasks
from task1 import MLP
from task2 import CNN

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0,1] and reshape
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

# Split training into 80% train, 20% validation
X_train, X_val, y_train_onehot, y_val_onehot, y_train, y_val = train_test_split(
    X_train, y_train_onehot, y_train, test_size=0.2, random_state=42
)

# Flatten images for MLP
X_train_flat = X_train.reshape(X_train.shape[0], -1).T
X_val_flat = X_val.reshape(X_val.shape[0], -1).T
X_test_flat = X_test.reshape(X_test.shape[0], -1).T

# Instantiate models (corrected constructor calls)
mlp = MLP(input_size=28*28, hidden_sizes=[128, 64], output_size=10)  # Adjusted MLP constructor call
cnn = CNN(input_shape=(28, 28, 1), num_classes=10)  # Adjusted CNN constructor call, assuming input shape for CNN

# Training parameters
epochs = 10
learning_rate = 0.01
batch_size = 32

# Training function
def train_model(model, X_train, y_train, X_val, y_val, model_type="MLP"):
    """Train either MLP or CNN using gradient descent."""
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(X_train.shape[1])
        X_train_shuffled, y_train_shuffled = X_train[:, indices], y_train[:, indices]
        
        # Mini-batch training
        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_train_shuffled[:, i:i+batch_size]
            y_batch = y_train_shuffled[:, i:i+batch_size]

            # Forward pass
            y_pred = model.forward(X_batch)

            # Compute loss (Mean Squared Error)
            loss = np.mean((y_pred - y_batch) ** 2)

            # Backward pass
            d_loss = 2 * (y_pred - y_batch) / y_batch.shape[1]
            model.backward(d_loss, learning_rate)

        # Evaluate on validation set
        y_train_pred = model.forward(X_train)
        y_val_pred = model.forward(X_val)
        
        train_loss = np.mean((y_train_pred - y_train) ** 2)
        val_loss = np.mean((y_val_pred - y_val) ** 2)

        train_accuracy = accuracy_score(np.argmax(y_train, axis=0), np.argmax(y_train_pred, axis=0))
        val_accuracy = accuracy_score(np.argmax(y_val, axis=0), np.argmax(y_val_pred, axis=0))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - {model_type}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Train MLP
print("\nTraining MLP...")
mlp_train_losses, mlp_val_losses, mlp_train_acc, mlp_val_acc = train_model(mlp, X_train_flat, y_train_onehot.T, X_val_flat, y_val_onehot.T, "MLP")

# Train CNN
print("\nTraining CNN...")
cnn_train_losses, cnn_val_losses, cnn_train_acc, cnn_val_acc = train_model(cnn, X_train, y_train_onehot.T, X_val, y_val_onehot.T, "CNN")

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_type="MLP"):
    """Evaluates the model and plots a confusion matrix."""
    y_pred = model.forward(X_test)
    y_pred_labels = np.argmax(y_pred, axis=0)
    y_true_labels = np.argmax(y_test, axis=0)
    
    acc = accuracy_score(y_true_labels, y_pred_labels)
    print(f"{model_type} Test Accuracy: {acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_type} Confusion Matrix")
    plt.savefig(f"plots/3_{model_type}_confusion_matrix.png")
    plt.show()

# Evaluate models
evaluate_model(mlp, X_test_flat, y_test_onehot.T, "MLP")
evaluate_model(cnn, X_test, y_test_onehot.T, "CNN")

# Plot training curves
def plot_training_curves(train_losses, val_losses, train_acc, val_acc, model_type):
    """Plots loss and accuracy curves.""" 
    plt.figure(figsize=(12,5))

    # Loss curve
    plt.subplot(1,2,1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_type} Training Loss")
    plt.legend()

    # Accuracy curve
    plt.subplot(1,2,2)
    plt.plot(range(epochs), train_acc, label="Train Accuracy")
    plt.plot(range(epochs), val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_type} Accuracy")
    plt.legend()
    plt.savefig(f"plots/3_{model_type}_Accuracy.png")
    plt.show()

# Plot curves for MLP and CNN
plot_training_curves(mlp_train_losses, mlp_val_losses, mlp_train_acc, mlp_val_acc, "MLP")
plot_training_curves(cnn_train_losses, cnn_val_losses, cnn_train_acc, cnn_val_acc, "CNN")
