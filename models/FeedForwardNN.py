import numpy as np

class FeedForwardNN:
    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.9, weight_decay=0.0, dropout_rate=0.0):
        """
        Initialize the multi-layer feed-forward neural network.

        Parameters:
        layer_sizes (list): List of integers representing the number of neurons in each layer.
        learning_rate (float): Learning rate for gradient descent.
        momentum (float): Momentum for weight updates.
        weight_decay (float): L2 regularization coefficient.
        dropout_rate (float): Fraction of neurons to drop during training.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.velocity_w = []
        self.velocity_b = []

        self.error_history_train = []
        self.error_history_test = []

        # Xavier initialization for weights
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.velocity_w.append(np.zeros_like(self.weights[-1]))
            self.velocity_b.append(np.zeros_like(self.biases[-1]))

    def forward(self, X):
        """
        Perform forward propagation.

        Parameters:
        X (ndarray): Input data.

        Returns:
        activations (list): Activations for all layers.
        """
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = np.dot(activations[-1], W) + b
            # Hidden layers
            if i < len(self.weights) - 1:  
                A = self.relu(Z)
                # Apply dropout
                if self.dropout_rate > 0:  
                    dropout_mask = np.random.rand(*A.shape) > self.dropout_rate
                    A *= dropout_mask
                    A /= (1 - self.dropout_rate)
            else:  
                # Output layer
                A = self.softmax(Z)
            activations.append(A)
        return activations

    def inference(self, X, y_true):
        """
        Perform inference on a dataset and evaluate its performance.

        Parameters:
        X (ndarray): Input data.
        y_true (ndarray): True labels.

        Returns:
        tuple: A tuple containing:
            - outputs (ndarray): The raw outputs (softmax probabilities) from the final layer.
            - predictions (ndarray): Predicted class labels.
            - confusion_matrix (ndarray): The confusion matrix.
            - accuracy (float): Overall accuracy.
        """
        outputs = self.forward(X)[-1]
        predictions = np.argmax(outputs, axis=1)

        # Calculate confusion matrix
        num_classes = self.layer_sizes[-1]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true, predictions):
            confusion_matrix[int(true), int(pred)] += 1

        # Calculate accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        return outputs, predictions, confusion_matrix, accuracy
    
    def backward(self, activations, y_true):
        """
        Perform backward propagation and update weights and biases.

        Parameters:
        activations (list): Activations for all layers.
        y_true (ndarray): True labels.
        """

        # Convert true labels to one-hot encoding
        y_true_onehot = np.eye(self.layer_sizes[-1])[y_true]

        # Compute output layer delta
        batch_size = y_true.shape[0]
        deltas = [activations[-1] - y_true_onehot]

        # Backpropagate through layers
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.relu_derivative(activations[i])
            deltas.insert(0, delta)

        # Update weights and biases with momentum and weight decay
        for i in range(len(self.weights)):
            grad_w = np.dot(activations[i].T, deltas[i]) + self.weight_decay * self.weights[i]
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)

            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * grad_w
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grad_b

            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]


    def calculate_error_fraction(self, X, y):
        """
        Calculate error fraction for a dataset.

        Parameters:
        X (ndarray): Input data.
        y (ndarray): True labels.

        Returns:
        float: Error fraction.
        """
        predictions = self.predict(X)
        incorrect_predictions = np.sum(predictions != y)
        error_fraction = incorrect_predictions / len(y)
        return error_fraction
    

    def train(self, X, y, epochs, batch_size, early_stopping=False, patience=5, X_test=None, y_test=None):
        """
        Train the network using mini-batch gradient descent.

        Parameters:
        X (ndarray): Training data.
        y (ndarray): Training labels.
        epochs (int): Number of epochs.
        batch_size (int): Mini-batch size.
        validation_data (tuple): Tuple of validation data (X_val, y_val).
        early_stopping (bool): Whether to use early stopping.
        patience (int): Number of epochs to wait for improvement before stopping.
        """
        best_loss = float('inf')
        wait = 0

        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                activations = self.forward(X_batch)
                self.backward(activations, y_batch)
            
            # Monitor loss and accuracy
            if epoch % 10 == 0 or epoch == epochs - 1:
                loss = self.compute_loss(self.forward(X)[-1], y)

                self.error_history_train.append(self.calculate_error_fraction(X, y))
                #print(f"Epoch {epoch}, Loss: {loss:.4f}, and Error Fraction: {self.error_history_train[-1]:.4f}")

                if X_test is not None and y_test is not None:                
                    val_loss = self.compute_loss(self.forward(X_test)[-1], y_test)
                    self.error_history_test.append(self.calculate_error_fraction(X_test, y_test))
                    #print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, and Error Fraction: {self.error_history_test[-1]:.4f}")
                    
                    if early_stopping:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            wait = 0
                        else:
                            wait += 1
                            if wait > patience:
                                print("Early stopping triggered.")
                                break

    def predict(self, X):
        """
        Make predictions on input data.

        Parameters:
        X (ndarray): Input data.

        Returns:
        ndarray: Predicted labels.
        """
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute the categorical cross-entropy loss.

        Parameters:
        y_pred (ndarray): Predicted probabilities.
        y_true (ndarray): True labels (one-hot encoded).

        Returns:
        float: Loss value.
        """
        # Convert y_true to one-hot encoding
        y_true_onehot = np.eye(y_pred.shape[1])[y_true]  

        # Compute cross-entropy loss
        return -np.mean(np.sum(y_true_onehot * np.log(y_pred + 1e-8), axis=1))
