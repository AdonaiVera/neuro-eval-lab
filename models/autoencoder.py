import numpy as np
import matplotlib.pyplot as plt

class AutoencoderNN:
    def __init__(self, input_size, hidden_size, learning_rate=0.01, momentum=0.9):
        """
        Initialize the autoencoder neural network.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, input_size) * 0.1
        self.bias_output = np.zeros((1, input_size))

        # Momentum terms
        self.velocity_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_bias_output = np.zeros_like(self.bias_output)

    def reconstruction_loss(self, y_pred, y_true):
        """
        Calculate the reconstruction loss.
        """
        diff = y_true - y_pred
        return 0.5 * np.mean(np.sum(diff**2, axis=1))

    def forward(self, X):
        """
        Perform forward propagation.
        """
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.output_input  # Reconstruction task doesn't need activation
        return self.output

    def backward(self, X, y_true):
        """
        Perform backward propagation and update weights and biases.
        """
        batch_size = X.shape[0]
        num_pixels = X.shape[1]

        # Output layer gradients
        output_error = (self.output - y_true) / num_pixels  # Scale by 784
        grad_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        grad_bias_output = np.sum(output_error, axis=0, keepdims=True)

        # Hidden layer gradients
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_input)
        grad_weights_input_hidden = np.dot(X.T, hidden_error)
        grad_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Momentum updates
        self.velocity_hidden_output = self.momentum * self.velocity_hidden_output - self.learning_rate * grad_weights_hidden_output
        self.velocity_bias_output = self.momentum * self.velocity_bias_output - self.learning_rate * grad_bias_output
        self.velocity_input_hidden = self.momentum * self.velocity_input_hidden - self.learning_rate * grad_weights_input_hidden
        self.velocity_bias_hidden = self.momentum * self.velocity_bias_hidden - self.learning_rate * grad_bias_hidden

        # Apply updates
        self.weights_hidden_output += self.velocity_hidden_output
        self.bias_output += self.velocity_bias_output
        self.weights_input_hidden += self.velocity_input_hidden
        self.bias_hidden += self.velocity_bias_hidden

    def reconstruction_loss(self, y_pred, y_true):
        """
        Calculate the reconstruction loss.
        """
        diff = y_true - y_pred
        return 0.5 * np.mean(np.sum(diff**2, axis=1))
    
    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, threshold=0.00001):
        """
        Train the autoencoder and evaluate MRE for the training and test sets.
        """
        history_train = []
        history_test = []

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]

            # Mini-batch gradient descent
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                self.forward(X_batch)
                self.backward(X_batch, X_batch)

            # Evaluate performance every 10 epochs or final epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                train_mre = self.reconstruction_loss(self.forward(X_train), X_train)
                test_mre = self.reconstruction_loss(self.forward(X_test), X_test)
                history_train.append(train_mre)
                history_test.append(test_mre)
                print(f"Epoch {epoch}: Train MRE: {train_mre:.4f}, Test MRE: {test_mre:.4f}")

            # Early stopping
            if train_mre < threshold:
                print(f"Early stopping at epoch {epoch}. Train MRE: {train_mre:.4f}")
                break

        return history_train, history_test

    def inference(self, X):
        """
        Perform inference on a dataset and calculate reconstruction errors.

        Parameters:
        X (ndarray): Input data (e.g., training or test set).

        Returns:
        tuple: A tuple containing:
            - Reconstructed outputs (ndarray)
            - Reconstruction errors for each input (ndarray)
            - Mean reconstruction error (float)
        """
        # Forward pass
        reconstructed_outputs = self.forward(X)

        # Calculate reconstruction error for each sample
        reconstruction_errors = np.sum((X - reconstructed_outputs) ** 2, axis=1) / X.shape[1]  # Average pixel error

        # Calculate the mean reconstruction error (MRE)
        mean_reconstruction_error = np.mean(reconstruction_errors)

        print(f"Mean Reconstruction Error: {mean_reconstruction_error:.4f}")

        return reconstructed_outputs, reconstruction_errors, mean_reconstruction_error
    
    def visualize_hidden_weights(self):
        """
        Visualize the hidden neuron weights as 28x28 grayscale images.
        """
        print("\nVisualizing hidden neuron weights...")
        num_neurons = self.hidden_size
        for i in range(num_neurons):
            weights = self.weights_input_hidden[:, i]
            normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())  # Normalize to [0, 1]
            image = normalized_weights.reshape(28, 28)

            plt.imshow(image, cmap='gray')
            plt.title(f"Hidden Neuron {i+1}")
            plt.axis('off')
            plt.show()


    def calculate_per_digit_stats(self, X_test, y_test):
        """
        Calculate MRE and standard deviation for each digit.
        """
        unique_digits = np.unique(y_test)
        stats = {}

        for digit in unique_digits:
            digit_indices = np.where(y_test == digit)[0]
            X_digit = X_test[digit_indices]

            # Calculate MRE and standard deviation
            errors = np.sum((self.forward(X_digit) - X_digit) ** 2, axis=1)
            mre = np.mean(errors)
            std_dev = np.std(errors)
            stats[int(digit)] = {'mre': mre, 'std_dev': std_dev}

        # Overall MRE
        overall_errors = np.sum((self.forward(X_test) - X_test) ** 2, axis=1)
        stats['overall'] = {
            'mre': np.mean(overall_errors),
            'std_dev': np.std(overall_errors)
        }

        print("\nPer-Digit MRE and Standard Deviation:")
        for digit, values in stats.items():
            if digit == 'overall':
                print(f"Overall - MRE: {values['mre']:.4f}, Std Dev: {values['std_dev']:.4f}")
            else:
                print(f"Digit {digit} - MRE: {values['mre']:.4f}, Std Dev: {values['std_dev']:.4f}")

        return stats
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    def calculate_per_digit_mre(self, X_test, y_test):
        """
        Calculate the MRE for each digit in the test set and overall.

        Parameters:
        X_test (ndarray): Test images.
        y_test (ndarray): Test labels.

        Returns:
        dict: Dictionary with MRE for each digit and the overall MRE.
        """
        unique_digits = np.unique(y_test)
        per_digit_mre = {}
        overall_loss = []

        for digit in unique_digits:
            # Filter images for the current digit
            digit_indices = np.where(y_test == digit)[0]
            X_digit = X_test[digit_indices]

            # Calculate MRE for the current digit
            digit_loss = self.reconstruction_loss(self.forward(X_digit), X_digit)
            per_digit_mre[int(digit)] = digit_loss
            overall_loss.extend(digit_loss)

        # Calculate overall MRE
        overall_mre = np.mean(overall_loss)
        per_digit_mre['overall'] = overall_mre

        print("\nPer-Digit MRE:")
        for digit, mre in per_digit_mre.items():
            if digit == 'overall':
                print(f"Overall MRE: {mre:.4f}")
            else:
                print(f"Digit {digit} MRE: {mre:.4f}")

        return per_digit_mre