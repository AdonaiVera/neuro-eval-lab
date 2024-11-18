import numpy as np
import pandas as pd
import math

class NeuralNetwork:
    """
    Implementation of a dense neural network for classification tasks.
    """
    def __init__(self, input_size=784, hidden_size=100, num_hidden_layers=1, output_size=10, learning_rate=0.01, batch_size=32, num_epochs=100, beta=0.9, autoencoder=True):
        """
        Initialize network architecture, weights, and velocity for optimization.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.layer_dimensions = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
        self.learning_rate=learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta = beta
        self.autoencoder = autoencoder

        # Initialize weights
        self.weights = {}
        for l in range(1, len(self.layer_dimensions)):
            self.weights[f'W{l}'] = np.random.randn(self.layer_dimensions[l], self.layer_dimensions[l - 1]) * 0.01
            self.weights[f'b{l}'] = np.zeros((self.layer_dimensions[l], 1))

        # Initialize velocity for momentum optimization
        self.velocity = {
            f'dW{l + 1}': np.zeros_like(self.weights[f'W{l + 1}'])
            for l in range(len(self.layer_dimensions) - 1)
        }
        self.velocity.update({
            f'db{l + 1}': np.zeros_like(self.weights[f'b{l + 1}'])
            for l in range(len(self.layer_dimensions) - 1)
        })

        self.inputs = None
        self.labels = None

    def forward_step(self, previous_activation, W, b, activation_function):
        """
        Implements a single forward step for one layer.
        """
        def linear_forward(A, W, b):
            Z = np.dot(W, A) + b
            return Z, (A, W, b)

        if activation_function == "sigmoid":
            Z, linear_cache = linear_forward(previous_activation, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation_function == "relu":
            Z, linear_cache = linear_forward(previous_activation, W, b)
            A, activation_cache = self.relu(Z)

        return A, (linear_cache, activation_cache)

    def forward_pass(self, X, weights):
        """
        Perform a full forward pass through the network.
        """
        caches = []
        activation = X
        num_layers = len(weights) // 2

        for l in range(1, num_layers):
            previous_activation = activation
            activation, cache = self.forward_step(
                previous_activation, weights[f'W{l}'], weights[f'b{l}'], activation_function='relu')
            caches.append(cache)

        output_activation, cache = self.forward_step(
            activation, weights[f'W{num_layers}'], weights[f'b{num_layers}'], activation_function='sigmoid')
        caches.append(cache)
        return output_activation, caches

    def linear_backward(self, gradient_Z, cache):
        """
        Perform a single backward step for the linear portion of the layer.
        """
        A_previous, W, _ = cache
        num_samples = A_previous.shape[1]

        gradient_W = np.dot(gradient_Z, A_previous.T) / num_samples
        gradient_b = np.sum(gradient_Z, axis=1, keepdims=True) / num_samples
        gradient_A_previous = np.dot(W.T, gradient_Z)

        return gradient_A_previous, gradient_W, gradient_b

    def relu_backward(self, gradient_A, cache):
        """
        Compute backward pass for ReLU activation.
        """
        Z = cache
        gradient_Z = np.array(gradient_A, copy=True)
        gradient_Z[Z <= 0] = 0
        return gradient_Z

    def sigmoid_backward(self, gradient_A, cache):
        """
        Compute backward pass for sigmoid activation.
        """
        Z = cache
        sigmoid_output = 1 / (1 + np.exp(-Z))
        gradient_Z = gradient_A * sigmoid_output * (1 - sigmoid_output)
        return gradient_Z

    def backward_step(self, gradient_A, cache, activation_function):
        """
        Implements a single backward step for one layer.
        """
        linear_cache, activation_cache = cache

        if activation_function == "relu":
            gradient_Z = self.relu_backward(gradient_A, activation_cache)
        elif activation_function == "sigmoid":
            gradient_Z = self.sigmoid_backward(gradient_A, activation_cache)

        gradient_A_previous, gradient_W, gradient_b = self.linear_backward(gradient_Z, linear_cache)
        return gradient_A_previous, gradient_W, gradient_b

    def backward_pass(self, predicted, actual, caches):
        """
        Perform a full backward pass through the network.
        """
        gradients = {}
        num_layers = len(caches)

        actual = actual.reshape(predicted.shape)

        if self.autoencoder:
            gradient_output = 2*(predicted - actual)
        else:
            gradient_output = - (np.divide(actual, predicted) - np.divide(1 - actual, 1 - predicted))

        current_cache = caches[-1]
        gradients[f'dA{num_layers}'], gradients[f'dW{num_layers}'], gradients[f'db{num_layers}'] = self.backward_step(
            gradient_output, current_cache, "sigmoid")

        for l in reversed(range(num_layers - 1)):
            current_cache = caches[l]
            gradient_A_prev, gradient_W, gradient_b = self.backward_step(
                gradients[f'dA{l + 2}'], current_cache, "relu")
            gradients[f'dA{l + 1}'] = gradient_A_prev
            gradients[f'dW{l + 1}'] = gradient_W
            gradients[f'db{l + 1}'] = gradient_b

        return gradients

    def update_weights(self, weights, gradients, velocity, learning_rate):
        """
        Update weights and biases using momentum optimization.
        """
        num_layers = len(weights) // 2
        for l in range(num_layers):
            velocity[f'dW{l + 1}'] = self.beta * velocity[f'dW{l + 1}'] + (1 - self.beta) * gradients[f'dW{l + 1}']
            velocity[f'db{l + 1}'] = self.beta * velocity[f'db{l + 1}'] + (1 - self.beta) * gradients[f'db{l + 1}']
            weights[f'W{l + 1}'] -= learning_rate * velocity[f'dW{l + 1}']
            weights[f'b{l + 1}'] -= learning_rate * velocity[f'db{l + 1}']

        return weights, velocity

    def create_mini_batches(self, X, Y, batch_size=64, seed=0):
        """
        Create mini-batches for stochastic gradient descent.
        """
        np.random.seed(seed)
        num_samples = X.shape[1]
        mini_batches = []

        shuffled_indices = np.random.permutation(num_samples)
        shuffled_X = X[:, shuffled_indices]

        if self.autoencoder:
            shuffled_Y = Y[:, shuffled_indices]
        else:
            shuffled_Y = Y[:, shuffled_indices].reshape(Y.shape)
            

        num_batches = math.floor(num_samples / batch_size)
        for i in range(num_batches):
            mini_batch_X = shuffled_X[:, i * batch_size:(i + 1) * batch_size]
            mini_batch_Y = shuffled_Y[:, i * batch_size:(i + 1) * batch_size]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        if num_samples % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_batches * batch_size:]
            mini_batch_Y = shuffled_Y[:, num_batches * batch_size:]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        return mini_batches
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Compute Mean Squared Error (MSE) between true and predicted values.
        """
        # Ensure the inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute the squared differences
        squared_differences = (y_pred - y_true) ** 2

        # Compute the mean of the squared differences
        mse = np.mean(squared_differences)
        return mse
    
    def calculate_error_fraction(self, y_true, y_pred):
        """
        Calculate the error fraction using the winner-take-all approach.
        """
        # Convert to class indices if one-hot encoded
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        # Calculate misclassification rate
        num_misclassified = np.sum(y_true != y_pred)
        error_fraction = num_misclassified / len(y_true)
        return error_fraction

    def compute_cost(self, actual, predicted):
        """
        Compute the cost function for the neural network.
        """
        num_samples = actual.shape[1]
        total_error = 0
        for true_values, predicted_values in zip(actual.T, predicted.T):
            total_error += np.square(true_values - predicted_values).mean()
        return total_error / num_samples
    
    def train(self, X, Y, X_test, Y_test):
        """
        Train the neural network using mini-batch gradient descent and momentum optimization.
        """
        self.inputs = X.T
        self.labels = Y.T
        errors_train = []
        errors_test = []

        # Error fraction before training
        predictions_train = self.predict(self.inputs.T)
        predictions_test = self.predict(X_test)
        initial_train_error = self.calculate_error_fraction(self.labels.T, predictions_train)
        initial_test_error = self.calculate_error_fraction(Y_test, predictions_test)
        print(f"Initial Training Error: {initial_train_error}")
        print(f"Initial Test Error: {initial_test_error}")

        for epoch in range(self.num_epochs):
            mini_batches = self.create_mini_batches(self.inputs, self.labels, self.batch_size)

            for mini_batch_X, mini_batch_Y in mini_batches:
                # Forward propagation
                activations, caches = self.forward_pass(mini_batch_X, self.weights)
                # Backward propagation
                gradients = self.backward_pass(activations, mini_batch_Y, caches)
                # Update weights
                self.weights, self.velocity = self.update_weights(self.weights, gradients, self.velocity, self.learning_rate)

            # Logging error for visualization
            if epoch % 10 == 0:
                predictions_train = self.predict(self.inputs.T)
                predictions_test = self.predict(X_test)
                if self.autoencoder:
                    train_error = self.compute_cost(self.labels.T, predictions_train)
                    test_error = self.compute_cost(Y_test, predictions_test)
                else:
                    train_error = self.calculate_error_fraction(self.labels.T, predictions_train)
                    test_error = self.calculate_error_fraction(Y_test, predictions_test)

                print(f"Epoch {epoch}: Training Error: {train_error}, Test Error: {test_error}")
                errors_train.append(train_error)
                errors_test.append(test_error)

            if test_error <= 0.001:
                print("Early stopping...")
                break

        print(f"Final Training Error: {errors_train[-1]}")
        print(f"Final Test Error: {errors_test[-1]}")

        return self.weights, errors_train, errors_test

    def predict(self, X):
        """
        Predict labels for input data using trained weights.
        """
        activations, _ = self.forward_pass(X.T, self.weights)
        return self.apply_threshold(activations.T)

    def apply_threshold(self, predictions):
        """
        Apply thresholding to predictions.
        """
        predictions[predictions >= 0.75] = 1
        predictions[predictions <= 0.25] = 0
        return predictions
    
    def sigmoid(self, Z):
        """
        Sigmoid activation function.
        """
        A = 1 / (1 + np.exp(-Z))
        return A, Z
    
    def relu(self, Z):
        """
        ReLU activation function.
        """
        A = np.maximum(0, Z)
        return A, Z
    