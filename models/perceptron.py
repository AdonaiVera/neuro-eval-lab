import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size=784, learning_rate=0.01):
        # Initialize weights randomly between 0 and 0.5, including weight for bias
        self.weights = np.random.rand(input_size + 1) * 0.5 
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.error_history = []
    
    def save_weights(self, filepath):
        # Save the initial weights to a text file
        np.savetxt(filepath, self.weights)
        print(f"Initial weights saved to {filepath}")
    
    def predict(self, x):
        # Add bias input (x_0 = 1) to the input vector
        x_with_bias = np.insert(x, 0, 1) 
        # Compute the weighted sum (s(t))
        s = np.dot(self.weights, x_with_bias)
        # Apply the step function: output is 1 if s > 0, otherwise 0
        return 1 if s > 0 else 0
    
    def train(self, train_images, train_labels, max_epochs=15):
        for epoch in range(max_epochs):
            errors = 0
            for i in range(len(train_labels)):
                # Make a prediction
                prediction = self.predict(train_images[i])
                # Calculate the error (difference between true label and prediction)
                error = train_labels[i] - prediction
                # Update weights if there's an error
                if error != 0:
                    # Update weights according to the perceptron learning rule
                    x_with_bias = np.insert(train_images[i], 0, 1) 
                    self.weights += self.learning_rate * error * x_with_bias
                    errors += 1
            # Calculate error fraction after this epoch
            error_fraction = errors / len(train_labels)
            self.error_history.append(error_fraction)
            print(f"Epoch {epoch + 1}, Error fraction: {error_fraction:.4f}")
            # Stop training if error fraction is sufficiently low
            if error_fraction < 0.01 and epoch > 5:
                print("Training complete with low error fraction.")
                break

    def get_error_history(self):
        return self.error_history

    def plot_error_history(self):
        # Plot the error fraction over epochs
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.error_history) + 1), self.error_history, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Error Fraction")
        plt.title("Training Error Fraction vs. Epoch")
        plt.grid()
        plt.show()