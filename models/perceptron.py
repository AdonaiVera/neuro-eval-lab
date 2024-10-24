import numpy as np

class Perceptron:
    def __init__(self, input_size=784):
        # Initialize weights randomly between 0 and 0.5, including weight for bias
        self.weights = np.random.rand(input_size + 1) * 0.5 
        self.input_size = input_size
    
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