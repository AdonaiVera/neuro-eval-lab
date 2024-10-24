import numpy as np
from models.perceptron import Perceptron
from methods.evaluator_metrics import calculate_error_fraction, calculate_metrics

def main():
    # Define paths for data and weights
    train_image_file = 'data/train_images.txt'
    train_label_file = 'data/train_labels.txt'
    test_image_file = 'data/test_images.txt'
    test_label_file = 'data/test_labels.txt'
    initial_weights_file = 'data/initial_weights.txt'
    
    # Load training and test data
    train_images = np.loadtxt(train_image_file)
    train_labels = np.loadtxt(train_label_file)
    test_images = np.loadtxt(test_image_file)
    test_labels = np.loadtxt(test_label_file)
    
    # Create and initialize the perceptron
    perceptron = Perceptron(input_size=784)
    perceptron.save_weights(initial_weights_file)
    
    # Calculate error fraction on the training set (before training)
    print("\nCalculating error fraction on the training set:")
    calculate_error_fraction(perceptron, train_images, train_labels)
    
    # Calculate metrics on the test set (before training)
    print("\nCalculating performance metrics on the test set:")
    calculate_metrics(perceptron, test_images, test_labels)

if __name__ == "__main__":
    main()
