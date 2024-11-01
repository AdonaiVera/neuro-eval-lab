import numpy as np
from models.perceptron import Perceptron
from methods.preprocess import prepare_data_for_perceptrons
from methods.evaluator_metrics import (
    calculate_error_fraction,
    calculate_metrics,
    evaluate_bias_range,
    plot_metrics_comparison,
    plot_metrics_vs_bias,
    plot_roc_curve,
    plot_weights_as_heatmaps,
    classify_challenge_set,
    plot_classification_counts,
    calculate_balanced_accuracy_error,
    plot_metrics_comparison_for_all_digits,
    plot_all_perceptrons_training_error
)

def main_first_problem():
    # Define paths for data and weights
    train_image_file = 'data/train_images.txt'
    train_label_file = 'data/train_labels.txt'
    test_image_file = 'data/test_images.txt'
    test_label_file = 'data/test_labels.txt'
    challenge_image_file = 'data/challenge_images.txt'
    challenge_label_file = 'data/challenge_labels.txt'
    initial_weights_file = 'data/initial_weights.txt'
    final_weights_file = 'data/final_weights.txt'
    
    # Load training and test data
    train_images = np.loadtxt(train_image_file)
    train_labels = np.loadtxt(train_label_file)
    test_images = np.loadtxt(test_image_file)
    test_labels = np.loadtxt(test_label_file)
    challenge_images = np.loadtxt(challenge_image_file)
    challenge_labels = np.loadtxt(challenge_label_file)
    
    # Create and initialize the perceptron 
    '''
    Testing multiples learning rate to get the best results:
    LR: 0.01
    Trained model performance on the test set:
    Error Fraction: 0.0100
    Precision: 0.9900
    Recall: 0.9900
    F1 Score: 0.9900
    ---
    LR: 0.001
    Trained model performance on the test set:
    Error Fraction: 0.0250
    Precision: 0.9612
    Recall: 0.9900
    F1 Score: 0.9754
    '''
    perceptron = Perceptron(input_size=784, learning_rate=0.01)
    perceptron.save_weights(initial_weights_file)
    
    # Calculate and display the metrics
    initial_error_fraction = calculate_error_fraction(perceptron, test_images, test_labels)
    initial_precision, initial_recall, initial_f1_score = calculate_metrics(perceptron, test_images, test_labels)

    print("\nUntrained model performance on the test set:")
    print(f"Error Fraction: {initial_error_fraction:.4f}")
    print(f"Precision: {initial_precision:.4f}")
    print(f"Recall: {initial_recall:.4f}")
    print(f"F1 Score: {initial_f1_score:.4f}")

    # Train the perceptron
    print("\nTraining the perceptron...")
    perceptron.train(train_images, train_labels)
    
    # Save the final weights after training
    perceptron.save_weights(final_weights_file)

    # Plot training error fraction history
    print("\nPlotting training error fraction over epochs...")
    perceptron.plot_error_history()
    
    # Evaluate the trained perceptron on the test set
    print("\nEvaluating the trained perceptron on the test set:")
    final_error_fraction = calculate_error_fraction(perceptron, test_images, test_labels)
    final_precision, final_recall, final_f1_score = calculate_metrics(perceptron, test_images, test_labels)

    print("\nTrained model performance on the test set:")
    print(f"Error Fraction: {final_error_fraction:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    print(f"F1 Score: {final_f1_score:.4f}")

    before_values = [initial_error_fraction, initial_precision, initial_recall, initial_f1_score]
    after_values = [final_error_fraction, final_precision, final_recall, final_f1_score]

    # Plotting before and after training metrics
    plot_metrics_comparison(before_values, after_values)

    # Evaluate metrics across a range of bias values
    print("\nEvaluating metrics for a range of bias values...")
    metrics = evaluate_bias_range(perceptron, test_images, test_labels)

    # Plot metrics vs. bias
    plot_metrics_vs_bias(metrics)
    
    # Plot ROC curve
    plot_roc_curve(metrics)

    # Calculate distances to the ideal point (0,1) on the ROC curve
    distances = [
        np.sqrt((1 - tp)**2 + fp**2)
        for tp, fp in zip(metrics["true_positives"], metrics["false_positives"])
    ]

    print("Check true positive and false positive values:")
    print(metrics["true_positives"])
    print(metrics["false_positives"])
    # Identify the best bias index based on the ROC curve
    best_roc_bias_idx = np.argmin(distances)
    best_roc_bias_value = metrics["bias_values"][best_roc_bias_idx]
    print(f"\nBest bias value based on ROC (θ*): {best_roc_bias_value:.4f} with TPR: {metrics['true_positives'][best_roc_bias_idx]:.4f} and FPR: {metrics['false_positives'][best_roc_bias_idx]:.4f}")


    # Load initial and final weights to display as heatmaps
    initial_weights = np.loadtxt(initial_weights_file)[1:]  # Exclude bias
    final_weights = np.loadtxt(final_weights_file)[1:]      # Exclude bias
    
    # Plot the weights
    print("\nPlotting initial and final weights as heatmaps...")
    plot_weights_as_heatmaps(
        weights_list=[initial_weights, final_weights],
        titles=["Initial Weights (28x28)", "Final Weights (28x28)"]
    )

     # Classify the challenge set and calculate the classification counts
    classification_counts = classify_challenge_set(perceptron, challenge_images, challenge_labels)
    
    # Display the classification results as a table
    plot_classification_counts(classification_counts)

def main_second_problem():
    # File paths (assumed to be in the 'data' folder)
    train_image_file = 'data/MNISTnumImages5000_balanced.txt'
    train_label_file = 'data/MNISTnumLabels5000_balanced.txt'

    # Prepare data for all 10 perceptrons
    training_data, test_data = prepare_data_for_perceptrons(
        image_file='data/MNISTnumImages5000_balanced.txt', 
        label_file='data/MNISTnumLabels5000_balanced.txt', 
        sampling_strategy='undersample'  # or 'undersample'
    )

    # Track the training error over epochs for each perceptron
    training_errors = {digit: [] for digit in range(10)}

    # Store metrics for each perceptron
    before_metrics = {'balanced_accuracy_error': [], 'precision': [], 'recall': [], 'f1_score': []}
    after_metrics = {'balanced_accuracy_error': [], 'precision': [], 'recall': [], 'f1_score': []}

    # Use the 9th perceptron to determine the number of epochs required
    perceptron_9 = Perceptron(input_size=784, learning_rate=0.01)
    train_images_9 = training_data[9]['images']
    train_labels_9 = training_data[9]['labels']
    
    perceptron_9.train(train_images_9, train_labels_9)
    # Plot training error fraction history
    print("\nPlotting training error fraction over epochs...")
    perceptron_9.plot_error_history()
 
    # Base on the trainig of perceptron 9, we can see that the error fraction is already low after 8 epochs
    max_epochs=8

    # Track the training error over epochs for each perceptron

    # Example usage
    for digit in range(10):
        print(f"Training Perceptron for digit {digit}...")
        # Initialize the perceptron for the current digit
        perceptron = Perceptron(input_size=784, learning_rate=0.01)

        # Get training and test data for the current perceptron
        images_train = training_data[digit]['images']
        labels_train = training_data[digit]['labels']
        images_test = test_data[digit]['images']
        labels_test = test_data[digit]['labels']

        # Calculate initial metrics on the test set before training
        initial_error_fraction = calculate_balanced_accuracy_error(perceptron, images_test, labels_test)
        initial_precision, initial_recall, initial_f1_score = calculate_metrics(perceptron, images_test, labels_test)
        
        
        # Store initial metrics for comparison later
        before_metrics['balanced_accuracy_error'].append(initial_error_fraction)
        before_metrics['precision'].append(initial_precision)
        before_metrics['recall'].append(initial_recall)
        before_metrics['f1_score'].append(initial_f1_score)
        
        print(f"Digit {digit} - Initial Test Set Metrics:")
        print(f"Error Fraction: {initial_error_fraction:.4f}")
        print(f"Precision: {initial_precision:.4f}")
        print(f"Recall: {initial_recall:.4f}")
        print(f"F1 Score: {initial_f1_score:.4f}")
        
        # Train the perceptron for the determined number of epochs
        perceptron.train(images_train, labels_train, max_epochs)
        training_errors[digit] = perceptron.get_error_history()
        final_error_fraction = calculate_balanced_accuracy_error(perceptron, images_test, labels_test)
        final_precision, final_recall, final_f1_score = calculate_metrics(perceptron, images_test, labels_test)

        # Store final metrics for comparison
        after_metrics['balanced_accuracy_error'].append(final_error_fraction)
        after_metrics['precision'].append(final_precision)
        after_metrics['recall'].append(final_recall)
        after_metrics['f1_score'].append(final_f1_score)
        
        print(f"\nDigit {digit} - Final Test Set Metrics after Training:")
        print(f"Balanced Accuracy Error: {final_error_fraction:.4f}")
        print(f"Precision: {final_precision:.4f}")
        print(f"Recall: {final_recall:.4f}")
        print(f"F1 Score: {final_f1_score:.4f}")

    # Plot the metrics comparison for all digits
    plot_metrics_comparison_for_all_digits(before_metrics, after_metrics)

    # Plot the training error fraction history for all perceptrons
    print(training_errors)
    plot_all_perceptrons_training_error(training_errors)


if __name__ == "__main__":
    # Run the main function for the first problem
    #main_first_problem()

    # Run the main function for the second problem
    main_second_problem()

