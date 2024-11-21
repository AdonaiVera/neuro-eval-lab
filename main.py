import numpy as np
from models.perceptron import Perceptron
from models.neural_network import NeuralNetwork
from methods.preprocess import prepare_data_for_perceptrons, prepare_data_for_multiclass, prepare_data_for_multiclass_digits
from methods.evaluator_metrics import (
    calculate_error_fraction,
    calculate_metrics,
    calculate_error_fraction_multiple_class,
    evaluate_bias_range,
    plot_metrics_comparison,
    plot_metrics_vs_bias,
    plot_roc_curve,
    plot_weights_as_heatmaps,
    classify_challenge_set,
    plot_classification_counts,
    calculate_balanced_accuracy_error,
    plot_metrics_comparison_for_all_digits,
    plot_all_perceptrons_training_error,
    plot_error_fraction,
    plot_confusion_matrix,
    plot_error_fraction_encoders,
    plot_inference,
    plot_mre_comparison,
    plot_hidden_layer_features,
    calculate_per_digit_stats,
    visualize_hidden_neurons,
    plot_reconstruction_errors,
    plot_reconstructed_images
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
        image_file=train_image_file, 
        label_file=train_label_file, 
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

def main_feed_forward_nn():
    # File paths (assumed to be in the 'data' folder)
    train_image_file = 'data/MNISTnumImages5000_balanced.txt'
    train_label_file = 'data/MNISTnumLabels5000_balanced.txt'

    # Prepare balanced training and test datasets
    X_train, y_train, X_test, y_test = prepare_data_for_multiclass(train_image_file, train_label_file)

    # Initialize feed-forward neural network
    input_size=784
    hidden_size=200
    num_hidden_layers=1
    output_size=10
    learning_rate = 0.01
    epochs = 200
    batch_size = 32
    beta=0.9

    # Initialize the model
    model = NeuralNetwork(input_size, hidden_size, num_hidden_layers, output_size, learning_rate, batch_size, epochs, beta)
    
    # Train the model
    weights, errors_train, errors_test = model.train(X_train, y_train, X_test, y_test)

    # Plot error
    plot_error_fraction(errors_train, errors_test)

    # Prediction train and test    
    y_preds_train = model.predict(X_train)
    y_preds_test = model.predict(X_test)

    # List of class labels (e.g., for digit classification)
    class_labels = list(range(10))

    # Plot confusion matrices for training and test sets
    plot_confusion_matrix(y_true=y_train, y_pred=y_preds_train, class_labels=class_labels, title="Training Set Confusion Matrix")
    plot_confusion_matrix(y_true=y_test, y_pred=y_preds_test, class_labels=class_labels, title="Test Set Confusion Matrix")

def main_autoencoder_nn():
    # File paths (assumed to be in the 'data' folder)
    train_image_file = 'data/MNISTnumImages5000_balanced.txt'
    train_label_file = 'data/MNISTnumLabels5000_balanced.txt'

    # Prepare balanced training and test datasets
    X_train, y_train, X_test, y_test = prepare_data_for_multiclass(train_image_file, train_label_file)

    # Initialize feed-forward neural network
    input_size=784
    hidden_size=200
    num_hidden_layers=1
    output_size=10
    learning_rate = 0.01
    epochs = 200
    batch_size = 32
    beta=0.9

    # Initialize the model
    model = NeuralNetwork(input_size, hidden_size, num_hidden_layers, output_size, learning_rate, batch_size, epochs, beta)
    
    # Train the model
    weights, _, _ = model.train(X_train, y_train, X_test, y_test)


    # Initialize autoencoder
    output_size=784

    print("\nInitializing AutoencoderNN...")
    autoencoder = NeuralNetwork(input_size, hidden_size, num_hidden_layers, output_size, learning_rate, batch_size, epochs, beta, autoencoder=True)

    # Train the autoencoder
    weights_autoencoder, errors_train_autoencoder, errors_test_autoencoder = autoencoder.train(X_train, X_train, X_test, X_test)


    # Plot error
    plot_error_fraction(errors_train_autoencoder, errors_test_autoencoder, title="MRE Over Epochs Autoencoder")


    # Calculate per-digit MRE and standard deviation
    calculate_per_digit_stats(autoencoder, X_train, X_test)

    # Plot MRE comparison
    plot_mre_comparison(autoencoder, X_train, X_test)

    # Plot random inference images
    plot_inference(autoencoder, X_test)

    # Plot features
    plot_hidden_layer_features(weights["W1"], weights_autoencoder["W1"], num_neurons=20)

    # Visualize the features for the first 10 hidden neurons
    visualize_hidden_neurons(weights_autoencoder["W1"], num_neurons=2)

def main_transfer_learning():
    # File paths (assumed to be in the 'data' folder)
    train_image_file = 'data/MNISTnumImages5000_balanced.txt'
    train_label_file = 'data/MNISTnumLabels5000_balanced.txt'

    # Prepare balanced training and test datasets
    X_train, y_train, X_test, y_test = prepare_data_for_multiclass(train_image_file, train_label_file)

    # Initialize hyperparameters (from HW 4)
    input_size = 784
    hidden_size = 200
    num_hidden_layers = 1
    output_size = 784
    learning_rate = 0.01
    epochs = 200
    batch_size = 32
    beta = 0.9

    # Step 1: Train the autoencoder (if not already trained)
    print("Training Autoencoder...")
    autoencoder = NeuralNetwork(input_size, hidden_size, num_hidden_layers, input_size, learning_rate, batch_size, epochs, beta, autoencoder=True)
    weights_autoencoder, _, _ = autoencoder.train(X_train, X_train, X_test, X_test)

    # Save the input-to-hidden weights for transfer learning
    pre_trained_weights = weights_autoencoder["W1"]

    # Step 2: Case I - Freeze input-to-hidden weights and train hidden-to-output weights
    print("Starting Case I: Train hidden-to-output weights only...")
    output_size = 10

    model_case1 = NeuralNetwork(input_size, hidden_size, num_hidden_layers, output_size, learning_rate, batch_size, epochs, beta, frozen_layers=[1])
    # Set pre-trained weights for input-to-hidden
    model_case1.weights["W1"] = pre_trained_weights  

    # Train only the output layer
    _, errors_train_case_1, errors_test_case_1 = model_case1.train(X_train, y_train, X_test, y_test) 

    # Plot error
    plot_error_fraction(errors_train_case_1, errors_test_case_1)

    # Prediction train and test    
    y_preds_train = model_case1.predict(X_train)
    y_preds_test = model_case1.predict(X_test)

    # List of class labels (e.g., for digit classification)
    class_labels = list(range(10))

    # Plot confusion matrices for training and test sets
    plot_confusion_matrix(y_true=y_train, y_pred=y_preds_train, class_labels=class_labels, title="Training Set Confusion Matrix")
    plot_confusion_matrix(y_true=y_test, y_pred=y_preds_test, class_labels=class_labels, title="Test Set Confusion Matrix")


    # Step 3: Case II - Train both input-to-hidden and hidden-to-output weights
    print("Starting Case II: Train all layers...")
    model_case2 = NeuralNetwork(input_size, hidden_size, num_hidden_layers, output_size, learning_rate, batch_size, epochs, beta)
    
    # Set pre-trained weights for input-to-hidden
    model_case2.weights["W1"] = pre_trained_weights  

    # Train both layers
    _, errors_train_case_2, errors_test_case_2 = model_case2.train(X_train, y_train, X_test, y_test)  

    # Plot error
    plot_error_fraction(errors_train_case_2, errors_test_case_2)

    # Prediction train and test    
    y_preds_train = model_case2.predict(X_train)
    y_preds_test = model_case2.predict(X_test)

    # List of class labels (e.g., for digit classification)
    class_labels = list(range(10))

    # Plot confusion matrices for training and test sets
    plot_confusion_matrix(y_true=y_train, y_pred=y_preds_train, class_labels=class_labels, title="Training Set Confusion Matrix")
    plot_confusion_matrix(y_true=y_test, y_pred=y_preds_test, class_labels=class_labels, title="Test Set Confusion Matrix")

    print("Transfer learning completed successfully.")

def main_transfer_learning_different_datasets():
    # File paths (assumed to be in the 'data' folder)
    train_image_file = 'data/MNISTnumImages5000_balanced.txt'
    train_label_file = 'data/MNISTnumLabels5000_balanced.txt'

    # Prepare datasets for digits 0-4 and 5-9
    X_train_0_to_4, y_train_0_to_4, X_test_0_to_4, y_test_0_to_4 = prepare_data_for_multiclass_digits(
        train_image_file, train_label_file, digits=[0, 1, 2, 3, 4]
    )

    print(y_train_0_to_4)
    X_test_5_to_9, y_test_5_to_9 = prepare_data_for_multiclass_digits(
        train_image_file, train_label_file, digits=[5, 6, 7, 8, 9], test_only=True
    )

    # Initialize hyperparameters
    input_size = 784
    hidden_size = 200
    num_hidden_layers = 1
    output_size = 784
    learning_rate = 0.01
    epochs = 200
    batch_size = 32
    beta = 0.9

    # Train the autoencoder on digits 0-4
    print("Training autoencoder on digits 0-4...")
    autoencoder = NeuralNetwork(input_size, hidden_size, num_hidden_layers, output_size, learning_rate, batch_size, epochs, beta, autoencoder=True)
    weights_autoencoder, _, _ = autoencoder.train(X_train_0_to_4, X_train_0_to_4, X_test_0_to_4, X_test_0_to_4)

    # Calculate reconstruction error for digits 0-4
    print("\nCalculating reconstruction errors for digits 0-4...")
    mre_0_to_4 = []
    std_0_to_4 = []
    for digit in range(5):
        X_test_digit = X_test_0_to_4[np.argmax(y_test_0_to_4, axis=1) == digit]
        reconstructed = autoencoder.predict(X_test_digit)
        error = np.mean((X_test_digit - reconstructed) ** 2, axis=1)
        mre_0_to_4.append(np.mean(error))
        std_0_to_4.append(np.std(error))
        print(f"Digit {digit}: MRE = {mre_0_to_4[-1]:.6f}, Std Dev = {std_0_to_4[-1]:.6f}")

    # Calculate reconstruction error for digits 5-9
    print("\nCalculating reconstruction errors for digits 5-9...")
    mre_5_to_9 = []
    std_5_to_9 = []
    for digit in range(5, 10):
        X_test_digit = X_test_5_to_9[np.argmax(y_test_5_to_9, axis=1) == digit - 5]
        reconstructed = autoencoder.predict(X_test_digit)
        error = np.mean((X_test_digit - reconstructed) ** 2, axis=1)
        mre_5_to_9.append(np.mean(error))
        std_5_to_9.append(np.std(error))
        print(f"Digit {digit}: MRE = {mre_5_to_9[-1]:.6f}, Std Dev = {std_5_to_9[-1]:.6f}")

    # Combine results for digits 0-4 and 5-9
    digits = list(range(10))
    mre_all = mre_0_to_4 + mre_5_to_9
    std_all = std_0_to_4 + std_5_to_9

    # Display results in tabular format
    print("\nSummary of Reconstruction Errors:")
    print("Digit\tMRE\t\tStd Dev")
    for digit, mre, std in zip(digits, mre_all, std_all):
        print(f"{digit}\t{mre:.6f}\t{std:.6f}")

    # Plot results
    plot_reconstruction_errors(digits, mre_all, std_all)

    # Visualize original and reconstructed images for digits 0-4
    print("\nVisualizing original and reconstructed images for digits 5-9...")
    plot_reconstructed_images(autoencoder, X_test_5_to_9, y_test_5_to_9, digits=[5, 6, 7, 8, 9])

    print("\nTask completed: Transfer learning across datasets.")


if __name__ == "__main__":
    # Run the main function for the first problem
    #main_first_problem()

    # Run the main function for the second problem
    #main_second_problem()

    # Run the main function for the feed-forward neural network
    #main_feed_forward_nn()
    
    # Run the main autoencoder
    #main_autoencoder_nn()

    # Run the main transfer learning differents tasks
    #main_transfer_learning()

    # Run the main transfer learning differents datasrt
    main_transfer_learning_different_datasets()