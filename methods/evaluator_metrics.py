import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

def plot_reconstruction_errors(digits, mre, std):
    plt.errorbar(digits, mre, yerr=std, fmt='o', capsize=5, label="Reconstruction Error")
    plt.xlabel("Digit")
    plt.ylabel("Mean Reconstruction Error (MRE)")
    plt.title("Reconstruction Errors for Digits 0-4 and 5-9")
    plt.xticks(digits)
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_error_fraction(model, images, labels):
    incorrect_predictions = 0
    total_samples = len(labels)

    # Evaluate each sample
    for i in range(total_samples):
        prediction = model.predict(images[i])
        if prediction != labels[i]:
            incorrect_predictions += 1

    error_fraction = incorrect_predictions / total_samples
    accuracy = 1 - error_fraction
    print(f"Error fraction: {error_fraction:.4f}, Accuracy: {accuracy:.4f}")
    return error_fraction

def plot_mre_history(history_train, history_test):
    """
    Plot the MRE history for training and test sets over epochs.

    Parameters:
    history_train (list): Training MRE history.
    history_test (list): Test MRE history.
    """
    epochs = range(0, len(history_train) * 10, 10)  # Every 10th epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_train, label='Training MRE')
    plt.plot(epochs, history_test, label='Test MRE')
    plt.xlabel('Epochs')
    plt.ylabel('MRE')
    plt.title('Mean Reconstruction Error (MRE) Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_error_fraction_multiple_class(model, images, labels):
    """
    Calculate the error fraction for multi-class predictions.

    Parameters:
    model (object): The trained model for prediction.
    images (ndarray): Input images.
    labels (ndarray): True labels (class indices).

    Returns:
    float: Error fraction.
    """
    # Predict class indices for all samples
    predicted_labels=model.predict(images)

    # Calculate the number of incorrect predictions
    incorrect_predictions = np.sum(predicted_labels != labels)

    # Calculate error fraction
    error_fraction = incorrect_predictions / len(labels)

    # Optionally calculate accuracy for additional output
    accuracy = 1 - error_fraction
    print(f"Error fraction: {error_fraction:.4f}, Accuracy: {accuracy:.4f}")
    
    return error_fraction


def calculate_balanced_accuracy_error(model, images, labels):
    # Initialize counters for each class
    true_positives = false_positives = true_negatives = false_negatives = 0
    total_samples = len(labels)

    # Evaluate each sample
    for i in range(total_samples):
        prediction = model.predict(images[i])
        if labels[i] == 1:  
            if prediction == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:  
            if prediction == 0:
                true_negatives += 1
            else:
                false_positives += 1

    # Calculate recall for each class
    recall_class_1 = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    recall_class_0 = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    # Calculate balanced accuracy and error
    balanced_accuracy = (recall_class_1 + recall_class_0) / 2
    balanced_accuracy_error = 1 - balanced_accuracy

    print(f"Balanced Accuracy: {balanced_accuracy:.4f}, Balanced Accuracy Error: {balanced_accuracy_error:.4f}")
    return balanced_accuracy_error

def calculate_metrics_multi_class(model, images, labels, num_classes=10):
    """
    Calculate precision, recall, and F1 score for multi-class data.

    Parameters:
    model (object): The trained model for prediction.
    images (ndarray): Input images.
    labels (ndarray): True labels (one-hot encoded or class indices).
    num_classes (int): Number of classes.

    Returns:
    dict: Dictionary containing precision, recall, and F1 score for each class.
    """
    # Predict class indices for all samples
    predictions = np.argmax(model.predict(images), axis=1)
    true_labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    metrics = {"precision": [], "recall": [], "f1_score": []}

    for cls in range(num_classes):
        # True positives, false positives, and false negatives for each class
        tp = np.sum((predictions == cls) & (true_labels == cls))
        fp = np.sum((predictions == cls) & (true_labels != cls))
        fn = np.sum((predictions != cls) & (true_labels == cls))

        # Precision, Recall, and F1 Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1_score)

    # Print overall metrics for the dataset
    print(f"Precision (per class): {metrics['precision']}")
    print(f"Recall (per class): {metrics['recall']}")
    print(f"F1 Score (per class): {metrics['f1_score']}")

    return metrics

def calculate_metrics(model, images, labels):
    true_positives = false_positives = true_negatives = false_negatives = 0

    # Evaluate each sample
    for i in range(len(labels)):
        prediction = model.predict(images[i])
        true_label = labels[i]

        if prediction == 1 and true_label == 1:
            true_positives += 1
        elif prediction == 1 and true_label == 0:
            false_positives += 1
        elif prediction == 0 and true_label == 0:
            true_negatives += 1
        elif prediction == 0 and true_label == 1:
            false_negatives += 1

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"True Positives: {true_positives}, False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}, False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    return precision, recall, f1_score

def plot_confusion_matrix(y_true, y_pred, class_labels, title="Confusion Matrix"):
    """
    Plot a raw confusion matrix (counts) for multi-class classification.

    Parameters:
    y_true (ndarray): Ground truth labels (class indices or one-hot encoded).
    y_pred (ndarray): Predicted labels (class indices or probabilities).
    class_labels (list): List of class labels (e.g., [0, 1, ..., 9]).
    title (str): Title for the confusion matrix plot.
    """
    # Convert one-hot encoded labels to class indices if necessary
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def visualize_hidden_neurons(weights, num_neurons, img_dim=(28, 28), normalize=True):
    """
    Visualize the features learned by hidden neurons.

    Parameters:
    weights (ndarray): The weights of the hidden layer with shape (num_neurons, 784).
    num_neurons (int): Number of hidden neurons to visualize.
    img_dim (tuple): Dimensions of the reshaped image (default is 28x28 for MNIST).
    normalize (bool): Whether to normalize weights to the range [0, 255].

    Returns:
    None
    """
    # Normalize weights to [0, 1] or [0, 255] for visualization
    if normalize:
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        weights = (weights * 255).astype(np.uint8)

    # Select the number of neurons to visualize
    selected_neurons = weights[:num_neurons]

    # Create a figure for visualization
    fig, axes = plt.subplots(1, num_neurons, figsize=(num_neurons * 2, 2))
    fig.suptitle("Visualizations of Hidden Neuron Features", fontsize=16)

    for i, ax in enumerate(axes):
        # Reshape weights into a 28x28 image
        neuron_image = selected_neurons[i].reshape(img_dim)

        # Plot the image
        ax.imshow(neuron_image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Neuron {i + 1}")

    plt.tight_layout()
    plt.show()

def plot_hidden_layer_features(classifier_weights, autoencoder_weights, num_neurons=20):
    """
    Plot the feature images of hidden neurons from the autoencoder and feed-forward network.

    Parameters:
    autoencoder (AutoencoderNN): Trained autoencoder.
    feedforward_nn (FeedForwardNN): Trained feed-forward network.
    num_neurons (int): Number of neurons to select and visualize.
    """
    # Randomly select neurons to visualize
    idx = np.random.choice(classifier_weights.shape[0], num_neurons, replace=False)
    c_weights = classifier_weights[idx, :]
    a_weights = autoencoder_weights[idx, :]

    # Normalize weights for visualization
    c_weights_normalized = (c_weights - np.min(c_weights, axis=1, keepdims=True)) / np.ptp(c_weights, axis=1, keepdims=True)
    a_weights_normalized = (a_weights - np.min(a_weights, axis=1, keepdims=True)) / np.ptp(a_weights, axis=1, keepdims=True)

    # Reshape weights into 28x28 images
    c_images = c_weights_normalized.reshape(-1, 28, 28)
    a_images = a_weights_normalized.reshape(-1, 28, 28)

    # Create a single figure for both feature sets
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    fig.suptitle("Classifier vs Autoencoder Features", fontsize=16)

    for i in range(4):
        for j in range(5):
            # Plot classifier features (left 5 columns)
            ax = axes[i, j]
            ax.imshow(c_images[i * 5 + j], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f"C{i * 5 + j + 1}", fontsize=10)

            # Plot autoencoder features (right 5 columns)
            ax = axes[i, j + 5]
            ax.imshow(a_images[i * 5 + j], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f"A{i * 5 + j + 1}", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_mre_comparison(autoencoder, X_train, X_test):
    """
    Calculate and plot the Mean Reconstruction Error (MRE) of the final network
    for the training and test sets.
    """

    # Predict reconstructed outputs
    y_preds_train = autoencoder.predict(X_train)
    y_preds_test = autoencoder.predict(X_test)

    # Calculate Mean Reconstruction Error (MRE)
    mre_train = np.mean(np.sum((X_train - y_preds_train) ** 2, axis=1))
    mre_test = np.mean(np.sum((X_test - y_preds_test) ** 2, axis=1))

    # Store the results
    mre_results = {"Training Set MRE": mre_train, "Test Set MRE": mre_test}

    # Plot the MREs as side-by-side bars
    plt.figure(figsize=(8, 6))
    plt.bar(mre_results.keys(), mre_results.values(), color=['blue', 'orange'])
    plt.ylabel("Mean Reconstruction Error (MRE)")
    plt.title("MRE of the Final Network on Training and Test Sets")
    plt.show()

    return mre_results

def plot_reconstructed_images(autoencoder, X_test, y_test, digits, samples_per_digit=5):
    """
    For each digit, randomly select samples from the test set, get the reconstructed images,
    and plot the original and reconstructed images in two rows.

    Parameters:
    autoencoder (NeuralNetwork): The trained autoencoder model.
    X_test (ndarray): Test feature set.
    y_test (ndarray): Test labels (one-hot encoded).
    digits (list): List of digits to visualize (e.g., [5, 6, 7, 8, 9]).
    samples_per_digit (int): Number of samples to visualize per digit.
    """
    for digit in digits:
        # Get indices of samples for the current digit
        digit_indices = np.where(np.argmax(y_test, axis=1) == digit - 5)[0]
        
        # Randomly select `samples_per_digit` samples
        selected_indices = random.sample(list(digit_indices), samples_per_digit)
        selected_images = X_test[selected_indices]

        # Get reconstructed images
        reconstructed_images = autoencoder.predict(selected_images)

        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, samples_per_digit, figsize=(10, 4))
        fig.suptitle(f"Original and Reconstructed Images for Digit {digit}", fontsize=14)

        for i, (original, reconstructed) in enumerate(zip(selected_images, reconstructed_images)):
            # Original images (top row)
            axes[0, i].imshow(original.reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Original")

            # Reconstructed images (bottom row)
            axes[1, i].imshow(reconstructed.reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Reconstructed")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
def plot_inference(model, X_test, num_samples=8):
    """
    Plot the original and reconstructed images side by side for multiple samples.

    Parameters:
    images_input (ndarray): Original input images.
    images_output (ndarray): Reconstructed images from the network.
    num_samples (int): Number of samples to display (default: 8).
    """
    # Randomly select indices
    random_indices = np.random.randint(X_test.shape[0], size=num_samples)
    sample_X = X_test[random_indices, :]
    sample_pred = model.predict(sample_X)

    # Create a figure for the plots
    fig, ax = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle("Original (Top) and Reconstructed (Bottom) Images")

    for i in range(num_samples):
        # Plot original image
        ax[0][i].imshow(sample_X[i].reshape(28, 28), cmap='gray')
        ax[0][i].axis('off')

        # Plot reconstructed image
        ax[1][i].imshow(sample_pred[i].reshape(28, 28), cmap='gray')
        ax[1][i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_error_fraction_encoders(error_history_train, error_history_test, title="Error Fraction Over Epochs", xlabel="Epochs", ylabel="Error Fraction"):
    """
    Plot the Mean Reconstruction Error (MRE) over epochs for training and test sets.

    Parameters:
    error_history_train (list): List of training MRE values over epochs.
    error_history_test (list): List of test MRE values over epochs.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    epochs = list(range(0, len(error_history_train) * 10, 10))  # Assuming MRE is recorded every 10 epochs

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, error_history_train, label="Training Set MRE", marker="o")
    plt.plot(epochs, error_history_test, label="Test Set MRE", marker="s")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def calculate_per_digit_stats(model, X_train, X_test):
    """
    Compute training and testing costs per digit and display test set statistics (MRE ± σ).
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_cost = {}
    test_cost = {}
    test_stats = []

    # Compute training costs per digit
    for digit, start_index in enumerate(range(0, 4000, 400)):
        train_cost[digit] = model.compute_cost(
            train_preds[start_index:start_index + 400, ].T, 
            X_train[start_index:start_index + 400, ].T
        )

    # Compute testing costs per digit
    for digit, start_index in enumerate(range(0, 1000, 100)):
        segment_test_preds = test_preds[start_index:start_index + 100, ].T
        segment_X_test = X_test[start_index:start_index + 100, ].T

        test_cost[digit] = model.compute_cost(segment_test_preds, segment_X_test)

        # Calculate MRE and standard deviation for this digit
        errors = np.mean((segment_X_test - segment_test_preds) ** 2, axis=0)
        mean_error = np.mean(errors)
        std_dev = np.std(errors)

        test_stats.append({
            "Digit": digit,
            "MRE ± σ": f"{mean_error:.4f} ± {std_dev:.4f}",
            "Mean Error": mean_error,
            "Std Dev": std_dev
        })

    # Create a DataFrame for the test set statistics
    test_stats_df = pd.DataFrame(test_stats)

    # Print the test statistics table
    print("\nTest Set Reconstruction Errors (MRE ± σ):")
    print(test_stats_df[["Digit", "MRE ± σ"]].to_string(index=False))

    return train_cost, test_cost, test_stats_df

def plot_error_fraction(train_errors, test_errors, interval=10, title="Error Fraction Over Training Period"):
    """
    Plot the time series of error fractions for training and test sets.

    Parameters:
    train_errors (list): List of training error fractions.
    test_errors (list): List of test error fractions.
    interval (int): Interval at which errors were recorded (default: 10 epochs).
    title (str): Title of the plot.
    """
    # Number of recorded epochs
    epochs = len(train_errors)
    x_ticks = list(range(0, epochs * interval, interval))  # Generate epoch ticks based on the interval

    plt.figure(figsize=(10, 6))
    plt.plot(x_ticks, train_errors, label="Training Error Fraction", marker='o', linestyle='-')
    plt.plot(x_ticks, test_errors, label="Test Error Fraction", marker='s', linestyle='--')
    
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_perceptrons_training_error(training_errors):
    """
    Plot the training error over epochs for each perceptron on the same graph.
    
    Parameters:
    training_errors (dict): A dictionary where each key is a digit (0-9), and each value is a list of 
                            error values over epochs for that perceptron.
    """
    plt.figure(figsize=(12, 8))

    # Color map for 10 distinct colors for each perceptron
    colors = plt.cm.get_cmap('tab10', 10)  # 10 colors from the 'tab10' colormap

    for digit, errors in training_errors.items():
        plt.plot(errors, label=f'Digit {digit}', color=colors(digit), marker='o')

    # Adding labels, title, and legend
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.title('Training Error over Epochs for Each Perceptron')
    plt.legend(title="Perceptron for Digit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison_for_all_digits(before_metrics, after_metrics):
    metrics = ['balanced_accuracy_error', 'precision', 'recall', 'f1_score']
    metric_names = {
        'balanced_accuracy_error': 'Balanced Accuracy Error',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }
    
    # Loop through each metric to create a separate plot
    for metric in metrics:
        before_values = before_metrics[metric]
        after_values = after_metrics[metric]
        
        # X-axis positions for the digits
        x = np.arange(10)
        width = 0.35  # Width of each bar

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting before and after training values as paired bars
        ax.bar(x - width / 2, before_values, width, label='Before Training')
        ax.bar(x + width / 2, after_values, width, label='After Training')
        
        # Adding labels and title
        ax.set_xlabel('Digits')
        ax.set_ylabel(metric_names[metric])
        ax.set_title(f'{metric_names[metric]} Before and After Training for Each Digit')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Digit {i}' for i in range(10)])
        ax.legend()

        # Display values on each bar
        for i in range(10):
            ax.text(i - width / 2, before_values[i] + 0.02, f'{before_values[i]:.2f}', ha='center', va='bottom')
            ax.text(i + width / 2, after_values[i] + 0.02, f'{after_values[i]:.2f}', ha='center', va='bottom')
        
        # Show plot
        plt.tight_layout()
        plt.show()

def plot_metrics_comparison(before_values, after_values):
    metrics = ['Error Fraction', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))  # Label locations
    width = 0.35  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars for before and after values
    bars_before = ax.bar(x - width/2, before_values, width, label='Before Training')
    bars_after = ax.bar(x + width/2, after_values, width, label='After Training')
    
    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance Metrics on Test Set Before and After Training')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Function to add labels to bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom'
            )
    
    # Add labels to both sets of bars
    add_labels(bars_before)
    add_labels(bars_after)
    
    plt.show()

def plot_training_errors_over_epochs(training_errors):
    """
    Plot the training error for each perceptron over all epochs on the same graph.
    
    Parameters:
    training_errors (dict): Dictionary where each key is a digit (0-9), and each value is a list of error values
                            over epochs for that perceptron.
    """
    plt.figure(figsize=(12, 8))
    
    # Define a color map for differentiating perceptrons
    colors = plt.cm.get_cmap('tab10', 10)  # Using tab10 colormap for 10 perceptrons
    
    # Plot the training errors for each perceptron
    for digit, errors in training_errors.items():
        plt.plot(errors, label=f'Digit {digit}', color=colors(digit))
    
    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.title('Training Error over Epochs for Each Perceptron')
    plt.legend(title="Perceptron for Digit")
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()
    plt.show()

def plot_metrics_vs_bias(metrics):
    bias_values = metrics["bias_values"]
    error_fractions = metrics["error_fractions"]
    precisions = metrics["precisions"]
    recalls = metrics["recalls"]
    f1_scores = metrics["f1_scores"]
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    ax[0, 0].plot(bias_values, error_fractions, marker='o', color='b')
    ax[0, 0].set_title('Error Fraction vs Bias')
    ax[0, 0].set_xlabel('Bias (w0)')
    ax[0, 0].set_ylabel('Error Fraction')
    
    ax[0, 1].plot(bias_values, precisions, marker='o', color='g')
    ax[0, 1].set_title('Precision vs Bias')
    ax[0, 1].set_xlabel('Bias (w0)')
    ax[0, 1].set_ylabel('Precision')
    
    ax[1, 0].plot(bias_values, recalls, marker='o', color='r')
    ax[1, 0].set_title('Recall vs Bias')
    ax[1, 0].set_xlabel('Bias (w0)')
    ax[1, 0].set_ylabel('Recall')
    
    ax[1, 1].plot(bias_values, f1_scores, marker='o', color='purple')
    ax[1, 1].set_title('F1 Score vs Bias')
    ax[1, 1].set_xlabel('Bias (w0)')
    ax[1, 1].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(metrics):
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["false_positives"], metrics["true_positives"], marker='o', color='b')
    plt.title("ROC Curve for Different Bias Values")
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.grid(True)
    plt.show()

def plot_weights_as_heatmaps(weights_list, titles):
    """
    Plot multiple sets of weights as 28x28 heatmaps.

    Parameters:
    - weights_list: A list of 1D arrays, where each array represents the weights to be reshaped and plotted.
    - titles: A list of titles corresponding to each set of weights.

    Note: Each set of weights should exclude the bias term and have exactly 784 elements.
    """
    num_plots = len(weights_list)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    if num_plots == 1:
        axs = [axs]  # Ensure axs is iterable even if there's only one subplot

    for i, (weights, title) in enumerate(zip(weights_list, titles)):
        # Reshape the weights to a 28x28 matrix (excluding the bias term)
        weights_matrix = weights.reshape(28, 28)
        
        # Plot as a heatmap
        axs[i].imshow(weights_matrix, cmap='viridis')
        axs[i].set_title(title)
        axs[i].axis('off')
    
    plt.suptitle("Weight Heatmaps")
    plt.show()

def evaluate_bias_range(perceptron, test_images, test_labels, num_values=20):
    # Extract the trained bias weight (w0)
    original_w0 = perceptron.weights[0]

    print(f"Original bias weight (w0): {original_w0:.4f}")
    
    # Generate a range of 20 values around the original bias weight (10 lower, 10 higher)
    bias_values = np.linspace(original_w0 - 10, original_w0 + 10, num=num_values + 1)

    print(f"Evaluating bias values: {bias_values}")
    
    # Lists to store metrics for each bias value
    error_fractions = []
    precisions = []
    recalls = []
    f1_scores = []
    true_positives = []
    false_positives = []
    
    for bias in bias_values:
        # Temporarily set the bias weight to the current value
        perceptron.weights[0] = bias
        
        # Calculate metrics for this bias
        error_fraction = calculate_error_fraction(perceptron, test_images, test_labels)
        precision, recall, f1_score = calculate_metrics(perceptron, test_images, test_labels)
        
        # Track the number of true positives and false positives for ROC
        TP = sum((perceptron.predict(img) == 1) and (label == 1) for img, label in zip(test_images, test_labels))
        FP = sum((perceptron.predict(img) == 1) and (label == 0) for img, label in zip(test_images, test_labels))
        
        # Append the metrics
        error_fractions.append(error_fraction)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        true_positives.append(TP)
        false_positives.append(FP)
    
    # Reset bias to the original value
    perceptron.weights[0] = original_w0
    
    # Package metrics
    metrics = {
        "bias_values": bias_values,
        "error_fractions": error_fractions,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "true_positives": true_positives,
        "false_positives": false_positives
    }
    
    return metrics

def classify_challenge_set(perceptron, challenge_images, challenge_labels):
    # Initialize a 2x8 matrix to store counts (2 rows: classifications 0 and 1, 8 columns for digits 2 to 9)
    classification_counts = np.zeros((2, 8), dtype=int)

    # Loop through each image and its corresponding label in the challenge set
    for image, label in zip(challenge_images, challenge_labels):
        # Make a prediction using the perceptron
        prediction = int(perceptron.predict(image))  
        
        # If the label is between 2 and 9, we count it in the matrix
        if 2 <= label <= 9:
            column_index = int(label - 2)
            classification_counts[prediction, column_index] += 1

    return classification_counts

def plot_classification_counts(classification_counts):
    # Create a DataFrame for better visualization
    df = pd.DataFrame(
        classification_counts,
        index=["Classified as 0", "Classified as 1"],
        columns=[f"Digit {i}" for i in range(2, 10)]
    )

    print(df)
    
    # Plot the results as a grouped bar chart
    ax = df.T.plot(kind="bar", figsize=(10, 6), width=0.8, colormap="viridis")
    ax.set_xlabel("Digits (2 to 9)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Classification of Digits in the Challenge Set")
    plt.xticks(rotation=0)
    plt.legend(title="Classification")

    # Add the values inside each bar
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fmt='%d')

    plt.show()