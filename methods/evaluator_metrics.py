import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

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


def plot_confusion_matrix(conf_matrix, class_labels, title="Confusion Matrix"):
    """
    Visualize the confusion matrix using seaborn.

    Parameters:
    conf_matrix (ndarray): Confusion matrix (10x10 for digit classification).
    class_labels (list): List of class labels (e.g., [0, 1, 2, ..., 9]).
    title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def plot_inference(images_input, images_output, sample_idx=10):
    '''
    Plot the original and reconstructed images for a sample.
    '''
    
    # Plot original and reconstructed images for a sample
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(images_input[sample_idx].reshape(28, 28), cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(images_output[sample_idx].reshape(28, 28), cmap="gray")
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

def plot_error_fraction(network):
    """
    Plot the time series of error fractions for training and test sets from the network.

    Parameters:
    network (FeedForwardNN): The trained FeedForwardNN instance with recorded errors.
    """
    epochs = len(network.error_history_train)
    x_ticks = list(range(0, epochs * 10, 10))  # Assuming error history is recorded every 10 epochs

    plt.figure(figsize=(10, 6))
    plt.plot(x_ticks, network.error_history_train, label="Training Error Fraction", marker='o')
    plt.plot(x_ticks, network.error_history_test, label="Test Error Fraction", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction Over Training Period")
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