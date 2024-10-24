import numpy as np

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
