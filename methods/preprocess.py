from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import random
import os
import pandas as pd

def preprocess_mnist_data():
    # Define paths for input files
    image_file = 'data/MNISTnumImages5000_balanced.txt'
    label_file = 'data/MNISTnumLabels5000_balanced.txt'
    
    # Load images and labels
    images = np.loadtxt(image_file)
    labels = np.loadtxt(label_file)

    # Extract samples for digits 0 and 1
    digit_0_indices = np.where(labels == 0)[0]
    digit_1_indices = np.where(labels == 1)[0]

    # Split into training (400 each) and test sets (100 each)
    train_0_indices = digit_0_indices[:400]
    train_1_indices = digit_1_indices[:400]
    test_0_indices = digit_0_indices[400:500]
    test_1_indices = digit_1_indices[400:500]

    # Combine and randomize training set
    train_indices = np.concatenate([train_0_indices, train_1_indices])
    random.shuffle(train_indices)
    train_images = images[train_indices]
    train_labels = labels[train_indices]

    # Combine test set
    test_indices = np.concatenate([test_0_indices, test_1_indices])
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    # Prepare the challenge set with digits 2-9 (100 samples each)
    challenge_indices = []
    for digit in range(2, 10):
        indices = np.where(labels == digit)[0][:100]
        challenge_indices.extend(indices)
    challenge_images = images[challenge_indices]
    challenge_labels = labels[challenge_indices]

    # Define output paths for processed data
    output_dir = 'data'
    np.savetxt(os.path.join(output_dir, 'train_images.txt'), train_images)
    np.savetxt(os.path.join(output_dir, 'train_labels.txt'), train_labels, fmt='%d')
    np.savetxt(os.path.join(output_dir, 'test_images.txt'), test_images)
    np.savetxt(os.path.join(output_dir, 'test_labels.txt'), test_labels, fmt='%d')
    np.savetxt(os.path.join(output_dir, 'challenge_images.txt'), challenge_images)
    np.savetxt(os.path.join(output_dir, 'challenge_labels.txt'), challenge_labels, fmt='%d')

    print("Preprocessing completed. Data saved in the 'data' folder.")


def prepare_data_for_multiclass_digits(feature_file, label_file, digits=None, test_only=False):
    """
    Load, preprocess, and split data for multi-class classification, with optional digit filtering.

    Parameters:
    feature_file (str): Path to the features file (tab-separated).
    label_file (str): Path to the labels file.
    digits (list): List of digits to filter (default is None, meaning no filtering).
    test_only (bool): Whether to return only test data (default is False).

    Returns:
    tuple: A tuple containing:
        - X_train (ndarray): Features of the training data.
        - y_train (ndarray): One-hot encoded labels of the training data.
        - X_test (ndarray): Features of the testing data.
        - y_test (ndarray): One-hot encoded labels of the testing data.
    """
    # Load features and labels
    features = pd.read_csv(feature_file, sep="\t", header=None)
    labels = pd.read_csv(label_file, header=None)
    features['label'] = labels

    if digits is not None:
        features = features[features['label'].isin(digits)]

    # Initialize train and test dataframes
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Split data by label
    unique_labels = features['label'].unique()
    for label in unique_labels:
        df = features.loc[features['label'] == label]
        train_split = df.sample(frac=0.8, random_state=200)  # 80% training data
        test_split = df.drop(train_split.index)  # Remaining 20% for testing
        train_df = pd.concat([train_df, train_split])
        test_df = pd.concat([test_df, test_split])

    # Extract features and labels for training
    if not test_only:
        y_train = train_df['label'].values.reshape(-1, 1)
        X_train = train_df.drop(columns=['label']).values
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
    else:
        X_train, y_train = None, None

    # Extract features and labels for testing
    y_test = test_df['label'].values.reshape(-1, 1)
    X_test = test_df.drop(columns=['label']).values
    mlb = MultiLabelBinarizer()
    y_test = mlb.fit_transform(y_test)

    if test_only:
        return X_test, y_test
    else:
        return X_train, y_train, X_test, y_test


def prepare_data_for_multiclass(feature_file, label_file):
    """
    Load, preprocess, and split data for multi-class classification.

    Parameters:
    feature_file (str): Path to the features file (tab-separated).
    label_file (str): Path to the labels file.

    Returns:
    tuple: A tuple containing:
        - X_train (ndarray): Features of the training data.
        - y_train (ndarray): One-hot encoded labels of the training data.
        - X_test (ndarray): Features of the testing data.
        - y_test (ndarray): One-hot encoded labels of the testing data.
    """
    # Load features and labels
    features = pd.read_csv(feature_file, sep="\t", header=None)
    labels = pd.read_csv(label_file, header=None)
    features['label'] = labels

    # Initialize train and test dataframes
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Split data by label
    for i in range(0, 10):  # Assuming labels are 0 through 9
        df = features.loc[features['label'] == i]
        train_split = df.sample(frac=0.8, random_state=200)  # 80% training data
        test_split = df.drop(train_split.index)  # Remaining 20% for testing
        train_df = pd.concat([train_df, train_split])
        test_df = pd.concat([test_df, test_split])

    # Extract features and labels for training
    y_train = train_df['label'].values.reshape(-1, 1)
    X_train = train_df.drop(columns=['label']).values

    # Extract features and labels for testing
    y_test = test_df['label'].values.reshape(-1, 1)
    X_test = test_df.drop(columns=['label']).values

    # One-hot encode labels
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    return X_train, y_train, X_test, y_test

def prepare_data_for_perceptrons(image_file, label_file, num_perceptrons=10, sampling_strategy='oversample'):
    # Load images and labels
    images = np.loadtxt(image_file)
    labels = np.loadtxt(label_file)

    # Initialize dictionaries to hold training and test sets for each perceptron
    training_data = {}
    test_data = {}

    for digit in range(num_perceptrons):
        # Create binary labels for this perceptron
        binary_labels = (labels == digit).astype(int)
        
        # Split into training and test sets
        digit_indices = np.where(labels == digit)[0]
        other_indices = np.where(labels != digit)[0]
        
        # Select 400 samples of the target digit and 400 samples of other digits for training
        train_digit_indices = np.random.choice(digit_indices, 400, replace=False)
        train_other_indices = np.random.choice(other_indices, 3600, replace=False)
        train_indices = np.concatenate((train_digit_indices, train_other_indices))
        random.shuffle(train_indices)

        # Apply oversampling or undersampling strategy
        if sampling_strategy == 'oversample':
            # Oversample the minority class (1) to balance with the majority class (0)
            minority_indices = train_digit_indices
            majority_indices = train_other_indices
            oversampled_minority_indices = np.random.choice(minority_indices, len(majority_indices), replace=True)
            train_indices = np.concatenate((oversampled_minority_indices, majority_indices))
            random.shuffle(train_indices)
        
        elif sampling_strategy == 'undersample':
            # Undersample the majority class (0) to balance with the minority class (1)
            minority_indices = train_digit_indices
            majority_indices = np.random.choice(train_other_indices, len(minority_indices), replace=False)
            train_indices = np.concatenate((minority_indices, majority_indices))
            random.shuffle(train_indices)

        # Training set for this perceptron
        training_data[digit] = {
            "images": images[train_indices],
            "labels": binary_labels[train_indices]
        }

        # Test set for this perceptron (100 samples of target digit and 900 of other digits)
        test_digit_indices = np.setdiff1d(digit_indices, train_digit_indices)
        test_other_indices = np.setdiff1d(other_indices, train_other_indices)
        test_indices = np.concatenate((test_digit_indices, test_other_indices))

        test_data[digit] = {
            "images": images[test_indices],
            "labels": binary_labels[test_indices]
        }

    return training_data, test_data


# Run the preprocessing function
if __name__ == "__main__":
    preprocess_mnist_data()
