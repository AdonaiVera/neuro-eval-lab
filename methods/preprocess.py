import numpy as np
import random
import os

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

# Run the preprocessing function
if __name__ == "__main__":
    preprocess_mnist_data()
