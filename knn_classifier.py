import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import code.dataset as dataset
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt



# Function to get the features and lables from the loader in order to process them later
def get_features(loader):
    features, labels = [], []
    for sample in loader:
        features.append(sample['image'].view(sample['image'].size(0), -1))
        labels.append(sample['class'])
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features,labels

# Main kNN classifier function
def knn_classifier(train_loader, test_loader, k):
    print(f'Running KNN for k={k}...')
    # Extract features and labels from the training set
    train_features, train_labels = get_features(train_loader)
    # Extract features and labels from the test set
    test_features, test_labels = get_features(test_loader)
    # Initialize and train the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn_classifier.fit(train_features, train_labels)
    # Predict on the test set
    predictions = knn_classifier.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Calculate False Negatives (FN) rate
    fn_rate = fn / (fn + tp) if (fn + tn) > 0 else 0.0
    print(f'False Negatives Rate: {fn_rate * 100:.2f}%')

    # Calculate False Positives (FP) rate
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f'False Positives Rate: {fp_rate * 100:.2f}%')

    # Calculate F1 score
    f1 = f1_score(test_labels, predictions)
    print(f'F1 Score: {f1 * 100:.2f}%')

    return f1, fn_rate

#Function to implement manual n-fold cross validation
def manual_n_fold_cross_validation(data, n_folds):
    # Divide per total images
    total_samples = len(data)/5
    shuffled_array = create_shuffled_array(total_samples-1)

    folds = [[] for _ in range(n_folds)]  # Initialize empty lists for each fold
    count = 0
    for i, item in enumerate(shuffled_array):
        if count >= n_folds:
            count = 0
        # Have all version of an image in the same fold
        total_values = int(item*5 + 1)
        for i in range(0,5):
            folds[count].append(data[total_values-i])
        count += 1
    return folds

# Create a shuffled array of size N
def create_shuffled_array(size):
    original_array = np.arange(1, size + 1)  # Creates an array from 1 to size
    shuffled_array = np.random.permutation(original_array)
    return shuffled_array

# Compose train and test sets from the folds
def get_train_test_sets(folds, fold_index):
    # Combine all folds except the test fold to create training data
    train_data = [data for i, fold in enumerate(folds) if i != fold_index for data in fold]
    # Use the test fold as test data
    test_data = folds[fold_index]

    return train_data, test_data

if __name__ == '__main__':
    #path = "../"
    n_neighbors = [3, 5, 7, 9, 11, 13, 15, 17,19, 21]

    n_folds = 10
    # Split the dataset into train and test sets
    data = dataset.CBISDDSM(file="CBIS-DDSM/train-augmented.csv")
    folds = manual_n_fold_cross_validation(data, n_folds)

    f1_scores = []
    fn_rates = []
    # N-folds cross validation
    for i in range(n_folds):
        train_data, test_data = get_train_test_sets(folds, i)
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
        # Run KNN classifier
        f1, fn_rate = knn_classifier(train_loader, test_loader, n_neighbors[i])

        # Append results to lists
        f1_scores.append(f1)
        fn_rates.append(fn_rate)

    plt.subplot(1, 2, 1)
    plt.plot(n_neighbors, f1_scores, marker='o')
    plt.title('F1 Score vs. Number of Neighbors', fontsize=24)
    plt.xlabel('Number of Neighbors', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Plot False Negatives Rates
    plt.subplot(1, 2, 2)
    plt.plot(n_neighbors, fn_rates, marker='o', color='r')
    plt.title('False Negatives Rate vs. Number of Neighbors', fontsize=24)
    plt.xlabel('Number of Neighbors', fontsize=20)
    plt.ylabel('False Negatives Rate', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()
