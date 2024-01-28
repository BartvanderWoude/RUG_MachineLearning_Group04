import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from dataset import CBISDDSM
from sklearn.metrics import confusion_matrix
import numpy as np


def get_features(loader):
    features, labels = [], []
    for sample in loader:
        features.append(sample['image'].view(sample['image'].size(0), -1))
        labels.append(sample['class'].argmax(dim=1))
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features,labels

def knn_classifier(train_loader, test_loader, k):
    print(f'Running KNN for k={k}...')
    # Extract features and labels from the training set
    train_features, train_labels = get_features(train_loader)
    # Extract features and labels from the test set
    test_features, test_labels = get_features(test_loader)
    # Initialize and train the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
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
    fn_rate = fn / (fn + tn) if (fn + tn) > 0 else 0.0
    print(f'False Negatives Rate: {fn_rate * 100:.2f}%')

    # Calculate False Positives (FP) rate
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f'False Positives Rate: {fp_rate * 100:.2f}%')


def manual_n_fold_cross_validation(data, n_folds):
    total_samples = len(data)
    shuffled_array = create_shuffled_array(total_samples-1)

    folds = [[] for _ in range(n_folds)]  # Initialize empty lists for each fold
    count = 0
    for i, item in enumerate(shuffled_array):
        if count >= n_folds:
            count = 0
        folds[count].append(data[item])
        count += 1

    return folds

def create_shuffled_array(size):
    original_array = np.arange(1, size + 1)  # Creates an array from 1 to size
    shuffled_array = np.random.permutation(original_array)
    return shuffled_array


def get_train_test_sets(folds, fold_index):
    # Combine all folds except the test fold to create training data
    train_data = [data for i, fold in enumerate(folds) if i != fold_index for data in fold]
    # Use the test fold as test data
    test_data = folds[fold_index]

    return train_data, test_data

if __name__ == '__main__':
    path = "../"
    n_neighbors = [3, 5, 7, 9, 11]

    n_folds = 5
    # Split the dataset into train and test sets
    data = CBISDDSM(file="train2.csv", path=path)
    folds = manual_n_fold_cross_validation(data, n_folds)

    for i in range(n_folds):
        train_data, test_data = get_train_test_sets(folds, i)
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
        # Run KNN classifier
        knn_classifier(train_loader, test_loader, n_neighbors[i])