import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from dataset import CBISDDSM
from sklearn.metrics import confusion_matrix


def get_features(loader):
    train_features, train_labels = [], []
    for sample in loader:
        train_features.append(sample['image'].view(sample['image'].size(0), -1))
        train_labels.append(sample['class'].argmax(dim=1))
    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels, dim=0).numpy()
    return train_features,train_labels

def knn_classifier(train_loader, test_loader, k):
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

if __name__ == '__main__':
    path = "../"
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13]}


    # Split the dataset into train and test sets
    data = CBISDDSM(file="train2.csv", path=path)
    # 80% training, 20% testing
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

    # Run KNN classifier
    knn_classifier(train_loader, test_loader, k=3)