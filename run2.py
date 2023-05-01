import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

EPOCHS = 100
EPSILON = 5.0
LR = 1e-3              
BATCH_SIZE = 128

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform)
    test_ds = CIFAR10(root='data/', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)
    
    return trainloader, testloader

# Load CIFAR10 dataset
trainloader, testloader = load_data()

train_data, train_labels = next(iter(trainloader))
test_data, test_labels = next(iter(testloader))

# Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Add Laplace noise to train data to privatize it
X_train_noisy = X_train + np.random.laplace(0, 1/EPSILON, X_train.shape)

# Flatten and scale the data
X_train_flattened = X_train_noisy.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_options:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train_scaled, y_train)
    train_pred = clf.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    train_loss = 1.0 - train_acc

    test_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    test_loss = 1.0 - test_acc

    print("Kernel: {} - Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%"
            .format(kernel, train_loss, train_acc*100, test_loss, test_acc*100))

print("\n\n")

# Train and test the model using KNN and NBC and noised data

# Train KNN Classifier
n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)
train_pred_knn = knn.predict(X_train_scaled)
test_pred_knn = knn.predict(X_test_scaled)
train_accuracy_knn = accuracy_score(y_train, train_pred_knn)
train_loss_knn = 1.0 - train_accuracy_knn
test_accuracy_knn = accuracy_score(y_test, test_pred_knn)
test_loss_knn = 1.0 - test_accuracy_knn
print("KNN Classifier K: {} - Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%"
            .format(n_neighbors, train_loss, train_acc*100, test_loss, test_acc*100))

# Train Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
train_pred_nb = nb.predict(X_train_scaled)
test_pred_nb = nb.predict(X_test_scaled)
train_accuracy_nb = accuracy_score(y_train, train_pred_nb)
train_loss_nb = 1.0 - train_accuracy_nb
test_accuracy_nb = accuracy_score(y_test, test_pred_nb)
test_loss_nb = 1.0 - test_accuracy_nb
print("Naive Bayes Classifier - Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%"
            .format(train_loss, train_acc*100, test_loss, test_acc*100))