import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


"""

ДЗ має включати

розбиття train/test

нормалізація даних

зменшення вхідної розмірності до n=100 за допомогою PCA

Класифікація зображень за допомогою knn

оцінка точності моделі на тестовому датасеті результат оформити як pca_knn.py
"""
#Data load, already splitted
(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig, axes = plt.subplots(2, 10, figsize=(16, 6))
for i in range(20):
    axes[i // 10, i % 10].imshow(X_train[i], cmap='gray');
    axes[i // 10, i % 10].axis('off')
    axes[i // 10, i % 10].set_title(f"target: {y_train[i]}")

plt.tight_layout()
plt.show()
plt.clf()

# Custom function to reshape the data in 1d vector
def reshape_data(X):
    if X.ndim >= 3 :
        X=X.reshape(X.shape[0], -1)
        print("Data Reshaping..")
    return X

# Function for KNearestNeighbors pipeline with n PCA components and n neigbors
def make_pca_knn_pipeline(n_c,n_n,X_train, y_train):
    pipeline = make_pipeline(
        FunctionTransformer(reshape_data, validate=False),
        MinMaxScaler(),
        PCA(n_components=n_c),
        KNeighborsClassifier(n_neighbors=n_n),
    )
    # Fit the pipeline on the training data
    print(f"Fitting data to {n_c} components with {n_n} neighbors")
    pipeline.fit(X_train, y_train)
    return pipeline


# Create a pipeline with the FunctionTransformer for reshaping
neighbors = [1, 2, 3, 4, 5, 6, 8, 10, 12]
neighbors_accuracy = {}
knn_pipes = {}
for n in neighbors:
    pipeline = make_pca_knn_pipeline(100, n, X_train, y_train)
    knn_pipes[n] = pipeline
    # Evaluate the pipeline on the test data
    neighbors_accuracy[n] = pipeline.score(X_test, y_test)

for n in neighbors:
    print(f"Accuracy for {n} neighbors: {neighbors_accuracy[n]}")

# Optimal n neighbors
plt.scatter(neighbors, np.array(list(neighbors_accuracy.values())))
plt.xlabel('X-axis (Neighbors)')
plt.ylabel('Y-axis (Accuracy)')
plt.title('K means Accuracy Plot')
plt.show()
plt.clf()
# Sorting data on bins with labels
cluster_data = {}
label_data = {}
cluster_accuracy = {}
test_cluster_labels = knn_pipes[3].predict(X_test)

for label in range(10):
    cluster_data[label] = X_test[test_cluster_labels == label]
    label_data[label] = X_test[np.logical_and(test_cluster_labels == label, label == y_test)]
    print(len(cluster_data[label]), len(label_data[label]),f"TP part in bin(TP/Bin size) = {len(label_data[label])/len(cluster_data[label]):.3f}")


for cluster_label in range(10):
    # Finding ind of samples in the current cluster
    cluster_ind = np.where(test_cluster_labels == cluster_label)[0]
    # Extract true positive labels for samples in the current cluster
    true_labels_cluster = y_test[cluster_ind]

    # Determine the most frequent true label in the cluster
    most_frequent_label = np.argmax(np.bincount(true_labels_cluster))

    # Accuracy for the current cluster
    accuracy = accuracy_score(true_labels_cluster, np.full_like(true_labels_cluster, most_frequent_label))

    print(f"Accuracy claster{cluster_label} for most label {most_frequent_label}: Accuracy = {accuracy:.3f}")


plt.clf()
# concatenate images
images = []
for i in range(10):
    images.append(np.concatenate(cluster_data[i][:30],axis=1))

plt.imshow(np.concatenate(images,axis=0),extent=[0, 30, 10, 0])
plt.show()