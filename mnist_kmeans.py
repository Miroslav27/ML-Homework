import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score

#Data load, already splitted
(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig, axes = plt.subplots(2, 10, figsize=(16, 6))
for i in range(20):
    axes[i // 10, i % 10].imshow(X_train[i], cmap='gray');
    axes[i // 10, i % 10].axis('off')
    axes[i // 10, i % 10].set_title(f"target: {y_train[i]}")

plt.tight_layout()


# Custom function to reshape the data in 1d
def reshape_data(X):
    if X.ndim >= 3 :
        X=X.reshape(X.shape[0], -1)
        print("Data Reshaping..")
    return X

# Function for K means pipeline with n Clusters
def make_Kmeans_pipeline(n,X_train, y_train):
    pipeline = make_pipeline(
        FunctionTransformer(reshape_data, validate=False),
        MinMaxScaler(),
        KMeans(n_clusters=n)
    )
    # Fit the pipeline on the training data
    print(f"Fitting data to {n} clusters")
    pipeline.fit(X_train, y_train)
    return pipeline


# Create a pipeline with the FunctionTransformer for reshaping
clusters = [2, 4, 6, 8, 10, 12]
cluster_accuracy = {}
kmeans_pipes = {}
for n in clusters:
    kmeans_pipes[n] = make_Kmeans_pipeline(n,X_train, y_train)
    # Evaluate the pipeline on the test data
    cluster_accuracy[n] = kmeans_pipes[n].score(X_test, y_test)
for n in clusters:
    print(f"Accuracy for {n} clusters: {cluster_accuracy[n]}")

# where is Idris Elba?
plt.scatter(clusters, np.array(list(cluster_accuracy.values()))*-1)
plt.xlabel('X-axis (Clusters)')
plt.ylabel('Y-axis (Accuracy)')
plt.title('K means Accuracy Plot')
plt.show()


#Sorting data on bins with labels
cluster_data = {}
label_data = {}
cluster_accuracy = {}
test_cluster_labels = kmeans_pipes[10].predict(X_test)

for label in range(10):
    cluster_data[label] = X_test[test_cluster_labels == label]
    label_data[label] = X_test[np.logical_and(test_cluster_labels == label, label == y_test)]
    print(len(cluster_data[label]), len(label_data[label]))
    # cluster number do not correspond to label number

for cluster_label in range(10):
    #Finding ind of samples in the current cluster
    cluster_ind = np.where(test_cluster_labels == cluster_label)[0]
    #Extract true positive labels for samples in the current cluster
    true_labels_cluster = y_test[cluster_ind]

    #Determine the most frequent true label in the cluster
    most_frequent_label = np.argmax(np.bincount(true_labels_cluster))

    #Accuracy for the current cluster
    accuracy = accuracy_score(true_labels_cluster, np.full_like(true_labels_cluster, most_frequent_label))

    print(f"Cluster accuracy claster{cluster_label} for most label {most_frequent_label}: Accuracy = {accuracy:.2f}")
    fig, axes = plt.subplots(1, 10, figsize=(18, 7))
    for i in range(10):
        axes[i % 10].imshow(cluster_data[cluster_label][i], cmap='gray');
        axes[i % 10].axis('off')
        axes[i % 10].set_title(f"c:{cluster_label} l:{most_frequent_label} Ac:{accuracy:.1f}")
    plt.tight_layout()

# concatenate images horizontally
def concat_images(imgs_list):
    return np.concatenate(imgs_list, axis=1)

combined_image = concat_images(cluster_data[0][:10])
plt.imshow(combined_image)
plt.show()