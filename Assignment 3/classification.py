from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import cv2 as cv
import numpy as np
from quantization import get_hist

# X: [number of images x number of clusters]
# y: label for each image

class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, X, y):
        self.num_cluster = X.shape[1]
        self.X = np.copy(X)
        self.y = np.copy(y)
        return self
    def predict(self, X_test):
        y_predict = np.zeros((X_test.shape[0],))
        for i, x in enumerate(X_test):
            # distance between the histogram and all cluster
            norms = np.linalg.norm(self.X - x, axis=1)
            # minimum k indices
            idx = np.argpartition(norms, self.k)[:self.k]
            # minimum k distances
            dists = norms[idx]
            # corresponding labels
            labels = self.y[idx]
            values, counts = np.unique(labels, return_counts=True)
            # maximum occuring labels
            occurs = values[counts == np.max(counts)]
            if len(occurs) == 1:
                y_predict[i] = occurs[0]
            else:
                # multipl