from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import cv2 as cv
import numpy as np
from quantization import get_hist

# X: [number of images x number of clusters]
# y: label for each image

class KNN:
   