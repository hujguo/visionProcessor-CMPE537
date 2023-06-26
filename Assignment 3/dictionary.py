import cv2 as cv
import os
import numpy as np

def get_dictionary(dictname, num_cluster):
    if dictname == 'BOW':
        # https://docs.opencv.org/3.4.2/d4/d72/classcv_1_1BOWKMeansTrainer.html
        return cv.BOWKMeans