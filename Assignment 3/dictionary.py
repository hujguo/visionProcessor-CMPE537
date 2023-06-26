import cv2 as cv
import os
import numpy as np

def get_dictionary(dictname, num_cluster):
    if dictname == 'BOW':
        # https://docs.opencv.org/3.4.2/d4/d72/classcv_1_1BOWKMeansTrainer.html
        return cv.BOWKMeansTrainer(clusterCount=num_cluster)
    else:
        raise Exception('Invalid option')

def func_add_descriptors(impath, indices, dictionary, descriptor, desc_per_img=20):
    img = cv.imread(impath)
    assert img is not None
    _, desc = descriptor.detectAndCompute(img, None)
 