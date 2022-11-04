import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from sys import exit
import os
import random
from operator import sub

################################################################################
# Path to sets of images                                                       #
################################################################################

PREPATH = 'images/'
IMNAMES = {
    'cmpe_building': ['left_2', 'left_1', 'middle', 'right_1', 'right_2'],
    'north_campus': ['left_2', 'left_1', '