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
    'north_campus': ['left_2', 'left_1', 'middle', 'right_1', 'right_2'],
    'paris': ['paris_a', 'paris_b', 'paris_c',],
    'book': ['book_table', 'book_orig'] # test dataset
}
EXT = '.jpg'

NPYDIR = 'data/'

RESDIR = 'results/'

METHODS = ['left-to-right', 'middle-out', 'first-out-then-middle']

################################################################################
# Save and load data                                                           #
################################################################################

def read_imgs(dataset):
    assert(IMNAMES.get(dataset))
    # read images
    imgs = []
    for name in IMNAMES[dataset]:
        path = PREPATH + dataset + '/' + name + EXT
        img = cv.imread(path, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

# compress data
# no need to provide corresponding pairs for (j, i) where j > i
def save_points(dataset, points):
    N = len(IMNAMES.get(dataset))
    assert(N > 0)
    assert(len(points) == N)
    os.makedirs(NPYDIR, exist_ok=True)
    np.save(NPYDIR + dataset, np.array(points, dtype=object), allow_pickle=True)

# corresponding pointss for (j, i) where j > i are derived from (i, j) pair
def load_points(dataset):
    N = len(IMNAMES.get(dataset))
    assert(N > 0)
    points = np.load(NPYDIR + dataset + '.npy', allow_pickle=True)
    assert(len(points) == N)
    # fill in the blanks (we know A to B, obtain B to A)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            points[j][i] = [points[i][j][1], points[i][j][0]]
    return points

def save_image(name, img):
    os.makedirs(RESDIR, exist_ok=True)
    if cv.imwrite(RESDIR + name + EXT, cv.cvtColor(img, cv.COLOR_RGB2BGR)):
        print('Saved to', RESDIR + name + EXT)

def save_figure(fig, name):
    plt.figure(fig.number)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    fig.show()
    fig.savefig(RESDIR + name + '.png')
    mng.full_screen_toggle()
    print('Saved to ' + RESDIR + name + '.png')

################################################################################
# Corresponding points between images                                          #
################################################################################

def select_points(img, K):
    fig = plt.figure()
    plt.imshow(img)
    x = plt.ginput(K)   # (x, y) format
    plt.close(fig)
    return list(tuple(map(int, tup)) for tup in x)

def cv_mouse_click(event, x, y, flags, 