import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def cv_imshow_scale(winname, img):
    cv.namedWindow(winname, cv.WINDOW_KEEPRATIO)
    cv.imshow(winname, img)
    cv.waitKey()
    cv.destroyWindow(winname)

def plt_imshow(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def loop_images(func,   # function to be called on each image
                params, # func(impath, (taxon_id, img_id), *params)
                taxons_path='Dataset/Caltech20/training/',
                dry_run=True):
    taxons = os.listdir(taxons_path)
    for i_t, taxon in enumerate(taxons, start=1):
        taxon_path = os.path.join(taxons_path, taxon)
        imnames = os.listdir(taxon_path)
        for i_i, imname in enumerate(imnames, start=1):
            impath = os.path.join(taxon_path, imname)
            assert os.path.isfile(impath), 'Could not find %s' % (impath)
  