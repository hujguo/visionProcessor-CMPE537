
import os
import numpy as np

import descriptors
import dictionary
import quantization
import classification
import evaluation

# parameters
clustername = 'BOW'
num_cluster = 300
descname = 'SIFT'
#classifier_name, parameters = 'KNN', [5]
classifier_name, parameters = 'SVM', [200]
#classifier_name, parameters = 'MLP', [100, 'constant']
traindir = 'Dataset/Caltech20/training'
testdir = traindir + '/../testing'
# collect descriptors
descriptor = descriptors.get_descriptor(descname)
featuresdict = {}
taxons = os.listdir(traindir)
for taxon in taxons:
    taxonpath = os.path.join(traindir, taxon)
    print('Collecting features from', taxon)
    features = descriptors.features_in_dir(descriptor, taxonpath, print_progress=True)