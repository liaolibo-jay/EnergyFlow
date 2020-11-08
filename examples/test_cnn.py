# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np
import os.path
# energyflow imports
import energyflow as ef
from energyflow.archs import CNN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import cepc # uproot package for cepc
# attempt to import sklearn
try:
    from sklearn.metrics import roc_auc_score, roc_curve
except:
    print('please install scikit-learn in order to make ROC curves')
    roc_curve = False

# attempt to import matplotlib
try:
    import matplotlib.pyplot as plt
except:
    print('please install matploltib in order to make plots')
    plt = False

################################### SETTINGS ###################################

# data controls
num_data = 100000
val_frac, test_frac = 0.1, 0.15

# image parameters
R = 0.4
img_width = 2*R
npix = 33
nb_chan = 2
norm = True

# required network architecture parameters
input_shape = (npix, npix, nb_chan)
filter_sizes = [8, 4, 4]
num_filters = [8, 8, 8] # very small so can run on non-GPUs in reasonable time

# optional network architecture parameters
dense_sizes = [50]
pool_sizes = 2

# network training parameters
num_epoch = 2
batch_size = 100

################################################################################

# load data
X, y = cepc.load(num_data=num_data)

# convert labels to categorical
Y = to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')

# make jet images
images = np.asarray([pixelate(x, npix=npix, img_width=img_width, nb_chan=nb_chan, norm=norm,charged_counts_only=True) for x in X])

print('Done making jet images')

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = data_split(images, Y, val=val_frac, test=test_frac)

print('Done train/val/test split')

# preprocess by zero centering images and standardizing each pixel
X_train, X_val, X_test = standardize(*zero_center(X_train, X_val, X_test))

print('Finished preprocessing')
print('Model summary:')

# build architecture
hps = {'input_shape': input_shape,
       'filter_sizes': filter_sizes,
       'num_filters': num_filters,
       'dense_sizes': dense_sizes,
       'pool_sizes': pool_sizes}
cnn = CNN(hps)

# train model
cnn.fit(X_train, Y_train,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          verbose=1)

# get predictions on test data
preds = cnn.predict(X_test, batch_size=1000)

# get ROC curve if we have sklearn
if roc_curve:
    cnn_fp, cnn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

    # get area under the ROC curve
    auc = roc_auc_score(Y_test[:,1], preds[:,1])
    print()
    print('CNN AUC:', auc)
    print()

    # make ROC curve plot if we have matplotlib
    if plt:

        # get multiplicity and mass for comparison
        masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
        mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
        mass_fp, mass_tp, threshs = roc_curve(Y[:,1], -masses)
        mult_fp, mult_tp, threshs = roc_curve(Y[:,1], -mults)

        # some nicer plot settings 
        plt.rcParams['figure.figsize'] = (4,4)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.autolayout'] = True

        # plot the ROC curves
        plt.plot(cnn_tp, 1-cnn_fp, '-', color='black', label='CNN')
        plt.plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')
        plt.plot(mult_tp, 1-mult_fp, '-', color='red', label='Multiplicity')

        # axes labels
        plt.xlabel('Quark Jet Efficiency')
        plt.ylabel('Gluon Jet Rejection')

        # axes limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # make legend and show plot
        plt.legend(loc='lower left', frameon=False)
        plt.savefig("test_cnn.jpg")
        plt.show()
