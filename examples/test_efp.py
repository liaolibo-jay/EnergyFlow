"""An example involving Energy Flow Polynomials (EFPs) and a linear
classifier (Fisher's Linear Discriminant by default). First, the 
[`EFPSet`](../docs/efp/#efpset) class is used to compute the EFPs
up to the specified `dmax`, the default being `dmax=5`. Then linear
classifiers are trained for different numbers of EFPs as input, 
determined by taking all EFPs up to degree `d` with `d` from `1` 
to `dmax`. The output of the example is a plot of the ROC curves
for the classifiers with different numbers of EFP inputs.
"""

# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np

# energyflow imports
import energyflow as ef
from energyflow.archs import LinearClassifier
from energyflow.utils import data_split, standardize, to_categorical
import cepc

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
num_data = 15000
test_frac = 0.2

# efp parameters
dmax = 4
measure = 'ee'
beta = 0.2

# plotting
colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue', 'tab:black']

################################################################################

# load data
X, y = cepc.load(num_data)

print('Loaded b-quark and c-quark jets')

# calculate EFPs
print('Calculating d <= {} EFPs for {} jets... \n'.format(dmax, num_data), end='')
efpset = ef.EFPSet(('d<=', dmax), measure='ee', beta=beta)
masked_X = [x[x[:,0] > 0] for x in X]
#print(X[0])
#print(masked_X[0])
X = efpset.batch_compute(masked_X)
#print(X[0])
#print(X[1])
print('Done')

# train models with different numbers of EFPs as input
rocs = []
for d in range(1, dmax+1):

    # build architecture
    model = LinearClassifier(linclass_type='lda')

    # select EFPs with degree <= d
    X_d = X[:,efpset.sel(('d<=', d))]

    # do train/val/test split 
    (X_train, X_test, y_train, y_test) = data_split(X_d, y, val=0, test=test_frac)
    print('Done train/val/test split')

    # train model
    model.fit(X_train, y_train)

    # get predictions on test data
    preds = model.predict(X_test)

    # get ROC curve if we have sklearn
    if roc_curve:
        rocs.append(roc_curve(y_test, preds[:,1]))

        # get area under the ROC curve
        auc = roc_auc_score(y_test, preds[:,1])
        print()
        print('EFPs d <= {} AUC:'.format(d), auc)
        print()

# make ROC curve plot if we have matplotlib
if plt:

    # some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # iterate over the ROC curves and plot them
    for i,d in enumerate(range(1, dmax+1)):
        plt.plot(rocs[i][1], 1-rocs[i][0], '-', color=colors[i], 
                                                label='LDA: d <= {} EFPs'.format(d))

    # axes labels
    plt.xlabel('B Quark Jet Efficiency')
    plt.ylabel('C Quark Jet Rejection')

    # axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # make legend and show plot
    plt.legend(loc='lower left', frameon=False)
    plt.show()
    plt.savefig("cepc.jpg")
