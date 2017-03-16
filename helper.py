from __future__ import print_function
import argparse
import logging
import numpy as np
import sys
import utils as U
import reader as R
import models as M
import pickle as pk
import os.path
from time import time

logger = logging.getLogger(__name__)

def getDevFoldDrawCards(x, y):
    train_x, train_y, dev_x, dev_y = [], [], [], []

    for i in range(len(x)):
        if i%20 == 0:
            dev_x.append(x[i])
            dev_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(dev_x), np.array(dev_y)

def getFoldDrawCards(fold, x, y):
    train_x, train_y, dev_x, dev_y, test_x, test_y = [], [], [], [], [], []
    validation_fold = fold+1
    if validation_fold > 9: validation_fold = 0
    for i in range(len(x)):
        if i%10 == fold:
            test_x.append(x[i])
            test_y.append(y[i])
        elif i%10 == validation_fold:
            dev_x.append(x[i])
            dev_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(dev_x), np.array(dev_y), np.array(test_x), np.array(test_y)

def getFoldBulkCut(fold, x, y):
    train_x, train_y, dev_x, dev_y, test_x, test_y = [], [], [], [], [], []
    thresholdValid = float(fold) * 0.1
    thresholdTest = thresholdValid + 0.1
    if (thresholdValid == 1.0):
        thresholdValid = 0.0
    if (thresholdTest == 1.0):
        thresholdTest = 0.0

    bottomValid = thresholdValid*len(x)
    topValid = (thresholdValid+0.1)*len(x)
    bottomTest = thresholdTest*len(x)
    topTest = (thresholdTest+0.1)*len(x)

    for i in range(len(x)):
        if (i < topValid) and (i >= bottomValid):
            dev_x.append(x[i])
            dev_y.append(y[i])
        elif (i < topTest) and (i >= bottomTest):
            test_x.append(x[i])
            test_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(dev_x), np.array(dev_y), np.array(test_x), np.array(test_y)
