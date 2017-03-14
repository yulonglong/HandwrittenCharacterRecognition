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

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='LogisticRegression', help="Model type for classification")
# parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The output folder")

args = parser.parse_args()
model_type = args.model_type

out_dir = args.out_dir_path
out_dir = out_dir + "-" + model_type

U.mkdir_p(out_dir)
U.set_logger(out_dir)
U.print_args(args)

train, x, y = R.read_dataset(args.train_path, mode='training')
# test, test_x, test_y = R.read_dataset(args.test_path, mode='testing')

def getFold(fold, x, y):
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(len(x)):
        if i%10 == fold:
            test_x.append(x[i])
            test_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

totalF1, totalRecall, totalPrecision = 0.0, 0.0, 0.0

for i in range(10):
    train_x, train_y, test_x, test_y = getFold(i,x,y)
    logger.info(" === Fold %i ===" % i)
    f1, recall, precision = M.run_model(train_x, train_y, test_x, test_y, model_type, class_weight='balanced')

    totalF1 += f1
    totalRecall += recall
    totalPrecision += precision

logger.info("================================================")

logger.info('[AVERAGE OVERALL TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f' % (totalF1/10.0, totalRecall/10.0, totalPrecision/10.0))
logger.info('Total time taken = %d seconds' % totalTime)
