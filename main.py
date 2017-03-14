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
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=10, help="Number of epochs for Neural Net")

args = parser.parse_args()
model_type = args.model_type

out_dir = args.out_dir_path
out_dir = out_dir + "-" + model_type

U.mkdir_p(out_dir)
U.mkdir_p(out_dir + '/data')
U.mkdir_p(out_dir + '/preds')
U.mkdir_p(out_dir + '/models')
U.mkdir_p(out_dir + '/models/best_weights')
U.set_logger(out_dir)
U.print_args(args)

train, x, y = R.read_dataset(args.train_path, model=model_type)
# test, test_x, test_y = R.read_dataset(args.test_path)

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

totalAccuracy = 0.0

for i in range(10):
    train_x, train_y, test_x, test_y = getFold(i,x,y)
    logger.info(" === Fold %i ===" % i)
    accuracy = M.run_model(train_x, train_y, test_x, test_y, model_type, args, out_dir=out_dir, class_weight='balanced')
    totalAccuracy += accuracy

logger.info("================================================")

logger.info('[AVERAGE OVERALL TEST]  Acc: %.3f' % (totalAccuracy/10.0))