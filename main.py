from __future__ import print_function
import argparse
import logging
import numpy as np
from time import time
import sys
import utils as U
import reader as R
import pickle as pk
import os.path

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The output folder")

args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir)
U.set_logger(out_dir)
U.print_args(args)

train, train_x, train_y = R.read_dataset(args.train_path, mode='training')
test, test_x, test_y = R.read_dataset(args.test_path, mode='testing')

# all the arrays have to be numpy array before passing them into this function
def do_logistic_regression(train_x, train_y, test_x, test_y, class_weight=None):
	from sklearn import datasets, neighbors, linear_model, svm
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import recall_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import f1_score

	score_average = 'micro'

	logger.info("Start training...")
	# logreg = linear_model.LogisticRegression(class_weight=class_weight).fit(train_x, train_y)
	# logreg = svm.LinearSVC(class_weight=class_weight).fit(train_x, train_y)
	logreg = svm.SVC(class_weight=class_weight, degree=3).fit(train_x, train_y)
	logger.info("Finished training.")

	logger.info("Start predicting train set...")
	train_pred_y = logreg.predict(train_x)
	logger.info("Finished predicting train set.")
	logger.info("Start predicting test set...")
	test_pred_y = logreg.predict(test_x)
	logger.info("Finished predicting test set.")

	logger.info('[TRAIN] F1: %.3f, Recall: %.3f, Precision: %.3f' % (
			f1_score(train_y,train_pred_y,average=score_average), recall_score(train_y,train_pred_y,average=score_average), precision_score(train_y,train_pred_y,average=score_average)))
	logger.info('[TEST] F1: %.3f, Recall: %.3f, Precision: %.3f' % (
			f1_score(test_y,test_pred_y,average=score_average), recall_score(test_y,test_pred_y,average=score_average), precision_score(test_y,test_pred_y,average=score_average)))

	logger.info("================================================")
	# mail.send_email(out_dir+'/log.txt',"no-reply-lstm@yulonglong.com","a0080415@u.nus.edu","Logistic-Regression", extra_output)


do_logistic_regression(train_x, train_y, test_x, test_y, class_weight='balanced')
