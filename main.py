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
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='svm', help="Model type for classification")
# parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The output folder")

args = parser.parse_args()

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

# all the arrays have to be numpy array before passing them into this function
def do_logistic_regression(x, y, class_weight=None):
	from sklearn import datasets, neighbors, linear_model, svm
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import recall_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import f1_score

	score_average = 'micro'

	totalF1, totalRecall, totalPrecision = 0.0, 0.0, 0.0

	for i in range(10):
		train_x, train_y, test_x, test_y = getFold(i,x,y)
		logger.info("Fold %d" % i)
		logger.info("train_x.shape : " + str(train_x.shape))
		logger.info("train_y.shape : " + str(train_y.shape))
		logger.info("test_x.shape  : " + str(test_x.shape))
		logger.info("test_y.shape  : " + str(test_y.shape))

		logger.info("Start training...")
		if model_type == 'ARDRegression':
			model = linear_model.ARDRegression().fit(train_x, train_y)
		elif model_type == 'BayesianRidge':
			model = linear_model.BayesianRidge().fit(train_x, train_y)
		elif model_type == 'ElasticNet':
			model = linear_model.ElasticNet().fit(train_x, train_y)
		elif model_type == 'ElasticNetCV':
			model = linear_model.ElasticNetCV().fit(train_x, train_y)
		elif model_type == 'HuberRegressor':
			model = linear_model.HuberRegressor().fit(train_x, train_y)
		elif model_type == 'Lars':
			model = linear_model.Lars().fit(train_x, train_y)
		elif model_type == 'LarsCV':
			model = linear_model.LarsCV().fit(train_x, train_y)
		elif model_type == 'Lasso':
			model = linear_model.Lasso().fit(train_x, train_y)
		elif model_type == 'LassoCV':
			model = linear_model.LassoCV().fit(train_x, train_y)
		elif model_type == 'LassoLars':
			model = linear_model.LassoLars().fit(train_x, train_y)
		elif model_type == 'LassoLarsCV':
			model = linear_model.LassoLarsCV().fit(train_x, train_y)
		elif model_type == 'LassoLarsIC':
			model = linear_model.LassoLarsIC().fit(train_x, train_y)
		elif model_type == 'LinearRegression':
			model = linear_model.LinearRegression().fit(train_x, train_y)
		elif model_type == 'LogisticRegression':
			model = linear_model.LogisticRegression(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'LogisticRegressionCV':
			model = linear_model.LogisticRegressionCV(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'MultiTaskLasso':
			model = linear_model.MultiTaskLasso().fit(train_x, train_y)
		elif model_type == 'MultiTaskElasticNet':
			model = linear_model.MultiTaskElasticNet().fit(train_x, train_y)
		elif model_type == 'MultiTaskLassoCV':
			model = linear_model.MultiTaskLassoCV().fit(train_x, train_y)
		elif model_type == 'MultiTaskElasticNetCV':
			model = linear_model.MultiTaskElasticNetCV().fit(train_x, train_y)
		elif model_type == 'OrthogonalMatchingPursuit':
			model = linear_model.OrthogonalMatchingPursuit().fit(train_x, train_y)
		elif model_type == 'OrthogonalMatchingPursuitCV':
			model = linear_model.OrthogonalMatchingPursuitCV().fit(train_x, train_y)
		elif model_type == 'PassiveAggressiveClassifier':
			model = linear_model.PassiveAggressiveClassifier(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'PassiveAggressiveRegressor':
			model = linear_model.PassiveAggressiveRegressor().fit(train_x, train_y)
		elif model_type == 'Perceptron':
			model = linear_model.Perceptron(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'RandomizedLasso':
			model = linear_model.RandomizedLasso().fit(train_x, train_y)
		elif model_type == 'RandomizedLogisticRegression':
			model = linear_model.RandomizedLogisticRegression().fit(train_x, train_y)
		elif model_type == 'RANSACRegressor':
			model = linear_model.RANSACRegressor().fit(train_x, train_y)
		elif model_type == 'Ridge':
			model = linear_model.Ridge().fit(train_x, train_y)
		elif model_type == 'RidgeClassifier':
			model = linear_model.RidgeClassifier(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'RidgeClassifierCV':
			model = linear_model.RidgeClassifierCV(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'RidgeCV':
			model = linear_model.RidgeCV().fit(train_x, train_y)
		elif model_type == 'SGDClassifier':
			model = linear_model.SGDClassifier(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'SGDRegressor':
			model = linear_model.SGDRegressor().fit(train_x, train_y)
		elif model_type == 'TheilSenRegressor':
			model = linear_model.TheilSenRegressor().fit(train_x, train_y)
		elif model_type == 'lars_path':
			model = linear_model.lars_path().fit(train_x, train_y)
		elif model_type == 'lasso_path':
			model = linear_model.lasso_path().fit(train_x, train_y)
		elif model_type == 'lasso_stability_path':
			model = linear_model.lasso_stability_path().fit(train_x, train_y)
		elif model_type == 'logistic_regression_path':
			model = linear_model.logistic_regression_path(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'orthogonal_mp':
			model = linear_model.orthogonal_mp().fit(train_x, train_y)
		elif model_type == 'orthogonal_mp_gram':
			model = linear_model.orthogonal_mp_gram().fit(train_x, train_y)
		elif model_type == 'LinearSVC':
			model = svm.LinearSVC(class_weight=class_weight).fit(train_x, train_y)
		elif model_type == 'SVC':
			model = svm.SVC(class_weight=class_weight, degree=3).fit(train_x, train_y)
			
		logger.info("Finished training.")

		logger.info("Start predicting train set...")
		train_pred_y = model.predict(train_x)
		logger.info("Finished predicting train set.")
		logger.info("Start predicting test set...")
		test_pred_y = model.predict(test_x)
		logger.info("Finished predicting test set.")

		logger.info('[TRAIN] F1: %.3f, Recall: %.3f, Precision: %.3f' % (
				f1_score(train_y,train_pred_y,average=score_average), recall_score(train_y,train_pred_y,average=score_average), precision_score(train_y,train_pred_y,average=score_average)))
		logger.info('[TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f' % (
				f1_score(test_y,test_pred_y,average=score_average), recall_score(test_y,test_pred_y,average=score_average), precision_score(test_y,test_pred_y,average=score_average)))

		totalF1 += f1_score(test_y,test_pred_y,average=score_average)
		totalRecall += recall_score(test_y,test_pred_y,average=score_average)
		totalPrecision += precision_score(test_y,test_pred_y,average=score_average)

		logger.info("================================================")

	logger.info('[AVERAGE OVERALL TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f' % (totalF1/10.0, totalRecall/10.0, totalPrecision/10.0))

do_logistic_regression(x, y, class_weight='balanced')
