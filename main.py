from __future__ import print_function
import argparse
import codecs
import logging
import numpy as np
import sys
import utils as U
import helper as H
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
parser.add_argument("-tr", "--train-path", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='LogisticRegression', help="Model type for classification")
parser.add_argument("-ts", "--test-path", dest="test_path", type=str, metavar='<str>', default=None, help="The path to the test set")
parser.add_argument("-o", "--out", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The output folder")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=10, help="Number of epochs for Neural Net")
parser.add_argument("--test", dest="is_test", action='store_true', help="Flag to indicate testing (default=False)")

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

if args.is_test and args.test_path == None:
    logger.error("Please enter the path to the file for testing!")
    exit()

train, x, y = R.read_dataset(args.train_path, model=model_type)

# If it is a test
if args.is_test and args.test_path:
    test, test_x, test_y = R.read_dataset(args.test_path, model=model_type)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    train_x, train_y, dev_x, dev_y = H.getDevFoldDrawCards(x,y)
    logger.info("================ Testing ================")
    accuracy = M.run_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, args, out_dir=out_dir, class_weight='balanced')


    ##########################################################33
    ## Start reading output for submissioin
    #
    
    logger.info("Reading output and creating submission.csv file for kaggle...")
    
    preds_y = []
    with codecs.open(out_dir + '/preds/best_test_pred.txt', mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            preds_y.append(int(line))

    assert len(preds_y) == len(test)

    train_size = len(train)
    test_size = len(test)
    final_answer = [None] * (train_size + test_size + 1) # Id starts from 1, not 0
    
    for i in range(len(preds_y)):
        final_answer[int(test[i].id)] = chr(preds_y[i]+97)
    
    for i in range(len(train)):
        final_answer[int(train[i].id)] = train[i].y
        
    for i in range(len(final_answer)):
        if i == 0: continue # Id starts from 1, not 0
        assert final_answer[i] != None

    output = open(out_dir + '/preds/submission.csv' ,"w")
    output.write("Id,Prediction\n")
    for i in range(len(final_answer)):
        if i == 0: continue # Id starts from 1, not 0
        output.write(str(i) + "," + final_answer[i] + "\n")
    output.close()
    
    logger.info("Done! Please submit 'data/preds/submission.csv' to kaggle!")

# If it is to train
else:
    totalAccuracy = 0.0
    for i in range(10):
        train_x, train_y, dev_x, dev_y, test_x, test_y = H.getFoldDrawCards(i,x,y)
        logger.info("================ Fold %i ================" % i)
        accuracy = M.run_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, args, out_dir=out_dir, class_weight='balanced')
        totalAccuracy += accuracy

    logger.info("================================================")
    logger.info('[AVERAGE OVERALL TEST]  Acc: %.3f' % (totalAccuracy/10.0))