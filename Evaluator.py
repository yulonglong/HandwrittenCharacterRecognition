
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import utils as U

import logging

logger = logging.getLogger(__name__)

###############################################################################################################################
## Evaluator class
#

class Evaluator():
    
    def __init__(self, out_dir, train, dev, test, batch_size_eval=256, print_info=False):
        self.out_dir = out_dir
        self.batch_size_eval = batch_size_eval

        self.train_x, self.train_y, self.train_y_multi  = train[0], train[1], train[2]
        self.dev_x, self.dev_y, self.dev_y_multi  = dev[0], dev[1], dev[2]
        self.test_x, self.test_y, self.test_y_multi  = test[0], test[1], test[2]

        self.dev_y_org = self.dev_y.astype('int32')
        self.test_y_org = self.test_y.astype('int32')

        self.best_dev = -1
        self.best_test = -1
        self.best_dev_epoch = -1
        self.best_test_missed = -1
        self.best_test_missed_epoch = -1
        self.dump_ref_scores()

        self.best_dev_correct = 0
        self.best_test_correct = 0
    
    def dump_ref_scores(self):
        U.mkdir_p(self.out_dir + '/preds/')
        np.savetxt(self.out_dir + '/preds/' + '/dev_ref.' + '.txt', self.dev_y_org, fmt='%i')
        np.savetxt(self.out_dir + '/preds/' + '/test_ref.' + '.txt', self.test_y_org, fmt='%i')
    
    def dump_predictions(self, dev_pred, test_pred, epoch):
        U.mkdir_p(self.out_dir + '/preds/' + '/' + str(epoch))
        np.savetxt(self.out_dir + '/preds/' + '/' + str(epoch) + '/dev_pred_' + str(epoch) + '.' + '.txt', dev_pred, fmt='%.8f')
        np.savetxt(self.out_dir + '/preds/' + '/' + str(epoch) + '/test_pred_' + str(epoch) + '.' + '.txt', test_pred, fmt='%.8f')

    def dump_best_predictions(self, dev_pred, test_pred):
        U.mkdir_p(self.out_dir + '/preds/' + '/best')
        np.savetxt(self.out_dir + '/preds/' + '/best/best_dev_pred.' + '.txt', dev_pred, fmt='%i')
        np.savetxt(self.out_dir + '/preds/' + '/best/best_test_pred.' + '.txt', test_pred, fmt='%i')
    
    def process_predictions(self):
        train_pred = np.zeros(len(self.train_pred))
        for i in range (len(self.train_pred)):
            curr_max = -1
            curr_index_max = -1

            for j in range (len(self.train_pred[i])):
                if curr_max < self.train_pred[i,j]:
                    curr_max = self.train_pred[i,j]
                    curr_index_max = j

            train_pred[i] = curr_index_max

        dev_pred = np.zeros(len(self.dev_pred))
        for i in range (len(self.dev_pred)):
            curr_max = -1
            curr_index_max = -1

            for j in range (len(self.dev_pred[i])):
                if curr_max < self.dev_pred[i,j]:
                    curr_max = self.dev_pred[i,j]
                    curr_index_max = j

            dev_pred[i] = curr_index_max
        
        test_pred = np.zeros(len(self.test_pred))
        for i in range (len(self.test_pred)):
            curr_max = -1
            curr_index_max = -1

            for j in range (len(self.test_pred[i])):
                if curr_max < self.test_pred[i,j]:
                    curr_max = self.test_pred[i,j]
                    curr_index_max = j
    
            test_pred[i] = curr_index_max

        return train_pred, dev_pred, test_pred

    def get_correct_count(self, train_pred, dev_pred, test_pred):
        assert(len(train_pred) == len(self.train_y))
        train_correct = 0
        for i in range (len(self.train_y)):
            if (train_pred[i] == self.train_y[i]): # Answer key can contain multiple answers
                train_correct += 1

        assert(len(dev_pred) == len(self.dev_y))
        dev_correct = 0
        for i in range (len(self.dev_y)):
            if (dev_pred[i] == self.dev_y[i]): # Answer key can contain multiple answers
                dev_correct += 1

        assert(len(test_pred) == len(self.test_y))
        test_correct = 0
        for i in range (len(self.test_y)):
            if (test_pred[i] == self.test_y[i]): # Answer key can contain multiple answers
                test_correct += 1
                
        return (train_correct, dev_correct, test_correct)

    def evaluate_from_best_file(self):
        train_pred = np.loadtxt(self.out_dir + '/preds/' + '/best/best_train_pred.' + '.txt')
        dev_pred = np.loadtxt(self.out_dir + '/preds/' + '/best/best_dev_pred.' + '.txt')
        test_pred = np.loadtxt(self.out_dir + '/preds/' + '/best/best_test_pred.' + '.txt')

        (self.train_correct, self.dev_correct, self.test_correct) = self.get_correct_count(train_pred, dev_pred, test_pred)
        self.train_accuracy = float(self.train_correct)/float(len(self.train_y_multi))
        self.dev_accuracy = float(self.dev_correct)/float(len(self.dev_y_multi))
        self.test_accuracy = float(self.test_correct)/float(len(self.test_y_multi))

        self.best_dev_correct = self.dev_correct
        self.best_test_correct = self.test_correct

    def evaluate(self, model, epoch):
        self.train_pred = model.predict(self.train_x, batch_size=self.batch_size_eval)
        self.dev_pred = model.predict(self.dev_x, batch_size=self.batch_size_eval)
        self.test_pred = model.predict(self.test_x, batch_size=self.batch_size_eval)

        self.dump_predictions(self.dev_pred, self.test_pred, epoch)

        new_train_pred, new_dev_pred, new_test_pred = self.process_predictions()

        
        (self.train_correct, self.dev_correct, self.test_correct) = self.get_correct_count(new_train_pred, new_dev_pred, new_test_pred)
        self.train_accuracy = float(self.train_correct)/float(len(self.train_y_multi))
        self.dev_accuracy = float(self.dev_correct)/float(len(self.dev_y_multi))
        self.test_accuracy = float(self.test_correct)/float(len(self.test_y_multi))

        if (self.dev_accuracy > self.best_dev):
            self.best_dev = self.dev_accuracy
            self.best_test = self.test_accuracy
            self.best_dev_correct = self.dev_correct
            self.best_test_correct = self.test_correct
            self.best_dev_epoch = epoch
            model.save_weights(self.out_dir + '/models/best_weights/best_model_weights.' + '.h5', overwrite=True)
            self.dump_best_predictions(new_dev_pred, new_test_pred)

        if (self.test_accuracy > self.best_test_missed):
            self.best_test_missed = self.test_accuracy
            self.best_test_missed_epoch = epoch
            
        return self.best_test

    def get_current_count(self):
        return self.dev_correct, len(self.dev_y_multi), self.test_correct, len(self.test_y_multi)

    def get_best_count(self):
        return self.best_dev_correct, len(self.dev_y_multi), self.best_test_correct, len(self.test_y_multi)

    def print_info(self):
        logger.info('[TRAIN] Acc: %.3f' % (self.train_accuracy))
        logger.info('[DEV]   Acc: %.3f (Best @ %i: {%.3f})' % (self.dev_accuracy, self.best_dev_epoch, self.best_dev))
        logger.info('[TEST]  Acc: %.3f (Best @ %i: {%.3f})' % (self.test_accuracy, self.best_dev_epoch, self.best_test))
        logger.info('---------------------------------------------------------------------------------------')
