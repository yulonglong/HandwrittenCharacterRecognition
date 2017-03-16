from __future__ import print_function
import random
import codecs
import sys
import logging
import re
import glob
import numpy as np
import pickle as pk
import helper as H
import re # regex
import copy
from time import time

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def run_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, args, out_dir=None, class_weight=None):
    logger.info("train_x.shape : " + str(train_x.shape))
    logger.info("train_y.shape : " + str(train_y.shape))
    logger.info("dev_x.shape   : " + str(dev_x.shape))
    logger.info("dev_y.shape   : " + str(dev_y.shape))
    logger.info("test_x.shape  : " + str(test_x.shape))
    logger.info("test_y.shape  : " + str(test_y.shape))

    if model_type == 'nn':
        return run_nn_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, args, out_dir=out_dir, class_weight=class_weight)
    else:
        return run_simple_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, out_dir=out_dir, class_weight=class_weight)

# all the arrays have to be numpy array before passing them into this function
def run_simple_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, out_dir=None, class_weight=None):
    from sklearn import datasets, neighbors, linear_model, svm

    totalTime = 0

    startTrainTime = time()
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
    else:
        raise NotImplementedError('Model not implemented')

        
    logger.info("Finished training.")
    endTrainTime = time()
    trainTime = endTrainTime - startTrainTime
    logger.info("Training time : %d seconds" % trainTime)


    logger.info("Start predicting train set...")
    train_pred_y = model.predict(train_x)
    logger.info("Finished predicting train set.")
    logger.info("Start predicting test set...")
    test_pred_y = model.predict(test_x)
    logger.info("Finished predicting test set.")
    endTestTime = time()
    testTime = endTestTime - endTrainTime
    logger.info("Testing time : %d seconds" % testTime)
    totalTime += trainTime + testTime

    train_pred_y = np.round(train_pred_y)
    test_pred_y = np.round(test_pred_y)

    np.savetxt(out_dir + '/preds/best_test_pred' + '.txt', test_pred_y, fmt='%i')

    logger.info('[TRAIN] Acc: %.3f' % (accuracy_score(train_y, train_pred_y)))
    logger.info('[TEST]  Acc: %.3f' % (accuracy_score(test_y, test_pred_y)))

    return accuracy_score(test_y, test_pred_y)


####################################################################################################
#### BEGIN SECTION FOR NEURAL NET
##

def create_nn_model():
    from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, MaxPooling2D
    from keras.models import Model

    #Create your own input format (here 3x200x200)
    img_input = Input(shape=(1,16,8), name='image_input')

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=(1,16,8))(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Dropout(0.5)(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Dropout(0.5)(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dense(26, activation='softmax', name='predictions')(x)

    my_model = Model(input=img_input, output=x)
    my_model.summary()

    return my_model


def run_nn_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model_type, args, out_dir=None, class_weight=None):
    import keras.utils.np_utils as np_utils
    train_y_multi = np_utils.to_categorical(train_y, 26)
    dev_y_multi = np_utils.to_categorical(dev_y, 26)
    test_y_multi = np_utils.to_categorical(test_y, 26)

    ############################################################################################
    ## Set optimizers and compile model
    #

    import keras.optimizers as opt
    clipvalue = 0
    clipnorm = 10
    optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    loss = 'categorical_crossentropy'
    metric = 'accuracy'

    model = create_nn_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    logger.info('model compilation completed!')

    ###############################################################################################################################
    ## Training
    #

    from Evaluator import Evaluator

    evl = Evaluator(
        out_dir,
        (train_x, train_y, train_y_multi),
        (dev_x, dev_y, dev_y_multi), ### WARNING PLEASE CHANGE THIS TO PROPER DEV SET
        (test_x, test_y, test_y_multi)
    )

    logger.info('---------------------------------------------------------------------------------------')
    logger.info('Initial Evaluation:')
    evl.evaluate(model, -1)

    # Print and send email Init LSTM
    content = evl.print_info()

    total_train_time = 0
    total_eval_time = 0

    for ii in range(args.epochs):

        t0 = time()
        history = model.fit(train_x, train_y_multi, batch_size=32, nb_epoch=1, verbose=0)
        tr_time = time() - t0
        total_train_time += tr_time

        # Evaluate
        t0 = time()
        best_acc = evl.evaluate(model, ii)
        evl_time = time() - t0
        total_eval_time += evl_time

        logger.info('Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm)' % (ii, tr_time, tr_time/60.0, evl_time, evl_time/60.0))
        logger.info('[Train] loss: %.4f , metric: %.4f' % (history.history['loss'][0], history.history['acc'][0]))

        # Print and send email Epoch LSTM
        content = evl.print_info()

    return best_acc
