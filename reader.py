from __future__ import print_function
import random
import codecs
import sys
import nltk
import logging
import re
import glob
import numpy as np
import pickle as pk
import re # regex
import copy

logger = logging.getLogger(__name__)

class Alpha():
    def __init__ (self, id, y, nextId, position, x):
        self.id = id
        self.y = y
        self.nextId = nextId
        self.position = position
        self.x = x
        
    def print_letter(self):
        counter = 0
        for i in range(16):
            for j in range(8):
                if (self.x[counter] == '1'):
                    print('X', end='')
                else:
                    print(' ', end='')
                counter += 1
            print('')
        print('')

class_mapping = dict()
        
def read_dataset(train_path, mode='training'):
    training_instances = []
    x = []
    y = []
    counter = 0
    class_mapping_index = 0
    with codecs.open(train_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            if counter == 0:
                counter += 1
                continue

            tokens = re.split(',', line.rstrip())
            curr_instance = Alpha(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4:])
            
            # logger.info(curr_instance.y)
            # curr_instance.print_letter()
            # logger.info(list(ord(x) for x in curr_instance.y )[0] - 97)

            if mode == 'training':
                if not curr_instance.y in class_mapping:
                    class_mapping[curr_instance.y] = class_mapping_index
                    class_mapping_index += 1
            
            training_instances.append(curr_instance)

            x.append(list(float(xx) for xx in curr_instance.x))
            y.append(class_mapping[curr_instance.y])
            
    logger.info("Class mapping size %d", len(class_mapping))
    return training_instances, np.array(x), np.array(y)
            