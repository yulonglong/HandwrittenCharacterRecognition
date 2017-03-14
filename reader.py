from __future__ import print_function
import random
import codecs
import sys
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
        self.x = self.get_x(x)
        self.x2D = self.get_2D_x(x)

    def get_x(self, x):
        curr_x = list(float(xx) for xx in x)
        return curr_x

    def get_2D_x(self, x):
        new_x = []
        counter = 0
        for i in range(16):
            new_x_j = []
            for j in range(8):
                new_x_j_k = [ float(x[counter]) ]
                new_x_j.append(new_x_j_k)
                counter += 1
            new_x.append(new_x_j)
        return new_x
        
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
        
def read_dataset(train_path, model='default'):
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
            
            training_instances.append(curr_instance)

            if model == 'nn':
                x.append(curr_instance.x2D)
            else:
                x.append(curr_instance.x)
                
            y.append(ord(curr_instance.y)-97)

    return training_instances, x, y
