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
        
def read_dataset(args):
    training_instances = []
    with codecs.open(args.train_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            tokens = re.split(',', line.rstrip())
            curr_instance = Alpha(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4:])
            
            logger.info(curr_instance.y)
            curr_instance.print_letter()
            
            training_instances.append(curr_instance)
            