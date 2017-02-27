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
parser.add_argument("-o", "--out", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The output folder")

args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir)
U.set_logger(out_dir)
U.print_args(args)

R.read_dataset(args)
