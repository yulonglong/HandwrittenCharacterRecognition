#!/bin/bash

THEANO_FLAGS="device=gpu0,mode=FAST_RUN,floatX=float32" python main.py -tr data/train.csv -o output -t nn --epochs 20
