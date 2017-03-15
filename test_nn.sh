#!/bin/bash

THEANO_FLAGS="device=gpu0,mode=FAST_RUN,floatX=float32" python main.py -tr data/train.csv -ts data/test.csv --test -o output -t nn --epochs 20

