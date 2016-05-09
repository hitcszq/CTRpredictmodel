import theano
import os
import sys
import os.path
import struct
import numpy as np
import theano.tensor as T
import time
def read():
	testdirpath = "/media/new/Data/tangbo/kaggle_display_ad_challenge/convert2#_after_gbdt_dataSet/testSet/after_gbdt_test.bin"
	testfile = open(testdirpath)
	(count) = struct.unpack('i',testfile.read(4))
	print count
read()
