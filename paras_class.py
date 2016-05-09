import theano
#import sklearn
#import sklearn.metrics
import numpy as np
#import itertools
from numpy import *	#zeros(..)
import theano.tensor as T
import theano.tensor.nnet as nnet
#from train import Feature_group_list
from lookup_table import Lookup_table
#from compiler.ast import flatten
from feature_group_list import Feature_group_list
#import pycuda.gpuarray as gpuarray
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
#import time


class paras:
	def __init__(self,input_groupid,lookup_table,pattern_groupid_list):
		self.input_nodenum = input_groupid*lookup_table.embedding_length
		
		self.mu = 0
		self.sigm = 0.1
		self.k_size = 3
		self.hid1_nodenum = 400
		self.hid2_nodenum = 100
		self.output_nodenum = 2
				
		self.theta1 = np.random.normal(self.mu,self.sigm,self.input_nodenum*self.hid1_nodenum).reshape(self.input_nodenum,self.hid1_nodenum).astype(np.float32)
		self.b1 = np.random.normal(self.mu,self.sigm,self.hid1_nodenum).reshape(1,self.hid1_nodenum).astype(np.float32)
		self.theta2 = np.random.normal(self.mu,self.sigm,self.hid1_nodenum*self.hid2_nodenum).reshape(self.hid1_nodenum,self.hid2_nodenum).astype(np.float32)
		self.b2 =  np.random.normal(self.mu,self.sigm,self.hid2_nodenum).reshape(1,self.hid2_nodenum).astype(np.float32)
		self.theta3 = np.random.normal(self.mu,self.sigm,self.hid2_nodenum*self.output_nodenum).reshape(self.hid2_nodenum,self.output_nodenum).astype(np.float32)
		self.b3 =  np.random.normal(self.mu,self.sigm,self.output_nodenum).reshape(1,self.output_nodenum).astype(np.float32)
		self.vs = np.random.randn(self.hid2_nodenum,self.k_size).astype(np.float32)
		

		self.lamda = 0.1
		self.alpha = 0.001
		self.MAX_Threads_EachBlock_NUM = 800
#		FM model V matrix also theano.shared()
#		self.table_list = theano.shared(lookup_table.table_list)
		#+
        	self.lookup_table = lookup_table
		self.pattern_group_id_list = pattern_groupid_list
        	self.partOfTable = None  #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
