import theano
import os
import sys
import os.path
import struct
import numpy as np
import theano.tensor as T
import time

class Train_feature_group_list:
	def __init__(self,trainFileDir):
		self.list_target = []
		self.list_group = []
		self.list_feature = []
		self.maxGroupIndex = {}  #1:2,2:2,3:2,4:7,5:543354523,6:345323.....
		self.max_groupid = 69
		self.input_groupid = 0
		self.pattern_groupid_list = []
		self.count_of_file = 0
		self.total_instance_num = 0
		self.BlockTrainFile = open(trainFileDir,'rb')
		(self.count_of_file,) = struct.unpack('i',self.BlockTrainFile.read(4))
		for i in xrange(self.max_groupid):
			self.maxGroupIndex[i+1] = 0  #groupid from :1,2,3....28

	def readBlockFile(self,maxValue):

		for i in range(maxValue):
			(target,max_group_id) = struct.unpack('di',self.BlockTrainFile.read(8+4))
			self.list_target.append(target)
			tuple_group = struct.unpack('%di'%(max_group_id),self.BlockTrainFile.read(4*max_group_id))
			self.list_group.append(tuple_group)
			tuple_feature = struct.unpack('%di'%(max_group_id),self.BlockTrainFile.read(4*max_group_id))
			self.list_feature.append(tuple_feature)

		
				
		
