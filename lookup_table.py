import theano
import theano.tensor as T
#from train import Feature_group_list 
import numpy as np
from feature_group_list import Feature_group_list
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
class Lookup_table:
#	table = T.matrix()

#	table_list = []
#	group_list = []
#	table_counts = 0
	def __init__(self,max_groupidx):
		self.embedding_length = 8
		self.total_length = 0
		mu = 0
		sigm = 0.01
		self.group_offset = {}

		self.groupid_offset = {}
		maxGroupIndex = max_groupidx
		for k in maxGroupIndex.keys():
			if maxGroupIndex[k] == 0:
				continue
			self.total_length += (maxGroupIndex[k]+1)*self.embedding_length



		self.central_array =  np.random.normal(mu,sigm,self.total_length).astype(np.float32) 
	

		'''		
		value = 0
		for i in xrange(self.total_length):
			self.central_array[i] = value
			value += 1.0
		'''


		count = 0
		num = 0
		for k in maxGroupIndex.keys():
			if maxGroupIndex[k] == 0:
				continue
			else:
				self.group_offset[k] = count * self.embedding_length
				self.groupid_offset[k] = num * self.embedding_length
				count += (maxGroupIndex[k]+1)
				num += 1
		print 'init lookupTable finished!'
	def QueryPos(self,groupid,featureid):
		offset = self.group_offset[groupid]
		pos = offset + featureid*self.embedding_length
		return pos

