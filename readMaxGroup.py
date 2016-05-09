import theano
import os
import sys
import os.path
import struct
import numpy as np
import theano.tensor as T
import time


#train_set_rootdir="/media/new/Data/AdData/data"
class ReadMaxGroup:
	def __init__(self):	
		self.maxGroupIndex = {}  #1:2,2:2,3:2,4:7,5:543354523,6:345323.....
		self.max_groupid = 23
		self.input_groupid = 0
		self.pattern_groupid_list = []
		self.count_of_file = 0
		self.total_instance_num = 0
	def read_max_group(self,rootdir):

		for i in xrange(self.max_groupid):
			self.maxGroupIndex[i+1] = 0  #groupid from :1,2,3....28

		train_set_rootdir=rootdir
		idx = 0
		for parent,dirnames,filenames in os.walk(train_set_rootdir):
			for filename in filenames:
				if filename=="2015-04-09_part-00113" or filename=="2015-04-10_part-00175":
					print 'continue above file!'
					continue

				full_filename = os.path.join(parent,filename)
				binfile = open(full_filename,'rb')
				(self.count_of_file,) = struct.unpack('i',binfile.read(4))
				self.total_instance_num += self.count_of_file
				binfile.close()
		print self.total_instance_num
		print '\n'
		
		for parent,dirnames,filenames in os.walk(train_set_rootdir):
			for filename in filenames:
#				print "filename : "+os.path.join(parent,filename)
				if filename=="2015-04-09_part-00113" or filename=="2015-04-10_part-00175":
					print 'continue above file!'
					continue
				full_filename = os.path.join(parent,filename)
				binaryfile = open(full_filename,'rb')
				(self.count_of_file,) = struct.unpack('i',binaryfile.read(4))
				
				for sample_idx in range(self.count_of_file):
					(target,max_group_id) = struct.unpack('di',binaryfile.read(8+4))
					tuple_group = struct.unpack('%di'%(max_group_id),binaryfile.read(4*max_group_id))
					tuple_feature = struct.unpack('%di'%(max_group_id),binaryfile.read(4*max_group_id))

					for groupid_incre in xrange(len(tuple_group)):
						groupid = tuple_group[groupid_incre]
						if tuple_feature[groupid_incre]>self.maxGroupIndex[groupid]:
							self.maxGroupIndex[groupid] = tuple_feature[groupid_incre]


#				print '\n'
				binaryfile.close()
#				break # only process one file
		for i in xrange(self.max_groupid):
			if self.maxGroupIndex[i+1] != 0:
				self.input_groupid += 1
#			self.maxGroupIndex[i+1] += 1#self.maxGroupIndex[i+1]
			print 'selfmaxGroupIndex[%d]\'s count is %d:'%(i+1,self.maxGroupIndex[i+1])
		for k in self.maxGroupIndex.keys():
			if self.maxGroupIndex[k] != 0:
				self.pattern_groupid_list.append(k)
		return 0

