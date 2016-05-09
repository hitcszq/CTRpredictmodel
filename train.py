#coding:utf-8
import itertools
from feature_group_list import Feature_group_list
from lookup_table import Lookup_table 
from Class_CNN import CNN
import theano
import os
import sys
import os.path
import struct
#import numpy as np
from numpy import *
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
#import pycuda.autoinit
from pycuda.compiler import SourceModule
import theano.tensor as T
import time
from readMaxGroup import ReadMaxGroup
from train_feature_group_list import Train_feature_group_list
import mpi4py.MPI as MPI
from paras_class import paras
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
P0 = 0
P1 = 1
P2 = 2
batchsize = 1000
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start_time = time.time()
global context

def _finish_up():
    global context
    context.pop()
    context = None

    from pycuda.tools import clear_context_caches
    clear_context_caches()
    

if rank==0:
    sys.setrecursionlimit(1000000)


    readMaxGroup = ReadMaxGroup()
    res = readMaxGroup.read_max_group("/home/zjp/multigpu/dataset/train")
    #res = readMaxGroup.read_max_group("/media/new/Data/tangbonew/kaggle_avazu_pCTR_compitition/converted2#_tr_va_te_dataset/valid")

    pattern_groupid_list = readMaxGroup.pattern_groupid_list
    print 'res:%d \n'%(res)
    print 'pattern_groupid_list:'
    print pattern_groupid_list
    print 'readMaxGroup.input_groupid:'
    print readMaxGroup.input_groupid
    lookup_table = Lookup_table(readMaxGroup.maxGroupIndex)
    print 'lookup_table\'s table_count:%d\n'%(lookup_table.total_length)
    
    paras_instance=paras(readMaxGroup.input_groupid,lookup_table,pattern_groupid_list)
    #paras_instance_t = paras_instance
    for j in range(20):
	#paras_instance_t = pickle.dumps(paras_instance, True)
        comm.send(paras_instance, dest = P1)
        comm.send(paras_instance, dest = P2)
	paras_instance.lookup_table = None
        batch_num = readMaxGroup.total_instance_num/batchsize
        if readMaxGroup.total_instance_num % batchsize !=0:
            batch_num += 1
        for itert in range(batch_num):
	    #if itert == batch_num-1:
	#	batchsize = readMaxGroup.total_instance_num / 2 % batchsize
	    if itert % 100 ==0:
		print 'iter %d, complete mini-batchs:%d, take time %s' % (j, itert, time.time()-start_time) 
            status = MPI.Status()
            delta = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
            recv_rank = status.Get_source()
            #print 'recevice from %d' % recv_rank
            #更新参数
	    paras_instance.vs = paras_instance.vs - delta['vs_delta']	
            paras_instance.theta1 = paras_instance.theta1 + delta['theta1_delta']
            paras_instance.theta2 = paras_instance.theta2 + delta['theta2_delta']
            paras_instance.theta3 = paras_instance.theta3 + delta['theta3_delta']
        
            paras_instance.b1 = paras_instance.b1 + delta['b1_delta']
            paras_instance.b2 = paras_instance.b2 + delta['b2_delta']
            paras_instance.b3 = paras_instance.b3 + delta['b3_delta']
            
            partOfTable = delta['delta_x_value']
            vec_length = lookup_table.embedding_length
            for i in xrange(delta['batchsize']):
                group = delta['list_group_batch'][i]			
                feature = delta['list_feature_batch'][i]
                count = 0
                for groupid in group:
                    featureid = feature[count]
                    offset = lookup_table.groupid_offset[groupid]
                    #print batchsize    
                    pos = lookup_table.QueryPos(groupid,featureid)
                    lookup_table.central_array[pos:pos+vec_length] -= paras_instance.alpha * delta['delta_x_value'][i][offset:offset+vec_length]
                    partOfTable[i][offset:offset+vec_length] = lookup_table.central_array[pos:pos+vec_length]
                    count += 1
	    #paras_instance_t = pickle.dumps(paras_instance)
            #print 'send to %d' % recv_rank
            paras_instance.partOfTable = partOfTable
            if recv_rank == P1:
                comm.send(paras_instance, dest = P1)
            else:
                comm.send(paras_instance, dest = P2)
	#paras_instance.lookup_table = lookup_table
    	status = MPI.Status()
    	delta = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
    	recv_rank = status.Get_source()
    	if recv_rank == P1:
        	comm.send(paras_instance, dest = P1)
    	else:
        	comm.send(paras_instance, dest = P2)
	paras_instance.lookup_table = lookup_table
	print 'P0 complete'
        comm.send(readMaxGroup, dest=P1)
    print 'P0 complete'    
else:
#--------------------------choice of GPU----------------
    cuda.init()
    device = cuda.Device(rank-1)
    context = device.make_context()
#--------------------------------------------------------
    cnn = CNN()

    origin_batchsize = batchsize
    epochs = 20
    epoch = 0
    ite=  0
    stop = False
    sum_time = 0
    while epoch < epochs:
        t0 = time.time()
        count = 0
	batchsize = 1000
        block_size = 100000
        #READ DIFFERENT DISK
        if rank == 1:
            train_feature_group_list = Train_feature_group_list("/media/new/Data/train1.dat")
        else:
            train_feature_group_list = Train_feature_group_list("/media/new/DATA2/train2.dat")
	#readMaxGroup = ReadMaxGroup()
        #res = readMaxGroup.read_max_group("/home/zjp/multigpu/dataset/train")
        train_feature_group_list.input_groupid = 23 ############ this should be general ##############
        train_feature_group_list.readBlockFile(block_size)
        trainInstanceNum = 40428966/2########### this should be general ############
        blocks = float(trainInstanceNum) / block_size
        blocks = int(blocks)+1
        print "blocks"
        print blocks

        for b in range(0,blocks):
            print("block:%s"%(b))
            if b==blocks-1 and  trainInstanceNum%block_size!=0:
                block_size = trainInstanceNum % block_size
            #cnn.feature_group_list = train_feature_group_list
            #batchsize = 100
            #origin_batchsize = 1000
            #chunk = int( round( float(block_size) / batchsize ) )
            chunk = block_size / batchsize
	    if block_size % batchsize != 0:
		chunk += 1
            for batch_index in range(0,chunk):
                #----------------receive paras, every mini-batch------------------------
                #print '%d recevice from 0' % rank
                paras_recv = comm.recv(source = P0)
                if b == 0 and batch_index == 0:
                    cnn.set_paras(paras_recv)
                else:
                    cnn.setPartOfTable(batchsize, list_group_batch, list_feature_batch, paras_recv)
                #-------------------------train----------------------------------------------
                if batch_index==chunk-1 and block_size%batchsize!=0:
                    batchsize = block_size%batchsize
                
                [(trainx,trainy,list_feature_batch,list_group_batch)] = cnn.load_data(train_feature_group_list,batch_index,origin_batchsize,batchsize)

                #t_before_train_time = time.time()         
                delta_paras=cnn.train(batchsize,trainx,trainy,list_feature_batch,list_group_batch)
                #t_after_train_time = time.time()
		#delta_paras = delta_paras.copy()
		delta_paras['batchsize'] = batchsize
                #---------after one mini-batch, send delta to process 0--------------
                #print '%d send to 0' % rank
                comm.send(delta_paras, dest = P0)
                
                #print 'each iter,training takes time(seconds):'
                #print t_after_train_time-t_before_train_time
                
#count += 1
#               if count%100==0:
#one_hundred_batch = time.time()
#                   print("rank %d, epoch:%s,load and train data,finished:%s"%(rank, epoch,float(count)/chunk))
            
            train_feature_group_list.list_target = []
            train_feature_group_list.list_feature = []
            train_feature_group_list.list_group = []
            label = train_feature_group_list.BlockTrainFile.tell()
    #		train_feature_group_list.BlockTrainFile.close()
            if b==blocks-1:
                break
            train_feature_group_list.BlockTrainFile.seek(label,0)
            if b==blocks-2 and trainInstanceNum % block_size!=0:
                train_feature_group_list.readBlockFile( trainInstanceNum % block_size )
                continue
            train_feature_group_list.readBlockFile(block_size)
        
        comm.recv(source = P0)
        train_feature_group_list.BlockTrainFile.close()
        
	print 'P%d complete' % rank
        '''
        reading test file by blocking,because test file is scalar.
        '''
        if rank == 1:
            # print "---------Start predicting----------"

            # cnn.pred_value_file = open("/home/zjp/multigpu/test_prediction/pred_value_epoch%d.txt"%(epoch),'a')

            # test_feature_group_list = Train_feature_group_list("/home/zjp/multigpu/dataset/test/test.bin")
            # test_feature_group_list.input_groupid = 23
            # test_instance_num = test_feature_group_list.count_of_file
            
            # cnn_test = CNN()
            # paras_test=paras(23,cnn.lookup_table,cnn.pattern_group_id_list)
            # cnn_test.set_paras(paras_test)
            
            
            # test_block_size = 100000
            # origin_test_block_size = test_block_size
            # blocks = test_instance_num / test_block_size
            # if test_instance_num % test_block_size != 0:
                # blocks += 1
 
            # for block_index in range(0,blocks):
                # if block_index==blocks-1 and test_instance_num%test_block_size!=0:
                    # test_block_size = test_instance_num%test_block_size
                # test_feature_group_list.readBlockFile(test_block_size)
       
                # print len(test_feature_group_list.list_target)
                # print test_block_size
                # print origin_test_block_size
                # [(testx,testy,list_feature_batch,list_group_batch)] = cnn_test.load_data(test_feature_group_list,\
                    # 0,origin_test_block_size,test_block_size)

                # auc = cnn.calc_auc(testx,testy,batchsize,list_feature_batch)
                # test_feature_group_list.list_target = []
                # test_feature_group_list.list_feature = []
                # test_feature_group_list.list_group = []
                # print 'epoch :%d ,AUC:%s'%(epoch,auc)
            # cnn.pred_value_file.close()
            # test_feature_group_list.BlockTrainFile.close()
            readMaxGroup = comm.recv(source=P0)
            print("epoch:%s,start calculating... "%(epoch,))
            cnn.pred_value_file = open("FM_NN_pred_value_blocktmp.txt",'a')
            
            batchsize = 1000
            test_block_size = 10000
            test_block_size_backup = test_block_size
            test_train_feature_group_list = Train_feature_group_list("/home/zjp/test.bin")
            test_train_feature_group_list.input_groupid = readMaxGroup.input_groupid
#            test_trainInstanceNum = readMaxGroup.total_instance_num
	    test_trainInstanceNum = test_train_feature_group_list.count_of_file
            test_blocks = float(test_trainInstanceNum) / test_block_size
            test_blocks = int(test_blocks)+1

            train_labels = []
            train_preds = []
	    auc_preds_list = []

            trainset_log_loss = 0
            for b in range(0,test_blocks):
                if b==test_blocks-1 and  test_trainInstanceNum%test_block_size!=0:
                    test_block_size = test_trainInstanceNum % test_block_size
                test_train_feature_group_list.readBlockFile(test_block_size)
                cnn.feature_group_list = test_train_feature_group_list
                [(test_trainx,test_trainy,test_list_feature_batch,test_list_group_batch)] = cnn.load_data(test_train_feature_group_list, 0, test_block_size_backup,test_block_size)
                train_labels.append(test_trainy)
                auc = cnn.calc_auc(test_trainx,test_trainy,batchsize,test_list_feature_batch)
                train_preds.append(cnn.pred)

	        for value_list in cnn.pred:
			prd_value = value_list[1]
	                prd_value = float(prd_value)
			auc_preds_list.append(prd_value)

                test_train_feature_group_list.list_target = []
                test_train_feature_group_list.list_feature = []
                test_train_feature_group_list.list_group = []
                print("calc log_loss:blocks:%s"%(b,))

            cnn.pred_value_file.close()
            test_train_feature_group_list.BlockTrainFile.close()
            train_labels = list(itertools.chain.from_iterable(train_labels))
            train_preds =  list(itertools.chain.from_iterable(train_preds))

            trainset_log_loss = log_loss(train_labels,train_preds)
	    te_auc = roc_auc_score(train_labels,auc_preds_list)

            trainloss_file=open("./log/trainloss.txt",'a')
	    auc_file = open("./log/trainauc.txt",'a')

            trainloss_file.write('%s,'%(trainset_log_loss))
	    auc_file.write('%s'%(te_auc))

            trainloss_file.close()
            auc_file.close()
        epoch += 1
        t1 = time.time()
        print 'rank %d, this epoch takes time(seconds):' % rank
        print t1-t0
        sum_time += (t1-t0)
        batchsize = origin_batchsize
    import atexit
    atexit.register(_finish_up)
    print 'all epochs take time(seconds):'
    print sum_time
