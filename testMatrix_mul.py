import theano
import sklearn
import sklearn.metrics
import numpy as np
import itertools
from numpy import *	#zeros(..)
import theano.tensor as T
import theano.tensor.nnet as nnet
#from train import Feature_group_list
from lookup_table import Lookup_table
from compiler.ast import flatten
from feature_group_list import Feature_group_list
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
def test_AUC():
	mod = SourceModule(
	"""
		__global__ void mat_mul(float* z,float* x, float* y,int* x_width,int* y_width)
		{
			int t_idx = threadIdx.x;
			int idx = (blockDim.x*blockIdx.x) + t_idx;
			float sum = 0;
			int xw = x_width[0];
			int yw = y_width[0];
			for(int e=0;e<xw;e++)
			{
				sum += x[idx/yw*xw+e]*y[yw*e+t_idx];
			}
			z[idx] = sum;
		}

		__global__ void Matrix_multi(float* z,float* x,float* y,int* x_width,int* y_width)
		{
			int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
			int idy = (blockIdx.y*blockDim.y) + threadIdx.y;
			int thread_idx = ((gridDim.x*blockDim.x-1)*idy)+idx;
			float sum = 0;
			int xw = x_width[0];
			int yw = y_width[0];
			if(idx<5 && idy<5)
			{
			for(int e=0;e<xw;e++)
			{
				sum += x[thread_idx/yw*xw+e]*y[yw*e+idx];
			}
			z[thread_idx] = sum;
			}
		
		}
		
		__global__ void matrix_tran(float* a_t,float* a,int* rows,int* cols,int* line_remainders)
		{
			int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
			int idy = (blockIdx.y*blockDim.y) + threadIdx.y;
			int eachLineRemaindNum = line_remainders[0];
			int cols_num = cols[0];
			int rows_num = rows[0];
			int thread_idx = ((gridDim.x*blockDim.x-eachLineRemaindNum)*idy) + idx;
			int sz_row = thread_idx/cols_num;
			int sz_col = thread_idx%cols_num;
			if(idx<cols_num)
			{
			a_t[thread_idx] = a[sz_row+sz_col * rows_num];
			}
		}
		__global__ void BigCOLMatrix_mul(float* z,float* x, float* y,int x_height,int x_width,int y_width,int line_remainders)
		{
			int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
			int idy = (blockIdx.y*blockDim.y) + threadIdx.y;
			int eachLineRemaindNum = line_remainders;
			int thread_idx = ((gridDim.x*blockDim.x-eachLineRemaindNum )*idy) + idx;
			float sum = 0;
			int xw = x_width;
			int yw = y_width;
			int xHeight = x_height;
			if(idx<yw && idy<xHeight)
			{
				for(int e=0;e<xw;e++)
				{
					sum += x[thread_idx/yw*xw+e]*y[yw*e+idx];
				}
				z[thread_idx] = sum;
			}
		}

	"""
	)
	
	'''

	'''
	mat_mul = mod.get_function("mat_mul")
	Matrix_multi = mod.get_function("Matrix_multi")
	matrix_tran = mod.get_function("matrix_tran")
	BigCOLMatrix_mul = mod.get_function("BigCOLMatrix_mul")
	'''
	z = gpuarray.to_gpu(np.zeros((100,400)).astype(np.float32))
#	x = gpuarray.to_gpu(np.asarray([[1,1],[1,1]]).astype(np.float32))
#	y = gpuarray.to_gpu(np.asarray([[1,1,1,1],[1,1,1,1]]).astype(np.float32))
	x = gpuarray.to_gpu(np.ones((100,2000)).astype(np.float32))
	y = gpuarray.to_gpu(np.ones((2000,400)).astype(np.float32))

	x_width = gpuarray.to_gpu(np.array([2000]).astype(np.int32))
	y_width = gpuarray.to_gpu(np.array([400]).astype(np.int32))
	mat_mul(z,x,y,x_width,y_width,block=(400,1,1),grid=(100,1))
	print z
	'''
	z = gpuarray.to_gpu(np.zeros((5,5)).astype(np.float32))
	x = gpuarray.to_gpu(np.asarray([[1,2,1],[1,1,2],[1,1,2],[2,2,1],[1,2,3]]).astype(np.float32))
	y = gpuarray.to_gpu(np.asarray([[2,1,3,1,1],[1,5,6,1,1],[2,2,1,1,3]]).astype(np.float32))
#	x = gpuarray.to_gpu(np.ones((400,1000)).astype(np.float32))
#	y = gpuarray.to_gpu(np.ones((1000,1010)).astype(np.float32))
	x_width = gpuarray.to_gpu(np.array([3]).astype(np.int32))
	y_width = gpuarray.to_gpu(np.array([5]).astype(np.int32))
	

	a = np.asarray([[1,2,1],[1,1,2],[1,1,2],[2,2,1],[1,2,3]]).astype(np.float32)
	b = np.asarray([[2,1,3,1,1],[1,5,6,1,1],[2,2,1,1,3]]).astype(np.float32)

	Matrix_multi(z,cuda.In(a),cuda.In(b),x_width,y_width,block=(2,2,1),grid=(3,3))
#	Matrix_multi(z,x,y,x_width,y_width,block=(2,2,1),grid=(3,3))
	print z
	x_t = gpuarray.to_gpu(np.zeros((3,5)).astype(np.float32))
	gpu_rows = gpuarray.to_gpu(np.array([3]).astype(np.int32))
	gpu_cols = gpuarray.to_gpu(np.array([5]).astype(np.int32))
	gpu_remainders = gpuarray.to_gpu(np.array([1]).astype(np.int32))
	matrix_tran(x_t,x,gpu_rows,gpu_cols,gpu_remainders,block=(2,2,1),grid=(3,2))
	print x_t


	z = gpuarray.to_gpu(np.zeros((4,7)).astype(np.float32))
	a = np.asarray([
	[1,2,3,4,5,6,7],
	[1,2,2,3,3,4,4],
	[2,2,3,1,1,2,2],
	[0,0,1,2,3,1,1]
#	[1,2,3,1,2,3,1]
	
	]).astype(np.float32)
	b = np.asarray([
	[7,6,5,4,3,2,1],
	[1,1,2,2,3,4,5],
	[1,2,2,3,3,4,5],
	[2,1,1,1,2,2,3],
	[1,1,2,1,2,1,1],
	[1,2,3,3,1,1,1],
	[2,1,1,2,2,1,2]
	]).astype(np.float32)

	BigCOLMatrix_mul(z,cuda.In(a),cuda.In(b),np.int32(4),np.int32(7),np.int32(7),np.int32(1),block=(2,2,1),grid=(4,3))
	print z

test_AUC()
