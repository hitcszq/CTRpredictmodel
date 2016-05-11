#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 nohup python **.py > a.txt &
#THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 nohup python **.py > a.txt &
#!/bin/bash
true>./log/trainauc.txt
true>./log/trainloss.txt
mpirun -np 3 python train.py
