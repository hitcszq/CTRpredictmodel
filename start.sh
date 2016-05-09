THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 nohup python **.py > a.txt &
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 nohup python **.py > a.txt &
