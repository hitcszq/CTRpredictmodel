import mpi4py.MPI as MPI

P0 = 0
P1 = 1
P2 = 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == P0:
    #初始化参数，Class_CNN中的属性，封装成一个结构体
    #Class parameters
    comm.send(parameters, dest = P1)
    comm.send(parameters, dest = P2)
    while(True):
        status = MPI.Status()
        comm.recv(source = )
        