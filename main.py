from mpi4py import MPI
import datetime
import math

def f(x):
    return 5*x**3 + 3*x**2 + 4*x + 20

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    soma = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        soma += f(a + i * h)
    return soma * h

def butterfly_sum(comm, local_value):
    rank = comm.Get_rank()
    size = comm.Get_size()
    step = 1
    while step < size:
        partner = rank ^ step
        if partner < size:
            partner_value = comm.sendrecv(local_value, dest=partner, source=partner)
            if rank < partner:
                local_value += partner_value
        step *= 2
    return local_value

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a = 0.0
b = 1000000.0
n = 10000000

if rank == 0:
    wt = MPI.Wtime()
    start_time = datetime.datetime.now()

# Broadcast de n para todos os processos
val = comm.bcast(n if rank == 0 else None, root=0)

parte_n = int(val / size)
inicio = parte_n * rank
fim = inicio + parte_n if rank != size - 1 else val

local_integral = trapezoidal_rule(a + (b - a) * inicio / val, a + (b - a) * fim / val, parte_n)

# MPI.Reduce
start_reduce = MPI.Wtime()
total_integral_reduce = comm.reduce(local_integral, op=MPI.SUM, root=0)
end_reduce = MPI.Wtime()

# Butterfly
start_butterfly = MPI.Wtime()
total_integral_butterfly = butterfly_sum(comm, local_integral)
end_butterfly = MPI.Wtime()

if rank == 0:
    wt = MPI.Wtime() - wt
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    print("Resultado da integral com MPI.Reduce:", total_integral_reduce)
    print("Tempo de execução do MPI.Reduce:", end_reduce - start_reduce)

    print("Resultado da integral com Butterfly:", total_integral_butterfly)
    print("Tempo de execução do Butterfly:", end_butterfly - start_butterfly)

    print("Tempo total de execução (datetime):", execution_time)
