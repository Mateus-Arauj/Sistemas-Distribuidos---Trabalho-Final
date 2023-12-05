from mpi4py import MPI
import datetime

def f(x):
    return 5*x**3 + 3*x**2 + 4*x + 20

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    soma = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        soma += f(a + i * h)
    return soma * h

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

val = [0]
a = 0.0
b = 1000000.0
n = 10000000

if rank == 0:
    wt = MPI.Wtime()
    start_time = datetime.datetime.now()

    val[0] = n
    val = comm.bcast(val, root=0)  # Broadcast para todos os processos

    parte_n = int(n / size)
    inicio = parte_n * rank
    fim = inicio + parte_n
    if rank == size - 1:
        fim = n

    local_integral = trapezoidal_rule(a + (b - a) * inicio / n, a + (b - a) * fim / n, parte_n)
    total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)  # Reduce para soma das integrais

    wt = MPI.Wtime() - wt
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    print("Resultado da integral:", total_integral)
    print("Tempo de execução (MPI.Wtime):", wt)
    print("Tempo de execução (datetime):", execution_time)

else:
    val = comm.bcast(val, root=0)  # Recebe o broadcast do mestre

    parte_n = int(val[0] / size)
    inicio = parte_n * rank
    fim = inicio + parte_n
    if rank == size - 1:
        fim = val[0]

    local_integral = trapezoidal_rule(a + (b - a) * inicio / val[0], a + (b - a) * fim / val[0], parte_n)
    comm.reduce(local_integral, op=MPI.SUM, root=0)  # Participa do reduce para soma das integrais
