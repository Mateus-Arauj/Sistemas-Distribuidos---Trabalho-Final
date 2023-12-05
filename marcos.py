# integral_mpi.py
from mpi4py import MPI
import numpy as np
import time

# Definição da função f(x)
def f(x):
    return 5 * x**3 + 3 * x**2 + 4 * x + 20

# Algoritmo do Trapézio
def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        integral += f(x)
    return integral * h

# Função para medir o tempo de execução
def measure_execution_time(function, *args):
    start_time = time.time()
    function(*args)
    end_time = time.time()
    return end_time - start_time

# Processamento pelo mestre
def master_processing(a, b, n, size, comm):
    local_n = n // size
    local_a = a + rank * local_n * (b - a) / n
    local_b = local_a + local_n * (b - a) / n

    if rank == 0:
        master_integral = trapezoidal_rule(a, b - (size - 1) * (b - a) / size, n - local_n * (size - 1))
        total_integral = master_integral
        for i in range(1, size):
            slave_integral = comm.recv(source=i, tag=77)
            total_integral += slave_integral
    else:
        slave_integral = trapezoidal_rule(local_a, local_b, local_n)
        comm.send(slave_integral, dest=0, tag=77)

    if rank == 0:
        print("Resultado da integral com processamento pelo mestre:", total_integral)

# Método Butterfly
def butterfly_method(local_a, local_b, local_n, rank, size, comm):
    local_integral = trapezoidal_rule(local_a, local_b, local_n)
    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            if rank + step < size:
                temp_integral = comm.recv(source=rank + step, tag=77)
                local_integral += temp_integral
        elif rank % step == 0:
            comm.send(local_integral, dest=rank - step, tag=77)

        step *= 2
        comm.Barrier()

    if rank == 0:
        print("Resultado da integral com método butterfly:", local_integral)

# Programa principal
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    a = 0.0
    b = 1000000.0
    n = 10000000

    # Medir o tempo para o processamento pelo mestre
    if rank == 0:
        master_time = measure_execution_time(master_processing, a, b, n, size, comm)
        print(f"Tempo com processamento pelo mestre ({size} processos): {master_time} segundos")

    # Medir o tempo para o método butterfly
    local_n = n // size
    local_a = a + rank * local_n * (b - a) / n
    local_b = local_a + local_n * (b - a) / n

    butterfly_time = measure_execution_time(butterfly_method, local_a, local_b, local_n, rank, size, comm)
    if rank == 0:
        print(f"Tempo com método butterfly ({size} processos): {butterfly_time} segundos")
