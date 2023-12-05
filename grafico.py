import matplotlib.pyplot as plt

# Supondo que você tenha uma lista com os números de processos e outra com os tempos correspondentes
num_processos = [1, 2, 4, 8, 16]
tempos_execucao = [tempo1, tempo2, tempo3, tempo4, tempo5]  # Substitua por seus tempos coletados

plt.plot(num_processos, tempos_execucao, marker='o')
plt.title('Tempo de Execução do Cálculo da Integral em Função do Número de Processos')
plt.xlabel('Número de Processos')
plt.ylabel('Tempo de Execução (s)')
plt.grid(True)
plt.show()