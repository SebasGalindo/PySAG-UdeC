# examples/simple_sum_problem.py
import numpy as np
# Asumimos que estamos ejecutando este script desde la carpeta raíz 'PySAG-UdeC'
# y que 'src' está en el path de Python, o que hemos instalado PySAG.
# Por ahora, para que funcione sin instalar, ajustaremos el path:
import sys
import os
# Añadir la carpeta 'src' al path para encontrar 'PySAG'
# Esto es una solución temporal para ejecutar ejemplos antes de instalar la librería
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Sube un nivel a PySAG-UdeC
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from PySAG import GA # Ahora esto debería funcionar

# 1. Definir la función de Fitness
# Queremos maximizar la suma de los genes de un individuo.
# Cada gen estará entre 0 y 10. Un individuo tendrá 5 genes.
# La solución óptima es [10, 10, 10, 10, 10] con fitness 50.

GENE_LOW = 0
GENE_HIGH = 10
NUM_GENES = 5

def fitness_function(individual):
    return np.sum(individual)

# 2. Instanciar la clase GA
ga_instance = GA(
    fitness_func=fitness_function,
    num_genes=NUM_GENES,
    population_size=50,
    num_generations=100,
    num_parents_mating=10,
    initial_pop_args={'gene_low': GENE_LOW, 'gene_high': GENE_HIGH}, # Argumentos para initial_population_uniform
    mutation_args={'gene_low': GENE_LOW, 'gene_high': GENE_HIGH, 'mutation_rate': 0.05}, # Argumentos para mutation_random_gene_uniform
    keep_elitism_percentage=0.1
)

# 3. Ejecutar el AG
best_solution, best_fitness = ga_instance.run()

# (Opcional) Graficar el fitness
ga_instance.plot_fitness()