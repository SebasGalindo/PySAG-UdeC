"""Ejemplo de Algoritmo Genético para resolver el problema de la suma."""

import numpy as np

# Importar los módulos de operadores
from PySAG import GA, crossover, initialization, mutation, selection

# 1. Definir la función de Fitness
# Queremos maximizar la suma de los genes de un individuo.
# Cada gen estará entre 0 y 10. Un individuo tendrá 5 genes.
# La solución óptima es [10, 10, 10, 10, 10] con fitness 50.

GENE_LOW = 0
GENE_HIGH = 10
NUM_GENES = 5


def fitness_function(individual):
    """Función de fitness que maximiza la suma de los genes."""
    return np.sum(individual)


# 2. Instanciar la clase GA
ga_instance = GA(
    fitness_func=fitness_function,
    num_genes=NUM_GENES,
    population_size=50,
    num_generations=100,
    num_parents_mating=10,
    initial_population_func=initialization.initial_population_uniform,
    initial_pop_args={"gene_low": GENE_LOW, "gene_high": GENE_HIGH},
    selection_func=selection.selection_roulette_wheel,
    crossover_func=crossover.crossover_single_point,
    crossover_probability=0.95,
    mutation_func=mutation.mutation_random_gene_uniform,
    mutation_args={"gene_low": GENE_LOW, "gene_high": GENE_HIGH, "mutation_rate": 0.05},
    keep_elitism_percentage=0.1,
)

# 3. Ejecutar el AG
best_solution, best_fitness = ga_instance.run()

# (Opcional) Graficar el fitness
ga_instance.plot_fitness()
