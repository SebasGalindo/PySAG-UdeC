import random  # Aunque no se usa aquí directamente, puede ser útil para futuras funciones

import numpy as np


def initial_population_uniform(population_size, num_genes, gene_low, gene_high):
    """
    Crea una población inicial con genes distribuidos uniformemente entre low y high.
    Asume genes numéricos.
    """
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(low=gene_low, high=gene_high, size=num_genes)
        population.append(individual)
    return population


def initial_population_binary(population_size, num_genes, **kwargs):
    """
    Crea una población inicial con genes binarios (0 o 1).
    """
    population = []
    for _ in range(population_size):
        individual = np.random.randint(0, 2, size=num_genes)
        population.append(individual)
    return population
