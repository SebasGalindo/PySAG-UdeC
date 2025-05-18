import random

import numpy as np


def mutation_random_gene_uniform(
    individual, gene_low, gene_high, mutation_rate, **kwargs
):
    """
    Muta genes aleatorios del individuo, reemplazándolos con un valor uniforme.
    """
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = np.random.uniform(low=gene_low, high=gene_high)
    return mutated_individual


def mutation_bit_flip(individual, mutation_rate, **kwargs):
    """
    Mutación por inversión de bit (para representaciones binarias).
    """
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual


def mutation_swap(individual, mutation_rate, **kwargs):
    """
    Mutación por intercambio (Swap Mutation).
    """
    mutated_individual = individual.copy()
    if random.random() < mutation_rate and len(mutated_individual) >= 2:
        idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
        mutated_individual[idx1], mutated_individual[idx2] = (
            mutated_individual[idx2],
            mutated_individual[idx1],
        )
    return mutated_individual
