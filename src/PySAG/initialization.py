"""
Modulo de inicialización.

Este módulo contiene funciones para inicializar poblaciones en algoritmos genéticos.
Entrega poblaciones iniciales para algoritmos genéticos.
Las posibles funciones de inicialización son:
    - initial_population_uniform:
        Inicializa una población con genes distribuidos uniformemente entre low y high.
    - initial_population_binary:
        Inicializa una población con genes binarios (0 o 1).
"""

import numpy as np


def initial_population_uniform(population_size, num_genes, gene_low, gene_high):
    """
    Función para inicializar una población.

    Esta función crea una población inicial con genes distribuidos
    uniformemente entre low y high. Se asume que los genes son numéricos.

    Args:
        population_size: Tamaño de la población.
        num_genes: Número de genes por individuo.
        gene_low: Valor mínimo para los genes.
        gene_high: Valor máximo para los genes.

    Returns:
        Una lista de individuos, donde cada individuo es un array de genes.
    """
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(low=gene_low, high=gene_high, size=num_genes)
        population.append(individual)
    return population


def initial_population_binary(population_size, num_genes, **kwargs):
    """
    Función para inicializar una población con genes binarios.

    Esta función crea una población inicial con genes binarios (0 o 1).

    Args:
        population_size: Tamaño de la población.
        num_genes: Número de genes por individuo.

    Returns:
        Una lista de individuos, donde cada individuo es un array de genes binarios.
    """
    population = []
    for _ in range(population_size):
        individual = np.random.randint(0, 2, size=num_genes)
        population.append(individual)
    return population
