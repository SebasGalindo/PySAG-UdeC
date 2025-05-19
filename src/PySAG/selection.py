"""
Modulo de selección.

Este módulo proporciona varios métodos de selección que pueden ser utilizados
en algoritmos genéticos.
Todas las funciones están optimizadas con Numba para mejorar el rendimiento.

Posibles funciones de selección:
    - selection_roulette_wheel:
        Selección por ruleta. Los individuos son seleccionados con una probabilidad
        proporcional a su fitness. Asume que todos los fitness son positivos.
    - selection_tournament:
        Selección por torneo. Se eligen 'tournament_size' individuos al azar,
        y el mejor de ellos se convierte en padre. Se repite 'num_parents' veces.
    - selection_rank:
        Selección por Rango. Los individuos son ordenados por fitness y se les asigna
        un rango. La probabilidad de selección es proporcional a su rango.
"""

import random

import numpy as np


def selection_roulette_wheel(population, fitness_values, num_parents, **kwargs):
    """
    Selección por ruleta.

    Los individuos son seleccionados con una probabilidad proporcional a su fitness.
    Se asume que todos los fitness son positivos.

    Args:
        population: Población de individuos.
        fitness_values: Valores de fitness de cada individuo.
        num_parents: Número de padres a seleccionar.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una lista de padres seleccionados.

    Raises:
        ValueError: Si la población está vacía.
    """
    fitness_sum = np.sum(fitness_values)
    if fitness_sum == 0:
        probabilities = np.ones(len(population)) / len(population)
    else:
        probabilities = fitness_values / fitness_sum

    selected_parents_indices = np.random.choice(
        len(population), size=num_parents, p=probabilities, replace=True
    )
    selected_parents = [population[i] for i in selected_parents_indices]
    return selected_parents


def selection_tournament(
    population, fitness_values, num_parents, tournament_size=3, **kwargs
):
    """
    Selección por torneo.

    Se eligen 'tournament_size' individuos al azar,
    y el mejor de ellos se convierte en padre. Se repite 'num_parents' veces.

    Args:
        population: Población de individuos.
        fitness_values: Valores de fitness de cada individuo.
        num_parents: Número de padres a seleccionar.
        tournament_size: Tamaño del torneo (número de individuos a evaluar).
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una lista de padres seleccionados.

    Raises:
        ValueError: Si la población está vacía.
    """
    selected_parents = []
    population_indices = list(range(len(population)))
    if not population_indices:  # Si la población está vacía
        return []
    if tournament_size > len(
        population
    ):  # Ajustar si el tamaño del torneo es mayor que la población
        tournament_size = len(population)

    for _ in range(num_parents):
        tournament_contender_indices = random.sample(
            population_indices, tournament_size
        )
        tournament_fitnesses = [fitness_values[i] for i in tournament_contender_indices]

        winner_index_in_tournament = np.argmax(tournament_fitnesses)
        winner_overall_index = tournament_contender_indices[winner_index_in_tournament]
        selected_parents.append(population[winner_overall_index])
    return selected_parents


def selection_rank(population, fitness_values, num_parents, **kwargs):
    """
    Selección por Rango.

    Los individuos son ordenados por fitness y se les asigna un rango.
    La probabilidad de selección es proporcional a su rango.

    Args:
        population: Población de individuos.
        fitness_values: Valores de fitness de cada individuo.
        num_parents: Número de padres a seleccionar.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una lista de padres seleccionados.

    Raises:
        ValueError: Si la población está vacía.
    """
    sorted_indices = np.argsort(fitness_values)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(population) + 1)

    rank_sum = np.sum(ranks)
    if rank_sum == 0:
        probabilities = np.ones(len(population)) / len(population)
    else:
        probabilities = ranks / rank_sum

    selected_parents_indices = np.random.choice(
        len(population), size=num_parents, p=probabilities, replace=True
    )
    selected_parents = [population[i] for i in selected_parents_indices]
    return selected_parents
