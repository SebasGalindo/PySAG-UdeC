import random

import numpy as np


def selection_roulette_wheel(population, fitness_values, num_parents, **kwargs):
    """
    Selección por ruleta. Los individuos son seleccionados con una probabilidad
    proporcional a su fitness. Asume que todos los fitness son positivos.
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
    Selección por torneo. Se eligen 'tournament_size' individuos al azar,
    y el mejor de ellos se convierte en padre. Se repite 'num_parents' veces.
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
    Selección por Rango. Los individuos son ordenados por fitness y se les asigna
    un rango. La probabilidad de selección es proporcional a su rango.
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
