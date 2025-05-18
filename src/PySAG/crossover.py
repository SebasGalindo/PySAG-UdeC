import random

import numpy as np


def crossover_single_point(parent1, parent2, **kwargs):
    """
    Cruce de un solo punto.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1) - 1)
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))
    return offspring1, offspring2


def crossover_two_points(parent1, parent2, **kwargs):
    """
    Cruce de dos puntos.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")
    if len(parent1) < 3:
        return parent1.copy(), parent2.copy()

    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)

    offspring1 = np.concatenate(
        (parent1[:point1], parent2[point1:point2], parent1[point2:])
    )
    offspring2 = np.concatenate(
        (parent2[:point1], parent1[point1:point2], parent2[point2:])
    )
    return offspring1, offspring2


def crossover_uniform(parent1, parent2, mix_probability=0.5, **kwargs):
    """
    Cruce uniforme. Para cada gen, se elige al azar de cuál padre tomarlo.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")

    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    for i in range(len(parent1)):
        if random.random() < mix_probability:
            offspring2[i] = parent1[i]
        else:
            offspring1[i] = parent2[i]
    return offspring1, offspring2
