"""Módulo que implementa operadores de cruce para algoritmos genéticos.

Este módulo proporciona varias estrategias de cruce que pueden ser utilizadas
en algoritmos genéticos. Todas las funciones están optimizadas con Numba
para mejorar el rendimiento.
"""

from typing import Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def crossover_single_point(
    parent1: NDArray[np.float64], parent2: NDArray[np.float64], **kwargs
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Realiza un cruce de un punto entre dos padres.

    Este operador selecciona un punto de cruce aleatorio y crea dos descendientes
    intercambiando las secciones de los padres después de este punto.

    Args:
        parent1: Primer padre como array de numpy.
        parent2: Segundo padre como array de numpy.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una tupla con dos arrays de numpy que representan a los dos descendientes.

    Raises:
        ValueError: Si los padres tienen longitudes diferentes.

    Example:
        >>> import numpy as np
        >>> parent1 = np.array([1.0, 2.0, 3.0, 4.0])
        >>> parent2 = np.array([5.0, 6.0, 7.0, 8.0])
        >>> child1, child2 = crossover_single_point(parent1, parent2)
        >>> child1.shape == parent1.shape
        True
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


@njit(cache=True)
def crossover_two_points(
    parent1: NDArray[np.float64], parent2: NDArray[np.float64], **kwargs
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Realiza un cruce de dos puntos entre dos padres.

    Este operador selecciona dos puntos de cruce aleatorios y crea dos descendientes
    intercambiando la sección entre estos dos puntos.

    Args:
        parent1: Primer padre como array de numpy.
        parent2: Segundo padre como array de numpy.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una tupla con dos arrays de numpy que representan a los dos descendientes.

    Raises:
        ValueError: Si los padres tienen longitudes diferentes o menos de 3 elementos.

    Example:
        >>> import numpy as np
        >>> parent1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> parent2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        >>> child1, child2 = crossover_two_points(parent1, parent2)
        >>> child1.shape == parent1.shape
        True
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")
    if len(parent1) < 3:
        return parent1.copy(), parent2.copy()

    point1 = np.random.randint(1, len(parent1) - 1)
    point2 = np.random.randint(point1 + 1, len(parent1))

    child1 = np.concatenate(
        (parent1[:point1], parent2[point1:point2], parent1[point2:])
    )
    child2 = np.concatenate(
        (parent2[:point1], parent1[point1:point2], parent2[point2:])
    )
    return child1, child2


@njit(cache=True)
def crossover_uniform(
    parent1: NDArray[np.float64],
    parent2: NDArray[np.float64],
    mix_probability: float = 0.5,
    **kwargs,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Realiza un cruce uniforme entre dos padres.

    Para cada gen, se elige al azar de cuál padre tomarlo, con una
    probabilidad dada.

    Args:
        parent1: Primer padre como array de numpy.
        parent2: Segundo padre como array de numpy.
        mix_probability: Probabilidad de tomar el gen del primer padre.
                         Debe estar entre 0 y 1. Por defecto es 0.5.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una tupla con dos arrays de numpy que representan a los dos descendientes.

    Raises:
        ValueError: Si los padres tienen longitudes diferentes.

    Example:
        >>> import numpy as np
        >>> parent1 = np.array([1.0, 2.0, 3.0, 4.0])
        >>> parent2 = np.array([5.0, 6.0, 7.0, 8.0])
        >>> child1, child2 = crossover_uniform(parent1, parent2, mix_probability=0.5)
        >>> child1.shape == parent1.shape
        True
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")

    child1 = parent1.copy()
    child2 = parent2.copy()
    mask = np.random.random(len(parent1)) < mix_probability

    # Usar operaciones vectorizadas para mejor rendimiento
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]

    return child1, child2
