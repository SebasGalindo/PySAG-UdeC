"""
Módulo de Selección para Algoritmos Genéticos.

Este módulo proporciona varios métodos de selección que pueden ser utilizados
en algoritmos genéticos. Todas las funciones están optimizadas con Numba
para mejorar el rendimiento y cuentan con validación de entradas y manejo
de excepciones robusto.

Funciones de Selección Disponibles:
    - selection_roulette_wheel:
        Selección proporcional al fitness (ruleta).
    - selection_tournament:
        Selección por torneo entre un subconjunto de individuos.
    - selection_rank:
        Selección basada en el ranking de fitness de los individuos.
    - selection_stochastic_universal_sampling (SUS):
        Variante de la ruleta que reduce el azar en la asignación de cupos.
    - selection_random:
        Selección aleatoria simple de individuos.
"""

from typing import Any, List

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .exceptions import (
    RangeError,
    SelectionError,
    TypeValidationError,
    validate_parameter,
)


def _validate_selection_inputs(
    population: List[NDArray[np.float64]],
    fitness_values: NDArray[np.float64],
    num_parents: int,
    population_size_min: int = 1,
) -> None:
    """
    Valida los parámetros comunes de las funciones de selección.

    Args:
        population: Lista de individuos (arrays de NumPy).
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.
        population_size_min: Tamaño mínimo permitido para la población.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        ValueError: Si la población está vacía, las longitudes no coinciden,
                    o num_parents es inválido.
        RangeError: Si num_parents está fuera del rango permitido.
    """
    if not isinstance(population, list):
        raise TypeValidationError("population", population, List[NDArray[np.float64]])
    if not all(isinstance(ind, np.ndarray) for ind in population):
        raise TypeValidationError(
            "population",
            "algunos elementos no son np.ndarray",
            List[NDArray[np.float64]],
        )
    validate_parameter(fitness_values, "fitness_values", NDArray[np.float64])
    validate_parameter(num_parents, "num_parents", int)

    if not population:
        raise ValueError("La población no puede estar vacía.")
    if len(population) < population_size_min:
        raise ValueError(
            f"La población debe tener al menos {population_size_min} individuos."
        )
    if len(population) != len(fitness_values):
        raise ValueError(
            "La longitud de la población y de los valores de fitness debe ser la misma."
        )
    if num_parents <= 0:
        raise RangeError("num_parents", num_parents, min_val=1)


@njit(cache=True)
def _roulette_wheel_selection_impl(
    fitness_values: NDArray[np.float64], num_parents: int
) -> NDArray[np.int_]:
    """
    Implementación Numba para la selección por ruleta.

    Selecciona 'num_parents' individuos de la población de forma aleatoria,
    con probabilidad proporcional a sus valores de fitness.
    Optimizada con decorador @njit de numba para mejorar el rendimiento.

    Args:
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.

    Returns:
        Lista de índices de los individuos seleccionados.

    Raises:
        ValueError: Si todos los valores de fitness son 0.
    """
    fitness_sum = np.sum(fitness_values)
    if fitness_sum == 0:
        # Si todos los fitness son 0, la probabilidad es uniforme
        probabilities = np.ones(len(fitness_values), dtype=np.float64) / len(
            fitness_values
        )
    else:
        probabilities = fitness_values / fitness_sum

    # np.random.choice no es soportado directamente en Numba
    # con probabilidades no uniformes
    # de la misma manera que en NumPy puro para devolver índices.
    # Implementamos una versión simple de la selección por ruleta:
    selected_indices = np.empty(num_parents, dtype=np.int_)
    cumulative_prob = np.cumsum(probabilities)
    for i in range(num_parents):
        random_val = np.random.random()
        for j in range(len(cumulative_prob)):
            if random_val <= cumulative_prob[j]:
                selected_indices[i] = j
                break
    return selected_indices


def selection_roulette_wheel(
    population: List[NDArray[np.float64]],
    fitness_values: NDArray[np.float64],
    num_parents: int,
    **kwargs: Any,
) -> List[NDArray[np.float64]]:
    """
    Selección por ruleta.

    Los individuos son seleccionados con una probabilidad proporcional a su fitness.
    Se asume que todos los valores de fitness son no negativos.

    Args:
        population: Lista de individuos (arrays de NumPy) que componen la población.
        fitness_values: Array de NumPy con los valores de fitness de cada individuo.
                        Deben ser no negativos.
        num_parents: Número de padres a seleccionar.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Lista de individuos seleccionados como padres.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        ValueError: Si la población está vacía, las longitudes no coinciden,
                    o si algún valor de fitness es negativo.
        RangeError: Si num_parents es inválido.
        SelectionError: Si ocurre un error inesperado durante la selección.

    Example:
        >>> import numpy as np
        >>> pop = [np.array([1,2]), np.array([3,4]), np.array([5,6]), np.array([7,8])]
        >>> fit = np.array([0.1, 0.4, 0.3, 0.2])
        >>> parents = selection_roulette_wheel(pop, fit, 2)
        >>> len(parents)
        2
        >>> isinstance(parents[0], np.ndarray)
        True
    """
    _validate_selection_inputs(population, fitness_values, num_parents)
    if np.any(fitness_values < 0):
        str_err = "Todos los valores de fitness deben ser"
        str_err += " no negativos para la selección por ruleta."
        raise ValueError(str_err)

    try:
        selected_indices = _roulette_wheel_selection_impl(fitness_values, num_parents)
        selected_parents = [population[i].copy() for i in selected_indices]
    except Exception as e:
        raise SelectionError(
            f"Error durante la selección por ruleta: {str(e)}",
            details={
                "population_size": len(population),
                "num_parents": num_parents,
                "original_error": str(e),
            },
        ) from e
    return selected_parents


@njit(cache=True)
def _tournament_selection_impl(
    fitness_values: NDArray[np.float64],
    num_parents: int,
    tournament_size: int,
    population_size: int,
) -> NDArray[np.int_]:
    """
    Implementación Numba para la selección por torneo.

    Selecciona 'num_parents' individuos de la población de forma aleatoria,
    con probabilidad proporcional a sus valores de fitness.
    optimizada con decorador @njit de numba para mejorar el rendimiento.

    Args:
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.
        tournament_size: Tamaño del torneo.
        population_size: Tamaño de la población.

    Returns:
        Lista de índices de los individuos seleccionados.

    Raises:
        ValueError: Si todos los valores de fitness son 0.
    """
    selected_indices = np.empty(num_parents, dtype=np.int_)
    population_indices = np.arange(population_size)

    for i in range(num_parents):
        if (
            tournament_size > population_size
        ):  # Numba no permite que el error se eleve desde aquí
            # Esta condición se manejará en la función principal
            pass

        contender_indices = np.random.choice(
            population_indices, size=tournament_size, replace=False
        )

        winner_in_tournament = -1
        best_fitness_in_tournament = -np.inf

        for contender_idx in contender_indices:
            if fitness_values[contender_idx] > best_fitness_in_tournament:
                best_fitness_in_tournament = fitness_values[contender_idx]
                winner_in_tournament = contender_idx
        selected_indices[i] = winner_in_tournament
    return selected_indices


def selection_tournament(
    population: List[NDArray[np.float64]],
    fitness_values: NDArray[np.float64],
    num_parents: int,
    tournament_size: int = 3,
    **kwargs: Any,
) -> List[NDArray[np.float64]]:
    """
    Selección por torneo.

    Se eligen 'tournament_size' individuos al azar, y el mejor de ellos (mayor fitness)
    se convierte en padre. Este proceso se repite 'num_parents' veces.

    Args:
        population: Lista de individuos (arrays de NumPy).
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.
        tournament_size: Número de individuos que participan en cada torneo.
                         Por defecto es 3.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Lista de individuos seleccionados como padres.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        ValueError: Si la población tiene menos individuos que `tournament_size`.
        RangeError: Si num_parents o tournament_size son inválidos.
        SelectionError: Si ocurre un error inesperado durante la selección.

    Example:
        >>> import numpy as np
        >>> pop = [np.array([1,2]), np.array([3,4]), np.array([5,6]), np.array([7,8])]
        >>> fit = np.array([10, 40, 30, 20])
        >>> parents = selection_tournament(pop, fit, 2, tournament_size=2)
        >>> len(parents)
        2
    """
    _validate_selection_inputs(
        population, fitness_values, num_parents, population_size_min=1
    )
    validate_parameter(tournament_size, "tournament_size", int, min_val=1)

    if tournament_size > len(population):
        raise ValueError(
            f"El tamaño del torneo ({tournament_size}) no puede ser mayor "
            f"que el tamaño de la población ({len(population)})."
        )
    if tournament_size == 0:
        raise RangeError(
            "tournament_size",
            tournament_size,
            min_val=1,
            details="Tournament size no puede ser cero.",
        )

    try:
        selected_indices = _tournament_selection_impl(
            fitness_values, num_parents, tournament_size, len(population)
        )
        selected_parents = [population[i].copy() for i in selected_indices]
    except Exception as e:
        raise SelectionError(
            f"Error durante la selección por torneo: {str(e)}",
            details={
                "population_size": len(population),
                "num_parents": num_parents,
                "tournament_size": tournament_size,
                "original_error": str(e),
            },
        ) from e
    return selected_parents


@njit(cache=True)
def _rank_selection_impl(
    fitness_values: NDArray[np.float64], num_parents: int
) -> NDArray[np.int_]:
    """
    Implementación Numba para la selección por ranking.

    Optimizada con decorador @njit de numba para mejorar el rendimiento.

    Args:
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.

    Returns:
        Lista de índices de los individuos seleccionados.
    """
    # argsort devuelve los índices que ordenarían el array
    sorted_indices_fitness = np.argsort(fitness_values)

    # Crear ranks (el mejor fitness obtiene el mayor rango)
    ranks = np.empty_like(sorted_indices_fitness, dtype=np.float64)
    ranks[sorted_indices_fitness] = np.arange(1, len(fitness_values) + 1)

    rank_sum = np.sum(ranks)
    if rank_sum == 0:  # Todos los rangos son 0 (imposible con arange(1, N+1))
        # o la población es 0 (manejado antes).
        # Esto es más una guarda por si los rangos fueran negativos o cero.
        probabilities = np.ones(len(fitness_values), dtype=np.float64) / len(
            fitness_values
        )
    else:
        probabilities = ranks / rank_sum

    # Implementación de np.random.choice con probabilidades para Numba
    selected_indices = np.empty(num_parents, dtype=np.int_)
    cumulative_prob = np.cumsum(probabilities)
    for i in range(num_parents):
        random_val = np.random.random()
        # Encuentra el primer índice j tal que random_val <= cumulative_prob[j]
        # Esto asegura que los individuos con mayor probabilidad (mayor rango)
        # tengan más chance de ser seleccionados.
        # Dado que las probabilidades se basan en 'ranks', que a su vez se derivan
        # de 'sorted_indices_fitness', la selección final de 'j' aquí
        # es un índice directo en la población original.
        selected_idx = 0
        for j in range(len(cumulative_prob)):
            if random_val <= cumulative_prob[j]:
                selected_idx = j
                break
        selected_indices[i] = selected_idx

    return selected_indices


def selection_rank(
    population: List[NDArray[np.float64]],
    fitness_values: NDArray[np.float64],
    num_parents: int,
    **kwargs: Any,
) -> List[NDArray[np.float64]]:
    """
    Selección por Rango.

    Los individuos son ordenados según su fitness y se les asigna un rango.
    La probabilidad de selección es proporcional a este rango
    (mayor rango, mayor probabilidad).
    Este método puede evitar la dominancia prematura de individuos con fitness muy alto.

    Args:
        population: Lista de individuos (arrays de NumPy).
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Lista de individuos seleccionados como padres.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        ValueError: Si la población está vacía o las longitudes no coinciden.
        RangeError: Si num_parents es inválido.
        SelectionError: Si ocurre un error inesperado durante la selección.

    Example:
        >>> import numpy as np
        >>> pop = [np.array([1,2]), np.array([3,4]), np.array([5,6]), np.array([7,8])]
        >>> fit = np.array([10, 40, 30, 20]) # Ranks: 10 (1), 20 (2), 30 (3), 40 (4)
        >>> parents = selection_rank(pop, fit, 2)
        >>> len(parents)
        2
    """
    _validate_selection_inputs(population, fitness_values, num_parents)

    try:
        selected_indices = _rank_selection_impl(fitness_values, num_parents)
        selected_parents = [population[i].copy() for i in selected_indices]
    except Exception as e:
        raise SelectionError(
            f"Error durante la selección por rango: {str(e)}",
            details={
                "population_size": len(population),
                "num_parents": num_parents,
                "original_error": str(e),
            },
        ) from e
    return selected_parents


@njit(cache=True)
def _stochastic_universal_sampling_impl(
    fitness_values: NDArray[np.float64], num_parents: int
) -> NDArray[np.int_]:
    """
    Implementación Numba para Stochastic Universal Sampling (SUS).

    Selecciona 'num_parents' individuos de la población de forma aleatoria,
    con probabilidad proporcional a sus valores de fitness.
    Optimizada con decorador @njit de numba para mejorar el rendimiento.

    Args:
        fitness_values: Array de NumPy con los valores de fitness.
        num_parents: Número de padres a seleccionar.

    Returns:
        Lista de índices de los individuos seleccionados.
    """
    fitness_sum = np.sum(fitness_values)
    if fitness_sum == 0:
        # Aquí, si todos son cero, la probabilidad es uniforme.
        return np.random.randint(0, len(fitness_values), size=num_parents)

    selected_indices = np.empty(num_parents, dtype=np.int_)

    # Distancia entre punteros
    pointer_distance = fitness_sum / num_parents

    # Posición inicial del primer puntero (aleatoria entre 0 y pointer_distance)
    start_pointer = np.random.uniform(0, pointer_distance)

    pointers = (
        np.arange(num_parents, dtype=np.float64) * pointer_distance + start_pointer
    )

    cumulative_fitness = 0.0
    current_member = 0
    for i_pointer in range(num_parents):
        # Moverse a través de los miembros hasta que el puntero sea superado
        while cumulative_fitness < pointers[i_pointer]:
            if current_member >= len(fitness_values):  # salvaguarda
                current_member = len(fitness_values) - 1  # seleccionar el ultimo
                break
            cumulative_fitness += fitness_values[current_member]
            current_member += 1  # Preparar para el siguiente miembro
        selected_indices[i_pointer] = (
            current_member - 1
        )  # El miembro que superó el puntero
        if selected_indices[i_pointer] < 0:  # Asegurarse de que no sea negativo
            selected_indices[i_pointer] = 0

    return selected_indices


def selection_stochastic_universal_sampling(
    population: List[NDArray[np.float64]],
    fitness_values: NDArray[np.float64],
    num_parents: int,
    **kwargs: Any,
) -> List[NDArray[np.float64]]:
    """
    Selección por Muestreo Universal Estocástico (SUS).

    Es una técnica de selección que minimiza el factor de suerte inherente en la
    selección por ruleta. Utiliza un único giro de una "ruleta" con N punteros
    equidistantes para seleccionar N individuos. Esto asegura que el número de
    veces que un individuo es seleccionado es más cercano a su proporción esperada.
    Se asume que todos los valores de fitness son no negativos.

    Args:
        population: Lista de individuos (arrays de NumPy).
        fitness_values: Array de NumPy con los valores de fitness.
        Deben ser no negativos.
        num_parents: Número de padres a seleccionar.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Lista de individuos seleccionados como padres.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        ValueError: Si la población está vacía, las longitudes no coinciden,
                    o si algún valor de fitness es negativo.
        RangeError: Si num_parents es inválido.
        SelectionError: Si ocurre un error inesperado durante la selección.

    Example:
        >>> import numpy as np
        >>> pop = [np.array([1,2]), np.array([3,4]), np.array([5,6]), np.array([7,8])]
        >>> fit = np.array([0.1, 0.4, 0.3, 0.2])
        >>> parents = selection_stochastic_universal_sampling(pop, fit, 2)
        >>> len(parents)
        2
    """
    _validate_selection_inputs(population, fitness_values, num_parents)
    if np.any(fitness_values < 0):
        raise ValueError(
            "Todos los valores de fitness deben ser no negativos para SUS."
        )
    if num_parents == 0:
        pass

    try:
        selected_indices = _stochastic_universal_sampling_impl(
            fitness_values, num_parents
        )
        selected_parents = [population[i].copy() for i in selected_indices]
    except Exception as e:
        raise SelectionError(
            f"Error durante la selección SUS: {str(e)}",
            details={
                "population_size": len(population),
                "num_parents": num_parents,
                "original_error": str(e),
            },
        ) from e
    return selected_parents


@njit(cache=True)
def _random_selection_impl(population_size: int, num_parents: int) -> NDArray[np.int_]:
    """
    Implementación Numba para la selección aleatoria simple.

    Optimizada con decorador @njit de numba para mejorar el rendimiento.

    Args:
        population_size: Tamaño de la población.
        num_parents: Número de padres a seleccionar.

    Returns:
        Lista de índices de los individuos seleccionados.
    """
    return np.random.randint(0, population_size, size=num_parents)


def selection_random(
    population: List[NDArray[np.float64]],
    fitness_values: NDArray[
        np.float64
    ],  # No se usa, pero se mantiene por consistencia de firma
    num_parents: int,
    **kwargs: Any,
) -> List[NDArray[np.float64]]:
    """
    Selección Aleatoria Simple.

    Selecciona 'num_parents' individuos de la población de forma completamente
    aleatoria, con reemplazo. No considera los valores de fitness.

    Args:
        population: Lista de individuos (arrays de NumPy).
        fitness_values: Array de NumPy con los valores de fitness (no utilizado).
        num_parents: Número de padres a seleccionar.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Lista de individuos seleccionados como padres.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        ValueError: Si la población está vacía o las longitudes no coinciden.
        RangeError: Si num_parents es inválido.
        SelectionError: Si ocurre un error inesperado durante la selección.

    Example:
        >>> import numpy as np
        >>> pop = [np.array([1,2]), np.array([3,4]), np.array([5,6]), np.array([7,8])]
        >>> fit = np.array([0.1, 0.4, 0.3, 0.2]) # Fitness no se usa
        >>> parents = selection_random(pop, fit, 3)
        >>> len(parents)
        3
    """
    _validate_selection_inputs(population, fitness_values, num_parents)

    try:
        selected_indices = _random_selection_impl(len(population), num_parents)
        selected_parents = [population[i].copy() for i in selected_indices]
    except Exception as e:
        raise SelectionError(
            f"Error durante la selección aleatoria: {str(e)}",
            details={
                "population_size": len(population),
                "num_parents": num_parents,
                "original_error": str(e),
            },
        ) from e
    return selected_parents
