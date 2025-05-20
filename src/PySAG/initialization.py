"""
Módulo de Inicialización de Poblaciones para Algoritmos Genéticos.

Este módulo proporciona funciones para crear la población inicial de individuos
(cromosomas) para un algoritmo genético. Las funciones están diseñadas para ser
flexibles y eficientes, utilizando Numba para la optimización cuando es aplicable.

Funciones de Inicialización Disponibles:
    - init_random_uniform:
        Genera una población con genes de valor real o entero muestreados de una
        distribución uniforme, especificado mediante el parámetro `dtype`.
    - init_random_binary:
        Genera una población con genes binarios (0 o 1).
    - init_random_permutation:
        Genera una población donde cada individuo es una permutación de enteros
        (útil para problemas de ordenamiento).
"""

from typing import Any, List, Type, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .exceptions import (
    InitializationError,
    RangeError,
    TypeValidationError,
    validate_parameter,
)


@njit(cache=True)
def _generate_float_uniform_population_impl(
    pop_size: int, chromosome_length: int, low: float, high: float
) -> List[NDArray[np.float64]]:
    """
    Generar población uniforme de flotantes.

    Implementación Numba para generar población uniforme de flotantes.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Número de genes en cada cromosoma (individuo).
        low: Límite inferior del rango para los valores de los genes.
        high: Límite superior del rango para los valores de los genes.
              (Exclusivo para flotantes, inclusivo para enteros).

    Returns:
        Una lista de individuos (arrays de NumPy) del `dtype` especificado.
    """
    population = []
    for _ in range(pop_size):
        # np.random.uniform genera valores en [low, high)
        individual = np.random.uniform(low, high, size=chromosome_length)
        population.append(individual)  # Por defecto es float64
    return population


@njit(cache=True)
def _generate_integer_uniform_population_impl(
    pop_size: int, chromosome_length: int, low: int, high: int
) -> List[NDArray[np.int_]]:  # Numba puede usar np.int_ o np.int64
    """
    Generar población uniforme de enteros.

    Implementación Numba para generar población uniforme de enteros.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Número de genes en cada cromosoma (individuo).
        low: Límite inferior del rango para los valores de los genes.
        high: Límite superior del rango para los valores de los genes.
              (Exclusivo para flotantes, inclusivo para enteros).

    Returns:
        Una lista de individuos (arrays de NumPy) del `dtype` especificado.
    """
    population = []
    # np.random.randint(low, high_exclusive, size=...)
    # Para rango [low, high] inclusivo, el segundo parámetro debe ser high + 1
    for _ in range(pop_size):
        individual = np.random.randint(low, high + 1, size=chromosome_length)
        population.append(
            individual.astype(np.int_)
        )  # Asegurar tipo int_ (plataforma-dependiente int)
    return population


@njit(cache=True)
def _generate_binary_population_impl(
    pop_size: int, chromosome_length: int, p_one: float
) -> List[NDArray[np.int_]]:
    """
    Generar población binaria.

    Implementación Numba para generar población binaria.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Número de genes en cada cromosoma (individuo).
        p_one: Probabilidad de que un gen sea 1.

    Returns:
        Una lista de individuos (arrays de NumPy) del `dtype` especificado.
    """
    population = []
    for _ in range(pop_size):
        individual = np.random.random(size=chromosome_length) < p_one
        population.append(
            individual.astype(np.int_)
        )  # Convertir booleano a int (0 o 1)
    return population


@njit(cache=True)
def _generate_permutation_population_impl(
    pop_size: int, chromosome_length: int
) -> List[NDArray[np.int_]]:
    """
    Generar población de permutaciones de enteros.

    Implementación Numba para generar población de permutaciones de enteros.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Número de genes en cada cromosoma (individuo).

    Returns:
        Una lista de individuos (arrays de NumPy) del `dtype` especificado.
    """
    population = []
    base_array = np.arange(
        chromosome_length, dtype=np.int_
    )  # Especificar dtype para claridad
    for _ in range(pop_size):
        individual = np.random.permutation(base_array)
        population.append(individual)  # Ya es del tipo correcto (np.int_)
    return population


def init_random_uniform(
    pop_size: int,
    chromosome_length: int,
    low: Union[float, int],
    high: Union[float, int],
    dtype: Type[
        Union[
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.int_,
        ]
    ] = np.float64,  # type: ignore
    **kwargs: Any,
) -> List[NDArray[Any]]:
    """
    Función para crear una población con genes con distribución uniforme.

    Genera una población con genes de valor real o entero muestreados de una
    distribución uniforme. El tipo de dato (flotante o entero) y su precisión
    se determinan por el parámetro `dtype`.

    - Si `dtype` es flotante (e.g., `np.float64`, `np.float32`):
        Los genes se muestrean de U(low, high), donde `high` es exclusivo.
    - Si `dtype` es entero (e.g., `np.int_`, `np.int32`, `np.int64`):
        Los genes se muestrean uniformemente del rango `[low, high]`,
        donde `high` es inclusivo.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Número de genes en cada cromosoma (individuo).
        low: Límite inferior del rango para los valores de los genes.
        high: Límite superior del rango para los valores de los genes.
              (Exclusivo para flotantes, inclusivo para enteros).
        dtype: Tipo de dato NumPy deseado para los genes.
               Debe ser un tipo flotante (e.g., `np.float64`, `np.float32`)
               o entero (e.g., `np.int_`, `np.int32`, `np.int64`).
               Por defecto es `np.float64`.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una lista de individuos (arrays de NumPy) del `dtype` especificado.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos,
                             o `dtype` no es un tipo NumPy numérico soportado.
        RangeError: Si `pop_size` o `chromosome_length` no son positivos,
                    o si los rangos `low`, `high` son inválidos para el `dtype`.
        InitializationError: Si ocurre un error inesperado durante la inicialización.

    Example:
        >>> # Población de flotantes (por defecto dtype=np.float64)
        >>> pop_float = init_random_uniform(5, 3, 0.0, 1.0)
        >>> len(pop_float)
        5
        >>> pop_float[0].shape
        (3,)
        >>> pop_float[0].dtype
        dtype('float64')

        >>> # Población de enteros de 32 bits
        >>> pop_int32 = init_random_uniform(5, 4, 0, 10, dtype=np.int32)
        >>> len(pop_int32)
        5
        >>> pop_int32[0].shape
        (4,)
        >>> pop_int32[0].dtype
        dtype('int32')
        >>> np.all(pop_int32[0] >= 0) and np.all(pop_int32[0] <= 10)
        True
    """
    validate_parameter(pop_size, "pop_size", int, min_val=1)
    validate_parameter(chromosome_length, "chromosome_length", int, min_val=1)
    validate_parameter(low, "low", (float, int))  # type: ignore
    validate_parameter(high, "high", (float, int))  # type: ignore

    # Validación de dtype
    valid_float_dtypes = [np.float32, np.float64]
    valid_int_dtypes = [np.int32, np.int64, np.int_]

    is_float_type = dtype in valid_float_dtypes
    is_int_type = dtype in valid_int_dtypes

    if not (is_float_type or is_int_type):
        allowed_dtypes_str = ", ".join(
            [t.__name__ for t in valid_float_dtypes + valid_int_dtypes]
        )
        raise TypeValidationError(
            "dtype",
            str(dtype),
            f"uno de los tipos NumPy soportados: {allowed_dtypes_str}",
        )

    # Validación de rango low/high según dtype
    if is_float_type and float(low) >= float(high):
        raise RangeError(
            "low",
            low,
            max_val=float(high),
            inclusive=False,  # type: ignore
            details="""Para tipos flotantes,
            'low' debe ser estrictamente menor que 'high'.""",
        )
    if is_int_type and int(low) > int(high):
        raise RangeError(
            "low",
            low,
            max_val=int(high),
            inclusive=True,  # type: ignore
            details="""Para tipos enteros,
            'low' no debe ser mayor que 'high'.""",
        )

    try:
        if is_float_type:
            # _generate_float_uniform_population_impl devuelve float64
            population_raw = _generate_float_uniform_population_impl(
                pop_size, chromosome_length, float(low), float(high)
            )
            if dtype == np.float32:  # Solo castear si se pide float32 específicamente
                population = [ind.astype(np.float32) for ind in population_raw]
            else:  # Es np.float64
                population = population_raw
        else:  # is_int_type
            # _generate_integer_uniform_population_impl devuelve np.int_
            population_raw = _generate_integer_uniform_population_impl(
                pop_size, chromosome_length, int(low), int(high)
            )
            # Castear al dtype entero específico
            # si es necesario y diferente de np.int_ default
            if population_raw[0].dtype != np.dtype(dtype):
                population = [
                    ind.astype(dtype) for ind in population_raw
                ]  # type: ignore
            else:
                population = population_raw
    except Exception as e:
        raise InitializationError(
            f"Error durante la inicialización uniforme: {str(e)}",
            details={  # type: ignore
                "pop_size": pop_size,
                "chromosome_length": chromosome_length,
                "low": low,
                "high": high,
                "dtype": str(dtype),
                "original_error": str(e),
            },
        ) from e
    return population  # type: ignore


def init_random_binary(
    pop_size: int,
    chromosome_length: int,
    p_one: float = 0.5,
    **kwargs: Any,
) -> List[NDArray[np.int_]]:
    """
    Genera una población con genes binarios (0 o 1).

    Cada gen se establece en 1 con probabilidad `p_one`
    y en 0 con probabilidad `1 - p_one`.
    Los genes son de tipo `np.int_`.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Número de genes en cada cromosoma.
        p_one: Probabilidad de que un gen individual sea 1. Por defecto es 0.5.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una lista de individuos (arrays de NumPy de tipo `np.int_`), donde cada
        individuo representa un cromosoma binario.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        RangeError: Si `pop_size` o `chromosome_length` no son positivos,
                    o si `p_one` no está en el rango [0, 1].
        InitializationError: Si ocurre un error inesperado durante la inicialización.

    Example:
        >>> pop_bin = init_random_binary(10, 8)
        >>> len(pop_bin)
        10
        >>> pop_bin[0].shape
        (8,)
        >>> pop_bin[0].dtype # np.int_ es int32 o int64 según la plataforma
        dtype('int...')
        >>> np.all((pop_bin[0] == 0) | (pop_bin[0] == 1))
        True
    """
    validate_parameter(pop_size, "pop_size", int, min_val=1)
    validate_parameter(chromosome_length, "chromosome_length", int, min_val=1)
    validate_parameter(p_one, "p_one", float, min_val=0.0, max_val=1.0)  # type: ignore

    try:
        # _generate_binary_population_impl devuelve una lista de NDArray[np.int_]
        population = _generate_binary_population_impl(
            pop_size, chromosome_length, p_one
        )
    except Exception as e:
        raise InitializationError(
            f"Error durante la inicialización binaria: {str(e)}",
            details={  # type: ignore
                "pop_size": pop_size,
                "chromosome_length": chromosome_length,
                "p_one": p_one,
                "original_error": str(e),
            },
        ) from e
    return population


def init_random_permutation(
    pop_size: int,
    chromosome_length: int,
    **kwargs: Any,
) -> List[NDArray[np.int_]]:
    """
    Genera una población donde cada individuo es una permutación de enteros.

    Cada cromosoma será una permutación de los enteros
    `0, 1, ..., chromosome_length - 1`.
    Este método está diseñado para problemas de optimización combinatoria (e.g., TSP)
    donde los individuos representan un orden o secuencia de elementos discretos.
    Los genes son de tipo `np.int_`.

    Args:
        pop_size: Número de individuos en la población.
        chromosome_length: Longitud de la permutación (número de elementos a permutar).
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una lista de individuos (arrays de NumPy de tipo `np.int_`), donde cada
        individuo es una permutación de enteros.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        RangeError: Si `pop_size` o `chromosome_length` no son positivos.
        InitializationError: Si ocurre un error inesperado durante la inicialización.

    Example:
        >>> pop_perm = init_random_permutation(5, 4)
        >>> len(pop_perm)
        5
        >>> pop_perm[0].shape
        (4,)
        >>> pop_perm[0].dtype # np.int_ es int32 o int64 según la plataforma
        dtype('int...')
        >>> import numpy as np
        >>> np.array_equal(np.sort(pop_perm[0]), np.arange(4))
        True
    """
    validate_parameter(pop_size, "pop_size", int, min_val=1)
    validate_parameter(chromosome_length, "chromosome_length", int, min_val=1)

    try:
        population = _generate_permutation_population_impl(pop_size, chromosome_length)
    except Exception as e:
        raise InitializationError(
            f"Error durante la inicialización de permutaciones: {str(e)}",
            details={  # type: ignore
                "pop_size": pop_size,
                "chromosome_length": chromosome_length,
                "original_error": str(e),
            },
        ) from e
    return population
