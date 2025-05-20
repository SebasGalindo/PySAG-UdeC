"""
Módulo que implementa operadores de cruce para algoritmos genéticos.

Este módulo proporciona varias estrategias de cruce que pueden ser utilizadas
en algoritmos genéticos. Las funciones están diseñadas para ser flexibles y
eficientes, utilizando Numba para la optimización del núcleo computacional.

Operadores de Cruce Disponibles:
    - crossover_single_point: Cruce de un solo punto.
    - crossover_two_points: Cruce de dos puntos.
    - crossover_uniform: Cruce uniforme, gen a gen.
    - crossover_arithmetic: Cruce aritmético para valores numéricos.
    - crossover_order_ox1: Cruce de orden (OX1) para permutaciones.
"""

from typing import Any, Tuple, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .exceptions import (
    CrossoverError,
    ParameterError,
    TypeValidationError,
    validate_parameter,
)

# Tipo genérico para los cromosomas, permitiendo flexibilidad (float, int, bool)
ChromosomeType = TypeVar("ChromosomeType", bound=np.generic)
ParentNDArray = NDArray[ChromosomeType]


def _validate_parents(
    parent1: ParentNDArray, parent2: ParentNDArray, min_length: int = 1
) -> None:
    """
    Valida los dos padres para operaciones de cruce.

    Args:
        parent1: Primer padre.
        parent2: Segundo padre.
        min_length: Longitud mínima requerida para los cromosomas.

    Raises:
        TypeValidationError: Si los padres no son arrays de NumPy.
        ParameterError: Si los padres no tienen la misma forma, no son 1D,
                        o su longitud es menor que `min_length`.

    """
    validate_parameter(parent1, "parent1", np.ndarray)  # type: ignore
    validate_parameter(parent2, "parent2", np.ndarray)  # type: ignore

    if parent1.shape != parent2.shape:
        raise ParameterError(
            f"Los padres deben tener la misma forma. "
            f"P1 tiene forma {parent1.shape}, P2 tiene forma {parent2.shape}."
        )
    if parent1.ndim != 1:
        raise ParameterError(
            f"Los padres deben ser arrays 1D. Dimensión encontrada: {parent1.ndim}."
        )
    if len(parent1) < min_length:
        raise ParameterError(
            f"La longitud de los padres debe ser al menos {min_length}. "
            f"Longitud encontrada: {len(parent1)}."
        )
    if parent1.dtype != parent2.dtype:
        # Advertencia o error opcional. NumPy manejará la promoción de tipos,
        # pero puede ser inesperado. Por ahora, permitimos diferentes dtypes numéricos.
        # Para permutaciones, se esperarán enteros.
        pass


@njit(cache=True)
def _crossover_single_point_impl(
    p1: ParentNDArray, p2: ParentNDArray
) -> Tuple[ParentNDArray, ParentNDArray]:
    """
    Implementación Numba para cruce de un punto.

    Args:
        p1: Primer padre.
        p2: Segundo padre.

    Returns:
        Una tupla (child1, child2) con los dos descendientes.

    """
    point = np.random.randint(1, len(p1))  # Punto de cruce entre 1 y len-1

    child1_parts = (p1[:point], p2[point:])
    child2_parts = (p2[:point], p1[point:])

    # np.concatenate es soportado por Numba y maneja dtypes.
    return np.concatenate(child1_parts), np.concatenate(child2_parts)


@njit(cache=True)
def _crossover_two_points_impl(
    p1: ParentNDArray, p2: ParentNDArray
) -> Tuple[ParentNDArray, ParentNDArray]:
    """
    Implementación Numba para cruce de dos puntos.

    Args:
        p1: Primer padre.
        p2: Segundo padre.

    Returns:
        Una tupla (child1, child2) con los dos descendientes.
    """
    # Generar dos puntos distintos, point1 < point2
    point1 = np.random.randint(1, len(p1) - 1)  # de 1 a len-2
    point2 = np.random.randint(point1 + 1, len(p1))  # de point1+1 a len-1

    child1_parts = (p1[:point1], p2[point1:point2], p1[point2:])
    child2_parts = (p2[:point1], p1[point1:point2], p2[point2:])

    return np.concatenate(child1_parts), np.concatenate(child2_parts)


@njit(cache=True)
def _crossover_uniform_impl(
    p1: ParentNDArray, p2: ParentNDArray, mix_prob: float
) -> Tuple[ParentNDArray, ParentNDArray]:
    """
    Implementación Numba para cruce uniforme.

    Args:
        p1: Primer padre.
        p2: Segundo padre.
        mix_prob: Probabilidad de intercambiar genes entre padres.

    Returns:
        Una tupla (child1, child2) con los dos descendientes.
    """
    child1 = p1.copy()
    child2 = p2.copy()

    for i in range(len(p1)):
        if np.random.random() < mix_prob:
            # Intercambiar genes en esta posición
            temp = child1[i]
            child1[i] = child2[i]
            child2[i] = temp

    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(len(p1)):
        if np.random.random() < mix_prob:
            gene1 = c1[i]
            c1[i] = c2[i]
            c2[i] = gene1
    return c1, c2


@njit(cache=True)
def _crossover_arithmetic_impl(
    p1: NDArray[np.float64], p2: NDArray[np.float64], alpha: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Implementación Numba para cruce aritmético.

    Args:
        p1: Primer padre.
        p2: Segundo padre.
        alpha: Factor de cruce.

    Returns:
        Una tupla (child1, child2) con los dos descendientes.
    """
    child1 = alpha * p1 + (1.0 - alpha) * p2
    child2 = (1.0 - alpha) * p1 + alpha * p2
    return child1, child2


@njit(cache=True)
def _crossover_order_ox1_impl(
    p1: NDArray[np.int_], p2: NDArray[np.int_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Implementación Numba para cruce de orden OX1.

    Args:
        p1: Primer padre.
        p2: Segundo padre.

    Returns:
        Una tupla (child1, child2) con los dos descendientes.

    """
    size = len(p1)
    child1 = np.empty_like(p1)
    child2 = np.empty_like(p2)

    # Paso 1: Seleccionar dos puntos de cruce
    start, end = np.sort(np.random.choice(np.arange(size + 1), size=2, replace=False))

    # Paso 2: Copiar el segmento del primer padre
    # al primer hijo, y del segundo al segundo
    child1[start:end] = p1[start:end]
    child2[start:end] = p2[start:end]

    # Paso 3: Llenar los genes restantes para child1
    # Elementos en p1[start:end]
    p1_segment_set = set(p1[start:end])
    current_p2_idx = end
    current_child1_idx = end
    filled_count = 0
    while filled_count < size - (end - start):
        if p2[current_p2_idx % size] not in p1_segment_set:
            child1[current_child1_idx % size] = p2[current_p2_idx % size]
            current_child1_idx += 1
            filled_count += 1
        current_p2_idx += 1
        if current_p2_idx > end + size:
            break

    # Paso 4: Llenar los genes restantes para child2
    p2_segment_set = set(p2[start:end])
    current_p1_idx = end
    current_child2_idx = end
    filled_count = 0
    while filled_count < size - (end - start):
        if p1[current_p1_idx % size] not in p2_segment_set:
            child2[current_child2_idx % size] = p1[current_p1_idx % size]
            current_child2_idx += 1
            filled_count += 1
        current_p1_idx += 1
        if current_p1_idx > end + size:
            break

    return child1, child2


def crossover_single_point(
    parent1: ParentNDArray, parent2: ParentNDArray, **kwargs: Any
) -> Tuple[ParentNDArray, ParentNDArray]:
    """
    Realiza un cruce de un solo punto entre dos padres.

    Selecciona un punto de cruce aleatorio y crea dos descendientes
    intercambiando las secciones de los padres después de este punto.
    Funciona con cualquier tipo de dato en los arrays NumPy.

    Args:
        parent1: Primer padre (array NumPy 1D).
        parent2: Segundo padre (array NumPy 1D, misma forma y tipo que parent1).
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una tupla (child1, child2) con los dos descendientes.

    Raises:
        TypeValidationError: Si los padres no son arrays de NumPy.
        ParameterError: Si los padres no tienen la misma forma, no son 1D,
                        o su longitud es menor que 2.
        CrossoverError: Si ocurre un error inesperado durante el cruce.

    Example:
        >>> import numpy as np
        >>> p1 = np.array([1, 2, 3, 4, 5])
        >>> p2 = np.array([6, 7, 8, 9, 10])
        >>> c1, c2 = crossover_single_point(p1, p2)
        >>> c1.shape == p1.shape and c2.shape == p2.shape
        True
        >>> (c1 == p1).all() # Puede ser False si ocurre cruce
        False

    """
    _validate_parents(parent1, parent2, min_length=1)
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    try:
        # Numba maneja bien dtypes numéricos y booleanos para slicing/concatenación.
        child1, child2 = _crossover_single_point_impl(parent1, parent2)
    except Exception as e:
        raise CrossoverError(
            f"Error durante el cruce de un solo punto: {str(e)}",
            details={
                "p1_dtype": str(parent1.dtype),
                "p2_dtype": str(parent2.dtype),
            },
        ) from e
    return child1, child2


def crossover_two_points(
    parent1: ParentNDArray, parent2: ParentNDArray, **kwargs: Any
) -> Tuple[ParentNDArray, ParentNDArray]:
    """
    Realiza un cruce de dos puntos entre dos padres.

    Selecciona dos puntos de cruce aleatorios y crea dos descendientes
    intercambiando la sección entre estos dos puntos.
    Funciona con cualquier tipo de dato en los arrays NumPy.

    Args:
        parent1: Primer padre (array NumPy 1D).
        parent2: Segundo padre (array NumPy 1D, misma forma y tipo que parent1).
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una tupla (child1, child2) con los dos descendientes.

    Raises:
        TypeValidationError: Si los padres no son arrays de NumPy.
        ParameterError: Si los padres no tienen la misma forma, no son 1D,
                        o su longitud es menor que 3.
        CrossoverError: Si ocurre un error inesperado durante el cruce.

    Example:
        >>> import numpy as np
        >>> p1 = np.array([1, 2, 3, 4, 5, 6])
        >>> p2 = np.array([10, 20, 30, 40, 50, 60])
        >>> c1, c2 = crossover_two_points(p1, p2)
        >>> c1.shape == p1.shape and c2.shape == p2.shape
        True
    """
    _validate_parents(parent1, parent2, min_length=2)
    if len(parent1) < 3:
        return parent1.copy(), parent2.copy()

    try:
        child1, child2 = _crossover_two_points_impl(parent1, parent2)
    except Exception as e:
        raise CrossoverError(
            f"Error durante el cruce de dos puntos: {str(e)}",
            details={
                "p1_dtype": str(parent1.dtype),
                "p2_dtype": str(parent2.dtype),
            },
        ) from e
    return child1, child2


def crossover_uniform(
    parent1: ParentNDArray,
    parent2: ParentNDArray,
    mix_probability: float = 0.5,
    **kwargs: Any,
) -> Tuple[ParentNDArray, ParentNDArray]:
    """
    Realiza un cruce uniforme entre dos padres.

    Para cada gen, se decide con `mix_probability` si los genes de los padres
    se intercambian para esa posición en los hijos. Si no se intercambian,
    el hijo1 toma el gen del padre1 y el hijo2 del padre2. Si se intercambian,
    el hijo1 toma del padre2 y el hijo2 del padre1.
    Funciona con cualquier tipo de dato en los arrays NumPy.

    Args:
        parent1: Primer padre (array NumPy 1D).
        parent2: Segundo padre (array NumPy 1D, misma forma y tipo que parent1).
        mix_probability: Probabilidad de intercambiar los genes de los padres
                         para una posición dada. Debe estar entre 0 y 1.
                         Por defecto es 0.5.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una tupla (child1, child2) con los dos descendientes.

    Raises:
        TypeValidationError: Si los padres no son arrays de NumPy.
        ParameterError: Si los padres no tienen la misma forma o no son 1D.
        RangeError: Si `mix_probability` está fuera del rango [0, 1].
        CrossoverError: Si ocurre un error inesperado durante el cruce.

    Example:
        >>> import numpy as np
        >>> p1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> p2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        >>> c1, c2 = crossover_uniform(p1, p2, mix_probability=0.5)
        >>> c1.shape == p1.shape and c2.shape == p2.shape
        True

    """
    _validate_parents(parent1, parent2, min_length=1)
    validate_parameter(
        mix_probability,
        "mix_probability",
        float,
        min_val=0.0,
        max_val=1.0,
    )  # type: ignore

    if len(parent1) == 0:
        return parent1.copy(), parent2.copy()

    try:
        child1, child2 = _crossover_uniform_impl(parent1, parent2, mix_probability)
    except Exception as e:
        raise CrossoverError(
            f"Error durante el cruce uniforme: {str(e)}",
            details={  # type: ignore
                "p1_dtype": str(parent1.dtype),
                "p2_dtype": str(parent2.dtype),
                "mix_probability": mix_probability,
            },
        ) from e
    return child1, child2


def crossover_arithmetic(
    parent1: NDArray[np.float64],
    parent2: NDArray[np.float64],
    alpha: float = 0.5,
    **kwargs: Any,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Realiza un cruce aritmético entre dos padres.

    Produce dos descendientes mediante una combinación lineal de los padres:
    `child1 = alpha * parent1 + (1 - alpha) * parent2`
    `child2 = (1 - alpha) * parent1 + alpha * parent2`
    Este operador es adecuado para cromosomas con genes de valor real o entero.
    Si se usan enteros, el resultado será flotante y puede requerir conversión.

    Args:
        parent1: Primer padre (array NumPy 1D de tipo numérico).
        parent2: Segundo padre (array NumPy 1D, misma forma y tipo que parent1).
        alpha: Factor de ponderación para la combinación lineal.
               Un valor común es 0.5 (promedio simple).
               Debe estar entre 0 y 1 para interpolación.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una tupla (child1, child2) con los dos descendientes (arrays de flotantes).

    Raises:
        TypeValidationError: Si los padres no son arrays de NumPy, o si sus `dtype`
                             no son numéricos (flotante o entero).
        ParameterError: Si los padres no tienen la misma forma o no son 1D.
        RangeError: Si `alpha` está fuera del rango [0, 1] (opcional, pero común).
        CrossoverError: Si ocurre un error inesperado durante el cruce.

    Example:
        >>> import numpy as np
        >>> p1 = np.array([1.0, 2.0, 3.0])
        >>> p2 = np.array([4.0, 5.0, 6.0])
        >>> c1, c2 = crossover_arithmetic(p1, p2, alpha=0.5)
        >>> c1
        array([2.5, 3.5, 4.5])
        >>> c2
        array([2.5, 3.5, 4.5])
        >>> p_int1 = np.array([1, 2, 3])
        >>> p_int2 = np.array([4, 6, 8])
        >>> c_int1, c_int2 = crossover_arithmetic(
            p_int1.astype(float), p_int2.astype(float), alpha=0.2
        )
        >>> c_int1 # Resultado es flotante
        array([3.4, 5.2, 7. ])

    """
    _validate_parents(parent1, parent2, min_length=0)

    # Validar que los dtypes son numéricos
    if not (
        np.issubdtype(parent1.dtype, np.number)
        and np.issubdtype(parent2.dtype, np.number)
    ):
        raise TypeValidationError(
            "parent1 y parent2 dtypes",
            f"P1: {parent1.dtype}, P2: {parent2.dtype}",
            "tipos numéricos de NumPy (integer o floating)",
        )
    p1_float = parent1.astype(np.float64, copy=False)
    p2_float = parent2.astype(np.float64, copy=False)

    validate_parameter(alpha, "alpha", float)

    if len(parent1) == 0:
        return p1_float.copy(), p2_float.copy()

    try:
        child1, child2 = _crossover_arithmetic_impl(p1_float, p2_float, alpha)
    except Exception as e:
        raise CrossoverError(
            f"Error durante el cruce aritmético: {str(e)}",
            details={  # type: ignore
                "p1_dtype": str(parent1.dtype),
                "p2_dtype": str(parent2.dtype),
                "alpha": alpha,
            },
        ) from e
    return child1, child2


def crossover_order_ox1(
    parent1: NDArray[np.int_], parent2: NDArray[np.int_], **kwargs: Any
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Realiza un cruce de orden (OX1) entre dos padres de permutación.

    Este operador es adecuado para cromosomas que representan permutaciones de enteros.
    1. Se seleccionan dos puntos de cruce aleatorios.
    2. El segmento del primer padre entre estos puntos se copia al primer hijo.
    3. Los genes restantes se toman del segundo padre en el orden en que aparecen,
       omitiendo los genes ya presentes del primer padre.
    4. El proceso se invierte para crear el segundo hijo.

    Args:
        parent1: Primer padre (array NumPy 1D de enteros, debe ser una permutación).
        parent2: Segundo padre (array NumPy 1D de enteros, misma forma y tipo,
                 debe ser una permutación de los mismos elementos).
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una tupla (child1, child2) con los dos descendientes (permutaciones).

    Raises:
        TypeValidationError:
            Si los padres no son arrays de NumPy o no son de tipo entero.
        ParameterError:
            Si los padres no tienen la misma forma, no son 1D,
            o su longitud es menor que 1 (o 3 para un cruce significativo).
            También si no parecen ser permutaciones válidas (opcional).
        CrossoverError:
            Si ocurre un error inesperado durante el cruce.

    Example:
        >>> import numpy as np
        >>> p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> p2 = np.array([8, 6, 4, 2, 7, 5, 3, 1])
        >>> # Los puntos de cruce son aleatorios, el resultado varía.
        >>> # Para un ejemplo determinista, necesitaríamos fijar los puntos de cruce.
        >>> c1, c2 = crossover_order_ox1(p1, p2)
        >>> c1.shape == p1.shape and c2.shape == p2.shape
        True
        >>> np.array_equal(np.sort(c1), np.sort(p1))
        True
        >>> np.array_equal(np.sort(c2), np.sort(p2))
        True

    """
    _validate_parents(parent1, parent2, min_length=1)

    if not (
        np.issubdtype(parent1.dtype, np.integer)
        and np.issubdtype(parent2.dtype, np.integer)
    ):
        raise TypeValidationError(
            "parent1 y parent2 dtypes para OX1",
            f"P1: {parent1.dtype}, P2: {parent2.dtype}",
            "tipos enteros de NumPy",
        )

    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    try:
        p1_int = parent1.astype(np.int_, copy=False)
        p2_int = parent2.astype(np.int_, copy=False)
        child1, child2 = _crossover_order_ox1_impl(p1_int, p2_int)
    except Exception as e:
        raise CrossoverError(
            f"Error durante el cruce de orden OX1: {str(e)}",
            details={
                "p1_dtype": str(parent1.dtype),
                "p2_dtype": str(parent2.dtype),
            },
        ) from e
    return child1, child2
