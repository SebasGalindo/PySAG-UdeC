"""Módulo que implementa operadores de mutación para algoritmos genéticos.

Este módulo proporciona varias estrategias de mutación que pueden ser utilizadas
en algoritmos genéticos. Todas las funciones están optimizadas con Numba
para mejorar el rendimiento.
"""

from typing import Any, Callable, Dict

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .exceptions import (
    MutationError,
    TypeValidationError,
    validate_parameter,
)


def _validate_mutation_inputs(
    individual: NDArray, mutation_rate: float, param_name: str = "individual"
) -> None:
    """Valida los parámetros comunes de las funciones de mutación.

    Args:
        individual: Individuo a mutar.
        mutation_rate: Tasa de mutación (debe estar entre 0 y 1).
        param_name: Nombre del parámetro para mensajes de error.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
    """
    # Validar que el individuo sea un array de NumPy
    if not isinstance(individual, np.ndarray):
        raise TypeValidationError(
            param_name=param_name, param_value=individual, expected_type=np.ndarray
        )

    # Validar que el array no esté vacío
    if individual.size == 0:
        raise ValueError(f"El parámetro '{param_name}' no puede ser un array vacío")

    # Validar mutation_rate
    validate_parameter(
        mutation_rate,
        name="mutation_rate",
        expected_type=(int, float),
        min_val=0.0,
        max_val=1.0,
    )


def _apply_mutation(
    individual: NDArray,
    mutation_func: Callable[[NDArray], NDArray],
    mutation_rate: float,
) -> NDArray:
    """Aplica una función de mutación a un individuo con una cierta probabilidad.

    Args:
        individual: Individuo a mutar.
        mutation_func: Función de mutación a aplicar.
        mutation_rate: Probabilidad de que ocurra la mutación.

    Returns:
        Una copia del individuo con la mutación aplicada.
    """
    try:
        if np.random.random() < mutation_rate:
            return mutation_func(individual)
        return individual.copy()
    except Exception as e:
        raise MutationError(
            f"Error al aplicar la mutación: {str(e)}",
            details={
                "mutation_func": mutation_func.__name__,
                "individual_shape": getattr(individual, "shape", "N/A"),
                "individual_dtype": getattr(individual, "dtype", "N/A"),
                "mutation_rate": mutation_rate,
                "original_error": str(e),
            },
        ) from e


@njit(cache=True)
def _mutation_random_gene_uniform_impl(
    individual: NDArray[np.float64], gene_low: float, gene_high: float
) -> NDArray[np.float64]:
    """Implementación núcleo de la mutación uniforme."""
    mutated = individual.copy()
    for i in prange(len(mutated)):
        if np.random.random() < 0.5:  # 50% de probabilidad por gen
            mutated[i] = np.random.uniform(gene_low, gene_high)
    return mutated


def mutation_random_gene_uniform(
    individual: NDArray[np.float64],
    gene_low: float,
    gene_high: float,
    mutation_rate: float = 0.01,
    **kwargs: Dict[str, Any],
) -> NDArray[np.float64]:
    """Muta genes aleatorios del individuo, reemplazándolos con un valor uniforme.

    Args:
        individual: Individuo a mutar.
        gene_low: Límite inferior para los valores de los genes.
        gene_high: Límite superior para los valores de los genes.
        mutation_rate: Probabilidad de que un gen mute. Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una copia del individuo con los genes mutados.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        RangeError: Si los valores están fuera de los rangos permitidos.
        MutationError: Si ocurre un error durante la mutación.

    Example:
        >>> import numpy as np
        >>> individual = np.array([1.0, 2.0, 3.0, 4.0])
        >>> mutated = mutation_random_gene_uniform(individual, 0.0, 10.0, 0.5)
        >>> mutated.shape == individual.shape
        True
    """
    # Validar parámetros de entrada
    _validate_mutation_inputs(individual, mutation_rate)

    # Validar que gene_low <= gene_high
    if gene_low > gene_high:
        raise ValueError(
            f"gene_low ({gene_low}) no puede ser mayor que gene_high ({gene_high})"
        )

    # Aplicar mutación
    try:
        if mutation_rate <= 0:
            return individual.copy()

        if mutation_rate >= 1.0:
            # Si la tasa de mutación es 1.0, mutamos todos los genes
            return _mutation_random_gene_uniform_impl(individual, gene_low, gene_high)

        # Aplicar mutación con la tasa especificada
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.uniform(gene_low, gene_high)
        return mutated

    except Exception as e:
        raise MutationError(
            f"Error en mutación uniforme: {str(e)}",
            details={
                "gene_low": gene_low,
                "gene_high": gene_high,
                "mutation_rate": mutation_rate,
                "individual_dtype": individual.dtype,
                "original_error": str(e),
            },
        ) from e


@njit(cache=True)
def _mutation_bit_flip_impl(individual: NDArray[np.int_]) -> NDArray[np.int_]:
    """Implementación núcleo de la mutación de bit flip."""
    mutated = individual.copy()
    idx = np.random.randint(0, len(mutated))
    mutated[idx] = 1 - mutated[idx]
    return mutated


def mutation_bit_flip(
    individual: NDArray[np.int_], mutation_rate: float = 0.01, **kwargs: Dict[str, Any]
) -> NDArray[np.int_]:
    """Invierte bits aleatorios en un individuo de representación binaria.

    Args:
        individual: Individuo a mutar (debe ser binario: 0s y 1s).
        mutation_rate: Probabilidad de que un bit mute. Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una copia del individuo con los bits invertidos.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        RangeError: Si los valores están fuera de los rangos permitidos.
        MutationError: Si ocurre un error durante la mutación.
        ValueError: Si el individuo no es binario.

    Example:
        >>> import numpy as np
        >>> individual = np.array([0, 1, 0, 1])
        >>> mutated = mutation_bit_flip(individual, 0.5)
        >>> mutated.shape == individual.shape
        True
    """
    # Validar parámetros de entrada
    _validate_mutation_inputs(individual, mutation_rate)

    # Validar que el individuo sea binario
    if not np.all(np.isin(individual, [0, 1])):
        raise ValueError("El individuo debe ser binario (contener solo 0s y 1s)")

    # Aplicar mutación
    try:
        if mutation_rate <= 0:
            return individual.copy()

        if mutation_rate >= 1.0:
            # Si la tasa de mutación es 1.0, invertimos todos los bits
            return 1 - individual

        # Aplicar mutación con la tasa especificada
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < mutation_rate
        mutated[mask] = 1 - mutated[mask]
        return mutated

    except Exception as e:
        raise MutationError(
            f"Error en mutación bit flip: {str(e)}",
            details={
                "mutation_rate": mutation_rate,
                "individual_dtype": individual.dtype,
                "original_error": str(e),
            },
        ) from e


@njit(cache=True)
def _mutation_swap_impl(individual: NDArray) -> NDArray:
    """Implementación núcleo de la mutación de intercambio."""
    idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
    mutated = individual.copy()
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated


def mutation_swap(
    individual: NDArray, mutation_rate: float = 0.01, **kwargs: Dict[str, Any]
) -> NDArray:
    """Intercambia dos genes aleatorios en el individuo.

    Args:
        individual: Individuo a mutar.
        mutation_rate: Probabilidad de que ocurra la mutación. Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una copia del individuo con dos genes intercambiados.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        RangeError: Si los valores están fuera de los rangos permitidos.
        MutationError: Si ocurre un error durante la mutación.
        ValueError: Si el individuo tiene menos de 2 elementos.

    Example:
        >>> import numpy as np
        >>> individual = np.array([1, 2, 3, 4, 5])
        >>> mutated = mutation_swap(individual, 1.0)  # 100% de probabilidad
        >>> set(mutated) == set(individual)  # Mismos elementos, orden diferente
        True
    """
    # Validar parámetros de entrada
    _validate_mutation_inputs(individual, mutation_rate)

    # Validar que el individuo tenga al menos 2 elementos
    if len(individual) < 2:
        return individual.copy()

    # Aplicar mutación
    try:
        if mutation_rate <= 0:
            return individual.copy()

        if mutation_rate >= 1.0 or len(individual) == 2:
            # Si la tasa es 1.0 o solo hay 2 elementos,
            # intercambiamos dos posiciones aleatorias
            idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
            mutated = individual.copy()
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
            return mutated

        # Para tasas de mutación menores a 1.0, aplicamos con probabilidad
        if np.random.random() < mutation_rate:
            return _mutation_swap_impl(individual)

        return individual.copy()

    except Exception as e:
        raise MutationError(
            f"Error en mutación swap: {str(e)}",
            details={
                "mutation_rate": mutation_rate,
                "individual_shape": individual.shape,
                "individual_dtype": individual.dtype,
                "original_error": str(e),
            },
        ) from e


@njit(cache=True)
def _mutation_gaussian_impl(
    individual: NDArray[np.float64], mu: float, sigma: float
) -> NDArray[np.float64]:
    """Implementación núcleo de la mutación gaussiana."""
    noise = np.random.normal(mu, sigma, size=len(individual))
    return individual + noise


def mutation_gaussian(
    individual: NDArray[np.float64],
    mu: float = 0.0,
    sigma: float = 1.0,
    mutation_rate: float = 0.01,
    **kwargs: Dict[str, Any],
) -> NDArray[np.float64]:
    """Añade ruido gaussiano a los genes del individuo.

    Args:
        individual: Individuo a mutar.
        mu: Media de la distribución normal. Por defecto es 0.0.
        sigma: Desviación estándar de la distribución normal. Por defecto es 1.0.
        mutation_rate: Probabilidad de que un gen mute. Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados en esta función).

    Returns:
        Una copia del individuo con ruido gaussiano añadido.

    Raises:
        TypeValidationError: Si los tipos de los parámetros son incorrectos.
        RangeError: Si los valores están fuera de los rangos permitidos.
        MutationError: Si ocurre un error durante la mutación.
        ValueError: Si sigma es negativo.

    Example:
        >>> import numpy as np
        >>> individual = np.array([1.0, 2.0, 3.0, 4.0])
        >>> mutated = mutation_gaussian(individual, sigma=0.1, mutation_rate=0.5)
        >>> mutated.shape == individual.shape
        True
    """
    # Validar parámetros de entrada
    _validate_mutation_inputs(individual, mutation_rate)

    # Validar que sigma no sea negativo
    if sigma < 0:
        raise ValueError(f"sigma no puede ser negativo, se recibió: {sigma}")

    # Aplicar mutación
    try:
        if mutation_rate <= 0:
            return individual.copy()

        if mutation_rate >= 1.0:
            # Si la tasa es 1.0, aplicamos ruido a todos los genes
            return _mutation_gaussian_impl(individual, mu, sigma)

        # Aplicar mutación con la tasa especificada
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < mutation_rate
        if np.any(mask):
            noise = np.random.normal(mu, sigma, size=np.sum(mask))
            mutated[mask] += noise

        return mutated

    except Exception as e:
        raise MutationError(
            f"Error en mutación gaussiana: {str(e)}",
            details={
                "mu": mu,
                "sigma": sigma,
                "mutation_rate": mutation_rate,
                "individual_dtype": individual.dtype,
                "original_error": str(e),
            },
        ) from e
