"""
Módulo que implementa operadores de mutación para algoritmos genéticos.

Este módulo proporciona varias estrategias de mutación que pueden ser utilizadas
en algoritmos genéticos. Las funciones están diseñadas para ser flexibles y
eficientes, utilizando Numba para la optimización del núcleo computacional.

Operadores de Mutación Disponibles:
    - mutation_bit_flip: Invierte bits aleatorios en un individuo binario.
    - mutation_random_gene_uniform: Reemplaza genes con valores de una dist. uniforme.
    - mutation_gaussian: Añade ruido gaussiano a genes numéricos.
    - mutation_swap: Intercambia dos genes aleatorios en el individuo.
    - mutation_inversion: Invierte un segmento del cromosoma (para permutaciones).
"""

from typing import Any, Dict, Optional, TypeVar, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .exceptions import (
    MutationError,
    ParameterError,
    RangeError,
    TypeValidationError,
    validate_parameter,
)

# Tipo genérico para los cromosomas
ChromosomeType = TypeVar("ChromosomeType", bound=np.generic)
IndividualNDArray = NDArray[ChromosomeType]


def _validate_mutation_inputs(
    individual: IndividualNDArray,
    mutation_rate: float,
    param_name: str = "individual",
    check_empty: bool = True,
    expected_ndim: int = 1,
) -> None:
    """
    Valida los parámetros comunes de las funciones de mutación.

    Args:
        individual: Individuo a mutar.
        mutation_rate: Tasa de mutación.
        param_name: Nombre del parámetro 'individual' para mensajes de error.
        check_empty: Si es True, valida que el individuo no esté vacío.
        expected_ndim: Dimensión esperada para el array del individuo.

    Raises:
        TypeValidationError: Si el tipo del individuo es incorrecto.
        ParameterError: Si el individuo está vacío o no tiene la dimensión esperada.
        RangeError: Si `mutation_rate` está fuera del rango [0, 1].
    """
    validate_parameter(individual, param_name, np.ndarray)  # type: ignore
    validate_parameter(
        mutation_rate,
        "mutation_rate",
        (float, int),
        min_val=0.0,
        max_val=1.0,
    )  # type: ignore

    if individual.ndim != expected_ndim:
        raise ParameterError(
            f"El parámetro '{param_name}' debe ser un array {expected_ndim}D. "
            f"Dimensión encontrada: {individual.ndim}."
        )
    if check_empty and individual.size == 0:
        raise ParameterError(
            f"El parámetro '{param_name}' no puede ser un array vacío."
        )


@njit(cache=True)
def _mutation_bit_flip_impl(
    individual_copy: NDArray[np.int_], mutation_rate: float
) -> NDArray[np.int_]:
    """
    Implementación Numba: Invierte bits según la tasa de mutación por gen.

    Args:
        individual_copy: Copia del individuo a mutar.
        mutation_rate: Tasa de mutación.

    Returns:
        El individuo mutado.
    """
    for i in range(len(individual_copy)):
        if np.random.random() < mutation_rate:
            individual_copy[i] = 1 - individual_copy[i]
    return individual_copy


@njit(cache=True)
def _mutation_random_gene_uniform_impl(
    individual_copy: IndividualNDArray,
    gene_low: float,
    gene_high: float,
    mutation_rate: float,
    is_float_type: bool,
) -> IndividualNDArray:
    """
    Implementación Numba: Reemplaza genes con valor uniforme según tasa por gen.

    Args:
        individual_copy: Copia del individuo a mutar.
        gene_low: Valor mínimo para el reemplazo.
        gene_high: Valor máximo para el reemplazo.
        mutation_rate: Tasa de mutación.
        is_float_type: Si es True, el individuo es de tipo float.

    Returns:
        El individuo mutado.
    """
    for i in range(len(individual_copy)):
        if np.random.random() < mutation_rate:
            if is_float_type:
                individual_copy[i] = np.random.uniform(gene_low, gene_high)
            else:  # Es entero
                # np.random.randint es [low, high)
                # para [low, high] inclusivo, usar high + 1
                individual_copy[i] = np.random.randint(
                    int(gene_low), int(gene_high) + 1
                )
    return individual_copy


@njit(cache=True)
def _mutation_gaussian_impl(
    individual_copy: NDArray[np.float64],
    mu: float,
    sigma: float,
    mutation_rate: float,
) -> NDArray[np.float64]:
    """
    Implementación Numba: Añade ruido gaussiano según tasa por gen.

    Args:
        individual_copy: Copia del individuo a mutar.
        mu: Media de la distribución normal.
        sigma: Desviación estándar de la distribución normal.
        mutation_rate: Tasa de mutación.

    Returns:
        El individuo mutado.
    """
    for i in range(len(individual_copy)):
        if np.random.random() < mutation_rate:
            noise = np.random.normal(mu, sigma)
            individual_copy[i] += noise
    return individual_copy


@njit(cache=True)
def _mutation_swap_impl(individual_copy: IndividualNDArray) -> IndividualNDArray:
    """
    Implementación Numba: Intercambia dos genes aleatorios.

    Args:
        individual_copy: Copia del individuo a mutar.

    Returns:
        El individuo mutado.
    """
    idx1, idx2 = np.random.choice(len(individual_copy), size=2, replace=False)

    gene1 = individual_copy[idx1]
    individual_copy[idx1] = individual_copy[idx2]
    individual_copy[idx2] = gene1
    return individual_copy


@njit(cache=True)
def _mutation_inversion_impl(individual_copy: IndividualNDArray) -> IndividualNDArray:
    """
    Implementación Numba: Invierte un segmento aleatorio del cromosoma.

    Args:
        individual_copy: Copia del individuo a mutar.

    Returns:
        El individuo mutado.
    """
    idx1, idx2 = np.sort(np.random.choice(len(individual_copy), size=2, replace=False))

    segment = individual_copy[idx1 : idx2 + 1]
    individual_copy[idx1 : idx2 + 1] = segment[::-1]
    return individual_copy


def mutation_bit_flip(
    individual: NDArray[np.int_],
    mutation_rate: float = 0.01,
    **kwargs: Dict[str, Any],
) -> NDArray[np.int_]:
    """
    Invierte bits aleatorios en un individuo de representación binaria.

    Cada bit tiene una probabilidad `mutation_rate` de ser invertido.

    Args:
        individual: Individuo a mutar (array NumPy 1D de enteros, 0s y 1s).
        mutation_rate: Probabilidad de que cada bit mute. Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una copia del individuo con los bits posiblemente invertidos.

    Raises:
        TypeValidationError:
        Si el individuo no es un array de NumPy o no es de tipo entero.

        ParameterError:
        Si el individuo no es 1D.

        RangeError:
        Si `mutation_rate` está fuera del rango [0, 1].

        ValueError:
        Si el individuo no es binario (no contiene solo 0s y 1s).

        MutationError:
        Si ocurre un error inesperado durante la mutación.

    Example:
        >>> import numpy as np
        >>> ind = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        >>> mutated = mutation_bit_flip(ind, mutation_rate=0.5)
        >>> mutated.shape == ind.shape
        True
        >>> # El resultado es probabilístico, pero algunos bits deberían cambiar.
    """
    _validate_mutation_inputs(individual, mutation_rate, check_empty=False)

    if not np.issubdtype(individual.dtype, np.integer):
        raise TypeValidationError(
            "individual.dtype", str(individual.dtype), "np.integer"
        )

    if individual.size > 0 and not np.all(np.isin(individual, [0, 1])):
        raise ValueError(
            "El individuo para 'mutation_bit_flip'"
            + " debe ser binario (contener solo 0s y 1s)."
        )

    if mutation_rate == 0.0 or individual.size == 0:
        return individual.copy()

    try:
        # La función Numba maneja la iteración y la tasa de mutación por gen.
        mutated_individual = _mutation_bit_flip_impl(individual.copy(), mutation_rate)
    except Exception as e:
        raise MutationError(
            f"Error durante la mutación de inversión de bits: {str(e)}",
            details={  # type: ignore
                "mutation_rate": mutation_rate,
                "individual_shape": individual.shape,
                "individual_dtype": str(individual.dtype),
                "original_error": str(e),
            },
        ) from e
    return mutated_individual


def mutation_random_gene_uniform(
    individual: IndividualNDArray,
    gene_low: Union[float, int],
    gene_high: Union[float, int],
    mutation_rate: float = 0.01,
    **kwargs: Dict[str, Any],
) -> IndividualNDArray:
    """
    Muta genes aleatorios reemplazándolos con un valor de una distribución uniforme.

    Cada gen tiene una probabilidad `mutation_rate` de ser mutado.
    Si el `dtype` del individuo es flotante, se usa `np.random.uniform(low, high)`.
    Si es entero, se usa `np.random.randint(low, high + 1)`.

    Args:
        individual: Individuo a mutar (array NumPy 1D).
        gene_low: Límite inferior para los nuevos valores de los genes.
        gene_high: Límite superior para los nuevos valores de los genes.
                   (Exclusivo para flotantes, inclusivo para enteros).
        mutation_rate: Probabilidad de que cada gen mute. Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una copia del individuo con genes posiblemente mutados.

    Raises:
        TypeValidationError: Si `individual` no es array, o si `gene_low`/`gene_high`
                             no son numéricos.
        ParameterError: Si el individuo no es 1D.
        RangeError: Si `mutation_rate` está fuera de rango, o si `gene_low > gene_high`.
        MutationError: Si ocurre un error inesperado.

    Example:
        >>> import numpy as np
        >>> ind_float = np.array([1.0, 2.0, 3.0, 4.0])
        >>> mutated_float = mutation_random_gene_uniform(ind_float, 0.0, 10.0, 0.5)
        >>> ind_int = np.array([1, 2, 3, 4])
        >>> mutated_int = mutation_random_gene_uniform(ind_int, 0, 10, 0.5)
    """
    _validate_mutation_inputs(individual, mutation_rate, check_empty=False)
    validate_parameter(gene_low, "gene_low", (float, int))  # type: ignore
    validate_parameter(gene_high, "gene_high", (float, int))  # type: ignore

    is_float_type = np.issubdtype(individual.dtype, np.floating)
    is_int_type = np.issubdtype(individual.dtype, np.integer)

    if not (is_float_type or is_int_type):
        raise TypeValidationError(
            "individual.dtype",
            str(individual.dtype),
            "un tipo numérico (flotante o entero)",
        )

    if float(gene_low) > float(
        gene_high
    ):  # Comparar como flotantes para evitar problemas de tipo
        raise RangeError(
            "gene_low",
            gene_low,
            max_val=gene_high,  # type: ignore
            details="'gene_low' no puede ser mayor que 'gene_high'.",
        )

    if mutation_rate == 0.0 or individual.size == 0:
        return individual.copy()

    try:
        mutated_individual = _mutation_random_gene_uniform_impl(
            individual.copy(),
            float(gene_low),
            float(gene_high),
            mutation_rate,
            is_float_type,
        )
    except Exception as e:
        raise MutationError(
            f"Error durante la mutación uniforme de gen aleatorio: {str(e)}",
            details={  # type: ignore
                "gene_low": gene_low,
                "gene_high": gene_high,
                "mutation_rate": mutation_rate,
                "individual_shape": individual.shape,
                "individual_dtype": str(individual.dtype),
                "original_error": str(e),
            },
        ) from e
    return mutated_individual


def mutation_gaussian(
    individual: NDArray[np.float64],
    mu: float = 0.0,
    sigma: float = 1.0,
    mutation_rate: float = 0.01,
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
    **kwargs: Dict[str, Any],
) -> NDArray[np.float64]:
    """
    Añade ruido gaussiano a los genes del individuo.

    Cada gen tiene una probabilidad `mutation_rate` de ser mutado.
    Si muta, se le suma un valor muestreado de N(mu, sigma).
    Opcionalmente, los valores pueden ser recortados a un rango [clip_low, clip_high].

    Args:
        individual: Individuo a mutar (array NumPy 1D de flotantes).
        mu: Media de la distribución normal para el ruido. Por defecto es 0.0.
        sigma: Desviación estándar de la distribución normal. Debe ser no negativo.
               Por defecto es 1.0.
        mutation_rate: Probabilidad de que cada gen mute. Por defecto es 0.01.
        clip_low: Límite inferior opcional para recortar los valores mutados.
        clip_high: Límite superior opcional para recortar los valores mutados.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una copia del individuo con ruido gaussiano posiblemente añadido y recortado.

    Raises:
        TypeValidationError: Si `individual` no es array de flotantes, o `mu`/`sigma`
                             no son numéricos.
        ParameterError: Si el individuo no es 1D.
        RangeError: Si `mutation_rate` está fuera de rango, `sigma` es negativo,
                    o `clip_low > clip_high`.
        MutationError: Si ocurre un error inesperado.

    Example:
        >>> import numpy as np
        >>> ind = np.array([1.0, 2.0, 3.0, 4.0])
        >>> mutated = mutation_gaussian(
            ind,
            sigma=0.1,
            mutation_rate=0.5,
            clip_low=0.0,
            clip_high=5.0,
        )
    """
    _validate_mutation_inputs(individual, mutation_rate, check_empty=False)
    if not np.issubdtype(individual.dtype, np.floating):
        raise TypeValidationError(
            "individual.dtype", str(individual.dtype), "un tipo flotante de NumPy"
        )

    validate_parameter(mu, "mu", (float, int))  # type: ignore
    validate_parameter(sigma, "sigma", (float, int), min_val=0.0)  # type: ignore
    if clip_low is not None:
        validate_parameter(clip_low, "clip_low", (float, int))  # type: ignore
    if clip_high is not None:
        validate_parameter(clip_high, "clip_high", (float, int))  # type: ignore

    if clip_low is not None and clip_high is not None and clip_low > clip_high:
        raise RangeError(
            "clip_low",
            clip_low,
            max_val=clip_high,  # type: ignore
            details="'clip_low' no puede ser mayor que 'clip_high'.",
        )

    if mutation_rate == 0.0 or individual.size == 0:
        return individual.copy()

    try:
        mutated_individual = _mutation_gaussian_impl(
            individual.copy().astype(np.float64), float(mu), float(sigma), mutation_rate
        )
        if clip_low is not None or clip_high is not None:
            np.clip(mutated_individual, clip_low, clip_high, out=mutated_individual)

    except Exception as e:
        raise MutationError(
            f"Error durante la mutación gaussiana: {str(e)}",
            details={  # type: ignore
                "mu": mu,
                "sigma": sigma,
                "mutation_rate": mutation_rate,
                "clip_low": clip_low,
                "clip_high": clip_high,
                "individual_shape": individual.shape,
                "individual_dtype": str(individual.dtype),
                "original_error": str(e),
            },
        ) from e
    return mutated_individual


def mutation_swap(
    individual: IndividualNDArray,
    mutation_rate: float = 0.01,
    **kwargs: Dict[str, Any],
) -> IndividualNDArray:
    """
    Intercambia dos genes aleatorios en el individuo.

    La operación de intercambio (swap) ocurre en el individuo completo con
    una probabilidad `mutation_rate`. Si ocurre, se eligen dos genes
    distintos al azar y se intercambian sus posiciones.

    Args:
        individual: Individuo a mutar (array NumPy 1D).
        mutation_rate: Probabilidad de que la operación de intercambio ocurra.
                       Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una copia del individuo, con dos genes posiblemente intercambiados.

    Raises:
        TypeValidationError: Si `individual` no es un array de NumPy.
        ParameterError: Si el individuo no es 1D o tiene menos de 2 elementos
                        (necesario para un intercambio).
        RangeError: Si `mutation_rate` está fuera del rango [0, 1].
        MutationError: Si ocurre un error inesperado.

    Example:
        >>> import numpy as np
        >>> ind = np.array([1, 2, 3, 4, 5])
        >>> # Para asegurar que el swap ocurra para el ejemplo:
        >>> mutated = mutation_swap(ind, mutation_rate=1.0)
        >>> len(mutated) == len(ind) and set(mutated) == (
            set(ind) and not np.array_equal(mutated, ind)
        )
        True
    """
    _validate_mutation_inputs(individual, mutation_rate, check_empty=True)
    if individual.size < 2:
        return individual.copy()

    mutated_individual = individual.copy()
    if np.random.random() < mutation_rate:
        try:
            mutated_individual = _mutation_swap_impl(mutated_individual)
        except Exception as e:  # Captura errores de Numba o NumPy dentro de _impl
            raise MutationError(
                f"Error durante la mutación de intercambio: {str(e)}",
                details={  # type: ignore
                    "mutation_rate": mutation_rate,
                    "individual_shape": individual.shape,
                    "individual_dtype": str(individual.dtype),
                    "original_error": str(e),
                },
            ) from e
    return mutated_individual


def mutation_inversion(
    individual: IndividualNDArray,
    mutation_rate: float = 0.01,
    **kwargs: Dict[str, Any],
) -> IndividualNDArray:
    """
    Invierte un segmento aleatorio del cromosoma.

    Esta mutación es comúnmente usada para representaciones de permutación.
    La operación de inversión ocurre en el individuo completo con una
    probabilidad `mutation_rate`. Si ocurre, se seleccionan dos puntos
    aleatorios y el segmento entre ellos (inclusive) se invierte.

    Args:
        individual: Individuo a mutar (array NumPy 1D).
        mutation_rate: Probabilidad de que la operación de inversión ocurra.
                       Por defecto es 0.01.
        **kwargs: Argumentos adicionales (no utilizados).

    Returns:
        Una copia del individuo, con un segmento posiblemente invertido.

    Raises:
        TypeValidationError: Si `individual` no es un array de NumPy.
        ParameterError: Si el individuo no es 1D o tiene menos de 2 elementos
                        (necesario para una inversión significativa).
        RangeError: Si `mutation_rate` está fuera del rango [0, 1].
        MutationError: Si ocurre un error inesperado.

    Example:
        >>> import numpy as np
        >>> ind = np.array([1, 2, 3, 4, 5, 6])
        >>> # Para asegurar que la inversión ocurra para el ejemplo:
        >>> mutated = mutation_inversion(ind, mutation_rate=1.0)
        >>> len(mutated) == len(ind) and set(mutated) == set(ind)
        True
        >>> # Es probable que el orden cambie si la longitud es >= 2
    """
    _validate_mutation_inputs(individual, mutation_rate, check_empty=True)
    if individual.size < 2:
        return individual.copy()

    mutated_individual = individual.copy()
    if np.random.random() < mutation_rate:
        try:
            mutated_individual = _mutation_inversion_impl(mutated_individual)
        except Exception as e:
            raise MutationError(
                f"Error durante la mutación de inversión: {str(e)}",
                details={  # type: ignore
                    "mutation_rate": mutation_rate,
                    "individual_shape": individual.shape,
                    "individual_dtype": str(individual.dtype),
                    "original_error": str(e),
                },
            ) from e
    return mutated_individual
