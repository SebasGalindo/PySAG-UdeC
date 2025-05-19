"""Módulo que contiene excepciones personalizadas para PySAG.

Este módulo define excepciones específicas para manejar errores en algoritmos genéticos.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union

# Tipo genérico para los valores de los errores
T = TypeVar("T")


class PySAGError(Exception):
    """Clase base para todas las excepciones de PySAG.

    Args:
        message: Mensaje descriptivo del error.
        details: Información adicional sobre el error.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Inicializa la excepción con un mensaje y detalles opcionales."""
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Representación en cadena de la excepción."""
        if self.details:
            return f"{self.message} | Detalles: {self.details}"
        return self.message


class GeneticAlgorithmError(PySAGError):
    """Excepción base para errores en el algoritmo genético."""

    pass


class InitializationError(GeneticAlgorithmError):
    """Se produce cuando hay un error en la inicialización de la población."""

    pass


class SelectionError(GeneticAlgorithmError):
    """Se produce cuando hay un error en la selección de individuos."""

    pass


class CrossoverError(GeneticAlgorithmError):
    """Se produce cuando hay un error en el operador de cruce."""

    pass


class MutationError(GeneticAlgorithmError):
    """Se produce cuando hay un error en el operador de mutación."""

    pass


class FitnessEvaluationError(GeneticAlgorithmError):
    """Se produce cuando hay un error en la evaluación de la función de aptitud."""

    pass


class ParameterError(PySAGError):
    """Se produce cuando hay un error en los parámetros de entrada."""

    pass


class ValidationError(PySAGError):
    """Se produce cuando falla la validación de un valor o parámetro.

    Args:
        param_name: Nombre del parámetro que falló la validación.
        param_value: Valor que falló la validación.
        expected: Descripción del valor esperado.
    """  # D205, D400 corregidos aquí

    def __init__(
        self, param_name: str, param_value: Any, expected: str, **kwargs: Any
    ) -> None:
        """
        Inicializa la excepción.

        Args:
            param_name: Nombre del parámetro que falló la validación.
            param_value: Valor que falló la validación.
            expected: Descripción del valor esperado.

        """
        message = (
            f"Validación fallida para el parámetro '{param_name}'. "
            f"Se recibió: {param_value}. Se esperaba: {expected}"
        )
        details = {
            "param_name": param_name,
            "param_value": param_value,
            "expected": expected,
            **kwargs,
        }
        super().__init__(message, details)


class RangeError(ValidationError):
    """Se produce cuando un valor está fuera del rango permitido."""

    def __init__(
        self,
        param_name: str,
        param_value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        inclusive: bool = True,
    ) -> None:
        """Inicializa la excepción.

        Args:
            param_name: Nombre del parámetro.
            param_value: Valor del parámetro.
            min_val: Valor mínimo permitido.
            max_val: Valor máximo permitido.
            inclusive: Si el rango incluye los límites.
        """
        range_desc = []
        if min_val is not None:
            range_desc.append(f"{'>=' if inclusive else '>'} {min_val}")
        if max_val is not None:
            range_desc.append(f"{'<=' if inclusive else '<'} {max_val}")

        expected = f"valor en el rango: {' '.join(range_desc)}"

        super().__init__(
            param_name=param_name,
            param_value=param_value,
            expected=expected,
            min_val=min_val,
            max_val=max_val,
            inclusive=inclusive,
        )


class TypeValidationError(ValidationError):
    """Se produce cuando un valor tiene un tipo incorrecto."""

    def __init__(
        self, param_name: str, param_value: Any, expected_type: Union[Type, List[Type]]
    ) -> None:
        """Inicializa la excepción.

        Args:
            param_name: Nombre del parámetro.
            param_value: Valor del parámetro.
            expected_type: Tipo o lista de tipos esperados.
        """
        if isinstance(expected_type, list):
            expected = (
                f"uno de los tipos: {', '.join(t.__name__ for t in expected_type)}"
            )
        else:
            expected = f"tipo {expected_type.__name__}"

        actual_type = type(param_value).__name__

        super().__init__(
            param_name=param_name,
            param_value=f"{param_value} (tipo: {actual_type})",
            expected=expected,
            actual_type=actual_type,
            expected_type=expected_type,
        )


def validate_parameter(
    value: T,
    name: str,
    expected_type: Union[Type, List[Type]],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive: bool = True,
) -> T:
    """Valida un parámetro según tipo y rango.

    Args:
        value: Valor del parámetro a validar.
        name: Nombre del parámetro (para mensajes de error).
        expected_type: Tipo o tipos de Python esperados.
        min_val: Valor mínimo permitido (opcional).
        max_val: Valor máximo permitido (opcional).
        inclusive: Si es True, el rango incluye los límites.

    Returns:
        El valor validado.

    Raises:
        TypeValidationError: Si el tipo no es el esperado.
        RangeError: Si el valor está fuera del rango permitido.
    """
    # Validación de tipo
    expected_types = (
        [expected_type] if not isinstance(expected_type, list) else expected_type
    )
    if not any(isinstance(value, t) for t in expected_types):
        raise TypeValidationError(name, value, expected_types)

    # Validación de rango para números
    if isinstance(value, (int, float)) and (min_val is not None or max_val is not None):
        if min_val is not None and (value < min_val if inclusive else value <= min_val):
            raise RangeError(name, value, min_val, max_val, inclusive)
        if max_val is not None and (value > max_val if inclusive else value >= max_val):
            raise RangeError(name, value, min_val, max_val, inclusive)

    return value
