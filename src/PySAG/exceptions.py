"""Módulo que contiene excepciones personalizadas para PySAG.

Este módulo define excepciones específicas para manejar errores en algoritmos genéticos.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_origin

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
            "param_value": str(param_value),
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
        details: Optional[str] = None,
    ) -> None:
        """
        Inicializa la excepción.

        Args:
            param_name: Nombre del parámetro que falló la validación.
            param_value: Valor que falló la validación.
            min_val: Valor mínimo permitido (inclusive).
            max_val: Valor máximo permitido (inclusive).
            inclusive: Si True, el rango es inclusivo.
            details: Información adicional sobre el error.
        """
        range_desc_parts = []
        if min_val is not None:
            range_desc_parts.append(f"{'>=' if inclusive else '>'} {min_val}")
        if max_val is not None:
            range_desc_parts.append(f"{'<=' if inclusive else '<'} {max_val}")

        range_desc = (
            " y ".join(range_desc_parts) if range_desc_parts else "un valor válido"
        )
        expected = f"valor en el rango: {range_desc}"
        if details:
            expected += f" ({details})"

        super().__init__(
            param_name=param_name,
            param_value=param_value,
            expected=expected,
            min_val=min_val,
            max_val=max_val,
            inclusive=inclusive,
        )


def _format_expected_type(expected_type_val: Any) -> str:
    """Formatea un tipo esperado a una cadena legible."""
    origin = get_origin(expected_type_val)
    if origin:  # Es un tipo genérico como List[int], NDArray[float]
        return (
            str(expected_type_val).replace("typing.", "").replace("numpy.typing.", "")
        )
    elif hasattr(expected_type_val, "__name__"):
        return expected_type_val.__name__
    else:
        return str(expected_type_val)


class TypeValidationError(ValidationError):
    """Se produce cuando un valor tiene un tipo incorrecto."""

    def __init__(
        self,
        param_name: str,
        param_value: Any,
        expected_type: Union[Type, List[Any], Any],
    ) -> None:
        """
        Inicializa la excepción.

        Args:
            param_name: Nombre del parámetro que falló la validación.
            param_value: Valor que falló la validación.
            expected_type: Tipo esperado o lista de tipos esperados.
        """
        actual_type_name = type(param_value).__name__

        if isinstance(expected_type, list):
            expected_str = "uno de los tipos:\n" + ", ".join(
                [_format_expected_type(t) for t in expected_type]
            )
        else:
            expected_str = f"tipo {_format_expected_type(expected_type)}"

        super().__init__(
            param_name=param_name,
            param_value=f"(tipo: {actual_type_name})",
            expected=expected_str,
            actual_type=actual_type_name,
        )


def validate_parameter(
    value: T,
    name: str,
    expected_type: Union[Type, List[Any], Any],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive: bool = True,
) -> T:
    """
    Función de validación de parámetros.

    Args:
        value: Valor a validar.
        name: Nombre del parámetro.
        expected_type: Tipo esperado o lista de tipos esperados.
        min_val: Valor mínimo permitido (inclusive).
        max_val: Valor máximo permitido (inclusive).
        inclusive: Si True, el rango es inclusivo.

    Returns:
        El valor validado.

    Raises:
        TypeValidationError: Si el valor no cumple con el tipo esperado.
        RangeError: Si el valor está fuera del rango permitido.
    """
    raw_expected_types = (
        [expected_type] if not isinstance(expected_type, list) else expected_type
    )

    instance_check_types = []
    for et in raw_expected_types:
        origin = get_origin(et)
        if origin is not None:
            instance_check_types.append(origin)
        else:
            instance_check_types.append(et)

    if not instance_check_types or not any(
        isinstance(value, t) for t in instance_check_types if t is not Any
    ):  # type: ignore
        # Si t es Any, isinstance(value, Any) no es lo que queremos.
        # Any debería permitir cualquier tipo,
        # así que si Any está en los tipos esperados, pasa.
        # Esta condición se vuelve más compleja si 'Any'
        # es uno de los múltiples tipos esperados.
        # Por simplicidad, si Any es un expected_type,
        # se asume que la validación de tipo pasa.
        is_any_expected = any(et is Any for et in raw_expected_types)  # type: ignore

        if not is_any_expected:
            raise TypeValidationError(name, value, expected_type)

    # Validación de rango para números
    if isinstance(value, (int, float)) and (min_val is not None or max_val is not None):
        if min_val is not None:
            if inclusive and value < min_val:
                raise RangeError(
                    name, value, min_val=min_val, max_val=max_val, inclusive=inclusive
                )
            if not inclusive and value <= min_val:
                raise RangeError(
                    name, value, min_val=min_val, max_val=max_val, inclusive=inclusive
                )
        if max_val is not None:
            if inclusive and value > max_val:
                raise RangeError(
                    name, value, min_val=min_val, max_val=max_val, inclusive=inclusive
                )
            if not inclusive and value >= max_val:
                raise RangeError(
                    name, value, min_val=min_val, max_val=max_val, inclusive=inclusive
                )
    return value
