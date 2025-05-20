"""PySAG: Una librería simple de Algoritmos Genéticos en Python."""

from . import crossover, initialization, mutation, selection
from .exceptions import (
    GeneticAlgorithmError,
    InitializationError,
    MutationError,
    ParameterError,
    RangeError,
    SelectionError,
    TypeValidationError,
)
from .ga import GA

__version__ = "0.1.0"

__all__ = [
    "GA",
    "crossover",
    "initialization",
    "mutation",
    "selection",
    "GeneticAlgorithmError",
    "InitializationError",
    "MutationError",
    "ParameterError",
    "RangeError",
    "SelectionError",
    "TypeValidationError",
]
