# src/PySAG/__init__.py
"""
PySAG: Una librería simple de Algoritmos Genéticos en Python.
"""
# Importar los módulos de operadores para que sean accesibles
# como PySAG.selection, PySAG.crossover, etc.
from . import crossover, initialization, mutation, selection
from .ga import GA

__version__ = "0.0.1"  # Empezamos con una versión inicial
