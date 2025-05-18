import numpy as np
from numpy.testing import assert_array_equal # Especial para comparar arrays numpy
import random

from PySAG import operators # Importamos desde nuestra librería instalada (-e)

def test_crossover_single_point_basic():
    """Prueba el cruce de punto único con datos simples."""
    # Forzar el punto de cruce para predecir el resultado
    random.seed(42) # Hacemos la aleatoriedad predecible para el test
    parent1 = np.array([1, 1, 1, 1, 1])
    parent2 = np.array([2, 2, 2, 2, 2])

    # Si el punto de cruce (random.randint(1, 4)) fuera 3 con seed(42)...
    # (Verifica esto ejecutando random.seed(42); random.randint(1, len(parent1) - 1))
    # Debería ser 1 con seed 42
    expected_offspring1 = np.array([1, 2, 2, 2, 2]) # p1[:1] + p2[1:]
    expected_offspring2 = np.array([2, 1, 1, 1, 1]) # p2[:1] + p1[1:]

    offspring1, offspring2 = operators.crossover_single_point(parent1, parent2)

    assert_array_equal(offspring1, expected_offspring1, "El primer hijo no coincide")
    assert_array_equal(offspring2, expected_offspring2, "El segundo hijo no coincide")

def test_crossover_single_point_short():
    """Prueba el cruce con arrays cortos donde no debería haber cruce."""
    parent1 = np.array([1])
    parent2 = np.array([2])
    offspring1, offspring2 = operators.crossover_single_point(parent1, parent2)
    # Debería devolver copias de los padres
    assert_array_equal(offspring1, parent1)
    assert_array_equal(offspring2, parent2)
    assert id(offspring1) != id(parent1) # Asegura que son copias

# TODO: Añadir más tests para otros operadores (selección, mutación, inicialización)
