"""
Ejemplo de Algoritmo Genético para maximizar una función matemática continua.

Problema: Maximizar la función f(x, y) = sin(x) * cos(y) + (x+y)/10
donde x e y están en el rango [-10, 10].
El individuo tendrá 2 genes (x, y).

El algoritmo genético se ejecutará por 150 generaciones
y el mejor fitness debe aproximarse a 2.4238.
"""

import numpy as np

from PySAG import GA, crossover, initialization, mutation, selection

# 1. Definir la función de Fitness
GENE_LOW = -10.0
GENE_HIGH = 10.0
NUM_GENES = 2  # Dos variables: x, y


def fitness_function(individual: np.ndarray) -> float:
    """
    Función de fitness para el problema de maximización de una función matemática.

    Calcula el fitness de un individuo.
    f(x, y) = sin(x) * cos(y) + (x+y)/10
    Buscamos maximizar esta función.
    """
    if len(individual) != NUM_GENES:
        raise ValueError(f"El individuo debe tener {NUM_GENES} genes.")
    x = individual[0]
    y = individual[1]

    # Asegurarse de que los genes estén dentro del rango
    # (puede ser redundante si la mutación/inicialización lo manejan)
    x = np.clip(x, GENE_LOW, GENE_HIGH)
    y = np.clip(y, GENE_LOW, GENE_HIGH)

    return np.sin(x) * np.cos(y) + (x + y) / 10.0


# 2. Configurar y Instanciar la clase GA
population_size = 100
num_generations = 150
num_parents_mating = 20
crossover_prob = 0.85
elitism_percentage = 0.05
mutation_rate_for_gaussian = 0.1  # Tasa de mutación por gen para la mutación Gaussiana

print("Configurando el Algoritmo Genético para maximizar f(x,y)...")

ga_instance_math = GA(
    fitness_func=fitness_function,
    num_genes=NUM_GENES,
    population_size=population_size,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    # Inicialización: Uniforme para valores flotantes
    initial_population_func=initialization.init_random_uniform,
    initial_pop_args={"low": GENE_LOW, "high": GENE_HIGH, "dtype": np.float64},
    # Selección: Por Torneo
    selection_func=selection.selection_tournament,
    selection_args={
        "tournament_size": 5
    },  # Argumento específico para la selección por torneo
    # Cruce: Uniforme
    crossover_func=crossover.crossover_uniform,
    crossover_args={"mix_probability": 0.5},  # Argumento para cruce uniforme
    crossover_probability=crossover_prob,
    # Mutación: Gaussiana
    mutation_func=mutation.mutation_gaussian,
    mutation_args={
        "mu": 0.0,
        "sigma": 0.5,  # Desviación estándar
        "mutation_rate": mutation_rate_for_gaussian,
        "gene_low": GENE_LOW,  # Límites para asegurar que la mutación no se salga
        "gene_high": GENE_HIGH,
    },
    keep_elitism_percentage=elitism_percentage,
    random_seed=42,  # Para reproducibilidad
)

# 3. Ejecutar el AG
print("Ejecutando el Algoritmo Genético...")
best_solution, best_fitness = ga_instance_math.run()

# 4. Mostrar Resultados
if best_solution is not None:
    print(f"\nMejor solución encontrada: {best_solution}")
    print(f"Valor de la función (fitness): {best_fitness:.6f}")

    # Verificar el fitness recalculando
    recalculated_fitness = fitness_function(best_solution)
    print(f"Fitness recalculado para la mejor solución: {recalculated_fitness:.6f}")

    # (Opcional) Graficar el fitness
    ga_instance_math.plot_fitness(save_path="math_function_maximization_fitness.png")
else:
    print("No se encontró una solución.")

print("\nEjemplo de maximización de función matemática completado.")
