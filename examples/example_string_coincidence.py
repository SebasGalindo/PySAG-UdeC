"""
Ejemplo de Algoritmo Genético.

Objetivo: Maximizar el número de caracteres coincidentes con la cadena objetivo.
El algoritmo genético se ejecutará por 500 generaciones
y el mejor fitness debe aproximarse a la longitud de la cadena objetivo.
"""

import string

import numpy as np

from PySAG import GA, crossover, initialization, mutation, selection

# 1. Definición del Problema: Coincidencia de Cadenas
TARGET_STRING = "HelloPySAG"
ALLOWED_CHARACTERS = (
    string.ascii_letters + string.digits + " _"
)  # Caracteres permitidos
# Mapeo de caracteres a enteros y viceversa
CHAR_TO_INT = {char: i for i, char in enumerate(ALLOWED_CHARACTERS)}
INT_TO_CHAR = {i: char for i, char in enumerate(ALLOWED_CHARACTERS)}
NUM_POSSIBLE_GENE_VALUES = len(ALLOWED_CHARACTERS)

NUM_GENES = len(
    TARGET_STRING
)  # La longitud del cromosoma es la longitud de la cadena objetivo


def individual_to_string(individual: np.ndarray) -> str:
    """Convierte un individuo (array de ints) a una cadena de caracteres."""
    return "".join(
        [
            INT_TO_CHAR.get(int(gene_val % NUM_POSSIBLE_GENE_VALUES), "?")
            for gene_val in individual
        ]
    )


def fitness_function_string_match(individual: np.ndarray) -> float:
    """
    Función de fitness para el problema de coincidencia de cadenas.

    Args:
        individual: Individuo representado como un array NumPy de enteros.

    Returns:
        float: Fitness de la solución.
    """
    if len(individual) != NUM_GENES:
        raise ValueError(f"El individuo debe tener {NUM_GENES} genes.")

    proposed_string = individual_to_string(individual)

    matches = 0
    for i in range(NUM_GENES):
        if proposed_string[i] == TARGET_STRING[i]:
            matches += 1
    return float(matches)


# 2. Configurar y Instanciar la clase GA
population_size = 200  # Población más grande
num_generations = 500  # Más generaciones para un problema potencialmente más difícil
num_parents_mating = 40
crossover_prob = 0.7  # Probabilidad de cruce más baja
elitism_percentage = 0.02  # Elitismo muy bajo
mutation_rate_per_gene = 0.05  # Tasa de mutación por gen

print(f"Configurando el Algoritmo Genético para encontrar la cadena: '{TARGET_STRING}'")
print(
    f"Caracteres permitidos: '{ALLOWED_CHARACTERS}' (Total: {NUM_POSSIBLE_GENE_VALUES})"
)

ga_instance_string = GA(
    fitness_func=fitness_function_string_match,
    num_genes=NUM_GENES,
    population_size=population_size,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    initial_population_func=initialization.init_random_uniform,
    initial_pop_args={
        "low": 0,
        "high": NUM_POSSIBLE_GENE_VALUES - 1,
        "dtype": np.int_,
    },
    selection_func=selection.selection_random,
    selection_args={},
    crossover_func=crossover.crossover_single_point,
    crossover_args={},
    crossover_probability=crossover_prob,
    mutation_func=mutation.mutation_random_gene_uniform,
    mutation_args={
        "gene_low": 0,
        "gene_high": NUM_POSSIBLE_GENE_VALUES - 1,
        "mutation_rate": mutation_rate_per_gene,
        # El dtype del individuo se respetará por la función de mutación
    },
    keep_elitism_percentage=elitism_percentage,
    random_seed=777,  # Para reproducibilidad
)

# 3. Ejecutar el AG
print("Ejecutando el Algoritmo Genético...")
best_solution_indices, best_fitness = ga_instance_string.run()

# 4. Mostrar Resultados
if best_solution_indices is not None:
    best_string_found = individual_to_string(best_solution_indices)
    print(f"\nMejor cadena encontrada: '{best_string_found}'")
    print(f"Fitness (caracteres coincidentes): {int(best_fitness)} de {NUM_GENES}")

    if best_string_found == TARGET_STRING:
        print("¡Éxito! La cadena objetivo fue encontrada.")
    else:
        str_info = "La cadena objetivo no fue encontrada perfectamente,"
        str_info += " pero esta fue la mejor aproximación."
        print(str_info)

    ga_instance_string.plot_fitness(save_path="string_matching_fitness.png")
else:
    print("No se encontró una solución.")

print("\nEjemplo de coincidencia de cadenas completado.")
