"""
Ejemplo de Algoritmo Genético para resolver el Problema de la Mochila Binaria (0/1).

Objetivo: Maximizar el valor total de los ítems en la mochila sin exceder
          su capacidad máxima de peso.
Cada gen representa un ítem: 1 si se toma, 0 si no.

El algoritmo genético se ejecutará por 200 generaciones
y el mejor fitness debe aproximarse a 490.
"""

import sys

import numpy as np

try:
    from PySAG import GA, crossover, initialization, mutation, selection
except ImportError:
    print("Error: No se pudo importar la librería PySAG.")
    print("Asegúrate de que esté instalada o que la ruta a 'src' esté en PYTHONPATH.")
    sys.exit(1)

# 1. Definición del Problema de la Mochila
# Ítems: (valor, peso)
items_data = {
    "item1": (60, 10),
    "item2": (100, 20),
    "item3": (120, 30),
    "item4": (80, 15),
    "item5": (90, 25),
    "item6": (70, 12),
    "item7": (110, 22),
    "item8": (50, 8),
    "item9": (130, 35),
    "item10": (75, 18),
}
item_names = list(items_data.keys())
item_values = np.array([items_data[name][0] for name in item_names])
item_weights = np.array([items_data[name][1] for name in item_names])

KNAPSACK_CAPACITY = 100  # Capacidad máxima de peso de la mochila
NUM_ITEMS = len(item_names)  # Número de genes, uno por ítem


def fitness_function_knapsack(individual: np.ndarray) -> float:
    """
    Función de fitness para el problema de la mochila binaria.

    Calcula el fitness de una solución para el problema de la mochila.
    El individuo es un array binario (0 o 1).

    Args:
        individual: Individuo representado como un array NumPy binario.

    Returns:
        float: Fitness de la solución.
    """
    if len(individual) != NUM_ITEMS:
        raise ValueError(f"El individuo debe tener {NUM_ITEMS} genes (ítems).")
    if not np.all(np.logical_or(individual == 0, individual == 1)):
        # Penalizar fuertemente si no es binario,
        # aunque la inicialización/mutación deberían manejarlo
        # print(f"Advertencia: Individuo no binario encontrado: {individual}")
        return -10000  # Penalización muy alta

    total_value = np.sum(individual * item_values)
    total_weight = np.sum(individual * item_weights)

    # Penalización si se excede la capacidad de la mochila
    if total_weight > KNAPSACK_CAPACITY:
        # Penalización proporcional al exceso de peso
        # Cuanto más se exceda, peor es el fitness.
        # Alternativamente, se puede devolver 0 o un valor muy bajo.
        penalty = (total_weight - KNAPSACK_CAPACITY) * 10  # Factor de penalización
        return max(
            0, total_value - penalty
        )  # Asegurar que el fitness no sea negativo con esta penalización
        # return 0 # Opción más simple: inválido si excede
    else:
        return total_value


# 2. Configurar y Instanciar la clase GA
population_size = 80
num_generations = 200
num_parents_mating = 15
crossover_prob = 0.9
elitism_percentage = 0.15
mutation_rate_bit_flip = 0.02  # Tasa de mutación por gen para bit-flip

print("Configurando el Algoritmo Genético para el Problema de la Mochila Binaria...")

ga_instance_knapsack = GA(
    fitness_func=fitness_function_knapsack,
    num_genes=NUM_ITEMS,
    population_size=population_size,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    # Inicialización: Binaria, ya que cada gen es 0 o 1
    initial_population_func=initialization.init_random_binary,
    initial_pop_args={},  # No necesita argumentos extra
    # Selección: Por Rango (Rank Selection)
    selection_func=selection.selection_rank,
    selection_args={},  # No necesita argumentos extra
    # Cruce: De dos puntos (Two-Point Crossover)
    crossover_func=crossover.crossover_two_points,
    crossover_args={},  # No necesita argumentos extra
    crossover_probability=crossover_prob,
    # Mutación: Bit Flip (adecuada para representaciones binarias)
    mutation_func=mutation.mutation_bit_flip,
    mutation_args={"mutation_rate": mutation_rate_bit_flip},
    keep_elitism_percentage=elitism_percentage,
    random_seed=123,  # Para reproducibilidad
)

# 3. Ejecutar el AG
print("Ejecutando el Algoritmo Genético...")
best_solution, best_fitness = ga_instance_knapsack.run()

# 4. Mostrar Resultados
if best_solution is not None:
    print(f"\nMejor solución (selección de ítems): {best_solution.astype(int)}")
    selected_items_indices = np.where(best_solution == 1)[0]
    selected_item_names = [item_names[i] for i in selected_items_indices]

    str_items = "Ítems seleccionados: "
    str_items += f"{selected_item_names if selected_item_names else 'Ninguno'}"
    print(str_items)

    final_value = np.sum(best_solution * item_values)
    final_weight = np.sum(best_solution * item_weights)
    print(f"Valor total en la mochila: {final_value}")
    print(f"Peso total en la mochila: {final_weight} (Capacidad: {KNAPSACK_CAPACITY})")

    if final_weight > KNAPSACK_CAPACITY:
        str_adv = "¡Advertencia! La solución excede la capacidad de la mochila"
        str_adv += f"Fitness: {best_fitness}"
        print(str_adv)
    else:
        print(f"Fitness de la mejor solución: {best_fitness:.2f}")

    ga_instance_knapsack.plot_fitness(save_path="knapsack_binary_fitness.png")
else:
    print("No se encontró una solución.")

print("\nEjemplo del Problema de la Mochila Binaria completado.")
