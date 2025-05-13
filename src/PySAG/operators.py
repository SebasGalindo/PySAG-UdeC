# src/PySAG/operators.py
import random
import numpy as np # Usaremos numpy para operaciones numéricas eficientes

# --- Funciones de Inicialización de Población ---
def initial_population_uniform(population_size, num_genes, gene_low, gene_high):
    """
    Crea una población inicial con genes distribuidos uniformemente entre low y high.
    Asume genes numéricos.
    """
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(low=gene_low, high=gene_high, size=num_genes)
        population.append(individual)
    return population

# --- Funciones de Selección ---
def selection_roulette_wheel(population, fitness_values, num_parents):
    """
    Selección por ruleta. Los individuos son seleccionados con una probabilidad
    proporcional a su fitness. Asume que todos los fitness son positivos.
    """
    fitness_sum = np.sum(fitness_values)
    if fitness_sum == 0: # Evitar división por cero si todos los fitness son 0
        probabilities = np.ones(len(population)) / len(population)
    else:
        probabilities = fitness_values / fitness_sum

    selected_parents_indices = np.random.choice(
        len(population),
        size=num_parents,
        p=probabilities,
        replace=True # Se puede seleccionar el mismo padre varias veces
    )
    selected_parents = [population[i] for i in selected_parents_indices]
    return selected_parents

# --- Funciones de Cruce (Crossover) ---
def crossover_single_point(parent1, parent2):
    """
    Cruce de un solo punto.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud para el cruce.")
    if len(parent1) < 2: # No se puede cruzar si hay menos de 2 genes
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1) - 1)
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))
    return offspring1, offspring2

# --- Funciones de Mutación ---
def mutation_random_gene_uniform(individual, gene_low, gene_high, mutation_rate):
    """
    Muta genes aleatorios del individuo, reemplazándolos con un valor uniforme.
    """
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = np.random.uniform(low=gene_low, high=gene_high)
    return mutated_individual
