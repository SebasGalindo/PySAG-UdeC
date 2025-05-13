# src/PySAG/ga.py
import numpy as np
import random
from . import operators # Importamos nuestros operadores

class GA:
    def __init__(self,
                 fitness_func,
                 num_genes,
                 population_size=50,
                 num_generations=100,
                 num_parents_mating=10,
                 initial_population_func=operators.initial_population_uniform,
                 initial_pop_args={'gene_low': 0, 'gene_high': 1}, # Args para la inicialización
                 selection_func=operators.selection_roulette_wheel,
                 crossover_func=operators.crossover_single_point,
                 mutation_func=operators.mutation_random_gene_uniform,
                 mutation_args={'gene_low': 0, 'gene_high': 1, 'mutation_rate': 0.01},
                 keep_elitism_percentage=0.1): # Porcentaje de los mejores para pasar directamente

        self.fitness_func = fitness_func
        self.num_genes = num_genes
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating

        self.initial_population_func = initial_population_func
        self.initial_pop_args = initial_pop_args if initial_pop_args is not None else {}

        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_args = mutation_args if mutation_args is not None else {}
        self.keep_elitism_count = int(population_size * keep_elitism_percentage)


        self.population = None
        self.best_solutions_fitness = [] # Para guardar el fitness del mejor de cada generación
        self.best_solution_overall = None
        self.best_fitness_overall = -np.inf # Asumimos maximización

    def _initialize_population(self):
        """Inicializa la población usando la función y argumentos proporcionados."""
        self.population = self.initial_population_func(
            population_size=self.population_size,
            num_genes=self.num_genes,
            **self.initial_pop_args
        )

    def _calculate_population_fitness(self):
        """Calcula el fitness de toda la población actual."""
        fitness_values = np.array([self.fitness_func(individual) for individual in self.population])
        return fitness_values

    def run(self):
        """Ejecuta el algoritmo genético."""
        if self.population is None:
            self._initialize_population()

        for generation in range(self.num_generations):
            fitness_values = self._calculate_population_fitness()

            # Guardar el mejor de esta generación
            current_best_fitness = np.max(fitness_values)
            self.best_solutions_fitness.append(current_best_fitness)
            if current_best_fitness > self.best_fitness_overall:
                self.best_fitness_overall = current_best_fitness
                best_idx = np.argmax(fitness_values)
                self.best_solution_overall = self.population[best_idx]

            print(f"Generación {generation+1}: Mejor Fitness = {current_best_fitness:.4f}")

            # Elitismo: Guardar los mejores individuos
            elite_individuals = []
            if self.keep_elitism_count > 0:
                elite_indices = np.argsort(fitness_values)[-self.keep_elitism_count:]
                elite_individuals = [self.population[i] for i in elite_indices]


            # Selección
            parents = self.selection_func(self.population, fitness_values, self.num_parents_mating)

            # Cruce y Mutación para crear la nueva generación
            num_offspring_needed = self.population_size - self.keep_elitism_count
            offspring_population = []

            # Asegurarse de que haya suficientes padres para el cruce
            if len(parents) < 2:
                # Si no hay suficientes padres (ej. num_parents_mating=1),
                # se clonan para llenar la descendencia, y luego se mutan.
                idx = 0
                while len(offspring_population) < num_offspring_needed:
                    offspring_population.append(parents[idx % len(parents)].copy())
                    idx +=1
            else: # Proceso normal de cruce
                idx = 0
                while len(offspring_population) < num_offspring_needed:
                    parent1 = parents[idx % len(parents)]
                    parent2 = parents[(idx + 1) % len(parents)] # Asegura que no se salga de rango
                    idx += 2 # Avanzamos de dos en dos porque generamos dos hijos

                    # Podríamos añadir una probabilidad de cruce aquí
                    offspring1, offspring2 = self.crossover_func(parent1, parent2)
                    offspring_population.append(offspring1)
                    if len(offspring_population) < num_offspring_needed: # Añadir el segundo si aún hay espacio
                        offspring_population.append(offspring2)


            # Mutación de la descendencia
            mutated_offspring_population = []
            for individual in offspring_population:
                mutated_individual = self.mutation_func(individual, **self.mutation_args)
                mutated_offspring_population.append(mutated_individual)

            # Nueva población: elitismo + descendencia mutada
            self.population = elite_individuals + mutated_offspring_population[:num_offspring_needed] # Asegura tamaño exacto

        print("\nOptimización Finalizada.")
        print(f"Mejor fitness global encontrado: {self.best_fitness_overall:.4f}")
        print(f"Mejor solución global: {self.best_solution_overall}")
        return self.best_solution_overall, self.best_fitness_overall

    def plot_fitness(self):
        """Intenta graficar la evolución del fitness si matplotlib está disponible."""
        try:
            import matplotlib.pyplot as plt
            plt.plot(self.best_solutions_fitness)
            plt.title("Evolución del Fitness por Generación")
            plt.xlabel("Generación")
            plt.ylabel("Mejor Fitness")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib no está instalado. No se puede graficar. Instala con 'pip install matplotlib'")