# src/pysag/ga.py
import numpy as np
import random

# Importar los nuevos módulos de operadores
from . import initialization as default_init
from . import selection as default_selection
from . import crossover as default_crossover
from . import mutation as default_mutation

class GA:
    def __init__(self,
                 fitness_func,
                 num_genes,
                 population_size=50,
                 num_generations=100,
                 num_parents_mating=10,
                 
                 # Actualizar las funciones por defecto a sus nuevas ubicaciones
                 initial_population_func=default_init.initial_population_uniform,
                 initial_pop_args=None,

                 selection_func=default_selection.selection_roulette_wheel,
                 selection_args=None,

                 crossover_func=default_crossover.crossover_single_point,
                 crossover_args=None,
                 crossover_probability=0.9,

                 mutation_func=default_mutation.mutation_random_gene_uniform,
                 mutation_args=None,
                 
                 keep_elitism_percentage=0.1):

        # ... (el resto del __init__ y los métodos de la clase GA se mantienen igual que antes) ...
        # ... asegúrate de que la lógica interna para llamar a las funciones (ej. en el método run) ...
        # ... no necesita cambios, ya que las funciones en sí (ej. self.selection_func) ...
        # ... son pasadas como objetos.

        self.fitness_func = fitness_func
        self.num_genes = num_genes
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating

        self.initial_population_func = initial_population_func
        self.initial_pop_args = initial_pop_args if initial_pop_args is not None else {}

        self.selection_func = selection_func
        self.selection_args = selection_args if selection_args is not None else {}
        
        self.crossover_func = crossover_func
        self.crossover_args = crossover_args if crossover_args is not None else {}
        self.crossover_probability = crossover_probability

        self.mutation_func = mutation_func
        self.mutation_args = mutation_args if mutation_args is not None else {}
        if 'mutation_rate' not in self.mutation_args and \
           (self.mutation_func == default_mutation.mutation_random_gene_uniform or \
            self.mutation_func == default_mutation.mutation_bit_flip or \
            self.mutation_func == default_mutation.mutation_swap): # Solo para los que usan mutation_rate como arg directo
             self.mutation_args['mutation_rate'] = 0.01


        self.keep_elitism_count = int(population_size * keep_elitism_percentage)
        if self.num_parents_mating % 2 != 0 and self.keep_elitism_count % 2 != 0:
            # Esta advertencia puede ser demasiado específica, considera si es útil
            # print("Advertencia: La suma de num_parents_mating y keep_elitism_count es impar...")
            pass
        if self.num_parents_mating < 2 and self.crossover_func is not None:
            # print(f"Advertencia: num_parents_mating ({self.num_parents_mating}) es menor que 2...")
            pass

        self.population = None
        self.best_solutions_fitness = []
        self.best_solution_overall = None
        self.best_fitness_overall = -np.inf


    def _initialize_population(self):
        # (Sin cambios)
        self.population = self.initial_population_func(
            population_size=self.population_size,
            num_genes=self.num_genes,
            **self.initial_pop_args
        )

    def _calculate_population_fitness(self):
        # (Sin cambios)
        fitness_values = np.array([self.fitness_func(individual) for individual in self.population])
        return fitness_values

    def run(self):
        # (El método run no debería necesitar cambios en su lógica interna,
        # ya que opera sobre las funciones almacenadas en self.selection_func, etc.)
        if self.population is None:
            self._initialize_population()

        for generation in range(self.num_generations):
            fitness_values = self._calculate_population_fitness()

            current_best_fitness = np.max(fitness_values)
            self.best_solutions_fitness.append(current_best_fitness)
            if current_best_fitness > self.best_fitness_overall:
                self.best_fitness_overall = current_best_fitness
                best_idx = np.argmax(fitness_values)
                self.best_solution_overall = self.population[best_idx].copy()

            print(f"Generación {generation+1}/{self.num_generations}: Mejor Fitness = {current_best_fitness:.4f}")

            elite_individuals = []
            if self.keep_elitism_count > 0 and self.population: # Añadir check para población no vacía
                elite_indices = np.argsort(fitness_values)[-self.keep_elitism_count:]
                elite_individuals = [self.population[i].copy() for i in elite_indices]

            if not self.population: # Si la población se vació por alguna razón
                print("Error: Población vacía. Deteniendo el AG.")
                break # Salir del bucle de generaciones

            parents = self.selection_func(
                self.population,
                fitness_values,
                self.num_parents_mating,
                **self.selection_args
            )

            num_offspring_to_generate = self.population_size - self.keep_elitism_count
            offspring_population = []

            if not parents or (len(parents) < 2 and self.crossover_func is not None):
                parent_idx = 0
                while len(offspring_population) < num_offspring_to_generate:
                    # Si parents está vacío, tomar de la población actual aleatoriamente
                    source_pop_for_cloning = parents if parents else self.population
                    if not source_pop_for_cloning: break # No hay de dónde clonar

                    cloned_parent = source_pop_for_cloning[parent_idx % len(source_pop_for_cloning)].copy()
                    offspring_population.append(cloned_parent)
                    parent_idx +=1
            else:
                parent_idx = 0
                while len(offspring_population) < num_offspring_to_generate:
                    p1 = parents[parent_idx % len(parents)]
                    p2 = parents[(parent_idx + 1) % len(parents)]

                    if random.random() < self.crossover_probability:
                        offspring1, offspring2 = self.crossover_func(p1, p2, **self.crossover_args)
                        offspring_population.append(offspring1)
                        if len(offspring_population) < num_offspring_to_generate:
                            offspring_population.append(offspring2)
                    else:
                        offspring_population.append(p1.copy())
                        if len(offspring_population) < num_offspring_to_generate:
                             offspring_population.append(p2.copy())
                    parent_idx = (parent_idx + 2) # Avanza sin % aquí, se controla con el len(parents) en el acceso
                    if parent_idx >= len(parents): # Ciclar manualmente los padres si se acaban
                        parent_idx = 0


            mutated_offspring_population = []
            for individual in offspring_population:
                mutated_individual = self.mutation_func(individual, **self.mutation_args)
                mutated_offspring_population.append(mutated_individual)
            
            self.population = elite_individuals + mutated_offspring_population[:num_offspring_to_generate]

        print("\nOptimización Finalizada.")
        print(f"Mejor fitness global encontrado: {self.best_fitness_overall:.4f}")
        return self.best_solution_overall, self.best_fitness_overall


    def plot_fitness(self):
        # (Sin cambios)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,6))
            plt.plot(self.best_solutions_fitness, marker='o', linestyle='-', markersize=4)
            plt.title("Evolución del Fitness por Generación")
            plt.xlabel("Generación")
            plt.ylabel("Mejor Fitness")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib no está instalado. No se puede graficar. Instala con 'pip install pysag[plot]' o 'pip install matplotlib'")