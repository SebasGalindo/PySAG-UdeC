"""
Clase principal que implementa el Algoritmo Genético (AG).

Esta clase encapsula la lógica de un algoritmo genético estándar, permitiendo
la configuración de sus componentes clave como la inicialización, selección,
cruce y mutación a través de funciones personalizadas.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import crossover as default_crossover
from . import initialization as default_init
from . import mutation as default_mutation
from . import selection as default_selection
from .exceptions import (
    GeneticAlgorithmError,
    InitializationError,
    validate_parameter,
)

# Tipos para mayor claridad
IndividualType = NDArray[Any]  # Un individuo es un array NumPy
PopulationType = List[IndividualType]  # La población es una lista de individuos
FitnessFunctionType = Callable[[IndividualType], float]
OperatorFunctionType = Callable[..., Any]


class GA:
    """
    Clase principal que implementa el Algoritmo Genético (AG).

    Permite configurar cada etapa del AG (inicialización, evaluación, selección,
    cruce y mutación) y ejecutar el proceso evolutivo.
    """

    def __init__(
        self,
        fitness_func: FitnessFunctionType,
        num_genes: int,
        population_size: int = 50,
        num_generations: int = 100,
        num_parents_mating: int = 10,
        initial_population_func: OperatorFunctionType = (
            default_init.init_random_uniform
        ),
        initial_pop_args: Optional[Dict[str, Any]] = None,
        selection_func: OperatorFunctionType = (
            default_selection.selection_roulette_wheel
        ),
        selection_args: Optional[Dict[str, Any]] = None,
        crossover_func: Optional[
            OperatorFunctionType
        ] = default_crossover.crossover_single_point,
        crossover_args: Optional[Dict[str, Any]] = None,
        crossover_probability: float = 0.9,
        mutation_func: OperatorFunctionType = (
            default_mutation.mutation_random_gene_uniform
        ),
        mutation_args: Optional[Dict[str, Any]] = None,
        keep_elitism_percentage: float = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa el algoritmo genético con los parámetros dados.

        Args:
            fitness_func:
            Función que toma un individuo y devuelve su valor de fitness (float).

            num_genes:
            Número de genes en cada individuo.

            population_size:
            Tamaño de la población. Por defecto es 50.

            num_generations:
            Número de generaciones a ejecutar. Por defecto es 100.

            num_parents_mating:
            Número de individuos a seleccionar como padres
            para el cruce. Por defecto es 10.

            initial_population_func:
            Función para crear la población inicial.
            Por defecto es `init_random_uniform`.

            initial_pop_args:
            Argumentos adicionales para `initial_population_func`.

            selection_func:
            Función para seleccionar padres.
            Por defecto es `selection_roulette_wheel`.

            selection_args:
            Argumentos adicionales para `selection_func`.

            crossover_func:
            Función para realizar el cruce. Puede ser None para no usar cruce.
            Por defecto es `crossover_single_point`.

            crossover_args:
            Argumentos adicionales para `crossover_func`.

            crossover_probability:
            Probabilidad de que ocurra el cruce.
            Debe estar entre 0 y 1. Por defecto es 0.9.

            mutation_func:
            Función para realizar la mutación.
            Por defecto es `mutation_random_gene_uniform`.

            mutation_args:
            Argumentos adicionales para `mutation_func`.

            keep_elitism_percentage:
            Porcentaje de los mejores individuos de la
            generación actual que se pasan directamente
            a la siguiente. Debe estar entre 0 y 1.
            Por defecto es 0.1.

            random_seed:
            Semilla opcional para el generador de números aleatorios
            de NumPy y Python, para reproducibilidad.

        Raises:
            TypeValidationError: Si alguna de las funciones de operador no es callable.
            ParameterError: Si alguno de los parámetros numéricos está fuera de rango.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Validación de parámetros
        validate_parameter(
            fitness_func, "fitness_func", expected_type=Callable
        )  # type: ignore
        validate_parameter(num_genes, "num_genes", expected_type=int, min_val=1)
        validate_parameter(
            population_size, "population_size", expected_type=int, min_val=1
        )
        validate_parameter(
            num_generations, "num_generations", expected_type=int, min_val=0
        )
        validate_parameter(
            num_parents_mating, "num_parents_mating", expected_type=int, min_val=1
        )

        validate_parameter(
            initial_population_func,
            "initial_population_func",
            expected_type=Callable,
        )  # type: ignore
        validate_parameter(
            selection_func, "selection_func", expected_type=Callable
        )  # type: ignore
        if crossover_func is not None:
            validate_parameter(
                crossover_func, "crossover_func", expected_type=Callable
            )  # type: ignore
        validate_parameter(
            mutation_func, "mutation_func", expected_type=Callable
        )  # type: ignore

        validate_parameter(
            crossover_probability,
            "crossover_probability",
            expected_type=(float, int),
            min_val=0.0,
            max_val=1.0,
        )
        validate_parameter(
            keep_elitism_percentage,
            "keep_elitism_percentage",
            expected_type=(float, int),
            min_val=0.0,
            max_val=1.0,
        )

        self.fitness_func: FitnessFunctionType = fitness_func
        self.num_genes: int = num_genes
        self.population_size: int = population_size
        self.num_generations: int = num_generations
        self.num_parents_mating: int = num_parents_mating

        self.initial_population_func: OperatorFunctionType = initial_population_func
        self.initial_pop_args: Dict[str, Any] = (
            initial_pop_args if initial_pop_args is not None else {}
        )

        self.selection_func: OperatorFunctionType = selection_func
        self.selection_args: Dict[str, Any] = (
            selection_args if selection_args is not None else {}
        )

        self.crossover_func: Optional[OperatorFunctionType] = crossover_func
        self.crossover_args: Dict[str, Any] = (
            crossover_args if crossover_args is not None else {}
        )
        self.crossover_probability: float = crossover_probability

        self.mutation_func: OperatorFunctionType = mutation_func
        self.mutation_args: Dict[str, Any] = (
            mutation_args if mutation_args is not None else {}
        )

        elitism_count_float = self.population_size * keep_elitism_percentage
        self.keep_elitism_count: int = int(elitism_count_float)
        if (
            self.keep_elitism_count == 0
            and elitism_count_float > 0
            and population_size > 0
        ):
            pass  # Por ahora, se confía en la conversión a int directa.

        self.population: Optional[PopulationType] = None
        self.best_solutions_fitness: List[float] = []
        self.best_solution_overall: Optional[IndividualType] = None
        self.best_fitness_overall: float = -np.inf  # Asume maximización

    def _initialize_population(self) -> None:
        """
        Inicializa la población utilizando la función y argumentos especificados.

        Este método utiliza los parámetros `initial_population_func` y
        `initial_pop_args` definidos durante la instanciación de la clase GA.
        Modifica `self.population`.
        """
        self.population = self.initial_population_func(
            pop_size=self.population_size,
            chromosome_length=self.num_genes,
            **self.initial_pop_args,  # Aquí deben estar 'low', 'high', 'dtype'
        )
        if not isinstance(self.population, list) or not all(
            isinstance(ind, np.ndarray) for ind in self.population
        ):
            str_error = """La función de inicialización debe devolver
            una lista de arrays NumPy (individuos)."""
            raise InitializationError(str_error)  # type: ignore
        if len(self.population) != self.population_size:
            str_error = """La población inicializada
            tiene tamaño {len(self.population)}, """
            str_error += f"pero se esperaba {self.population_size}."
            raise InitializationError(str_error)  # type: ignore

    def _calculate_population_fitness(self) -> NDArray[np.float64]:
        """
        Calcula la aptitud de todos los individuos en la población actual.

        Returns:
            Un array NumPy con los valores de fitness de la población.

        Raises:
            GeneticAlgorithmError: Si la población no ha sido inicializada.
        """
        if self.population is None:
            str_error = """La población no ha sido
            inicializada antes de calcular el fitness."""
            raise GeneticAlgorithmError(str_error)  # type: ignore

        fitness_values: List[float] = []
        for individual in self.population:
            try:
                fitness_values.append(self.fitness_func(individual))
            except Exception as e:
                raise GeneticAlgorithmError(  # type: ignore
                    f"Error al calcular el fitness para el individuo {individual}: {e}",
                    details={"original_error": str(e)},  # type: ignore
                ) from e
        return np.array(fitness_values, dtype=np.float64)

    def run(self) -> Tuple[Optional[IndividualType], float]:
        """
        Ejecuta el algoritmo genético a través de las generaciones.

        Returns:
            Una tupla con la mejor solución global encontrada (Individuo) y
            su valor de fitness (float).

        Raises:
            GeneticAlgorithmError: Si ocurren errores irrecuperables durante
                                   la ejecución del AG.
        """
        if self.population is None:
            try:
                self._initialize_population()
            except InitializationError as e:
                str_error = "Error de inicialización:"
                str_error += f" {e}"
                raise GeneticAlgorithmError(str_error) from e  # type: ignore

        if self.population is None:
            str_error = "Error crítico: La población sigue siendo None "
            str_error += "después de la inicialización."
            print(str_error)
            return None, -np.inf

        for generation in range(self.num_generations):
            try:
                fitness_values = self._calculate_population_fitness()
            except GeneticAlgorithmError as e:
                print(
                    f"Error en la generación {generation + 1} calculando fitness: {e}"
                )
                # Podría decidir terminar o intentar continuar si es recuperable
                raise  # Re-elevar por ahora

            current_best_fitness_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_fitness_idx]
            self.best_solutions_fitness.append(current_best_fitness)

            if current_best_fitness > self.best_fitness_overall:
                self.best_fitness_overall = current_best_fitness
                self.best_solution_overall = self.population[
                    current_best_fitness_idx
                ].copy()

            print(
                f"Generación {generation + 1}/{self.num_generations}: "
                f"Mejor Fitness = {self.best_fitness_overall:.4f} "
                f"(Actual: {current_best_fitness:.4f})"
            )

            # Elitismo
            elite_individuals: PopulationType = []
            if self.keep_elitism_count > 0 and self.population:
                # Ordenar por fitness descendente y tomar los 'k' mejores
                elite_indices = np.argsort(fitness_values)[-self.keep_elitism_count :]
                elite_individuals = [self.population[i].copy() for i in elite_indices]

            # Selección de Padres
            try:
                # Asegurar que num_parents_mating no exceda population_size
                actual_num_parents_mating = min(
                    self.num_parents_mating, len(self.population)
                )
                if actual_num_parents_mating < 2 and self.crossover_func is not None:
                    str_error = "Advertencia: No hay suficientes padres"
                    str_error += f" ({actual_num_parents_mating}) para el cruce. "
                    str_error += "Se omitirá el cruce."
                    print(str_error)

                parents = self.selection_func(
                    self.population,
                    fitness_values,
                    actual_num_parents_mating,  # Usar el valor ajustado
                    **self.selection_args,
                )
            except Exception as e:  # Captura errores de selección
                str_error = "Error durante la selección en la generación"
                str_error += f" {generation + 1}: {e}"
                raise GeneticAlgorithmError(str_error) from e  # type: ignore

            if not parents:  # Si la selección no devuelve padres
                print(
                    "Advertencia: La selección no devolvió padres en la generación"
                    f" {generation + 1}. "
                    "Rellenando descendencia con clones de la población actual."
                )
                # Como fallback
                # clonar de la población existente para mantener el tamaño
                parents = [
                    random.choice(self.population).copy()
                    for _ in range(actual_num_parents_mating)
                ]
                if (
                    not parents and self.population
                ):  # Si la población también está vacía, es un problema mayor
                    str_error = (
                        "La población está vacía y no se pueden seleccionar padres."
                    )
                    raise GeneticAlgorithmError(str_error)  # type: ignore

            # Generación de Descendencia (Cruce y Mutación)
            num_offspring_to_generate = self.population_size - len(elite_individuals)
            offspring_population: PopulationType = []

            # Condición para cruce: función de cruce definida y suficientes padres
            can_crossover = self.crossover_func is not None and len(parents) >= 2

            current_parent_idx = 0
            while len(offspring_population) < num_offspring_to_generate:
                if not parents:  # No hay padres de donde generar descendencia
                    if (
                        self.population
                    ):  # Si la población original aún existe, clonar de ahí
                        offspring_population.append(
                            random.choice(self.population).copy()
                        )
                        if len(offspring_population) >= num_offspring_to_generate:
                            break
                        continue
                    else:  # No hay forma de generar más individuos
                        str_error = "Error: No hay padres ni población "
                        str_error += "base para generar descendencia."
                        break

                p1 = parents[current_parent_idx % len(parents)]

                if can_crossover and random.random() < self.crossover_probability:
                    p2 = parents[
                        (current_parent_idx + 1) % len(parents)
                    ]  # Siguiente padre para cruce
                    try:
                        offspring1, offspring2 = self.crossover_func(
                            p1.copy(), p2.copy(), **self.crossover_args
                        )
                    except Exception as e:  # Captura errores de cruce
                        str_error = "Error durante el cruce en la generación"
                        str_error += f" {generation + 1}: {e}"
                        raise GeneticAlgorithmError(str_error) from e  # type: ignore

                    offspring_population.append(offspring1)
                    if len(offspring_population) < num_offspring_to_generate:
                        offspring_population.append(offspring2)
                    current_parent_idx += 2  # Avanzar dos padres
                else:  # Clonación (sin cruce o probabilidad no cumplida)
                    offspring_population.append(p1.copy())
                    current_parent_idx += 1  # Avanzar un padre

            # Mutación de la descendencia generada
            mutated_offspring_population: PopulationType = []
            for i in range(len(offspring_population)):
                individual_to_mutate = offspring_population[i]  # Ya es una copia
                try:
                    mutated_individual = self.mutation_func(
                        individual_to_mutate, **self.mutation_args  # Pasar la copia
                    )
                except Exception as e:  # Captura errores de mutación
                    str_error = "Error durante la mutación en la generación"
                    str_error += f" {generation + 1}: {e}"
                    raise GeneticAlgorithmError(str_error) from e  # type: ignore
                mutated_offspring_population.append(mutated_individual)

            # Formar la nueva población
            self.population = elite_individuals + mutated_offspring_population
            # Asegurar que la población tenga el tamaño correcto,
            # truncando si es necesario
            self.population = self.population[: self.population_size]

        print("\nOptimización Finalizada.")
        if self.best_solution_overall is not None:
            print(f"Mejor fitness global encontrado: {self.best_fitness_overall:.4f}")
        else:
            print(
                "No se encontró ninguna solución (la población podría haber colapsado)."
            )

        return self.best_solution_overall, self.best_fitness_overall

    def plot_fitness(self, save_path: Optional[str] = None) -> None:
        """
        Grafica la evolución del mejor fitness a lo largo de las generaciones.

        Utiliza Matplotlib para generar la gráfica. Si Matplotlib no está
        instalado, imprime un mensaje de advertencia.

        Args:
            save_path: Ruta opcional para guardar la gráfica como archivo de imagen.
                       Ej: "fitness_evolution.png". Si es None, solo muestra la gráfica.
        """
        if not self.best_solutions_fitness:
            print("No hay datos de fitness para graficar (ejecute el AG primero).")
            return

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(self.best_solutions_fitness) + 1),  # Generaciones desde 1
                self.best_solutions_fitness,
                marker="o",
                linestyle="-",
                markersize=4,
                color="b",
                label="Mejor Fitness por Generación",
            )
            plt.axhline(
                y=self.best_fitness_overall,
                color="r",
                linestyle="--",
                label=f"Mejor Fitness Global: {self.best_fitness_overall:.4f}",
            )

            plt.title("Evolución del Fitness por Generación")
            plt.xlabel("Generación")
            plt.ylabel("Mejor Fitness")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Gráfica de fitness guardada en: {save_path}")
            else:
                plt.show()
        except ImportError:
            print(
                "Matplotlib no está instalado. No se puede graficar. "
                "Instálalo con: pip install matplotlib"
            )
        except Exception as e:
            print(f"Ocurrió un error al graficar el fitness: {e}")
