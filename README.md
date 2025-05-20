# PySAG-UdeC

Una librería simple y educativa de Algoritmos Genéticos en Python.

## Desarrolladores

**John Sebastián Galindo Hernández**

**Miguel Ángel Moreno Beltrán**

## Estado del Proyecto

Actualmente PySAG-UdeC se encuentra en la fase de desarrollo.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Características

* Framework básico para Algoritmos Genéticos.
* Permite funciones de fitness personalizadas.
* Operadores genéticos intercambiables (Selección, Cruce, Mutación).
* Optimización de funciones con Numba donde es aplicable.
* Manejo de excepciones personalizado.
* Capacidad para graficar la evolución del fitness.

## Instalación

Puedes instalar PySAG-UdeC usando pip (una vez que sea publicado oficialmente en PyPI):

```bash
pip install PySAG-UdeC
```

Para desarrollo se puede clonar el repositorio y realizar la instalación para desarrollo:

```bash
git clone https://github.com/SebasGalindo/PySAG-UdeC.git
cd PySAG-UdeC
pip install -e .[dev]
```

-----

## 📖 Documentación de la API

La librería PySAG-UdeC ofrece un conjunto de módulos para construir y ejecutar algoritmos genéticos.

### 🧬 Clase Principal `GA`

La clase `GA` es el núcleo de la librería y gestiona el flujo del algoritmo genético.

**Parámetros Principales:**

  * `fitness_func (Callable)`: Función que evalúa el fitness de un individuo.
  * `num_genes (int)`: Número de genes en cada cromosoma.
  * `population_size (int)`: Tamaño de la población (por defecto: 50).
  * `num_generations (int)`: Número de generaciones a ejecutar (por defecto: 100).
  * `num_parents_mating (int)`: Número de individuos a seleccionar como padres (por defecto: 10).
  * `initial_population_func (Callable)`: Función para crear la población inicial (por defecto: `initialization.init_random_uniform`).
  * `initial_pop_args (Optional[Dict])`: Argumentos para la función de inicialización.
  * `selection_func (Callable)`: Función para seleccionar padres (por defecto: `selection.selection_roulette_wheel`).
  * `selection_args (Optional[Dict])`: Argumentos para la función de selección.
  * `crossover_func (Optional[Callable])`: Función para el cruce (por defecto: `crossover.crossover_single_point`). Puede ser `None`.
  * `crossover_args (Optional[Dict])`: Argumentos para la función de cruce.
  * `crossover_probability (float)`: Probabilidad de cruce (0.0-1.0, por defecto: 0.9).
  * `mutation_func (Callable)`: Función para la mutación (por defecto: `mutation.mutation_random_gene_uniform`).
  * `mutation_args (Optional[Dict])`: Argumentos para la función de mutación.
  * `keep_elitism_percentage (float)`: Porcentaje de los mejores individuos a pasar a la siguiente generación (0.0-1.0, por defecto: 0.1).
  * `random_seed (Optional[int])`: Semilla para reproducibilidad (por defecto: `None`).

**Métodos Principales:**

  * `run() -> Tuple[Optional[IndividualType], float]`: Ejecuta el algoritmo genético y devuelve la mejor solución y su fitness.
  * `plot_fitness(save_path: Optional[str] = None)`: Grafica la evolución del mejor fitness por generación.

-----

### 🚀 Módulo de Inicialización (`PySAG.initialization`)

Este módulo provee funciones para crear la población inicial.

  * **`init_random_uniform(pop_size, chromosome_length, low, high, dtype=np.float64)`**: Genera una población con genes de valor real o entero muestreados de una distribución uniforme.
      * `dtype`: Puede ser `np.float32`, `np.float64`, `np.int32`, `np.int64`, `np.int_`.
  * **`init_random_binary(pop_size, chromosome_length, p_one=0.5)`**: Genera una población con genes binarios (0 o 1). `p_one` es la probabilidad de que un gen sea 1.
  * **`init_random_permutation(pop_size, chromosome_length)`**: Genera una población donde cada individuo es una permutación de enteros de `0` a `chromosome_length - 1`.

-----

### 👍 Módulo de Selección (`PySAG.selection`)

Este módulo contiene métodos para seleccionar individuos para la reproducción.

  * **`selection_roulette_wheel(population, fitness_values, num_parents)`**: Selección proporcional al fitness (los fitness deben ser no negativos).
  * **`selection_tournament(population, fitness_values, num_parents, tournament_size=3)`**: Se eligen `tournament_size` individuos al azar, y el mejor de ellos se convierte en padre.
  * **`selection_rank(population, fitness_values, num_parents)`**: Selección basada en el ranking de fitness de los individuos. La probabilidad de selección es proporcional al rango.
  * **`selection_stochastic_universal_sampling(SUS)(population, fitness_values, num_parents)`**: Variante de la ruleta que reduce el azar en la asignación de cupos (los fitness deben ser no negativos).
  * **`selection_random(population, fitness_values, num_parents)`**: Selección aleatoria simple de individuos, no considera el fitness.

-----

### ↔️ Módulo de Cruce (`PySAG.crossover`)

Este módulo implementa operadores de cruce.

  * **`crossover_single_point(parent1, parent2)`**

      * **Descripción**: Cruce de un solo punto. Se elige un punto de corte aleatorio. Los hijos intercambian el material genético después de ese punto.
      * **Ejemplo**:
          * `parent1 = np.array([0,0,1,1])`
          * `parent2 = np.array([1,1,0,0])`
          * Si el punto de corte (aleatorio) es `2` (entre el índice 1 y 2):
              * `child1` toma `[0,0]` de `parent1` y `[0,0]` de `parent2` -\> `[0,0,0,0]`
              * `child2` toma `[1,1]` de `parent2` y `[1,1]` de `parent1` -\> `[1,1,1,1]`
          * **Retorna (un posible resultado)**: `(np.array([0,0,0,0]), np.array([1,1,1,1]))`

  * **`crossover_two_points(parent1, parent2)`**

      * **Descripción**: Cruce de dos puntos. Se eligen dos puntos de corte aleatorios. Los hijos intercambian el material genético entre esos dos puntos.
      * **Ejemplo**:
          * `parent1 = np.array([0,0,1,1,0,0])`
          * `parent2 = np.array([1,1,0,0,1,1])`
          * Si los puntos de corte (aleatorios) son `2` y `4`:
              * `child1` toma `[0,0]` de `parent1`, `[0,0]` de `parent2`, `[0,0]` de `parent1` -\> `[0,0,0,0,0,0]`
              * `child2` toma `[1,1]` de `parent2`, `[1,1]` de `parent1`, `[1,1]` de `parent2` -\> `[1,1,1,1,1,1]`
          * **Retorna (un posible resultado)**: `(np.array([0,0,0,0,0,0]), np.array([1,1,1,1,1,1]))`

  * **`crossover_uniform(parent1, parent2, mix_probability=0.5)`**

      * **Descripción**: Cruce uniforme. Para cada gen, se decide con `mix_probability` si los genes de los padres se intercambian.
      * **Ejemplo**:
          * `parent1 = np.array([0,0,1,1])`
          * `parent2 = np.array([1,1,0,0])`
          * `mix_probability = 0.5`
          * Un posible resultado (el intercambio es aleatorio por gen):
              * Gen 0: no intercambia (`child1[0]=0, child2[0]=1`)
              * Gen 1: intercambia (`child1[1]=1, child2[1]=0`)
              * Gen 2: no intercambia (`child1[2]=1, child2[2]=0`)
              * Gen 3: intercambia (`child1[3]=0, child2[3]=1`)
              * `child1 = [0,1,1,0]`
              * `child2 = [1,0,0,1]`
          * **Retorna (un posible resultado)**: `(np.array([0,1,1,0]), np.array([1,0,0,1]))`

  * **`crossover_arithmetic(parent1, parent2, alpha=0.5)`**

      * **Descripción**: Cruce aritmético para valores numéricos. `child1 = alpha*p1 + (1-alpha)*p2`, `child2 = (1-alpha)*p1 + alpha*p2`.
      * **Ejemplo**:
          * `parent1 = np.array([1.0, 2.0, 10.0])`
          * `parent2 = np.array([4.0, 6.0, 0.0])`
          * `alpha = 0.5`
              * `child1 = 0.5*parent1 + 0.5*parent2 = [0.5, 1.0, 5.0] + [2.0, 3.0, 0.0] = [2.5, 4.0, 5.0]`
              * `child2 = 0.5*parent2 + 0.5*parent1 = [2.0, 3.0, 0.0] + [0.5, 1.0, 5.0] = [2.5, 4.0, 5.0]`
          * **Retorna**: `(np.array([2.5, 4.0, 5.0]), np.array([2.5, 4.0, 5.0]))`

  * **`crossover_order_ox1(parent1, parent2)`**

      * **Descripción**: Cruce de orden (OX1) para permutaciones. Un segmento de `parent1` se copia a `child1`. El resto de `child1` se llena con genes de `parent2` en orden, omitiendo los ya presentes. Proceso similar para `child2`.
      * **Ejemplo**:
          * `parent1 = np.array([1, 2, 3, 4, 5])`
          * `parent2 = np.array([5, 4, 1, 2, 3])`
          * Si los puntos de corte (aleatorios) para el segmento de `parent1` son `1` y `3` (segmento `[2,3,4]` de `parent1`):
              * `child1` toma `[2,3,4]` de `parent1`. `child1 = [_, 2, 3, 4, _]` (donde `_` son posiciones a llenar)
              * Elementos restantes de `parent2` en orden, omitiendo `2,3,4`: `5, 1`.
              * Llenando `child1` desde la posición `end` (índice 4) y envolviendo: `child1[4] = 5`, `child1[0] = 1`.
              * `child1` final: `[1, 2, 3, 4, 5]` (En este caso particular, debido a la naturaleza de los padres y los puntos de corte, `child1` podría terminar siendo igual a `parent1`. El proceso es correcto, pero el resultado puede variar significativamente con otros inputs/puntos.)
              * Para `child2`, si el segmento de `parent2` (índices 1 a 3) es `[4,1,2]`:
              * `child2` toma `[4,1,2]` de `parent2`. `child2 = [_, 4, 1, 2, _]`
              * Elementos restantes de `parent1` en orden, omitiendo `4,1,2`: `3, 5`.
              * `child2` final: `[3, 4, 1, 2, 5]`
          * **Retorna (un posible resultado para `child1` y `child2`)**: `(np.array([1,2,3,4,5]), np.array([3,4,1,2,5]))` (El resultado exacto depende de los puntos aleatorios)

-----

### 🔄 Módulo de Mutación (`PySAG.mutation`)

Este módulo proporciona operadores de mutación.

  * **`mutation_bit_flip(individual, mutation_rate=0.01)`**

      * **Descripción**: Invierte bits aleatorios en un individuo binario. Cada bit tiene una probabilidad `mutation_rate` de ser invertido.
      * **Ejemplo**:
          * `individual = np.array([0,1,0,1])`
          * `mutation_rate = 0.5`
          * Un posible resultado (cada bit muta con probabilidad 0.5):
              * Bit 0 (0) no muta.
              * Bit 1 (1) muta a 0.
              * Bit 2 (0) muta a 1.
              * Bit 3 (1) no muta.
              * `mutated_individual = [0,0,1,1]`
          * **Retorna (un posible resultado)**: `np.array([0,0,1,1])`

  * **`mutation_random_gene_uniform(individual, gene_low, gene_high, mutation_rate=0.01)`**

      * **Descripción**: Reemplaza genes con valores de una distribución uniforme entre `gene_low` y `gene_high`. Cada gen tiene `mutation_rate` de probabilidad de ser mutado.
      * **Ejemplo (enteros)**:
          * `individual = np.array([10, 20, 30, 40])`
          * `gene_low = 0`, `gene_high = 5` (inclusive para enteros)
          * `mutation_rate = 0.5`
          * Un posible resultado:
              * Gen 0 (10) muta (ej. a 3).
              * Gen 1 (20) no muta.
              * Gen 2 (30) muta (ej. a 1).
              * Gen 3 (40) no muta.
              * `mutated_individual = [3, 20, 1, 40]`
          * **Retorna (un posible resultado)**: `np.array([3, 20, 1, 40])`

  * **`mutation_gaussian(individual, mu=0.0, sigma=1.0, mutation_rate=0.01, clip_low=None, clip_high=None)`**

      * **Descripción**: Añade ruido gaussiano (N(mu, sigma)) a genes numéricos. Cada gen tiene `mutation_rate` de probabilidad de ser mutado. Los valores pueden ser recortados.
      * **Ejemplo**:
          * `individual = np.array([1.0, 2.5, 3.0])`
          * `mu = 0.0`, `sigma = 0.1`, `mutation_rate = 0.6`
          * Un posible resultado:
              * Gen 0 (1.0) muta: `1.0 + ruido` (ej. `ruido = 0.05`) -\> `1.05`
              * Gen 1 (2.5) no muta.
              * Gen 2 (3.0) muta: `3.0 + ruido` (ej. `ruido = -0.02`) -\> `2.98`
              * `mutated_individual = [1.05, 2.5, 2.98]`
          * **Retorna (un posible resultado)**: `np.array([1.05, 2.5, 2.98])`

  * **`mutation_swap(individual, mutation_rate=0.01)`**

      * **Descripción**: Intercambia dos genes aleatorios en el individuo. La operación ocurre con `mutation_rate` de probabilidad.
      * **Ejemplo**:
          * `individual = np.array([1,2,3,4,5])`
          * `mutation_rate = 1.0` (para asegurar que ocurra la mutación para el ejemplo)
          * Si los índices aleatorios elegidos son `0` y `3`:
              * `mutated_individual = [4,2,3,1,5]` (el gen en la posición 0 y 3 se intercambian)
          * **Retorna (un posible resultado)**: `np.array([4,2,3,1,5])`

  * **`mutation_inversion(individual, mutation_rate=0.01)`**

      * **Descripción**: Invierte un segmento aleatorio del cromosoma. La operación ocurre con `mutation_rate` de probabilidad.
      * **Ejemplo**:
          * `individual = np.array([1,2,3,4,5,6])`
          * `mutation_rate = 1.0` (para asegurar que ocurra la mutación para el ejemplo)
          * Si los índices aleatorios para el segmento son `1` y `4` (segmento `[2,3,4,5]`):
              * El segmento `[2,3,4,5]` se invierte a `[5,4,3,2]`
              * `mutated_individual = [1,5,4,3,2,6]`
          * **Retorna (un posible resultado)**: `np.array([1,5,4,3,2,6])`

-----

### 💡 Ejemplo de Uso: Maximización de una Función Matemática

A continuación, se muestra un ejemplo básico de cómo utilizar la clase `GA` para maximizar la función $f(x, y) = \\sin(x) \\cdot \\cos(y) + (x+y)/10$.

```python
import numpy as np
from PySAG import GA, crossover, initialization, mutation, selection

# 1. Definir la función de Fitness
GENE_LOW = -10.0
GENE_HIGH = 10.0
NUM_GENES = 2  # Dos variables: x, y

def fitness_function(individual: np.ndarray) -> float:
    """
    Función de fitness para el problema de maximización de una función matemática.
    f(x, y) = sin(x) * cos(y) + (x+y)/10
    """
    if len(individual) != NUM_GENES: #
        raise ValueError(f"El individuo debe tener {NUM_GENES} genes.") #
    x = individual[0] #
    y = individual[1] #
    x = np.clip(x, GENE_LOW, GENE_HIGH) #
    y = np.clip(y, GENE_LOW, GENE_HIGH) #
    return np.sin(x) * np.cos(y) + (x + y) / 10.0 #

# 2. Configurar y Instanciar la clase GA
population_size = 100 #
num_generations = 150 #
num_parents_mating = 20 #
crossover_prob = 0.85 #
elitism_percentage = 0.05 #
mutation_rate_for_gaussian = 0.1 #

print("Configurando el Algoritmo Genético para maximizar f(x,y)...") #

ga_instance_math = GA(
    fitness_func=fitness_function, #
    num_genes=NUM_GENES, #
    population_size=population_size, #
    num_generations=num_generations, #
    num_parents_mating=num_parents_mating, #
    initial_population_func=initialization.init_random_uniform, #
    initial_pop_args={"low": GENE_LOW, "high": GENE_HIGH, "dtype": np.float64}, #
    selection_func=selection.selection_tournament, #
    selection_args={"tournament_size": 5}, #
    crossover_func=crossover.crossover_uniform, #
    crossover_args={"mix_probability": 0.5}, #
    crossover_probability=crossover_prob, #
    mutation_func=mutation.mutation_gaussian, #
    mutation_args={
        "mu": 0.0, #
        "sigma": 0.5, #
        "mutation_rate": mutation_rate_for_gaussian, #
        "clip_low": GENE_LOW,
        "clip_high": GENE_HIGH,
    },
    keep_elitism_percentage=elitism_percentage, #
    random_seed=42, #
)

# 3. Ejecutar el AG
print("Ejecutando el Algoritmo Genético...") #
best_solution, best_fitness = ga_instance_math.run() #

# 4. Mostrar Resultados
if best_solution is not None: #
    print(f"\nMejor solución encontrada: {best_solution}") #
    print(f"Valor de la función (fitness): {best_fitness:.6f}") #
    recalculated_fitness = fitness_function(best_solution) #
    print(f"Fitness recalculado para la mejor solución: {recalculated_fitness:.6f}") #
    ga_instance_math.plot_fitness(save_path="math_function_maximization_fitness.png") #
else:
    print("No se encontró una solución.") #

print("\nEjemplo de maximización de función matemática completado.") #

```

*Extracto del ejemplo `example_math_function_maximization.py`. Los argumentos `gene_low` y `gene_high` en `mutation_args` para `mutation_gaussian` han sido renombrados a `clip_low` y `clip_high` para consistencia con la documentación de la función.*

-----

## ⚙️ Herramientas de Desarrollo

Este proyecto utiliza las siguientes herramientas para asegurar la calidad del código:

  * **Black**: Para formateo de código.
  * **isort**: Para organizar las importaciones.
  * **Flake8**: Para el linting de código, con los plugins:
      * `flake8-docstrings`
      * `flake8-import-order`

Configuradas a través de `.pre-commit-config.yaml`.

-----

## 📜 Licencia

Este proyecto está bajo la Licencia MIT.
[](https://opensource.org/licenses/MIT)