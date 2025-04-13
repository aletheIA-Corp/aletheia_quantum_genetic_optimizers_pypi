from aletheia_quantum_genetic_optimizers.variability_explossion_methods import CrazyVariabilityExplossion
from aletheia_quantum_genetic_optimizers.reproduction_methods import Reproduction
from aletheia_quantum_genetic_optimizers.population_methods import Population
from aletheia_quantum_genetic_optimizers.mutation_methods import Mutation
from aletheia_quantum_genetic_optimizers.individuals import Individual
from aletheia_quantum_genetic_optimizers.tournament_methods import *

from typing import Callable, Dict, Literal, List
from info_tools import InfoTools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class GenethicOptimizer:
    def __init__(self,
                 bounds_dict: Dict,
                 num_generations: int,
                 num_individuals: int,
                 max_qubits: int,
                 objective_function: Callable,
                 metric_to_optimize: Literal['accuracy', 'recall', 'specificity', 'f1', 'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other', 'r2'],
                 problem_restrictions: Literal['bound_restricted', 'totally_restricted'] = "bound_restricted",
                 return_to_origin: Literal['return_to_origin', 'no_return'] | None = None,
                 problem_type: Literal["minimize", "maximize"] = "minimize",
                 tournament_method: Literal["ea_simple"] = "ea_simple",
                 podium_size: int = 3,
                 mutate_probability: float = 0.25,
                 mutate_gen_probability: float = 0.2,
                 mutation_policy: Literal['soft', 'normal', 'hard'] = 'normal',
                 verbose: bool = True,
                 early_stopping_generations: Literal['gradient'] | int = 'gradient',
                 variability_explossion_mode: Literal['crazy'] = 'crazy',
                 variability_round_decimals: int = 3,
                 randomness_quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 randomness_service: Literal["aer", "ibm"] = "aer",
                 optimization_quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 optimization_service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy",
                 element_matrix: Dict[str, tuple] | None = None
                 ):
        """
        Clase principal para crear un algoritmo genético cuántico con el objetivo de resolver problemas de optimización
        combinatoria con distintos valores entre bounds y problemas de optimización combinatoria con valores concretos.

        :param bounds_dict: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        :param num_generations: Numero de generaciones que se van a ejecutar
        :param num_individuals: Numero de Individuos iniciales que se van a generar
        :param max_qubits: Numero máximo de qubits a emplear para reproducir individuos (define el numero entero y la parte decimal de los numeros enteros y flotantes que se quieren generar)
        :param objective_function: Función objetivo que se va a emplear para puntuar a cada individuo (debe retornar un float)
        :param metric_to_optimize: Metrica que se quiere optimizar ['accuracy', 'recall', 'specificity', 'f1',
        'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other'] -> other significa cualquier otra genérica.
        Por ejemplo, se puede utilizar other para un problema de optimización de tipo viajante de comercio.
        :param problem_restrictions: ['bound_restricted', 'totally_restricted'] Restricciones que se van a aplicar a la hora de crear individuos, reprocirlos y mutarlos
        :param return_to_origin: [Literal['return_to_origin', 'no_return'] | None] En caso de problemas totally_restricted es necesario saber si el problema termina en el origen o no es necesario que suceda esto
        :param problem_type: [minimize, maximize] Seleccionar si se quiere minimizar o maximizar el resultado de la función objetivo. Por ejemplo si usamos un MAE es minimizar,
         un Accuracy sería maximizar.
        :param tournament_method: [easimple, .....] Elegir el tipo de torneo para seleccionar los individuos que se van a reproducir.
        :param podium_size: Cantidad de individuos de la muestra que van a competir para elegir al mejor. Por ejemplo, si el valor es 3, se escogen iterativamente 3 individuos
        al azar y se selecciona al mejor. Este proceso finaliza cuando ya no quedan más individuos y todos han sido seleccionados o deshechados.
        :param mutate_probability:Tambien conocido como indpb ∈[0, 1]. Probabilidad de mutar que tiene cada gen. Una probabilidad de 0, implica que nunca hay mutación,
        una probabilidad de 1 implica que siempre hay mutacion.
        :param mutate_gen_probability: [float] Probabilidad de mute un gen
        :param mutation_policy: Literal['soft', 'normal', 'hard'] Política de mutación (liviana, estandar y agresiva),
        :param verbose: Variable que define si se pinta información extra en consola y si se generan los graficos de los circuitos cuánticos.
        :param early_stopping_generations: Cantidad de generaciones que van a transcurrir para que en caso de repetirse la moda del fitness, se active el modo variability_explosion
        :param variability_explossion_mode: Modo de explosion de variabilidad, es decir, que se va a hacer para intentar salir de un minimo local establecido
        :param variability_round_decimals: Decimales a los que redondear las estadisticas de cálculo de moda necesarias para la explosion de variabilidad. Por ejemplo,
        en un caso de uso que busque accuracy, podría ser con 2 o 3 decimales. para casos de uso que contengan números muy bajos, habría que agregar más.
        :param randomness_quantum_technology. Literal["simulator", "quantum_machine"]. Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro service. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.  Se aplica al cálculo de numeros random (primera generación).
        :param randomness_service: ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica. Se aplica al cálculo de numeros random (primera generación).
        :param optimization_quantum_technology. Literal["simulator", "quantum_machine"]. Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro service. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology. Se aplica al proceso de optimización.
        :param optimization_service. ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica. Se aplica al proceso de optimización.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Literal["ibm_quantum", "ibm_cloud"] | None. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        :param element_matrix: Dict[str, tuple] | None = Matriz de distancia utilizada para los problemas de optimización combinatoria de tipo TSP.
        """

        # -- Almacenamos propiedades
        self.bounds_dict: Dict = bounds_dict
        self.num_generations: int = num_generations
        self.num_individuals: int = num_individuals
        self.max_qubits: int = max_qubits
        self.objective_function: Callable = objective_function
        self.metric_to_optimize: Literal['accuracy', 'recall', 'specificity', 'f1', 'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other', 'r2'] = metric_to_optimize
        self.problem_restrictions: Literal['bound_restricted', 'totally_restricted'] = problem_restrictions
        self.return_to_origin: Literal['return_to_origin', 'no_return'] | None = return_to_origin
        self.problem_type: Literal["minimize", "maximize"] = problem_type
        self.tournament_method: str = tournament_method
        self.podium_size: int = podium_size
        self.mutate_probability: float = mutate_probability
        self.mutate_gen_probability: float = mutate_gen_probability
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = mutation_policy
        self.verbose: bool = verbose
        self.early_stopping_generations: int = early_stopping_generations if isinstance(early_stopping_generations, int) else max(int(self.num_generations * 0.05), 3)
        self.early_stopping_generations_executed: bool = False
        self.early_stopping_generations_executed_counter: int = 0
        self.variability_round_decimals: int = variability_round_decimals
        self.randomness_quantum_technology: Literal["simulator", "quantum_machine"] = randomness_quantum_technology
        self.randomness_service: Literal["aer", "ibm"] = randomness_service
        self.optimization_quantum_technology: Literal["simulator", "quantum_machine"] = optimization_quantum_technology
        self.optimization_service: Literal["aer", "ibm"] = optimization_service
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine
        self.element_matrix: Dict[str, tuple] | None = element_matrix

        # -- MEJOR INDIVIDUAL
        self.best_individual_values: dict | None = None

        # <editor-fold desc="DEFINICIÓN DE VARIABLES GENERALES DEL ALGORITMO GENÉTICO CUÁNTICO">

        # -- Instanciamos info tools para los prints
        self.IT: InfoTools = InfoTools()

        self.IT.header_print("Se definen las variables generales del AletheIA Quantum Genetic Optimizer")

        # -- Definimos la variable que controla la variabilidad de los resultados de la función objetivo por generación
        self.variability_explosion_starts_in_generation: int | None = None

        # -- Instanciamos la clase GenethicTournamentMethods (clase que define el torno y lo almacenamos)
        self.IT.sub_intro_print("Instanciando el tipo de torneo...")
        self.GTM: Tournament = self.get_tournament_method(self.verbose)

        # -- Instanciamos la clase que controla que la variabilidad (sin cambios --> aplicamos mutación agresiva)
        self.IT.sub_intro_print("Instanciando la monitorización de variabilidad de resultados...")

        match variability_explossion_mode:
            case 'crazy':
                self.VEM: CrazyVariabilityExplossion = CrazyVariabilityExplossion(self.early_stopping_generations,
                                                                                  self.problem_type,
                                                                                  self.variability_round_decimals,
                                                                                  self.verbose)
            case _:
                self.VEM: CrazyVariabilityExplossion = CrazyVariabilityExplossion(self.early_stopping_generations,
                                                                                  self.problem_type,
                                                                                  self.variability_round_decimals,
                                                                                  self.verbose)

        self.IT.info_print("Instanciación de la monitorización de variabilidad de resultados realizada con éxito")

        # -- Validamos los inputs
        self.IT.sub_intro_print("Validando los inputs...")
        self.validate_input_parameters()
        self.IT.info_print("Validación de los inputs realizada con éxito")

        if self.verbose:
            self.IT.header_print(f"Bounds_dict y valores a combinar")
            for k, v in self.bounds_dict.items():
                self.IT.info_print(f"{k}: {v}")

        # </editor-fold>

        # <editor-fold desc="CREACIÓN Y EVALUACIÓN DE LA FUNCIÓN DE COSTE DE LA GENERACIÓN 0">

        self.IT.header_print("Creamos la población 0")

        # -- Creamos el objeto poblacion y la poblacion inicial
        self.population: population = Population(self.bounds_dict,
                                                 self.num_individuals,
                                                 self.problem_restrictions,
                                                 self.variability_round_decimals)

        # -- Creamos las listas de individuos que vamos a ir usando
        self.population.create_population(quantum_technology=self.randomness_quantum_technology,
                                          service=self.randomness_service,
                                          qm_api_key=self.qm_api_key,
                                          qm_connection_service=self.qm_connection_service,
                                          quantum_machine=self.quantum_machine,
                                          max_qubits=self.max_qubits,
                                          element_matrix=self.element_matrix)

        if self.verbose:

            self.IT.sub_intro_print("Individuos de la generación 0")

            for individual in self.population.populuation_dict[str(0)]:

                match self.problem_restrictions:

                    case "bound_restricted":
                       self.IT.info_print(f"INDIVIDUO: {individual.get_individual_id()} --> PROPIEDADES: {individual.get_individual_values()} --> PARAMETROS: {individual.get_vqc_parameters_values()}")

                    case "totally_restricted":
                        self.IT.info_print(f"INDIVIDUO: {individual.get_individual_id()} --> PROPIEDADES: {individual.get_individual_values()}")

        # -- Pasamos a cada individuo de la generacion 0 por la funcion de coste
        if self.verbose:
            self.IT.header_print(f"Generacion 0")
            self.IT.sub_intro_print("Ejecutando funcion objetivo en los individuos.....")

        for individual in self.population.populuation_dict[str(0)]:
            individual.add_or_update_variable("individual_fitness", self.objective_function(individual))

        if self.verbose:
            self.IT.info_print("Funcion objetivo ejecutada correctamente")
            self.print_generation_info(self.population.populuation_dict[str(0)], 0)

        # </editor-fold>

        # <editor-fold desc="GENERACIONES: REPRODUCCIÓN Y MUTACIONES GENÉTICAS">

        self.IT.header_print(f"Evolución genética de la población ---> ACTIVADA")

        # -- Iteramos por cada generación empezando en la 1
        for gen in range(1, self.num_generations):

            self.IT.header_print(f"Empezando con la generación {gen}")

            # <editor-fold desc="DESARROLLO DEL TORNEO">

            # -- Pintamos a los ganadores de la generación anterior
            if self.verbose:
                self.IT.sub_intro_print(f"Ganadores de la generacion {gen}")

            # -- Ejecutamos el torneo para obtener los padres ganadores en base a los individuos de la generacion anterior
            winners_list: List[Individual] = self.GTM.run_tournament(individuals_list=self.population.populuation_dict[str(gen - 1)])

            if self.verbose:
                for winner in winners_list:

                    if self.problem_type == "bounds_restricted":
                        self.IT.info_print(f"ID: {winner.get_individual_id()} --> "
                                           f"WINNER: {winner.get_individual_values()} --> "
                                           f"FITNESS: {winner.get_individual_fitness()} --> "
                                           f"PARAMETROS VQC: {winner.get_vqc_parameters_values()}")

                    else:
                        self.IT.info_print(f"ID: {winner.get_individual_id()} --> "
                                           f"WINNER: {winner.get_individual_values()} --> "
                                           f"FITNESS: {winner.get_individual_fitness()}")


            # </editor-fold>

            # <editor-fold desc="REPRODUCCIÓN CUÁNTICA DE NUEVOS INDIVIDUOS">

            self.IT.sub_intro_print("Etapa de reproducción cuántica de nuevos individuos")

            # -- Creamos los hijos y los agregamos a la lista de individuos
            children_list: List[Individual] = Reproduction(winners_list=winners_list,
                                                           number_of_children=self.num_individuals,
                                                           problem_restrictions=self.problem_restrictions,
                                                           problem_type=self.problem_type,
                                                           return_to_origin=self.return_to_origin,
                                                           metric_to_optimize=self.metric_to_optimize,
                                                           verbose=self.verbose).run_reproduction(quantum_technology=self.optimization_quantum_technology,
                                                                                                  service=self.optimization_service,
                                                                                                  qm_api_key=self.qm_api_key,
                                                                                                  qm_connection_service=self.qm_connection_service,
                                                                                                  quantum_machine=self.quantum_machine,
                                                                                                  generations_fitness_statistics_df=self.population.generations_fitness_statistics_df,
                                                                                                  max_qubits=self.max_qubits)



            # </editor-fold>

            # <editor-fold desc="EVALUACIÓN DE CAMBIO DE POLÍTICA DE MUTACIÓN Y EJECUCIÓN">

            self.IT.sub_intro_print("Etapa de evaluación de cambio de política de mutación y ejecución de mutaciones")

            # -- Evaluamos si el modelo ha quedado en un minimo local, en caso afirmativo le damos una explosion de variabilidad
            m_proba, m_gen_proba, m_policy, early_stopping_generations_executed = self.VEM.evaluate_early_stopping(self.population.generations_fitness_statistics_df)

            # -- En caso de que una de las variables retornadas por evaluate_early_stopping sea None, todas son None
            if m_proba is not None:
                self.mutate_probability = m_proba
                self.mutate_gen_probability = m_gen_proba
                self.mutation_policy = m_policy
                self.early_stopping_generations_executed = early_stopping_generations_executed

            # -- En caso de darse las condiciones, inicia el CRAZY MODE
            if self.early_stopping_generations_executed and self.mutation_policy == "hard":
                self.IT.header_print("CRAZY MODE ON", "light_red")
                self.variability_explosion_starts_in_generation = gen

            # -- Mutamos los individuos
            children_list = Mutation(children_list, self.mutate_probability, self.mutate_gen_probability, self.mutation_policy, self.problem_restrictions, self.num_generations).run_mutation()

            # -- Agregamos los individuos al diccionario de poblacion en su generacion correspondiente
            self.population.add_generation_population(children_list, gen)

            # </editor-fold>

            # <editor-fold desc="EJECUCIÓN DE FUNCIÓN OBJETIVO">

            # -- Pasamos a cada individuo de la gen=gen por la funcion de coste
            self.IT.sub_intro_print("Ejecutando funcion objetivo en los individuos.....")

            for individual in self.population.populuation_dict[str(gen)]:
                if not individual.get_individual_malformation():
                    individual.add_or_update_variable("individual_fitness", self.objective_function(individual))

            # </editor-fold>

            # <editor-fold desc="ESTADÍSTICA GENERAL DE LA GENERACIÓN">

            self.IT.sub_intro_print("Etapa estadística de la generación")
            self.print_generation_info(self.population.populuation_dict[str(gen)], gen)

            if self.verbose:

                best_fitness = None
                for gen_n, ind_list in self.population.populuation_dict.items():
                    best_gen_ind = sorted(
                        [ind for ind in self.population.populuation_dict[gen_n] if ind.get_individual_fitness() is not None],
                        key=lambda ind: ind.get_individual_fitness(),
                        reverse=self.problem_type == 'maximize'
                    )[0]

                    self.IT.info_print(f"Mejor ind gen: {gen_n}: {best_gen_ind.get_individual_values()} - Fitness: {best_gen_ind.get_individual_fitness()}")

                    if self.problem_type == "maximize":
                        if best_fitness is None:
                            best_fitness = best_gen_ind.get_individual_fitness()
                        else:
                            if best_fitness < best_gen_ind.get_individual_fitness():
                                best_fitness = best_gen_ind.get_individual_fitness()
                    else:
                        if best_fitness is None:
                            best_fitness = best_gen_ind.get_individual_fitness()
                        else:
                            if best_fitness > best_gen_ind.get_individual_fitness():
                                best_fitness = best_gen_ind.get_individual_fitness()

                self.IT.info_print(f"Mejor ind TOTAL: Fitness: {best_fitness}", "light_magenta")

            # -- Evaluamos si despues de la explosion de variabilidad, han transcurrido las generaciones de margen. En caso afirmativo, salimos del bucle
            if self.VEM.stop_genetic_iterations(self.population.generations_fitness_statistics_df):
                break

            # </editor-fold>

        self.IT.header_print("POBLACIÓN FINAL")

        match self.problem_restrictions:

            case "bound_restricted":

                # -- Iteramos por generacion
                for generation, pops in self.population.populuation_dict.items():

                    if self.verbose:
                        self.IT.sub_intro_print(f"GENERACION: {generation}")

                    if self.verbose:

                        for individual in pops:
                            self.IT.info_print(f"ID: {individual.get_individual_id()} --> "
                                               f"VALORES: {individual.get_individual_values()} --> "
                                               f"ANGULOS: {individual.get_vqc_parameters_values()} --> "
                                               f"FITNESS: {individual.get_individual_fitness()}")

                    # -- Mostramos el mejor individuo de la generación
                    if pops:

                        match self.problem_type:
                            case "maximize":
                                self.best_individual_values = max(pops, key=lambda ind: ind.get_individual_fitness())
                            case "minimize":
                                self.best_individual_values = min(pops, key=lambda ind: ind.get_individual_fitness())
                            case _:
                                self.best_individual_values = None  # por si se define otro tipo no soportado

                        if self.best_individual_values:
                            self.IT.info_print(
                                f"MEJOR INDIVIDUO GENERACION: {generation} --> "
                                f"ID {self.best_individual_values.get_individual_id()} --> "
                                f"VALORES: {self.best_individual_values.get_individual_values()} --> "
                                f"ANGULOS: {self.best_individual_values.get_vqc_parameters_values()} --> "
                                f"FITNESS: {self.best_individual_values.get_individual_fitness()}", "light_magenta")

            case "totally_restricted":

                # -- Iteramos por generacion
                for generation, pops in self.population.populuation_dict.items():

                    # -- Mostramos el mejor individuo de la generación
                    if pops:

                        match self.problem_type:
                            case "maximize":
                                self.best_individual_values = max(pops, key=lambda ind: ind.get_individual_fitness())
                            case "minimize":
                                self.best_individual_values = min(pops, key=lambda ind: ind.get_individual_fitness())
                            case _:
                                self.best_individual_values = None  # por si se define otro tipo no soportado

                        if self.best_individual_values:
                            self.IT.info_print(
                                f"MEJOR INDIVIDUO GENERACION: {generation} --> "
                                f"ID {self.best_individual_values.get_individual_id()} --> "
                                f"VALORES: {self.best_individual_values.get_individual_values()} --> "
                                f"FITNESS: {self.best_individual_values.get_individual_fitness()}", "light_magenta")


        # </editor-fold>

    # <editor-fold desc="GETTER DE LOS VALORES DEL MEJOR INDIVIDUO">
    def get_best_individual_values(self):
        """
        Getter para retornar el mejor individuo
        :return:
        """
        return self.best_individual_values.get_individual_values()

    # </editor-fold>

    # <editor-fold desc="MÉTODOS AUXILIARES DE VALIDACIÓN, TORNEOS, CÁLCULOS ESTADÍSTICOS Y PLOTEO">

    def validate_input_parameters(self) -> bool:
        """
        Metodo para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # <editor-fold desc="Validamos Dicts" >
        if self.bounds_dict is None:
            raise ValueError("bounds_dict no existe -> None")

        # -- INTERVAL
        interval_bounds_dict: dict = {k: v for k, v in self.bounds_dict.items() if v["bound_type"] == "interval"}
        if not all(isinstance(valor, (int, float)) for param in interval_bounds_dict for key in ["limits", "malformation_limits"] if key in param for valor in param[key]):
            raise ValueError("bounds_dict: No todos los valores en los bounds_dict interval son int o float.")

        # -- PREDEFINED
        predefined_bounds_dict: dict = {k: v for k, v in self.bounds_dict.items() if v["bound_type"] == "predefined"}
        if not all(isinstance(valor, (int, float)) for param in predefined_bounds_dict for key in ["limits", "malformation_limits"] if key in param for valor in param[key]):
            raise ValueError("bounds_dict: No todos los valores en los bounds_dict interval son int o float.")

        if self.element_matrix is not None:
            if not all(isinstance(k, int) and isinstance(v, tuple) and len(v) >= 1 and all(isinstance(coord, (int, float)) for coord in v) for k, v in self.element_matrix.items()):
                raise ValueError("element_matrix: Las claves deben ser int y los valores deben ser tuplas con al menos un número (int o float).")

        # </editor-fold>

        # <editor-fold desc="Validamos callables" >
        # -- objective_function
        if not isinstance(self.objective_function, Callable):
            raise ValueError(f"self.objective_function: Debe ser Callable y su tipo es {type(self.objective_function)}")

        # </editor-fold>

        # <editor-fold desc="Validamos enteros">

        # -- num_generations, num_individuals, podium_size, early_stopping_generations, early_stopping_generations_executed_counter
        if not isinstance(self.num_generations, int):
            raise ValueError(f"self.num_generations: Debe ser un entero y su tipo es {type(self.num_generations)}")
        if not isinstance(self.num_individuals, int):
            raise ValueError(f"self.num_individuals: Debe ser un entero y su tipo es {type(self.num_individuals)}")
        if not isinstance(self.podium_size, int):
            raise ValueError(f"self.podium_size: Debe ser un entero y su tipo es {type(self.podium_size)}")
        if not isinstance(self.early_stopping_generations, int):
            raise ValueError(f"self.early_stopping_generations: Debe ser un entero y su tipo es {type(self.early_stopping_generations)}")
        if not isinstance(self.early_stopping_generations_executed_counter, int):
            raise ValueError(f"self.early_stopping_generations: Debe ser un entero y su tipo es {type(self.early_stopping_generations_executed_counter)}")
        if not isinstance(self.variability_round_decimals, int):
            raise ValueError(f"self.variability_round_decimals: Debe ser un entero y su tipo es {type(self.variability_round_decimals)}")

        # </editor-fold>

        # <editor-fold desc="Validamos flotantes">

        # -- mutate_probability, mutate_gen_probability
        if not isinstance(self.mutate_probability, float):
            raise ValueError(f"self.mutate_probability: Debe ser un float y su tipo es {type(self.mutate_probability)}")
        if not isinstance(self.mutate_gen_probability, float):
            raise ValueError(f"self.mutate_gen_probability: Debe ser un float y su tipo es {type(self.mutate_gen_probability)}")

        # </editor-fold>

        # <editor-fold desc="Validamos Literals/strings">

        # -- problem_type, problem_restrictions, tournament_method
        if not isinstance(self.problem_type, str):
            raise ValueError(f"self.problem_type: Debe ser un str y su tipo es {type(self.problem_type)}")
        if self.problem_type not in ["minimize", "maximize"]:
            raise ValueError(f'self.problem_type debe ser una opción de estas: ["minimize", "maximize"] y se ha pasado {self.problem_type}')

        if self.problem_restrictions != "bound_restricted":
            if not isinstance(self.return_to_origin, str) or self.return_to_origin is None:
                raise ValueError(f"self.problem_type: Debe ser un str y su tipo es {type(self.return_to_origin)}")

        if self.problem_restrictions != "bound_restricted":
            if self.return_to_origin not in ["return_to_origin", "no_return"] and self.problem_type is not None:
                raise ValueError(f'self.problem_type debe ser una opción de estas: ["return_to_origin", "return_to_origin"] y se ha pasado {self.return_to_origin}')
        if not isinstance(self.problem_restrictions, str):
            raise ValueError(f"self.problem_restrictions: Debe ser un str y su tipo es {type(self.problem_restrictions)}")
        if self.problem_restrictions not in ["bound_restricted", "totally_restricted"]:
            raise ValueError(f'self.problem_restrictions debe ser una opción de estas: ["bound_restricted", "totally_restricted"] y se ha pasado {self.problem_restrictions}')
        if not isinstance(self.mutation_policy, str):
            raise ValueError(f"self.mutation_policy: Debe ser un str y su tipo es {type(self.mutation_policy)}")
        if self.mutation_policy not in ["soft", "normal", "hard"]:
            raise ValueError(f'self.mutation_policy debe ser una opción de estas: ["soft", "normal", "hard"] y se ha pasado {self.mutation_policy}')
        if not isinstance(self.tournament_method, str):
            raise ValueError(f"self.tournament_method: Debe ser un str y su tipo es {type(self.tournament_method)}")

        # </editor-fold>>

        # <editor-fold desc="Validamos bools">

        # -- verbose, early_stopping_generations_executed
        if not isinstance(self.verbose, bool):
            raise ValueError(f"self.verbose: Debe ser un bool y su tipo es {type(self.verbose)}")
        if not isinstance(self.early_stopping_generations_executed, bool):
            raise ValueError(f"self.early_stopping_generations_executed: Debe ser un bool y su tipo es {type(self.early_stopping_generations_executed)}")

        # </editor-fold>

        return True

    def get_tournament_method(self, verbose):
        """
        Metodo que crea y retorna el tournament seleccionado
        :return:
        """
        match self.tournament_method:

            case "ea_simple":
                self.IT.info_print("Se ha configurado existosamente el torneo ea_simple")
                return EaSimple(self.podium_size, self.problem_type, verbose)

    def print_generation_info(self, individual_generation_list: List[Individual], generation: int) -> None:
        """
        Metodo para pintar información de la generación por consola
        :param individual_generation_list: [List[Individual]] Lista de individuos
        :param generation: [int] Número de la generación
        :return: None
        """
        self.IT.sub_intro_print("Información de los individuos y los fitness")
        for i, ind in enumerate([z for z in individual_generation_list if z.generation == generation]):
            self.IT.info_print(f"Individuo {ind.get_individual_id()}: {ind.get_individual_values()} - Generación: {ind.generation} - [Fitness]: {ind.get_individual_fitness()}")

        self.IT.sub_intro_print(f"Información de la evolución de las distribuciones en cada generación")
        self.IT.print_tabulate_df(self.population.get_generation_fitness_statistics(generation), row_print=self.num_generations+1)

        return None

    def plot_generation_stats(self) -> None:
        self.population.plot_generation_stats(self.variability_explosion_starts_in_generation)

    def plot_evolution_animated(self) -> None:
        self.population.plot_evolution_animated()

    def plot_evolution(self) -> None:
        self.population.plot_evolution()

    # </editor-fold>

__all__ = ['GenethicOptimizer']