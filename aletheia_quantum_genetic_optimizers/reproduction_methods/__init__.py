from aletheia_quantum_genetic_optimizers.quantum_technology import QuantumTechnology
from aletheia_quantum_genetic_optimizers.individuals import Individual
from singleton_tools import SingletonMeta

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit import ParameterVector, Parameter
from plotly.subplots import make_subplots
from typing import List, Literal, Dict
from itertools import combinations
from info_tools import InfoTools
from random import choices

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
import pandas as pd
import numpy as np
import itertools
import random

class Graph(metaclass=SingletonMeta):

    def __init__(self):

        self.graph: bool = True


class Reproduction:
    def __init__(self, winners_list: List[Individual],
                 number_of_children: int,
                 problem_restrictions: Literal['bound_restricted', 'totally_restricted'],
                 return_to_origin:[Literal['return_to_origin', 'no_return'] | None],
                 problem_type: Literal["minimize", "maximize"],
                 metric_to_optimize: Literal['accuracy', 'recall', 'specificity', 'f1', 'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other', 'r2'],
                 verbose: bool = True):

        """
        Clase | Objeto de reproducción de individuos
        :param winners_list: List[Individual] Lista de ganadores.
        :param number_of_children: [int] Cantidad de individuos que se quieren reproducir
        :param problem_restrictions: Literal['bound_restricted', 'totally_restricted'] Tipo de problema en cuanto a reestricciones
        :param return_to_origin:[Literal['return_to_origin', 'no_return'] | None] En caso de problemas totally_restricted es necesario saber si el problema termina en el origen o no es necesario que suceda esto
        :param problem_type: Literal["minimize", "maximize"] Tipo de problema en cuanto a maximización o minimización del resultado de la función objetivo
        :param metric_to_optimize: Literal['accuracy', 'recall', 'specificity', 'f1', 'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other'] Métrica a optimizar en la función objetivo
        :param verbose: [bool] Variable para definir si se pinta en consola | grafica
        """

        # -- Definimos las variables generales
        self.winners_list: List[Individual] = winners_list
        self.number_of_children: int = number_of_children
        self.problem_restrictions: Literal['bound_restricted', 'totally_restricted'] = problem_restrictions
        self.return_to_origin: Literal['return_to_origin', 'no_return'] | None = return_to_origin
        self.children_list: List[Individual] = []
        self.parents_generation: int = self.winners_list[0].generation
        self.problem_type: Literal['minimize', 'maximize'] = problem_type
        self.metric_to_optimize: Literal['accuracy', 'recall', 'specificity', 'f1', 'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other', 'r2'] = metric_to_optimize
        self.verbose: bool = verbose

        # -- Instanciamos clases de apoyo
        self.IT: InfoTools = InfoTools()
        self.QT: QuantumTechnology | None = None

        # -- Añadimos variables que luego determinaremos en run_reproduction
        self.optimization_quantum_technology: Literal["simulator", "quantum_machine"] | None = None
        self.optimization_service: Literal["aer", "ibm"] | None = None
        self.qm_api_key: str | None = None
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] | None = None
        self.generations_fitness_statistics_df: pd.DataFrame | None = None
        self.max_qubits: int | None = None

        # -- Añadimos un dataset para guardar los resultados de las normalizaciones entre binario, pi y bounds_dict
        self._reescaling_result_df: pd.DataFrame | None = None

        # -- Instanciamos la clase Graph (Singleton para evitar graficación excesiva con verbose True)
        self.GPH: Graph = Graph()

    def run_reproduction(self, quantum_technology: Literal["simulator", "quantum_machine"], service: Literal["aer", "ibm"], qm_api_key: str | None, qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None, quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"], generations_fitness_statistics_df: pd.DataFrame, max_qubits: int):
        """
        Metodo para ejecutar la reproducción de los mejores individuos
        :param quantum_technology. Literal["simulator", "quantum_machine"]. Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro service. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param service. ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. [str] API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Literal["ibm_quantum", "ibm_cloud"] | None. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        :param generations_fitness_statistics_df: [pd.Dataframe] Df que contiene la información estadística de todas.
        :param max_qubits: [int | None] Número máximo de qubits a utilizar.
        :return: La siguiente generación de individuos
        """

        self.optimization_quantum_technology: Literal["simulator", "quantum_machine"] = quantum_technology
        self.optimization_service: Literal["aer", "ibm"] = service
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine
        self.generations_fitness_statistics_df: pd.DataFrame = generations_fitness_statistics_df
        self.max_qubits: int | None = max_qubits

        self.QT: QuantumTechnology = QuantumTechnology(quantum_technology=self.optimization_quantum_technology,
                                                       service=self.optimization_service,
                                                       qm_api_key=self.qm_api_key,
                                                       qm_connection_service=self.qm_connection_service,
                                                       quantum_machine=self.quantum_machine)

        if self.verbose:
            self.IT.intro_print(f"RUN REPRODUCTION: La generacion {self.parents_generation} reproducirá la generación {self.parents_generation + 1}")

        # -- Agregamos a los 10 % de mejores padres no repetidos a la children list
        num_to_select = max(1, int(len(self.winners_list) * 0.15))

        # -- Seleccionamos los mejores individuos según el tipo de problema (maximize o minimize)
        sorted_winners = sorted(self.winners_list,
                                key=lambda ind: ind.get_individual_fitness(),
                                reverse=self.problem_type == "maximize")

        # -- Almacenamos los mejores individuos
        selected_individuals = sorted_winners[:num_to_select]

        # -- Añadimos estos individuos a la children_list
        self.children_list += selected_individuals

        # -- Después, creamos individuos idénticos a esos padres porque no podemos modificar su generación sin dejar descalza la generacion 0
        self.children_list = [
            Individual(
                bounds_dict=ind.bounds_dict,
                child_values=ind.get_child_values(),
                vqc=self.winners_list[0].get_vqc(),
                vqc_parameters=ind.get_vqc_parameters_values(),
                generation=self.parents_generation + 1,
                problem_restrictions=self.problem_restrictions,
                element_matrix=self.winners_list[0].get_element_matrix())
            for ind in self.children_list]

        # -- Tenemos en cuenta las restricciones del problema
        match self.problem_restrictions:
            case "bound_restricted":
                return self.bound_restricted_reproduction()
            case "totally_restricted":
                return self.totally_restricted_reproduction()

    def bound_restricted_reproduction(self):

        # -- Obtenemos el bounds_dict para que cada tipo de parametro tenga su propio cruce
        bounds_dict = self.winners_list[0].bounds_dict

        # -- Obtenemos la combinación posible de individuos ganadores
        combinations_of_individuals = list(itertools.permutations(self.winners_list, r=2))

        # -- Creamos un diccionario para almacenar los circuitos cuánticos de cada combinación y propiedad
        combo_circuits: dict = {}

        # -- Para cada una de las combinaciones de individuos
        for idx, combo in enumerate(combinations_of_individuals):

            combo_circuits[idx] = {}

            # -- Obtenemos los ids de los individuos
            ind1: int = combo[0].get_individual_id()
            ind2: int = combo[1].get_individual_id()

            if self.verbose:
                self.IT.sub_intro_print("Reproducción entre individuos")
                self.IT.info_print(f"Reproducción entre individuo {ind1} e individuo {ind2}")

            # -- Por cada par de individuos reconstruimos sus respectivos circuitos cuánticos y quitamos medidas
            ind1_vqc: QuantumCircuit = combo[0].get_vqc().copy()
            ind1_vqc.remove_final_measurements()

            ind2_vqc: QuantumCircuit = combo[1].get_vqc().copy()
            ind2_vqc.remove_final_measurements()

            # -- Agregamos a los circuitos nuevos parámetros para las métricas
            ind1_vqc = self.QT.add_new_qubit(ind1_vqc, Parameter("metric_0"), combo[0].bounds_dict, max_qubits=self.max_qubits)
            ind2_vqc = self.QT.add_new_qubit(ind2_vqc, Parameter("metric_1"), combo[1].bounds_dict, max_qubits=self.max_qubits)

            # -- Obtenemos los parámetros de los circuitos
            parameters: dict = {"0":ind1_vqc.parameters, "1":ind2_vqc.parameters}

            # -- Creamos un diccionario para almacenar los parámetros de los circuitos para luego asignarlos
            parent_circuits: dict = {}

            # -- Iteramos por cada padre
            for i in range(0, 2):

                # -- Normalizamos el resultado de la función objetivo de cada padre
                metric_to_optimize: dict = {f"{combo[i].get_individual_id()}": self.QT.normalize_metric(
                    generations_fitness_statistics_df=self.generations_fitness_statistics_df,
                    metric_name=self.metric_to_optimize,
                    value=combo[i].get_individual_fitness())}

                # -- Obtenemos los parametros theta de los qubits ("Y" y "Z")
                bound_angles = [angle for parameter in combo[0].bounds_dict.keys() for angle in
                                combo[i].get_vqc_parameters_values()[parameter]]

                # -- Añadimos los ángulos de los parámetros (incluido el valor de la función objetivo)
                parameters_dict: dict = {
                    param: value for param, value in zip([z for z in parameters[str(i)]],
                                                         [z for z in metric_to_optimize.values()] +
                                                         # [z / 2 for z in metric_to_optimize.values()] +
                                                         [z for z in bound_angles])
                }

                # -- Guardamos los parámetros de cada circuito
                parent_circuits[i] = {"parameters_dict": parameters_dict}

            # -- Asignamos los valores de las métricas
            ind1_vqc = ind1_vqc.assign_parameters(parent_circuits[0]["parameters_dict"]).copy()
            ind2_vqc = ind2_vqc.assign_parameters(parent_circuits[1]["parameters_dict"]).copy()

            # -- Creamos un circuito nuevo a partir de los padres y agregamos una puerta CNOT al final
            combined_qc: QuantumCircuit = self.QT.create_parent_vqc(bounds_dict, ind1_vqc, ind2_vqc).copy()

            if self.verbose and self.GPH.graph is True:
                self.QT.advanced_qc_plot(combined_qc)


            # -- Añadimos las mediciones al circuito cuántico parental creado (procedimientos especial):
            # -- 1. Multiplicamos los circuitos: Generamos una réplica por cada propiedad.
            # -- 2. Agregamos mediciones: En todos los qubits excepto en aquellos de los cuales no queremos propiedades.
            # -- Por ejemplo: Si queremos ejecutar un circuito para obtener el número binario que luego reescalaremos en
            # -- el rango -pi y pi, y luego entre los bounds, en particular para max_depth, entonces agregamos medidas
            # -- para todos los qubits, excepto para los qubits de otras propiedades (ej: n_estimators del padre 1 y 2).

            # -- Añadimos las medidas a las vqc concatenadas de los mejores padres
            combined_qc: dict = self.QT.adding_measurements_parent_vqc(bounds_dict=bounds_dict,
                                                                       combined_qc=combined_qc,
                                                                       verbose=self.verbose,
                                                                       graph=self.GPH.graph).copy()

            # -- Hacemos que no se pueda graficar de vuelta (evitar sobreproducción de imágenes)
            self.GPH.graph = False

            # -- Guardamos el circuito con las medidas en el diccionario de circuitos parentales
            for parameter, circuit in combined_qc.items():
                combo_circuits[idx][parameter] = circuit

            # -- Guardamos los ángulos/parámetros de los padres que generaron estos circuitos
            combo_circuits[idx]["individuals"] = {"individual_0": combo[0], "individual_1": combo[1]}

        # -- Almacenamos en un diccionario las listas de circuitos que pueden generar cada una de las propiedades
        prop_vqc_dict: dict = {}
        for idx, sub_dict in combo_circuits.items():
            for k, v in sub_dict.items():
                if k in [z for z in {key for d in combo_circuits.values() for key in d.keys()} if 'individuals' not in z]:
                    if k not in prop_vqc_dict.keys():
                        prop_vqc_dict[k] = [v]
                    else:
                        prop_vqc_dict[k].append(v)

        # -- Eliminamos la cantidad de circuitos que exceden el numero de hijos a reproducir
        first_key = next(iter(prop_vqc_dict))
        excess = len(prop_vqc_dict[first_key]) - self.number_of_children + len(self.children_list)

        if excess > 0:

            # -- Generamos índices aleatorios a eliminar
            indices_to_remove = sorted(random.sample(range(len(prop_vqc_dict[first_key])), excess), reverse=True)

            # -- Eliminamos esos índices de cada lista
            for k in prop_vqc_dict:
                for idx in indices_to_remove:
                    del prop_vqc_dict[k][idx]

        # -- Ejecutamos cada uno de los circuitos 20k veces
        results_dict: dict = {}
        for indv_prop, circuits in prop_vqc_dict.items():
            results_dict[indv_prop] = self.QT.execution_object.run(circuits, shots=20000)

        # -- TODO: Se podría implementar el código comentado y comentar el de abajo
        #  Nos quedamos por cada individuo con los 2 valores binarios que salieron más veces para cada propiedad
        """individual_dict: dict = {}
        for k, v in results_dict.items():
            for i, shot in enumerate(v):
                if i in [z for z in individual_dict.keys()]:
                    individual_dict[i]["binary"] = individual_dict[i]["binary"] | {k: [z for z in dict(sorted(shot.items(), key=lambda x: x[1], reverse=True)[:1]).keys()]}
                    individual_dict[i]["pi"] = individual_dict[i]["pi"] | {k: [self.QT.binary_to_float_and_normalize_pi([z]) for z in individual_dict[i]["binary"][k]]}
                    individual_dict[i]["real"] = individual_dict[i]["real"] | {k: [self.QT.rescaling_pi_to_integer_float(bounds_dict, k, individual_dict[i]["pi"][k][0])][0]}
                else:
                    individual_dict[i] = {"binary": {k: [z for z in dict(sorted(shot.items(), key=lambda x: x[1], reverse=True)[:1]).keys()]}}
                    individual_dict[i] = individual_dict[i] | {"pi": {k: [self.QT.binary_to_float_and_normalize_pi([z]) for z in individual_dict[i]["binary"][k]]}}
                    individual_dict[i] = individual_dict[i] | {"real": {k: [self.QT.rescaling_pi_to_integer_float(bounds_dict, k, individual_dict[i]["pi"][k][0])][0]}}"""

        individual_dict: dict = {}
        for k, v in results_dict.items():
            for i, shot in enumerate(v):

                if i in [z for z in individual_dict.keys()]:
                    individual_dict[i]["binary"] = individual_dict[i]["binary"] | {k: random.choice(list(shot.keys()))}
                    individual_dict[i]["pi"] = individual_dict[i]["pi"] | {k: [self.QT.binary_to_float_and_normalize_pi([individual_dict[i]["binary"][k]])]}
                    individual_dict[i]["real"] = individual_dict[i]["real"] | {k: [self.QT.rescaling_pi_to_integer_float(bounds_dict, k, individual_dict[i]["pi"][k][0])][0]}
                else:
                    individual_dict[i] = {"binary": {k: random.choice(list(shot.keys()))}}
                    individual_dict[i] = individual_dict[i] | {"pi": {k: [self.QT.binary_to_float_and_normalize_pi([individual_dict[i]["binary"][k]])]}}
                    individual_dict[i] = individual_dict[i] | {"real": {k: [self.QT.rescaling_pi_to_integer_float(bounds_dict, k, individual_dict[i]["pi"][k][0])][0]}}

        # -- Generamos un daframe con los resultados (binarios, angulos y propiedades)
        if self._reescaling_result_df is None:
            individual_dict_df: pd.DataFrame = pd.DataFrame(individual_dict).T
            individual_dict_df["parent_generation"] = self.winners_list[0].generation
            individual_dict_df["children_generation"] = self.winners_list[0].generation + 1
            self._reescaling_result_df = individual_dict_df
        else:
            individual_dict_df: pd.DataFrame = pd.DataFrame(individual_dict).T
            individual_dict_df["parent_generation"] = self.winners_list[0].generation
            individual_dict_df["children_generation"] = self.winners_list[0].generation + 1
            self._reescaling_result_df = pd.concat([self._reescaling_result_df, individual_dict_df])

        if self.verbose:
            self.IT.print_tabulate_df(self._reescaling_result_df, row_print=100, show_index=True)

        for i, row in self._reescaling_result_df[self._reescaling_result_df["parent_generation"] == self.winners_list[0].generation].iterrows():
            self.children_list.append(Individual(bounds_dict=bounds_dict,
                                                child_values=[z for z in row["real"].values()],
                                                vqc=self.winners_list[0].get_vqc(),
                                                vqc_parameters=row["pi"],
                                                generation=row["children_generation"],
                                                problem_restrictions=self.winners_list[0].problem_restrictions))

        for i in self.children_list:
            self.IT.info_print(f"INDIVIDUO {i.get_individual_id()} --> PROPIEDADES: {i.get_individual_values()} --> PARAMETROS: {i.get_vqc_parameters_values()}")

        if self.verbose:
            for i in self.children_list:
                self.IT.info_print(f"INDIVIDUO {i.get_individual_id()} --> PROPIEDADES: {i.get_individual_values()} --> PARAMETROS: {i.get_vqc_parameters_values()}")

        return self.children_list

    def totally_restricted_reproduction(self):

        self.IT.header_print("Iniciamos el proceso de reproducción de los individuos")

        # -- Obtenemos el bounds_dict para el momento de reproducir un individuo nuevo
        bounds_dict: Dict = self.winners_list[0].bounds_dict

        # ----------------------------------------------------------------------------------------
        # -- Separamos en clústers los individuos a fin de hacer computacionalmente viable el QAOA
        # ----------------------------------------------------------------------------------------

        # -- Obtenemos el número de elementos que conforman el individuo (Ej. ciudades del problema del TSP)
        num_elements: int = len([z for z in self.winners_list[0].get_element_matrix().keys()])

        # -- Definimos variables que utilizaremos en la generación del problema del QAOA
        zero_cluster: int | None = None
        contains_zero: bool = False

        # -- Iteramos el proceso de creación de individuos sin malformaciones hasta completar el número de hijos
        while len(self.children_list) < self.number_of_children:

            # -- Seleccionamos otro individuo al azar con el que se va a cruzar
            random_individual_selected_1: Individual = random.choice([ind for ind in self.winners_list])
            random_individual_selected_2: Individual = random.choice([ind for ind in self.winners_list if ind != random_individual_selected_1.get_individual_values()])

            if self.verbose:
                self.IT.sub_intro_print(f"Padres que se van a reproducir:")
                self.IT.info_print(f"Padre 1: {random_individual_selected_1.get_individual_values()}")
                self.IT.info_print(f"Padre 2: {random_individual_selected_2.get_individual_values()}")

            # -- Realizamos los cruces entre dos padres para sacar un hijo
            c1: List = self.ox1([z for z in random_individual_selected_1.get_individual_values().values()],
                          [z for z in random_individual_selected_2.get_individual_values().values()])

            # -- Separamos en clústeres [centroids no se usan parece]
            clusters = self.divide_ids_into_clusters_with_coords(
                ordered_ids=c1,
                id_to_coords=self.winners_list[0].get_element_matrix(),
                min_size=2,
                max_size=5
            )

            if self.verbose:
                self.IT.info_print(f"Se han creado {len([z for z in clusters])} clústers para {num_elements} elementos")

            self.IT.intro_print(f"Reproduciendo el individuo {len(self.children_list) + 1}")

            self.IT.sub_intro_print("Clústers")

            for cid, c in clusters.items():
                self.IT.info_print(f"Clúster {cid}: {list(c.keys())}")

            # -- Si el problema es de tipo return_to_origin == no_return identificamos el clúster con el primer elemento
            if self.return_to_origin == "no_return":

                # -- Identificamos el clúster que contiene el primer elemento (ej. la primera ciudad del TSP)
                zero_cluster: int = next(cid for cid, cluster in clusters.items() if 0 in cluster)

                if self.verbose:
                    self.IT.info_print(f"Modo no_return: El clúster con el primer elemento es el {zero_cluster}")

            # -- Definimos la lista con las combinaciones de elementos
            cluster_combinations: list = []

            # -- Iteramos por cada clúster
            for cid, cluster_element_matrix in clusters.items():
                route, dist, _ = None, None, None

                self.IT.sub_intro_print(f"Optimizando clúster {cid} --> {cluster_element_matrix}")

                if self.return_to_origin == "no_return":

                    # -- El primer clúster debe empezar desde la ciudad 0
                    contains_zero = cid == zero_cluster

                # -- Intentamos resolver el problema con el QAOA
                self.IT.info_print(f"Resolviendo el problema con QAOA")
                combination = self.QT.solve_tsp_with_qaoa(
                    element_matrix=cluster_element_matrix,
                    problem_type=self.problem_type,
                    contains_zero=contains_zero,
                    return_to_origin=self.return_to_origin,
                )

                # -- Si existe una combinación completa del QAOA y posee todos los elementos, la agregamos a la lista
                if combination and len(combination) == len([z for z in cluster_element_matrix.keys()]):
                    cluster_combinations.append(combination)
                    self.IT.info_print("Se ha agregado una combinación completa a la lista")

                # -- Si no existe una combinación completa del QAOA o no posee todos los elementos, agregamos una random
                if not combination or len(combination) != len([z for z in cluster_element_matrix.keys()]):
                    self.IT.info_print(f"[WARNING] No se encontró combinación válida en cluster. Generando ruta aleatoria...", "light_yellow")
                    combinations = list(cluster_element_matrix.keys())
                    random_combination = random.sample(combinations, len(combinations))

                    self.IT.info_print("Se ha agregado una combinación aleatoria completa a la lista", "light_yellow")
                    cluster_combinations.append(random_combination)

            # -- Combinamos todas las rutas de los clústers
            final_combination: list = [item for sublist in cluster_combinations for item in sublist]

            self.IT.info_print(f"La combinacion final de elementos es: {final_combination}")

            if self.verbose:
                self.plot_tsp_route(element_matrix=self.winners_list[0].get_element_matrix(),
                                    combination=final_combination,
                                    return_to_origin=self.return_to_origin)

            # -- Verificamos si la ruta final contiene todas las ciudades
            if len(final_combination) == len([z for z in self.winners_list[0].get_element_matrix().keys()]):
                # -- Crear el nuevo individuo con la ruta combinada
                new_child = Individual(bounds_dict=bounds_dict,
                                       child_values=final_combination,
                                       vqc=None,
                                       vqc_parameters=None,
                                       generation=self.winners_list[0].generation + 1,
                                       problem_restrictions=self.winners_list[0].problem_restrictions,
                                       element_matrix=self.winners_list[0].get_element_matrix())

                # -- Validamos que no existan individuos muy similares en la lista y que no tengan malformacion
                is_duplicate = any(existing_indv == new_child for existing_indv in self.children_list)
                if not is_duplicate and not new_child.get_individual_malformation():
                    self.children_list.append(new_child)
                    self.IT.info_print("Un nuevo individuo ha sido añadido a la lista con exito")

        # -- Mostramos información de los hijos
        for i in self.children_list:
            self.IT.info_print(
                f"INDIVIDUO {i.get_individual_id()} --> PROPIEDADES: {i.get_individual_values()} --> COMBINACION: {i.get_child_values()}")

        return self.children_list

    @staticmethod
    def ox1(parent1, parent2):
        """Aplica Order Crossover (OX1) a dos padres de un problema basado en permutaciones."""

        # -- Creamos una lista del tamaño de los genes y la inicializamos en None
        size = len(parent1)
        offspring = [None] * size

        # Seleccionamos un segmento aleatorio en el rango permitido, por ejemplo de [0,1,2,3,4,5] se selecciona [2,3,4]
        start, end = sorted(random.sample(range(size), 2))

        # Copiar segmento del primer padre
        offspring[start:end + 1] = parent1[start:end + 1]

        # Completar con genes del segundo padre en el mismo orden, evitando repeticiones
        p2_genes = [gene for gene in parent2 if gene not in offspring]

        # Rellenar los espacios vacíos manteniendo el orden
        idx = 0
        for i in range(size):
            if offspring[i] is None:
                offspring[i] = p2_genes[idx]
                idx += 1

        return offspring

    @staticmethod
    def divide_ids_into_clusters_with_coords(ordered_ids: List, id_to_coords: Dict, min_size: int = 2, max_size: int = 5):
        """
        Divide una lista de IDs en grupos de entre min_size y max_size,
        mapeando cada ID a sus coordenadas desde un diccionario dado.

        :param ordered_ids: Lista ordenada de IDs a agrupar
        :param id_to_coords: Diccionario con {id: (x, y)}
        :param min_size: Tamaño mínimo de grupo
        :param max_size: Tamaño máximo de grupo
        :return: Diccionario con clusters numerados y coordenadas mapeadas
        """
        total = len(ordered_ids)
        i = 0
        cluster_idx = 0
        clusters = {}

        while i < total:
            restante = total - i

            if min_size <= restante <= max_size:
                ids = ordered_ids[i:]
                clusters[cluster_idx] = {idx: id_to_coords[idx] for idx in ids}
                break

            if restante < min_size:
                if clusters:
                    for idx in ordered_ids[i:]:
                        clusters[cluster_idx - 1][idx] = id_to_coords[idx]
                else:
                    clusters[cluster_idx] = {idx: id_to_coords[idx] for idx in ordered_ids[i:]}
                break

            for tam in range(max_size, min_size - 1, -1):
                if (restante - tam) >= min_size or (restante - tam) == 0:
                    ids = ordered_ids[i:i + tam]
                    clusters[cluster_idx] = {idx: id_to_coords[idx] for idx in ids}
                    i += tam
                    cluster_idx += 1
                    break

        return clusters

    @staticmethod
    def plot_tsp_route(element_matrix: Dict[str, tuple], combination: List[int], return_to_origin: Literal['return_to_origin', 'no_return'] | None, title: str = "Ruta TSP"):
        """
        Gráfico para graficar combinaciones:
        - Fondo negro
        - Flechas de conexión
        - Pantalla completa
        - Combinación mostrada debajo del gráfico
        """

        pio.renderers.default = "browser"  # o 'notebook' si estás en Jupyter
        combination_copy = combination.copy()

        # -- Coordenadas
        x_coords = [element_matrix[node][0] for node in combination_copy]
        y_coords = [element_matrix[node][1] for node in combination_copy]

        # -- Cierre visual del ciclo si aplica

        if return_to_origin == "return_to_origin":
            if combination_copy[0] != combination_copy[-1]:
                x_coords.append(element_matrix[combination_copy[0]][0])
                y_coords.append(element_matrix[combination_copy[0]][1])
                combination_copy.append(combination_copy[0])

        # -- Subgráfico: uno para el gráfico, otro para la combinación
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.85, 0.15],
            vertical_spacing=0.05,
            specs=[[{"type": "scatter"}], [{"type": "table"}]]
        )

        # -- Flechas entre puntos
        for i in range(len(combination_copy) - 1):
            x0, y0 = element_matrix[combination_copy[i]]
            x1, y1 = element_matrix[combination_copy[i + 1]]
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x1", yref="y1", axref="x1", ayref="y1",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="cyan"
            )

        # -- Ruta
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines+markers',
            line=dict(color='cyan', width=2),
            marker=dict(size=8, color='white'),
            name='Ruta'
        ), row=1, col=1)

        # -- Nodos
        for node in combination_copy:
            x, y = element_matrix[node]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[str(node)],
                textfont=dict(color='white'),
                textposition="top center",
                marker=dict(size=10, color='orange'),
                name=f'Nodo {node}',
                showlegend=False
            ), row=1, col=1)

        # -- Tabla con la combinación
        fig.add_trace(go.Table(
            header=dict(values=["Combinación de nodos"],
                        fill_color='black',
                        font=dict(color='white', size=14),
                        align='left'),
            cells=dict(values=[[str(combination_copy)]],
                       fill_color='black',
                       font=dict(color='lightgreen', size=12),
                       align='left')
        ), row=2, col=1)

        # -- Estilo general
        fig.update_layout(
            title=title,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )

        fig.update_xaxes(showgrid=False, zeroline=False, color='white', row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, color='white', row=1, col=1)

        fig.show()

__all__ = ['Graph', 'Reproduction']


