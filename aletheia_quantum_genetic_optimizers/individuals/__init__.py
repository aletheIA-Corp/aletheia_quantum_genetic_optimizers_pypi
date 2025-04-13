from singleton_tools import SingletonMeta
from typing import Dict, Union, Tuple, List, Literal
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from info_tools import InfoTools

import numpy as np
import warnings
import random
import math



class IndividualCache(metaclass=SingletonMeta):

    def __init__(self):
        """
        Clase para dar persistencia a ciertas características de los individuos
        """

        # -- Id del individuo que se incrementa cada vez que se crea uno uno
        self.individual_id: int = 1


class IndividualMethods(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_individual(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], child_values: List | None) -> Dict[str, Union[int, float]]:
        pass

    @abstractmethod
    def exists_malformation(self, individual_values: Dict[str, Union[int, float]], bounds_dict: Dict[str, Tuple[Union[int, float]]]) -> bool:
        pass


class BoundRestrictedIndividualMethods(IndividualMethods):
    def __init__(self):
        super().__init__()

        self.IC: IndividualCache = IndividualCache()
        self.IT: InfoTools = InfoTools()

    def exists_malformation(self, individual_values: Dict[str, Union[int, float]], bounds_dict: Dict[str, Tuple[Union[int, float]]]) -> bool:
        """
        Metodo para saber si el individuo tiene valores fuera del rango
        :return: True si existe malformacion, False else
        """

        for k, v in individual_values.items():
            individual_value: int | float = individual_values[k]
            individual_restrictions: tuple = bounds_dict[k]["malformation_limits"]

            if individual_value < min(individual_restrictions) or individual_value > max(individual_restrictions):
                return True

        return False

    def create_individual(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], child_values: List | None) -> Dict[str, Union[int, float]]:

        # -- Calculamos la cantidad de qubits necesarios para cada propiedad del individuo y las almacenamos en un dict
        individual_values: Dict[str, Union[int, float]] = {}
        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores
        if child_values is None:

            for parameter, v in bounds_dict.items():
                if v["bound_type"] == "interval":
                    match v["type"]:
                        case "int":
                            individual_values[parameter] = int(self.generate_random_value((v["limits"][0], v["limits"][1]), v["type"]))
                        case "float":
                            individual_values[parameter] = float(self.generate_random_value((v["limits"][0], v["limits"][1]), v["type"]))

                else:
                    individual_values[parameter] = self.generate_possible_value(v["limits"], v["type"])
        else:
            for parameter, cv in zip([z for z in bounds_dict.keys()], child_values):
                if bounds_dict[parameter]["type"] == "int":
                    individual_values[parameter] = int(cv)
                else:
                    individual_values[parameter] = float(cv)

        return individual_values


    @staticmethod
    def generate_random_value(val_tuple: tuple, data_type: str):
        if data_type == "int":
            return random.randint(val_tuple[0], val_tuple[1])
        elif data_type == "float":
            return random.uniform(val_tuple[0], val_tuple[1])

    @staticmethod
    def generate_possible_value(val_tuple: tuple, data_type: str):
        if data_type == "int":
            return random.choice(val_tuple)
        elif data_type == "float":
            return random.choice(val_tuple)


class FullRestrictedIndividualMethods(IndividualMethods):

    def __init__(self):
        super().__init__()

    def create_individual(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], child_values: List | None) -> Dict[str, Union[int, float]]:
        individual_values: Dict[str, Union[int, float]] = {}
        # -- En caso de que no se le pasen los child_list de la generacion, se crean aleatoriamente los valores
        if child_values is None:

            for parameter, v in bounds_dict.items():
                individual_values[parameter] = self.generate_possible_value(v["limits"], individual_values)
        else:
            for parameter, cv in zip([z for z in bounds_dict.keys()], child_values):
                individual_values[parameter] = cv

        return individual_values

    def exists_malformation(self, individual_values: Dict[str, Union[int, float]], bounds_dict: Dict[str, Tuple[Union[int, float]]]) -> bool:
        """
        Metodo para saber si el individuo tiene valores fuera del rango
        :return: True si existe malformacion, False else
        """

        for k, v in individual_values.items():
            individual_value: int | float = individual_values[k]
            individual_restrictions: tuple = bounds_dict[k]["malformation_limits"]

            if individual_value < min(individual_restrictions) or individual_value > max(individual_restrictions):
                return True
            if len(list(individual_values.values())) > len(set(individual_values.values())):
                return True
        return False

    @staticmethod
    def generate_possible_value(val_tuple: tuple, individual_values):
        return random.choice([z for z in val_tuple if z not in individual_values.values()])


class Individual:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], child_values: np.ndarray | None,
                 vqc: QuantumCircuit | None , vqc_parameters: Dict[str, List[float]] | None, generation: int,
                 problem_restrictions: Literal['bound_restricted', 'totally_restricted'],
                 element_matrix: Dict[str, tuple] | None = None):
        """
        Clase que va a instanciar los distintos individuos que van a competir.
        :param bounds_dict: [str, Tuple[Union[int, float]]] Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        de uso y que desemboca en un individuo que se deshechará por tener una malformación. Por ejemplo, si estamos optimizando un learning_rate y la mutación nos da un valor
        superior a 1, ese individuo, se descarta antes de ser evaluado. ej. '{learning_rate: (0.000001, 1)}', si los supera, consideramos malformación.
        :param child_values: [np.ndarray | None] Valores a partir de los cuales generar los nuevos individuos.
        :param vqc: Circuito cuántico perteneciente al individuo.
        :param vqc_parameters: Dict[str, List[float]] Diccionario de parámetros para cada propiedad del individuo. Opcional para la primera generación.
        :param generation: [int] Numero de generación a la que pertenece el individuo.
        :param problem_restrictions: Literal['bound_restricted', 'totally_restricted'] Define si el individuo a crear requiere bounds o está completamente restringido.
        :param element_matrix: [Dict[str, tuple] | None] Matriz de distancia utilizada para los problemas de optimización combinatoria de tipo TSP.
        """

        # -- Instanciamos la clase cache de individuo para dar persistencia a ciertas propiedades
        self.IC: IndividualCache = IndividualCache()

        # -- Definimos el id del individuo
        self._id = self.IC.individual_id

        # -- Almacenamos parámetros en propiedades
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict

        # -- Almacenamos los valores que se utilizarán para generar el individuo (sus valores reales)
        self.child_values: numpy.ndarray | None = child_values

        # -- Almacenamos el circuito cuántico con el que se crea el individuo
        self._vqc: QuantumCircuit | None = vqc

        # -- Alamcenamos los valores de los parámetros de la VQC que generaron las propiedades del individuo
        self._vqc_parameters: Dict[str, List[float]] | None = vqc_parameters

        # -- Almacenamos en una propiedad la generación a la que pertenece el individuo
        self.generation: int = generation

        # -- Almacenamos en una propiedad las restricciones a aplicar
        self.problem_restrictions: Literal['bound_restricted', 'totally_restricted'] = problem_restrictions

        # -- Definimos la propiedad en la que almacenaré el valor que la función de coste ha tenido para este individuo
        self._individual_fitness: float | None = None

        # -- Almacenamos en IMETHODS la instancia que contiene los metodos para el tipo de problema en cuestion
        self.IMETHODS = BoundRestrictedIndividualMethods() if self.problem_restrictions == "bound_restricted" else FullRestrictedIndividualMethods()

        # -- Creamos la propiedad de valores del individuo
        self._individual_values: Dict[str, Union[int, float]] = self.IMETHODS.create_individual(self.bounds_dict, self.child_values) if self.child_values is not None else {}

        # -- Almacenamos en una propiedad si el individuo tiene una malformación
        self._malformation: bool = self.exists_malformation()

        # -- Alamcenamos una matriz de distancia
        self._element_matrix: Dict[str, tuple] | None = element_matrix

        # -- Aumentamos en uno el individual_id para el próximo individuo
        self.IC.individual_id +=1

    def __eq__(self, other, decimals=4):
        """
        Compara si dos individuos son iguales en base a sus propiedades con precisión decimal.

        :param other: Otro objeto de la clase Individual con el que se realizará la comparación.
        :param decimals: Número de decimales a considerar en la comparación (por defecto 4).

        :return: True si ambos individuos tienen las mismas propiedades con la precisión dada, False en caso contrario.
        """

        # Verificar que el otro objeto es de la clase Individual
        if not isinstance(other, Individual):
            return False

        # Obtener los valores de los individuos
        values_self = self.get_individual_values()
        values_other = other.get_individual_values()

        # Redondear los valores antes de compararlos
        rounded_self = {k: round(v, decimals) if isinstance(v, float) else v for k, v in values_self.items()}
        rounded_other = {k: round(v, decimals) if isinstance(v, float) else v for k, v in values_other.items()}

        return rounded_self == rounded_other

    def exists_malformation(self) -> bool:
        """
        Metodo para saber si el individuo tiene valores fuera del rango
        :return: True si existe malformacion, False else
        """

        # -- NOTA: El metodo varía en funcion de las restricciones de self.problem_restrictions
        return self.IMETHODS.exists_malformation(self._individual_values, self.bounds_dict)

    # <editor-fold desc="Getters y setters    --------------------------------------------------------------------------------------------------------------------------------">

    def get_individual_id(self) -> int:
        """
        Metodo que devuelve el id del individuo
        :return:
        """
        return self._id

    def get_individual_values(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que devuelve los valores del individuo en un diccionario. Por ejemplo, si viene asi: {learning_rate: 0.0125, batch_size: 34}
        :return:
        """
        return self._individual_values

    def get_vqc(self) -> QuantumCircuit:
        """
        Metodo que devuelve el circuito cuántico con el que se crearon las propiedades del individuo.
        :return: El circuito cuántico con el que se generaron las propiedades del individuo
        """
        return self._vqc

    def get_vqc_parameters_values(self) -> Dict[str, List[float]] | None:
        """
        Metodo que devuelve los valores de los parámetros que generaron las propiedades del individuo.
        :return: Los parámetros que generaron las propiedades
        """
        return self._vqc_parameters

    def get_individual_fitness(self) -> int | float:
        """
        Metodo que devuelve el resultado de la función objetivo.
        :return: Los parámetros que generaron las propiedades
        """
        return self._individual_fitness

    def get_individual_malformation(self) -> bool:
        """
        Metodo que devuelve el resultado de la función objetivo.
        :return: Los parámetros que generaron las propiedades
        """
        return self._malformation

    def get_element_matrix(self) -> Dict[str, tuple] | None:
        """
        Metodo que devuelve el resultado de la función objetivo.
        :return: Los parámetros que generaron las propiedades
        """
        return self._element_matrix

    def add_or_update_variable(self, var_name: str, value: int | float) -> None:
        """
        Agrega o actualiza una variable de instancia en el objeto Individual.

        :param var_name: Nombre de la variable de instancia.
        :param value: Valor de la variable (puede ser de cualquier tipo).
        """
        setattr(self, f"_{var_name}", value)

    def set_individual_value(self, parameter: str, new_value: float | int):
        self._individual_values[parameter] = new_value
        self._malformation = self.exists_malformation()

    def set_individual_angles(self, parameter: str, new_value: list):
        self.get_vqc_parameters_values()[parameter] = new_value

    def set_individual_angle(self, angles: list):
        self._vqc_parameters = angles

    def set_individual_values(self, new_value_list: List[float | int]):
        for key, new_value in zip(self._individual_values.keys(), new_value_list):
            self._individual_values[key] = new_value
        self._malformation = self.exists_malformation()

    def get_child_values(self):
        return self.child_values

    # </editor-fold>

__all__ = ['IndividualCache', 'IndividualMethods', 'BoundRestrictedIndividualMethods', 'FullRestrictedIndividualMethods', 'Individual']
