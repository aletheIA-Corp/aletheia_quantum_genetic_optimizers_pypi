from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Literal, cast, List, Dict, Iterator, Tuple, Any
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit import ParameterVector, Parameter
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_optimization import QuadraticProgram
from qiskit.compiler import transpile
from qiskit.providers import Backend
from qiskit.pulse import num_qubits
from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from sklearn.cluster import KMeans
from info_tools import InfoTools
from numpy import floating

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import sys


class QuantumSimulator:

    def __init__(self, service: Literal["aer", "ibm"] = "aer"):
        """
        :param service. Literal["aer"]. El servicio tecnológico con el cual se ejecuta la logica.
        """

        self.service: Literal["aer", "ibm"] = service
        self.sampler = None

        if self.service == "ibm":
            sys.exit(f"Se ha seleccionado el servicio {self.service} para ejecutar el simulador cuántico (usar: aer)")

        match self.service:

            case "aer":
                self.sampler = Sampler(AerSimulator())

    def run(self, qcs: List[QuantumCircuit], shots: int):
        """
        Metodo para ejecutar un simulador cuantico y obtener sus resultados
        :param qcs: Circuito cuantico que se quiere medir
        :param shots: Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # -- Definimos lista de resultados (homologado con metodo run de QuantumMachine)
        results: list = []

        # -- Definimos el sampler para ejecutar shots cantidad de veces el circuito cuantico especificado
        for qc in qcs:

            job = self.sampler.run([qc], shots=shots)

            # -- Lanzamos el job (tarea de ejecución del circuito cuántico) y obtenemos sus resultados
            job_result = job.result()[0].data.c
            results.append(job_result)

        # -- Obtenemos los resultados
        results = self.get_results(results)

        return results

    @staticmethod
    def get_results(results: List):
        """
        Obtenemos los resultados
        :param results: [List] Resultados
        :return: Lista de resultados
        """

        results_list: list = []

        # -- Accedemos a los valores de las mediciones del circuito cuantico
        for result in results:

            # -- Contamos la probabilidad de los resultados
            qc_ibm_results = result.get_counts()

            # -- Agregamos los resultados a lista de resultados
            results_list.append(qc_ibm_results)

        return results_list


class QuantumMachine:

    def __init__(self,
                 service: Literal["aer", "ibm"],
                 qm_api_key: str | None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"],
                 optimization_level: int = 1,
                 verbose: bool = True):
        """
        :param service. ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. str | None. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Literal["ibm_quantum", "ibm_cloud"] | None. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        :param optimization_level. Nivel de optimización del circuito cuántico
        """

        self.service: Literal["aer", "ibm"] = service
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine
        self.optimization_level: int = optimization_level
        self.verbose: bool = verbose

        # -- Instanciamos la clase para la consola
        self.IT = InfoTools()

        # -- Definimos la variable con la máquina elegida (puede ser igual a quantum_machine pero también la que resulte de least_busy)
        self.selected_machine: str | None = None
        self.ibm_machine_transpiler = None

        # -- Definimos el sampler de la máquina cuántica seleccionada
        self.sampler = None

        if self.service == "aer":
            self.IT.info_print(f"Se ha seleccionado el servicio {self.service} para ejecutar el ordenador cuantico (usar: ibm)", "light_red")
            sys.exit()

        match self.service:

            case "ibm":

                # -- Obtenemos el nombre de la máquina elegida y el transpilador de esa máquina
                if self.qm_connection_service == "ibm_quantum":

                    self.IT.sub_intro_print(f"Conectando con el servicio de computacion cuantica de IBM: ibm_quantum")

                    # -- Generamos el servicio de conexion
                    channel_type = Literal["ibm_cloud", "ibm_quantum", "local"]
                    channel = cast(channel_type, self.qm_connection_service)
                    self.service: QiskitRuntimeService = QiskitRuntimeService(channel=channel, token=self.qm_api_key)

                    # -- Chequeamos si la conexion ha sido exitosa
                    if self.service.active_account()["verify"]:
                        self.IT.info_print("Conexion realizada con exito")

                        self.IT.sub_intro_print("Datos de la cuenta")
                        _user_data: dict = self.service.usage()
                        _user_period_start: str = _user_data["period"]["start"]
                        _user_period_end: str = _user_data["period"]["end"]
                        _by_instance: str = _user_data["byInstance"][0]["instance"]
                        _user_quota: int = _user_data["byInstance"][0]["quota"]
                        _user_usage: int = _user_data["byInstance"][0]["usage"]
                        _user_pending_jobs: int = _user_data["byInstance"][0]["pendingJobs"]
                        _user_max_pending_jobs: int = _user_data["byInstance"][0]["maxPendingJobs"]

                        self.IT.info_print(f"Instancia de ejecucion: {_by_instance}")
                        self.IT.info_print(f"Cuota de ejecucion: {_user_quota}")
                        self.IT.info_print(f"Usos del usuario: {_user_usage}")
                        self.IT.info_print(f"Trabajos pendientes: {_user_pending_jobs} / {_user_max_pending_jobs}")

                        # -- Buscamos la maquina elegida
                        self.IT.sub_intro_print("Buscando ordenador cuantico...")
                        if self.quantum_machine == "least_busy":
                            self.IT.info_print(f"---> Buscando la maquina menos cargada...")
                            least_busy_machine = self.service.least_busy()
                            self.selected_machine = self.service.backend(least_busy_machine.name)
                            self.IT.info_print(f"---> La maquina menos cargada es: {self.selected_machine.name}")
                        else:
                            self.IT.info_print(f"---> Buscando la maquina {self.quantum_machine}...")
                            self.selected_machine: Backend = self.service.backend(self.quantum_machine)

                        self.IT.info_print(f"Numero de Qubits: {self.selected_machine.num_qubits}")
                        self.IT.info_print(f"Trabajos pendientes: {self.selected_machine.status().pending_jobs}")
                        self.IT.info_print(f"Operaciones permitidas: {self.selected_machine.operation_names}")
                        self.IT.info_print(f"Numero maximo de circuitos: {self.selected_machine.max_circuits}")

                        if self.selected_machine is not None:

                            # -- Generamos el transpilador o pass manager de la maquina elegida
                            self.connection_transpiler = generate_preset_pass_manager(backend=self.selected_machine, optimization_level=self.optimization_level)

                        else:
                            self.IT.info_print("El ordenador especificado no existe. FIN", "light_red")
                            sys.exit()

                elif self.qm_connection_service == "ibm_cloud":
                    pass

    def run(self, qc_list: List[QuantumCircuit], shots: int):

        """
        Metodo para ejecutar un ordenador cuantico y obtener sus resultados
        :param qc_list: List[QuantumCircuit]. Circuito cuantico que se quiere medir
        :param shots: int. Cantidad de veces que se ejecutara el circuito cuantico
        :return: Mediciones del circuito cuantico
        """

        # -- Transpilamos los circuitos antes de la ejecución
        transpiled_circuits = transpile(qc_list, backend=self.selected_machine,
                                        optimization_level=self.optimization_level)

        # -- Usamos el backend seleccionado para crear una sesión
        with Session(backend=self.selected_machine) as session:

            # -- Generamos el sampler de la máquina
            self.sampler = Sampler(mode=session)

            # -- Ejecutamos el trabajo con los circuitos transpilados y el número de shots
            job = self.sampler.run(transpiled_circuits, shots=shots)

            # -- Esperamos que el trabajo termine y obtenemos los resultados
            results = job.result()

        # -- Cerramos la sesión
        session.close()

        # -- Retornamos las cuasi-distribuciones de las mediciones
        if results is not None:
           results = self.get_results(results)
           return results

        else:
            self.IT.info_print("No se han podido obtener los resultados del ordenador cuántico. FIN", "light_red")
            sys.exit()

    @staticmethod
    def get_results(results: list):
        """
        Metodo para obtener la lista de resultados
        :param results: [list] Resultados a obtener
        :return: Lista de resultados
        """

        # -- Instaciamos una lista vacía para obtener los resultados
        results_list = []

        # -- Por cada resultado obtenemos la cantidad de repeticiones
        for result in results:
            counts = result.data.c.get_counts()
            results_list.append(counts)

        return results_list


class QuantumTechnology:

    def __init__(self,
                 quantum_technology: Literal["simulator", "quantum_machine"] = "simulator",
                 service: Literal["aer", "ibm"] = "aer",
                 qm_api_key: str | None = None,
                 qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = None,
                 quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = "least_busy",
                 max_qubits: int = 32
                 ):

        """
        Metodo que instancia un objeto QuantumTechnology, el cual puede ser un simulador o un conector a una máquina cuántica real
        :param quantum_technology. Literal["simulator", "quantum_machine"]. Tecnología cuántica con la que calculan la lógica. Si es simulator, se hará con un simulador definido en el
        parámetro service. Si es quantum_machine, el algoritmo se ejecutará en una máquina cuántica definida en el parámetro technology.
        :param service. ["aer", "ibm"] El servicio tecnológico con el cual se ejecuta la lógica.
        :param qm_api_key. API KEY para conectarse con el servicio de computación cuántica de una empresa.
        :param qm_connection_service. Literal["ibm_quantum", "ibm_cloud"] | None. Servicio específico de computación cuántica. Por ejemplo, en el caso de IBM pueden ser a la fecha ibm_quantum | ibm_cloud
        :param quantum_machine. Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]. Nombre del ordenador cuántico a utilizar. Por ejemplo, en el caso de IBM puede ser ibm_brisbane, ibm_kyiv, ibm_sherbrooke. Si se deja en least_busy,
        se buscará el ordenador menos ocupado para llevar a cabo la ejecución del algoritmo cuántico.
        """

        self.quantum_technology: Literal["simulator", "quantum_machine"] = quantum_technology
        self.service: Literal["aer", "ibm"] = service
        self.qm_api_key: str | None = qm_api_key
        self.qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None = qm_connection_service
        self.quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"] = quantum_machine
        self.max_qubits: int = max_qubits

        # -- TODO: Diccionario de tecnologias y servicios habilitados (actualizar periódicamente)
        self._allowed_quantum_tech: dict = {"simulator": ["aer"],
                                            "quantum_machine": ["ibm"],
                                            "quantum_machines": {"ibm": ["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"]},
                                            "quantum_services": {"ibm": ["ibm_quantum", "ibm_cloud"]}}

        # -- Validamos que los parámetros relacionados con la tecnología cuántica son correctos
        self.validate_input_parameters()

        # -- Generamos el objeto que ejecuta el algoritmo cuántico
        self.execution_object: QuantumSimulator | QuantumMachine | None = None

        match quantum_technology:

            case "simulator":
                self.execution_object: QuantumSimulator = QuantumSimulator(self.service)

            case "quantum_machine":
                self.execution_object: QuantumMachine = QuantumMachine(self.service, self.qm_api_key, self.qm_connection_service, self.quantum_machine, 1)

    # ---------------------------- Metodos para tecnología cuántica  ------------------------------------- #

    def get_quantum_technology(self):
        """
        Metodo getter para retorna el execution object (objeto que ejecuta un circuito en un ordenador cuántico o en un simulador cuántico)
        :return: self.execution_object
        """
        return self.execution_object

    def validate_input_parameters(self) -> bool:
        """
        Metodo para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # -- Validamos la tecnologia cuántica definida
        if self.quantum_technology == "simulator":
            if self.service not in self._allowed_quantum_tech["simulator"]:
                raise ValueError(f"self.quantum_technology_executor: La randomness_technology escogida es {self.quantum_technology}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['simulator']}")

        # -- Validamos el ordenador cuántico elegido
        if self.quantum_technology == "quantum_machine":

            if self.service not in self._allowed_quantum_tech["quantum_machine"]:
                raise ValueError(f"self.technology: La technology escogida es {self.service}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machine']}")

            if self.quantum_machine not in self._allowed_quantum_tech["quantum_machines"][f"{self.service}"]:
                raise ValueError(f"self.quantum_machine: La quantum_machine escogida es {self.quantum_machine}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_machines'][f'{self.service}']}")

            if self.qm_connection_service not in self._allowed_quantum_tech["quantum_services"][f"{self.service}"]:
                raise ValueError(f"self.qm_connection_service: El qm_connection_service escogido es {self.qm_connection_service}. "
                                 f"Por tanto, debe estar entre los siguientes: {self._allowed_quantum_tech['quantum_services'][f'{self.service}']}")

        return True

    @staticmethod
    def create_generator(bounds_dict: dict) -> Tuple[QuantumCircuit, ParameterVector]:
        """Metodo que crea un circuito cuántico parametrizado que actúa como generador"""

        # -- Definimos los parámetros a entrenar
        parameters = ParameterVector("θ", len(bounds_dict.keys()))

        # -- Definimos el circuito cuántico
        qc: QuantumCircuit = QuantumCircuit(len(bounds_dict.keys()), len(bounds_dict.keys()))

        # -- Generamos una estructura de reescalado a nivel horizontal
        for deep_level in range(1, 5):

            # -- Aplicamos rotación a la puertas parametrizada de cada qubit
            for i in range(len(bounds_dict.keys())):

                # -- Definimos el indice del qubit i para la rotación "Y" y "Z"
                ry_index = i
                rz_index = i

                # -- Aplicamos rotación en "Y"
                # qc.ry(parameters[ry_index] / deep_level, i)
                if i % 2 == 0:
                    qc.ry(parameters[ry_index] / deep_level, i)
                else:
                    qc.rz(parameters[rz_index] / deep_level, i)

            # -- Creamos entrelazamiento entre qubits adyacentes (CNOT between qubit i and qubit i+1)
            for i in range(len(bounds_dict.keys()) - 1):
                qc.cx(i, i + 1)

            # -- En algunos casos cerramos el circulo de entrelazamiento (CNOT entre el primero y el último qubit)
            if len(bounds_dict.keys()) > 2:
                qc.cx(control_qubit=0, target_qubit=len(bounds_dict.keys()) - 1)

        # -- Añadimos medidas a los qubits
        qc.measure(range(len(bounds_dict.keys())), range(len(bounds_dict.keys())))

        # -- Añadimos una barrera
        qc.barrier()

        return qc, parameters

    @staticmethod
    def create_generator_reproduction(bounds_dict: dict):
        # -- TODO: se podrían agregar qubits hacia abajo con alguna proporción de la función objetivo
        """Metodo que crea un circuito cuántico parametrizado que actúa como generador de la GAN"""

        # -- Definimos los parámetros a entrenar
        parameters = ParameterVector(name="θ", length=2 * len(bounds_dict.keys()))

        # -- Definimos el circuito cuántico
        qc: QuantumCircuit = QuantumCircuit(len(bounds_dict.keys()) + 1, len(bounds_dict.keys()) + 1)

        for deep_level in range(1, 5):

            # -- Aplicamos rotación a la puertas parametrizada de cada qubit
            for i in range(len(bounds_dict.keys())):

                # -- Aplicamos rotación en Y
                qc.ry(parameters[i] / deep_level, i)

                # -- Aplicamos rotación en Z
                qc.rz(parameters[i + len(bounds_dict.keys())] / deep_level, i)

            # -- Añadimos entrelazamiento entre los qubits con una puerta CNOT

            # -- Creamos entrelazamiento entre qubits adyacentes (CNOT between qubit i and qubit i+1)
            for i in range(len(bounds_dict.keys()) - 1):
                qc.cx(i, i + 1)

            # -- En algunos casos cerramos el circulo de entrelazamiento (CNOT between the first and last qubit)
            if len(bounds_dict.keys()) > 2:
                qc.cx(0, len(bounds_dict.keys()) - 1)

        parameter_metric = ParameterVector(name="metric", length=2)

        for deep_level in range(1, 5):

            # -- Aplicamos rotación a la puertas parametrizada de cada qubit
            for i in range(len(bounds_dict.keys()) - 1):

                # -- Aplicamos rotación en Y
                qc.ry(parameter_metric[i] / deep_level, len(bounds_dict.keys()))

                qc.rz(parameter_metric[i+1] / deep_level, len(bounds_dict.keys()))

            # -- Añadimos entrelazamiento entre los qubits con una puerta CNOT

            # -- Creamos entrelazamiento entre qubits adyacentes (CNOT between qubit i and qubit i+1)
            for i in range(0, len(bounds_dict.keys())):
                qc.cx(i, len(bounds_dict.keys()))

            # -- En algunos casos cerramos el circulo de entrelazamiento (CNOT between the first and last qubit)
            if len(bounds_dict.keys()) > 2:
                qc.cx(0, len(bounds_dict.keys()) - 1)

        # -- Añadimos medidas a los qubits
        qc.measure(range(len(bounds_dict.keys()) + 1), range(len(bounds_dict.keys()) + 1))

        # -- Añadimos una barrera
        qc.barrier()

        return qc, parameters, parameter_metric

    def generate_parameters(self, num_individuals: int, parameters: ParameterVector) -> List[List[float]]:

        """
        Metodo que genera las propiedades cuánticas de los individuos mediante circuitos cuánticos.
        Proceso: calcula cuántos qubits se necesitan por parámetro y ajusta valores según limitaciones malformantes.
        :param num_individuals: [int] Número de individuos que queremos generar.
        :param parameters: [ParameterVector] Parámetros del circuito generador de primeros padres.

        :return results_dict (dict): Diccionario con los valores generados para cada individuo.
        """

        # -- Creamos la lista de circuitos cuánticos
        _qc_list: List[QuantumCircuit] = []

        # -- Obtenemos la cantidad de parámetros
        _parameter_length: int = len([z for z in parameters])

        # -- Iteramos por cada individuo del total de individuos que se buscan generar
        for individual in range(0, num_individuals):

            # -- Iteramos por cada parámetro del diccionario de propiedades
            for parameter in range(0, _parameter_length):

                # -- Generamos el circuito cuantico de cada parametro
                _temp_qc: QuantumCircuit = self._generate_qc(num_qubits=self.max_qubits).copy()

                # -- Apendeamos el circuito cuántico generado a la lista de circuitos
                _qc_list.append(_temp_qc)

        # -- Ejecutamos todos los circuitos cuánticos bajo una misma sesion
        results = self.quantum_randomness(_qc_list)

        # -- Generamos un diccionario de resultados para adjudicar los parametros a cada individuo
        results_list: list = []

        # -- Llenamos el diccionario con los valores de _results
        for i, result in enumerate(results):
            results_list.append(self.binary_to_float_and_normalize_pi(result.keys()))

        # -- Retornamos los valores de los parámetros entre -pi y pi dividos por individuo
        # return self.dividir_lista(results_list, len([z for z in parameters]))
        return self.dividir_lista(results_list, 2)

    def generate_properties(self, vqc: QuantumCircuit, parameters: ParameterVector, vqc_parameters: List[list], num_qubits: int) -> np.ndarray:

        """
        Metodo para convertir los resultados de las VQC en valores normalizados de las propiedades
        :param vqc: Circuito cuántico variacional (VQC)
        :param parameters: Parámetros del circuito cuántico variacional (VQC)
        :param vqc_parameters: Parámetros con los que se instancias los parámetros del VQC
        :param num_qubits: Numero de qubits para obtener resultados
        :return: np.ndarray con los resultados normalizados de las propiedades
        """

        # -- Creamos todos los circuitos con sus respectivos parámetros
        _parameterized_circuits = []


        # -- Para conjunto de parámetros (en realidad -> ángulos de los parámetros)
        for circuit_parameters_values in vqc_parameters:
            _parameters_dict = {param: value for param, value in zip([z for z in parameters], circuit_parameters_values)}
            _parameterized_circuit = vqc.assign_parameters(_parameters_dict)
            _parameterized_circuits.append(_parameterized_circuit)

        # -- Ejecutamos el total de circuitos juntos en la misma sesión
        results: List[Dict[str, int]] = self.get_quantum_technology().run(_parameterized_circuits, shots=1024)

        # -- Extraemos resultados y los convertimos a propiedades
        all_properties: np.ndarray = np.zeros((len(vqc_parameters), num_qubits))

        for circuit_idx, counts in enumerate(results):
            for i in range(num_qubits):
                prob_one = sum(count for state, count in counts.items() if state[-(i + 1)] == '1')
                total = sum(counts.values())

                # -- Guardamos el resultado normalizado para el circuito correspondiente
                all_properties[circuit_idx, i] = prob_one / total if total > 0 else 0.5

        return all_properties

    @staticmethod
    def denormalize_hyperparameters(bounds_dict: Dict, all_hyperparameters: np.ndarray) -> np.ndarray:
        """
        Desnormaliza los hiperparámetros de acuerdo a los límites especificados en bounds_dict.
        :param bounds_dict: Diccionario de bounds del problema a resolver
        :param all_hyperparameters: Array de valores normalizados de los hiperparámetros.

        :return Matriz con los hiperparámetros desnormalizados.
        """

        # -- Convertimos las keys de bounds_dict a una lista para conservar el orden
        keys: list = list(bounds_dict.keys())

        # -- Inicializamos una matriz vacía para almacenar los resultados desnormalizados
        data_desnormalized = np.empty_like(all_hyperparameters)

        # -- Definimos variables en None
        min_val: int | float | None = None
        max_val: int | float | None = None

        # -- Para cada propiedad del diccionario
        for i, key in enumerate(keys):

            # -- Extraemos la información del diccionario
            bounds_info = bounds_dict[key]
            bound_type = bounds_info['bound_type']

            # -- Si el tipo de bound es interval obtenemos el valor mínimo y máximo
            if bound_type == "interval":
                min_val, max_val = bounds_info['limits']

            # -- Obtenemos los parámetros
            data_column = all_hyperparameters[:, i]

            # -- Desnormalizamos para un intervalo continuo
            if bound_type == 'interval':
                data_desnormalized[:, i] = data_column * (max_val - min_val) + min_val

                # -- Si el tipo de dato es 'int', lo convertimos a entero
                if bounds_info['type'] == 'int':
                    data_desnormalized[:, i] = np.round(data_desnormalized[:, i]).astype(int)

             # -- Desnormalizamos para un conjunto de valores predefinidos
            elif bound_type == 'predefined':
                predefined_values = np.array(bounds_info['limits'])

                # -- Aquí buscamos el valor más cercano en el arreglo de valores predefinidos
                indices = np.round(data_column * (len(predefined_values) - 1)).astype(int)
                data_desnormalized[:, i] = predefined_values[indices]

        return data_desnormalized

    @staticmethod
    def normalize_metric(metric_name, value, generations_fitness_statistics_df: pd.DataFrame):
        match metric_name:
            case 'accuracy' | 'recall' | 'specificity' | 'f1' | 'aur_roc' | 'precision' | 'negative_precision' | 'r2':
                # Normalizar directamente de [0, 1] a [-π, π]
                return -math.pi + 2 * math.pi * value
            case 'mae' | 'mse' :
                # Asumiendo que el valor está normalizado previamente a [0, 1]
                normalized_value = max(0, min(value, 1))  # Clamping entre 0 y 1
                return -math.pi + 2 * math.pi * normalized_value
            case 'other':
                # Si es "other", devolvemos un valor neutro o indicativo
                max_value = max(generations_fitness_statistics_df["max"].values)
                min_value = min(generations_fitness_statistics_df["min"].values)

                # -- Evita división por cero si todos los valores son iguales
                if max_value == min_value:
                    return 0.0

                # -- Normalizar value al rango [-π, π]
                normalized_value = -math.pi + 2 * math.pi * (value - min_value) / (max_value - min_value)
                return normalized_value

            case _:
                raise ValueError(f"Métrica desconocida: {metric_name}")

    @staticmethod
    def add_new_qubit(qc: QuantumCircuit, new_param: Parameter, bounds_dict: Dict, max_qubits: int) -> QuantumCircuit:

        """
        Añadimos nuevos qubits a un circuito cuántico.
        :param qc: [QuantumCircuit] Circuito cuántico al que queremos agregar qubits.
        :param new_param: [Parameter] Nuevo parámetros que queremos agregar al circuito.
        :param bounds_dict: [Dict] Diccionario de bounds con las propiedades y sus limites/malformaciones
        :param max_qubits: [int] Número máximo de qubits que se agregan al circuito // 2.
        """

        # -- Obtenemos el número existente de qubits
        num_existing_qubits: int = qc.num_qubits

        # -- Obtenemos las claves de bounds_dict == propiedades
        num_bounds_keys: int = len([z for z in bounds_dict.keys()])

        # -- Calculamos los qubits se deben agregar (división x 2 porque habrá 2 padres con misma cantidad de qubits)
        num_new_qubits = (max_qubits - num_bounds_keys) // 2

        # -- Creamos un nombre único para cada nuevo registro
        new_register = QuantumRegister(num_new_qubits, name="metrics")
        qc.add_register(new_register)

        # -- Aplicamos las rotaciones y CNOTs para cada nuevo qubit añadido
        for i in range(num_new_qubits):

            # -- La i representa el índice del nuevo qubit en el circuito
            qubit_index = num_existing_qubits + i

            if i == 0:
                division_factor = 1
            else:
                # El primero se divide por 2, el siguiente por 3, etc por dinamismo de propiedades
                division_factor = i + 1

            # -- Definimos los niveles de repetición escalada de las secciones del circuito que se repiten
            for deep_level in range(1, 5):

                # -- Añadimos las rotaciones en z y en y
                qc.ry(new_param / (deep_level * division_factor), qubit_index)

                if i % 2 == 0:
                    qc.ry(new_param / (deep_level * division_factor), qubit_index)
                else:
                    qc.rz(new_param / (deep_level * division_factor), qubit_index)
                # qc.rz(new_param / (deep_level * division_factor), qubit_index)

                # -- Entrelazamos con el último qubit anterior (si existe)
                if qubit_index > 0:
                    qc.cx(qubit_index - 1, qubit_index)

            # -- Conectamos con el primer qubit para cerrar el ciclo
            for idx in range(num_existing_qubits):
                qc.cx(qubit_index, idx)

        return qc

    @staticmethod
    def create_parent_vqc(bounds_dict: dict, vqc_1: QuantumCircuit, vqc_2: QuantumCircuit) -> QuantumCircuit:
        """
        Versión simplificada del circuito parental.
        Se enfoca en preservar diversidad y control sin entrelazamiento excesivo.
        """

        # Validación
        if vqc_1.num_qubits != vqc_2.num_qubits:
            raise ValueError("Ambos circuitos deben tener el mismo número de qubits.")

        num_qubits = vqc_1.num_qubits
        total_qubits = 2 * num_qubits
        num_props = len(bounds_dict)

        qr1 = QuantumRegister(num_qubits, "q1")
        qr2 = QuantumRegister(num_qubits, "q2")
        qc = QuantumCircuit(qr1, qr2)

        # Paso 1: Composición directa de padres
        qc.compose(vqc_1, qr1, inplace=True)
        qc.compose(vqc_2, qr2, inplace=True)
        qc.barrier(label="Padres directos")

        # Paso 2: Entrelazamiento simple 1:1 (mantener herencia directa)
        for i in range(num_props):
            qc.cx(qr1[i], qr2[i])
        qc.barrier(label="Entrelazamiento 1:1 básico")

        # Paso 3: Rotaciones suaves (introducir variación sin caos)
        angle = np.pi / 8
        for i in range(num_props):
            qc.ry(angle, qr1[i])
            qc.rz(angle, qr2[i])
        qc.barrier(label="Rotaciones suaves")

        # Paso 4: Toque de aleatoriedad mínima
        # H en qubits aleatorios solo si total_qubits > 4
        if total_qubits >= 4:
            aleatorios = random.sample(range(total_qubits), 2)
            for q in aleatorios:
                qc.h(q)
            qc.barrier(label="Aleatoriedad mínima")

        return qc

    def adding_measurements_parent_vqc(self, bounds_dict: dict, combined_qc: QuantumCircuit, verbose: bool, graph: bool) -> dict:

        """
        Añadimos las medidas a los circuitos cuánticos concatenados de los mejores padres
        :param bounds_dict: Diccionario con los bounds de las propiedades de los padres.
        :param combined_qc: Circuito cuántico concatenado (unión de los circuitos cuánticos de los mejores padres)
        circuitos por propiedad del bounds_dict
        :param verbose: Variable booleana que define si se pinta o no los circuitos en cada iteración
        :param graph: Variable booleana que define si se muestra el gráfico de circuitos con medidas.
        :return: [dict] Circuitos
        """

        # -- Determinamos el numero de qubits para las propiedades y el total de qubits por circuito
        num_property_qubits = len([z for z in bounds_dict.keys()])
        qubits_per_circuit = combined_qc.num_qubits // 2  # -- 2 porque tenemos dos circuitos cuánticos concatenados

        circuits: dict = {}

        # -- Iteramos por las propiedades de los bounds
        for pid, parameter in enumerate([z for z in bounds_dict.keys()]):

            # -- Creamos una copia del circuito combinado
            combined_qc_duplicated = combined_qc.copy()

            # -- Agregamos registros clásicos para guardar los resultados de las mediciones
            qc_meas: QuantumCircuit = QuantumCircuit(combined_qc_duplicated.num_qubits)
            qc_meas.compose(combined_qc_duplicated, inplace=True)

            # -- Creamos y agregamos un registro clásico para las mediciones
            cr: ClassicalRegister = ClassicalRegister(combined_qc_duplicated.num_qubits, name='c')
            qc_meas.add_register(cr)

            # -- Definimos qué qubit se va a medir por parámetro (agregamos qubits de propiedades -ambos circuitos-)
            qubits_to_measure: list = [pid, qubits_per_circuit + pid]

            # -- Agregamos el resto de qubits a medir del primer circuito (los qubits de las métricas)
            for i in range(num_property_qubits, qubits_per_circuit):
                qubits_to_measure.append(i)

            # -- Agregamos el resto de qubits a medir del segundo circuito (los qubits de las métricas)
            for i in range(qubits_per_circuit + num_property_qubits, 2 * qubits_per_circuit):
                qubits_to_measure.append(i)

            # -- Agregamos las medidas propiamente dicho para todos los qubits seleccionados
            for qubit_idx in qubits_to_measure:
                qc_meas.measure(qubit_idx, qubit_idx)

            circuits[parameter] = qc_meas

            if verbose and graph is True:
                self.advanced_qc_plot(qc_meas)

        return circuits

    @staticmethod
    def advanced_qc_plot(combined_qc: QuantumCircuit):
        """
        Metodo avanzado de graficación de circuitos cuánticos
        :param combined_qc: Circuito cuántico a plotear
        :return:
        """

        # -- Dibujamos el circuito
        fig, ax = plt.subplots(figsize=(20, 10))

        # -- Customizamos parámetros del gráfico
        combined_qc.draw(
            "mpl",
            fold=-1,  # -- Previene folding
            scale=0.8,  # -- Escalamos los elementos
            style={
                "fontsize": 48,  # -- 48 Fuente primaria
                "subfontsize": 36,  # -- 36 Fuente secundaria
                "circuit_label_margin": 0.1,  # -- Ajuste de márgenes
                "gate_offset": 0.2,  # -- Espacio entre puertas
                "showindex": True,  # -- Muestra el índice de cada qubit
                "figwidth": 20,  # -- Ancho de la figura en inches
                "dpi": 4096,  # -- Alta resolución
                "cregbundle": False  # -- No aprieta los registros clásicos
            },
            ax=ax
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _generate_qc(num_qubits=32) -> QuantumCircuit:
        """
        Genera un circuito cuántico que genera números aleatorios dentro de un rango especificado por bounds_dict.
        :param num_qubits: Número de qubits que se utilizan para generar un valor que luego será denormalizado (-pi/pi)

        Returns: QuantumCircuit: Circuito cuántico listo para ser ejecutado.
        """

        # -- Creamos el circuito cuántico con qubits y bits clásicos
        qc = QuantumCircuit(num_qubits, num_qubits)

        # -- Aplicamos puertas Hadamard a todos los qubits para crear superposición
        qc.h(range(num_qubits))

        # -- Añadimos medidas a todos los qubits
        qc.measure(range(num_qubits), range(num_qubits))

        return qc

    @staticmethod
    def binary_to_float_and_normalize_pi(binary_str):

        # -- Convertimos la cadena binaria a un entero
        int_value = int([z for z in binary_str][0], 2)

        # -- Normalizamos el entero a un valor entre 0 y 1
        normalized_value = int_value / (2 ** len([z for z in binary_str][0]) - 1)

        # -- Mapeamos el valor normalizado al rango [-pi, pi]
        mapped_value = -np.pi + 2 * np.pi * normalized_value

        # -- Redondeamos a 10 decimales
        return round(mapped_value, 10)

    @staticmethod
    def rescaling_pi_to_integer_float(bounds_dict: dict, parameter: str, pi_number: float):
        """
        Reescala un valor desde el rango [-π, π] al rango especificado en bounds_dict.

        Args:
            bounds_dict: Diccionario con los límites de cada parámetro
            parameter: Nombre del parámetro a reescalar
            pi_number: Valor en el rango [-π, π] que se desea reescalar

        Returns:
            El valor reescalado, como entero si los límites son enteros,
            o como flotante con 10 decimales si los límites son flotantes.
        """
        # Obtener los límites del parámetro
        min_bound = min(bounds_dict[parameter]["malformation_limits"])
        max_bound = max(bounds_dict[parameter]["malformation_limits"])

        # Rango actual de pi_number (-pi, pi)
        current_min, current_max = -np.pi, np.pi

        # Realizar el mapeo lineal desde [-pi, pi] al rango deseado
        mapped_value = min_bound + (pi_number - current_min) * (max_bound - min_bound) / (current_max - current_min)

        # Si los límites son enteros, redondear a entero. Si no, redondear a 10 decimales.
        if isinstance(min_bound, int) and isinstance(max_bound, int):
            return int(round(mapped_value))
        else:
            return round(mapped_value, 10)

    @staticmethod
    def dividir_lista(lista: List[float], tamanio_grupo: int) -> List[List[float]]:
        return [lista[i:i + tamanio_grupo] for i in range(0, len(lista), tamanio_grupo)]

    def quantum_randomness(self, qcs: List[QuantumCircuit]):
        """
        Ejecuta circuitos cuánticos para generar números aleatorios utilizando un simulador u ordenador cuántico real.
        :param qcs: (List[QuantumCircuit]) Lista de circuitos cuánticos a ejecutar.

        Return: results: (List[Dict[str: int]) Lista de resultados en formato binario.
        """

        # -- Instanciamos el ejecutor del circuito
        _executor = self.get_quantum_technology()

        # -- Ejecutamos una vez el circuito cuántico con el ejecutor
        result: list = _executor.run(qcs, 1)

        return result

    def get_permutations_hybrid(self, bounds_dict: tuple, log2: int, shots: int) -> List:
        """
        Metodo híbrido para generar individuos para problema de tipo totally_restricted.
        Se utiliza un circuito cuántico combinado con procesamiento clásico.

        :param bounds_dict: [Dict] Diccionario de bounds.
        :param log2: [int] Número de permutaciones a generar.
        :param shots: Número de ciudades (puede ser grande, como 10).

        :return: [List] Lista de permutaciones generadas (potencial individuo)
        """

        # -- Definimos el límite minimo y máximo de los valores que se pueden generar
        min_val = min(bounds_dict["limits"])
        max_val = max(bounds_dict["limits"])
        value_range = max_val - min_val + 1

        # -- Utilizamos un circuito cuántico para generar "semillas" aleatorias
        qc = QuantumCircuit(log2, log2)

        # -- Añadimos puertas Hadamard a todos los qubits
        for i in range(log2):
            qc.h(i)

        # -- Añadimos mediciones a todos los qubits
        qc.measure(range(log2), range(log2))

        # -- Ejecutamos el circuito y obtenemos sus resultados
        results: List[Dict[str, int]] = self.execution_object.run([qc], shots=shots)

        # -- Obtenemos la primera semilla válida del resultado
        bitstring = next(iter(results[0].keys()))
        seed = int(bitstring, 2)
        random.seed(seed)

        # -- Generamos la permutación aleatoria (un solo individuo)
        individual = list(range(value_range))
        random.shuffle(individual)

        return individual

    # ---------------------------- Metodo específicos del QAOA -------------------------------------- #
    @staticmethod
    def create_tsp_qubo(element_matrix: Dict,
                        problem_type: Literal['minimize', 'maximize'],
                        return_to_origin: Literal['return_to_origin', 'no_return'] | None,
                        contains_zero: bool) -> QuadraticProgram:
        """
        Convierte un problema TSP a un QUBO.
        Si el nodo 0 está en el clúster, se fuerza como origen.

        :param element_matrix: [Dict] Diccionario {ID: (x, y)} con nodos del TSP.
        :param problem_type: [Literal['minimize', 'maximize']] Define el tipo de problema si es de minimización o maximización.
        :param return_to_origin: [Literal['return_to_origin', 'no_return'] | None] Variable que define si el problema retorna al origen o no.
        :param contains_zero: Define si el clúster posee el valor 0 (ej. Origen del recorrido).
        :return: Problema QUBO generado.
        """

        # -- Mapeamos índices internos para trabajar más fácil en matrices
        ids = list(element_matrix.keys())
        id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}

        # -- Obtenemos la cantidad de elementos del clúster
        n = len(ids)

        # -- Matriz de distancias
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(
                        np.array(element_matrix[ids[i]]) - np.array(element_matrix[ids[j]])
                    )

        qp = QuadraticProgram("TSP")

        # -- Declaramos variables binarias x_{i}_{j}
        for i in range(n):
            for j in range(n):
                if i != j:
                    qp.binary_var(f'x_{i}_{j}')

        # -- Función objetivo: minimizamos o maximizamos la suma de valores (ej. Las distancias entre ciudades)
        linear_obj = {f'x_{i}_{j}': dist[i, j] for i in range(n) for j in range(n) if i != j}
        qp.minimize(linear=linear_obj) if problem_type == "minimize" else qp.maximize(linear=linear_obj)

        # -- Si el cluster contiene el nodo 0 y no debe retornar al origen...
        if return_to_origin == "no_return" and contains_zero:

            # -- Obtenemos el índice del elemento 0
            zero_index = id_to_index[0]

            # -- Forzamos la salida desde el nodo 0
            qp.linear_constraint({f'x_{zero_index}_{j}': 1 for j in range(n) if j != zero_index}, '==', 1)

            # -- Prohibimos el regreso al nodo 0
            for i in range(n):
                if i != zero_index:
                    # Fijamos directamente la variable a 0
                    qp.linear_constraint({f'x_{i}_{zero_index}': 1}, '==', 0)

            # -- Hacemos que el resto de los nodos tengan entrada y salida una vez
            for j in range(n):
                if j != zero_index:
                    qp.linear_constraint({f'x_{i}_{j}': 1 for i in range(n) if i != j}, '==', 1)

            for i in range(n):
                if i != zero_index:
                    qp.linear_constraint({f'x_{i}_{j}': 1 for j in range(n) if j != i}, '==', 1)

        else:
            # -- En el resto de casos... todos los nodos entran y salen una sola vez
            for j in range(n):
                qp.linear_constraint({f'x_{i}_{j}': 1 for i in range(n) if i != j}, '==', 1)

            for i in range(n):
                qp.linear_constraint({f'x_{i}_{j}': 1 for j in range(n) if j != i}, '==', 1)

        # -- Convertimos el problema a formato QUBO
        qubo_converter = QuadraticProgramToQubo()
        return qubo_converter.convert(qp)

    @staticmethod
    def decode_solution(x: Dict, n: int, contains_zero: bool = False):
        """
        Decodifica la solución del QUBO.
        - Evita ciclos (rutas cerradas), incluyendo volver al nodo inicial.
        - Si contains_zero es True, 0 se pone al inicio.
        - Completa nodos faltantes aleatoriamente.

        :param x: [Dict] Diccionario con claves tipo 'x_i_j' y valores 0/1.
        :param n: Número total de nodos.
        :param contains_zero: Si el nodo 0 debe estar al inicio.
        :return: Lista con el orden de los nodos.
        """

        # -- Creamos el diccionario de nodos: de -> a
        edges = {int(k.split('_')[1]): int(k.split('_')[2]) for k, v in x.items() if v > 0.5 and k.startswith('x_')}

        # -- Creamos un set para evitar duplicados y la lista de combinaciones
        repeated = set()
        combination = []

        # -- Elegimos el nodo inicial (si el clúster posee 0 no aseguramos que es el punto de partida)
        start = 0 if contains_zero else next(iter(edges), 0)
        current = start
        combination.append(current)
        repeated.add(current)

        # -- Construimos la combinación evitando ciclos
        while current in edges:
            next_node = edges[current]

            # -- Si ya fue utilizado, cortamos
            if next_node in repeated:
                break

            combination.append(next_node)
            repeated.add(next_node)
            current = next_node

        # -- Si el último nodo apunta al primero, eliminamos el cierre
        if len(combination) > 1 and edges.get(combination[-1], None) == combination[0]:
            del edges[combination[-1]]

        # -- Agregamos nodos faltantes aleatoriamente
        missing = [i for i in range(n) if i not in repeated]
        random.shuffle(missing)
        combination.extend(missing)

        # -- Si contains_zero, aseguramos que 0 esté al inicio
        if contains_zero and combination[0] != 0:
            if 0 in combination:
                combination.remove(0)
            combination.insert(0, 0)

        return combination

    def solve_tsp_with_qaoa(self, element_matrix: Dict, problem_type: Literal['minimize', 'maximize'], contains_zero: bool, return_to_origin: Literal['return_to_origin', 'no_return'] | None):
        """
        Metodo para resolver el problema de optimización combinatoria mediante un QAOA.
        :param element_matrix: [Dict] Matriz de elementos del individuo que se deben combinar de la mejor manera.
        :param problem_type: Literal['minimize', 'maximize'] Definimos si el problema es de minimización o maximización.
        :param contains_zero: [bool] Variable booleana para saber si el clúster empieza con 0.
        :param return_to_origin: [Literal['return_to_origin', 'no_return'] | None Variable booleana para saber si el problema debe volver al origen.

        :return: combination, total_value, element_indices
        """

        # -- Calculamos la cantidad de elementos en la matriz de elementos (elementos del clúster)
        n = len([z for z in element_matrix.keys()])


        # -- Ejecutamos que transforma el problema a resolver en un problema de tipo QUBO
        qubo: QuadraticProgram = self.create_tsp_qubo(element_matrix=element_matrix,
                                                      problem_type=problem_type,
                                                      return_to_origin=return_to_origin,
                                                      contains_zero=contains_zero)

        # -- TODO: Definir si es simulador u ordenador cuántico

        # -- Definimos la tecnología cuántica de ejecución del QAOA
        sampler = AerSampler()

        # -- Definimos el optimizador del QAOA
        optimizer = COBYLA(maxiter=1000)

        # -- Instanciamos el QAOA
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=5)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)

        # -- Resolvemos el problema qubo con el optimizador
        result = qaoa_optimizer.solve(qubo)

        # -- Obtenemos el resultado para cada variable
        x = result.variables_dict

        # -- Decodificamos la solución
        element_indices = self.decode_solution(x, n, contains_zero=contains_zero)

        # -- Si no retornamos indices de elementos se devuelve una lista vacía
        if len(element_indices) < 1:
            return [], float("inf"), []

        # -- Obtenemos los ids de los elementos
        element_ids = list(element_matrix.keys())

        # -- Remapeamos la combinación a los elementos originales
        combination = [element_ids[i] for i in element_indices]

        return combination

    # ---------------------------- Metodos específicos para TSP  ------------------------------------- #
    @staticmethod
    def cluster_distances(element_matrix: Dict[str, tuple], num_clusters: int):
        """
        Metodo para separar los elementos del individuo en clústers.
        :param element_matrix: Matriz con valores de los elementos del individuo (ej: distancias entre ciudades -TSP-).
        :param num_clusters: Número de clústers que se buscan generar.
        :return: Clústers y centroides de cada clúster.
        """

        # -- Obtenemos los valores de la matriz de elementos
        coords = np.array(list(element_matrix.values()))

        # -- Instanciamos el KMeans con el número de clústers pretendido
        kmeans = KMeans(n_clusters=num_clusters)

        # -- Ejecutamos el modelo no-supervisado de clústering
        labels = kmeans.fit_predict(coords)

        # -- Generamos los clísters
        clusters = {i: {} for i in range(num_clusters)}
        for distance_id, label in zip(element_matrix.keys(), labels):
            clusters[label][distance_id] = element_matrix[distance_id]

        return clusters, kmeans.cluster_centers_

    @staticmethod
    def plot__restricted_solution(cities, order, title="Ruta completa"):
        plt.figure(figsize=(8, 5))
        coords = [cities[i] for i in order]
        x, y = zip(*coords)
        plt.plot(x, y, 'o-b')
        for i, city in enumerate(order[:-1]):
            cx, cy = cities[city]
            plt.text(cx + 0.01, cy + 0.01, str(city), fontsize=9)
        plt.title(title)
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    @staticmethod
    def euclidean(p1: tuple, p2: tuple) -> float:
        """
        Metodo para medir distancias eucledianas
        :param p1: Conjunto de coordenadas del individuo 1
        :param p2: Conjunto de coordenadas del individuo 2
        :return: Distancias eucledianas
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

__all__ = ['QuantumSimulator', 'QuantumMachine', 'QuantumTechnology']
