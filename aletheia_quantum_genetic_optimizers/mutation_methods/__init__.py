from typing import List, Literal
from aletheia_quantum_genetic_optimizers.individuals import Individual
import numpy as np
import random


class Mutation:
    def __init__(self, individual_list: List[Individual],
                 mutate_probability: float,
                 mutate_gen_probability: float,
                 mutation_policy: Literal['soft', 'normal', 'hard'],
                 problem_restrictions: Literal['bound_restricted', 'totally_restricted'],
                 num_generations: int):
        """
        Inicializa una instancia de la clase encargada de gestionar una población de individuos en un algoritmo evolutivo.

        Parámetros:
        - individual_list (List[Individual]): Lista de individuos que conforman la población inicial.
        - mutate_probability (float): Probabilidad de que un individuo sufra mutación en una generación.
        - mutate_gen_probability (float): Probabilidad de que un gen específico dentro de un individuo sea mutado.
        - mutation_policy (Literal['soft', 'normal', 'hard']): Define la severidad de la mutación aplicada.
        - problem_restrictions (Literal['bound_restricted', 'totally_restricted']): Tipo de restricción aplicada sobre los valores posibles de los genes.
        - num_generations (int): Número total de generaciones que el algoritmo debe ejecutar.
        """

        self.individual_list: List[Individual] = individual_list
        self.mutate_probability: float = mutate_probability
        self.mutate_gen_probability: float = mutate_gen_probability
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = mutation_policy

        # -- Almaceno en una propiedad las restricciones a aplicar
        self.problem_restrictions: Literal['bound_restricted', 'totally_restricted'] = problem_restrictions
        self.bounds_dict = self.individual_list[0].bounds_dict
        self.num_generations: int = num_generations

    def run_mutation(self):
        match self.problem_restrictions:
            case "bound_restricted":
                return self.bound_restricted_mutation()
            case "totally_restricted":
                return self.totally_restricted_mutation()

    # <editor-fold desc="Mutaciones en funcion del self.problem_restrictions    --------------------------------------------------------------------------------------------------">

    def bound_restricted_mutation(self) -> List[Individual]:

        """
        Política de mutación para propiedades entre bounds
        :return: [List[Individual]] Lista de individuos
        """

        # -- Evaluamos la necesidad de mutar cada individuo de la lista de individuos de la generación
        for individual in self.individual_list:

            # -- Si no se debe mutar, entonces se continúa con el siguiente individuo
            if np.random.rand() >= self.mutate_probability:
                continue

            # -- Si se debe mutar el individuo...

            # -- Realizamos los cruces de cada gen
            for parameter, bound in self.bounds_dict.items():

                # -- Analizamos por tipo de bound
                match bound['bound_type']:

                    case 'predefined':

                        # -- Si debe mutar una propiedad | gen
                        if np.random.rand() < self.mutate_gen_probability:
                            mutated_parameter, mutated_angles = self.mutation_bit_flip(individual, parameter)
                            individual.set_individual_value(parameter, mutated_parameter)
                            individual.set_individual_angles(parameter, mutated_angles)

                    case 'interval':
                        if np.random.rand() < self.mutate_gen_probability:
                            mutated_parameter, mutated_angles = self.mutation_uniform(individual, parameter)
                            individual.set_individual_value(parameter, mutated_parameter)
                            individual.set_individual_angles(parameter, mutated_angles)

        return self.individual_list

    def totally_restricted_mutation(self):
        """Realiza una mutación por intercambio en la lista dada."""

        for individual in self.individual_list:
            if np.random.rand() >= self.mutate_probability:
                continue

            individual_values = [z for z in individual.get_individual_values().values()]

            if len(individual_values) < 2:
                return individual_values  # No se puede mutar si hay menos de 2 elementos
            i, j = random.sample(range(len(individual_values)), 2)  # Escoge dos índices distintos
            individual_values[i], individual_values[j] = individual_values[j], individual_values[i]  # Intercambia los valores

            individual.set_individual_values(individual_values)

        return self.individual_list

    # </editor-fold>

    # <editor-fold desc="Metodos de mutacion de genes    -------------------------------------------------------------------------------------------------------------------------">

    def mutation_bit_flip(self, individual: Individual, parameter: str):
        """
        Mutación de variables continuas dentro de un rango
        :param individual: [Individual] Individuo de la generación del cual se debe mutar una de sus propiedades|genes
        :param parameter: [str] Parámetro | Ángulos de rotación que se deben mutar
        :return:
        """
        total_generations = self.num_generations
        current_generation: int = individual.generation
        mutation_progress = 1 - (current_generation / total_generations)  # -- Reduce progresivamente

        # -- Definimos los límites de mutación en función de la política de mutación

        match self.mutation_policy:

            case 'soft':
                # -- Inicialmente tomamos el 40% del rango total, y lo vamos reduciendo con cada generación
                percentage_range = 0.4 * mutation_progress

            case 'normal':
                # -- Inicialmente tomamos el 60% del rango total, y lo vamos reduciendo con cada generación
                percentage_range = 0.6 * mutation_progress

            case 'hard' | _:
                # -- Mantenemos el rango completo sin reducción progresiva
                percentage_range = 0.8 * mutation_progress

        # -- Obtenemos los valores dentro de los límites permitidos
        possible_values = sorted(self.bounds_dict[parameter]["malformation_limits"])
        current_value = individual.get_individual_values()[parameter]

        # -- Determinamos el rango basado en la generación actual
        min_val, max_val = min(possible_values), max(possible_values)
        range_span = max_val - min_val
        lower_bound = max(min_val, current_value - (range_span * percentage_range))
        upper_bound = min(max_val, current_value + (range_span * percentage_range))

        # -- Filtramos valores dentro del rango calculado
        filtered_values = [z for z in possible_values if lower_bound <= z <= upper_bound and z != current_value]

        # -- Si no quedan valores posibles, tomar el más cercano válido
        if not filtered_values:
            filtered_values = [z for z in possible_values if z != current_value]

        # -- Mutamos las propiedades|hiperparámetros (numeros enteros|flotantes)
        mutated_parameter = float(np.random.choice(filtered_values)) if self.bounds_dict[parameter]["type"] == "float" else int(np.random.choice(filtered_values))

        # -- Calculamos el valor mutado de los ángulos en la escala [-pi, pi]

        # -- Aplicamos la proporción de cambio a cada ángulo, contemplando los cambios de signo
        modified_angles: List[float] = []

        for angle in individual.get_vqc_parameters_values()[parameter]:

            # -- Convertimos el valor actual a su escala angular correspondiente.
            current_angle = ((current_value - min_val) / range_span) * (2 * np.pi) - np.pi

            # -- Convertimos el valor mutado a su escala angular correspondiente.
            mutated_angle = ((mutated_parameter - min_val) / range_span) * (2 * np.pi) - np.pi

            # -- Calculamos la proporción de cambio entre ángulos.
            mutation_angle_change = mutated_angle - current_angle

            # -- Aplicamos la proporción de cambio al ángulo actual.
            modified_angle = angle + mutation_angle_change

            # -- Si la mutación cruza de positivo a negativo o viceversa, se debe reflejar en el ángulo.
            if (angle > 0 > modified_angle) or (angle < 0 < modified_angle):
                modified_angle = -modified_angle  # Invertimos el signo si cruza el cero

            # -- Normalizamos el ángulo al rango [-π, π].
            modified_angle = ((modified_angle + np.pi) % (2 * np.pi)) - np.pi

            # -- Agregamos el ángulo modificado a la lista de nuevos ángulos.
            modified_angles.append(float(modified_angle))

        return mutated_parameter, modified_angles

    def mutation_uniform(self, individual, parameter):
        """
        Realiza una mutación uniforme en valores enteros o reales.
        :param individual: Individuo que se quiere mutar en alguno de sus genes.
        :param parameter: Parámetro que se quiere modificar del individuo.

        :return: Parámetro mutado.
        """

        total_generations = self.num_generations
        current_generation: int = individual.generation
        mutation_progress = 1 - (current_generation / total_generations)  # Reduce progresivamente

        # -- Definimos los límites de mutación en función de la política
        match self.mutation_policy:

            case 'soft':
                percentage_range = 0.4 * mutation_progress

            case 'normal':
                percentage_range = 0.6 * mutation_progress

            case 'hard' | _:
                percentage_range = 0.8 * mutation_progress

        # -- Obtenemos los valores dentro de los límites permitidos
        possible_values = sorted(self.bounds_dict[parameter]["malformation_limits"])
        current_value = individual.get_individual_values()[parameter]

        # -- Determinamos el rango basado en la generación actual
        min_val, max_val = min(possible_values), max(possible_values)
        range_span = max_val - min_val
        lower_bound = max(min_val, current_value - (range_span * percentage_range))
        upper_bound = min(max_val, current_value + (range_span * percentage_range))

        # -- Ajustamos los límites para valores flotantes o enteros

        # -- Aseguramos que el valor mutado esté dentro de los límites definidos
        if self.bounds_dict[parameter]["type"] == "float":
            mutated_parameter = float(np.random.uniform(lower_bound, upper_bound))
        else:
            mutated_parameter = int(np.random.uniform(np.floor(lower_bound), np.ceil(upper_bound)))

        # -- Control adicional para evitar valores extremos no deseados
        mutated_parameter = int(np.clip(mutated_parameter, min_val, max_val)) if isinstance(mutated_parameter, int) else float(np.clip(mutated_parameter, min_val, max_val))

        # -- Modificamos los ángulos en la misma proporción
        modified_angles: List[float] = []

        for angle in individual.get_vqc_parameters_values()[parameter]:

            # -- Convertimos el valor actual a su escala angular correspondiente.
            current_angle = ((current_value - min_val) / range_span) * (2 * np.pi) - np.pi

            # -- Convertimos el valor mutado a su escala angular correspondiente.
            mutated_angle = ((mutated_parameter - min_val) / range_span) * (2 * np.pi) - np.pi

            # -- Calculamos la proporción de cambio entre ángulos.
            mutation_angle_change = mutated_angle - current_angle

            # -- Aplicamos la proporción de cambio al ángulo actual.
            modified_angle = angle + mutation_angle_change

            # -- Si la mutación cruza de positivo a negativo o viceversa, se debe reflejar en el ángulo.
            if (angle > 0 > modified_angle) or (angle < 0 < modified_angle):
                modified_angle = -modified_angle  # Invertimos el signo si cruza el cero

            # -- Normalizamos el ángulo al rango [-π, π].
            modified_angle = ((modified_angle + np.pi) % (2 * np.pi)) - np.pi

            # -- Agregamos el ángulo modificado a la lista de nuevos ángulos.
            modified_angles.append(float(modified_angle))

        return mutated_parameter, modified_angles

    # </editor-fold>

__all__ = ['Mutation']