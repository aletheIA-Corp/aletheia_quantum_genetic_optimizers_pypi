from abc import ABC, abstractmethod
from info_tools import InfoTools
from typing import Literal

import pandas as pd
import numpy as np



class VariabilityExplossion(ABC):
    def __init__(self, early_stopping_generations: int, problem_type: Literal['maximize', 'minimize'], round_decimals: int = 3, verbose: bool = False):
        """
        Constructor de la clase CrazyVariabilityExplossion.
        Este constructor inicializa los parámetros relacionados con el early stopping y la explosión de variabilidad,
        heredando de la clase base VariabilityExplossion.

        :param early_stopping_generations: [int] Número de generaciones que se deben esperar para activar la explosión de variabilidad si el min o max se repite.
        :param problem_type: [Literal] Tipo de problema a resolver, puede ser 'maximize' o 'minimize'.
        :param round_decimals: [int] Número de decimales para redondear en las estadísticas de variabilidad.
        :param verbose: [bool] Indica si se deben mostrar mensajes detallados sobre el estado de la explosión de variabilidad.
        """

        # -- Obtenemos las generaciones que voy a esperar para que si se repite el min o max, arrancar la explosion de variabilidad
        self.early_stopping_generations: int = early_stopping_generations

        # -- Almacenamos el tipo de problema e instancio InfoTools
        self.problem_type: Literal['maximize', 'minimize'] = problem_type
        self.IT: InfoTools = InfoTools()
        self.round_decimals: int = round_decimals
        self.verbose: int = verbose

        # -- Inicializamos propiedades de control de flujo
        self.early_stopping_generations_executed_counter: int = 0  # Contador de cuantas generacioes han transcurrido desde la ultima explosion de variabilidad
        self.total_early_stopping_generations_executed_counter: int = 0  # Contador de cuantas generacioes han transcurrido desde la primera explosion de variabilidad
        self.early_stopping_generations_executed: bool = False  # Boleana que indica si estamos en modo explosion de variabilidad

        # -- Inicializamos propiedades de mutacion (valores por defecto)
        self.mutate_probability: float = 0.4
        self.mutate_gen_probability: float = 0.25
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = 'soft'

    @abstractmethod
    def evaluate_early_stopping(self, generations_fitness_statistics_df: pd.DataFrame | None) -> None:
        """
        Metodo que evalúa si se deben activar las condiciones para el early stopping y la explosión de variabilidad.
        Si el min o max en las últimas generaciones ha sido constante, se activa la explosión de variabilidad.
        :param generations_fitness_statistics_df: [pd.DataFrame | None] DataFrame con las estadísticas de fitness de las generaciones.

        :return: tuple (m_proba, m_gen_proba, m_policy, early_stopping_generations_execute)
            - m_proba: Probabilidad de mutación de un individuo.
            - m_gen_proba: Probabilidad de mutación de un gen.
            - m_policy: Política de mutación (puede ser 'soft', 'normal', 'hard').
            - early_stopping_generations_execute: Indica si se debe ejecutar el early stopping.
        """
        pass

    @abstractmethod
    def execute_variability_explossion(self):
        """
        Metodo que ejecuta la explosión de variabilidad en caso de que el algoritmo genético haya quedado atrapado en un mínimo local.
        Este metodo incrementa las probabilidades de mutación para intentar escapar de un mínimo local y mejorar el rendimiento del algoritmo.

        :return: tuple (mutate_probability, mutate_gen_probability, mutation_policy, early_stopping_generations_executed)
            - mutate_probability: Probabilidad de mutación de un individuo.
            - mutate_gen_probability: Probabilidad de mutación de un gen.
            - mutation_policy: Política de mutación (puede ser 'soft', 'normal', 'hard').
            - early_stopping_generations_executed: Si se debe ejecutar el early stopping.
        """
        pass

    @abstractmethod
    def stop_genetic_iterations(self, generations_fitness_statistics_df: pd.DataFrame | None):
        """
        Metodo que evalúa si es necesario detener el proceso de evolución en base a las condiciones de early stopping.
        Si el algoritmo ha estado ejecutando la explosión de variabilidad y no ha mejorado el resultado en un número determinado de generaciones,
        se detiene la evolución.
        :param generations_fitness_statistics_df: [pd.DataFrame | None] DataFrame con las estadísticas de fitness de las generaciones.

        :return: [bool] Retorna True si se debe detener la evolución, False si se debe continuar.
        """
        pass

    @abstractmethod
    def print_variability_status(self):
        """
        Metodo que imprime el estado de la explosión de variabilidad.
        Muestra información sobre si la explosión está activada, cuántas generaciones lleva activa y si se ha mejorado el resultado.
        """
        pass


class CrazyVariabilityExplossion(VariabilityExplossion):
    def __init__(self, early_stopping_generations: int, problem_type: Literal['maximize', 'minimize'], round_decimals: int = 3, verbose: bool = False):

        # -- Obtengo las generaciones que voy a esperar para que si se repite el min o max, arrancar la explosion de variabilidad
        super().__init__(early_stopping_generations, problem_type, round_decimals, verbose)

    def evaluate_early_stopping(self, generations_fitness_statistics_df: pd.DataFrame | None) -> tuple:
        """
        Metodo para evaluar si se han cumplido las condiciones del early stopping
        :param generations_fitness_statistics_df: [pd.Dataframe] Df con la información general de la generación
        :return:
            m_proba: Probabilidad de mutación de un individuo
            m_gen_proba: Probabilidad de mutación de un gen
            m_policy: Política de mutación
            early_stopping_generations_execute: Si se debe ejecutar el early stopping
        """

        # -- Aplicamos la explosión de variabilidad, si en las últimas early_stopping_generations el max es igual
        if generations_fitness_statistics_df is not None:
            df: pd.DataFrame = generations_fitness_statistics_df
            df_tail = df.tail(self.early_stopping_generations)
            if df.shape[0] >= int(self.early_stopping_generations * 2):
                mode_values = df_tail["min"].values if self.problem_type == "minimize" else df_tail["max"].values
                if np.all(mode_values == mode_values[0]):
                    return self.execute_variability_explossion()

        # -- Si ya se ha ejecutado la explosion de variabilidad, se vuelve a ejecutar en cada iteracion
        if self.early_stopping_generations_executed:
            return self.execute_variability_explossion()

        # -- Si no ha pasado Nada, retornamos None
        return None, None, None, None

    def execute_variability_explossion(self):
        """
        Metodo para ejecutar una explosion de variabilidad en caso de que el genético haya quedado clavado en un mínimo. Esto pretende dar una oportunidad extra al modelo genético
        de encontrar un mínimo mejor cuando ha quedado atrapado en un mínimo local.
        :return:
            mutate_probability: Probabilidad de mutación de un individuo
            mutate_gen_probability: Probabilidad de mutación de un gen
            mutation_policy: Política de mutación
            early_stopping_generations_executed: Si se debe ejecutar el early stopping
        """

        if self.early_stopping_generations_executed:
            self.mutate_probability: float = 0.4
            self.mutate_gen_probability: float = 0.25
            self.mutation_policy: Literal['soft', 'normal', 'hard'] = 'soft'
        else:
            self.mutate_probability: float = 0.9
            self.mutate_gen_probability: float = 0.5
            self.mutation_policy: Literal['soft', 'normal', 'hard'] = 'hard'
            self.early_stopping_generations_executed = True

        return self.mutate_probability, self.mutate_gen_probability, self.mutation_policy, self.early_stopping_generations_executed

    def stop_genetic_iterations(self, generations_fitness_statistics_df: pd.DataFrame | None):
        """
        Metodo para que en caso de que si se ha ejecutado el execute_variability_explossion (early_stopping_generations_executed = True), y hayan pasado n generaciones, se detiene.
        :param generations_fitness_statistics_df: [pd.Dataframe] Df con los resultados estadísticos de los individuos

        :return: Valor booleano
        """

        if self.early_stopping_generations_executed_counter >= self.early_stopping_generations:

            # -- Si no ha mejorado en las vueltas, True, sino reseteamos el contador para darle margen
            df: pd.DataFrame = generations_fitness_statistics_df
            best_value = max(df["max"].values) if self.problem_type == "maximize" else min(df["max"].values)
            best_counter_value = max(df.tail(self.early_stopping_generations_executed_counter)["max"].values) if self.problem_type == "maximize" else (
                min(df.tail(self.early_stopping_generations_executed_counter)["max"].values))

            if self.problem_type == "maximize":
                if best_counter_value <= best_value:
                    self.IT.info_print("El CRAZY MODE no ha conseguido mejorar el resultado. Detenemos el proceso de evolución", "light_yellow")
                    return True
                else:
                    self.IT.info_print("El CRAZY MODE ha mejorado el mejor resultado. Le damos margen de generaciones", "light_blue")
                    self.early_stopping_generations_executed_counter = 0
                    return False
            else:
                if best_counter_value >= best_value:
                    self.IT.info_print("El CRAZY MODE no ha conseguido mejorar el resultado. Detenemos el proceso de evolución", "light_yellow")
                    return True
                else:
                    self.IT.info_print("El CRAZY MODE ha mejorado el mejor resultado. Le damos margen de generaciones", "light_blue")
                    self.early_stopping_generations_executed_counter = 0
                    return False

        if self.early_stopping_generations_executed:
            self.early_stopping_generations_executed_counter += 1
            self.total_early_stopping_generations_executed_counter += 1

        # -- En caso de verbose activo, mostramos la info
        if self.verbose:
            self.print_variability_status()

        return False

    def print_variability_status(self):
        """
        Metodo que imprime el estado de la explosión de variabilidad.
        Muestra información sobre si la explosión está activada, cuántas generaciones lleva activa y si se ha mejorado el resultado.
        """

        self.IT.sub_intro_print(f"Resumen de CrazyVariabilityExplossion")
        self.IT.info_print(f"CrazyVariabilityExplossion Activated: {self.early_stopping_generations_executed}",
                           "light_red" if self.early_stopping_generations_executed else "light_green")
        if self.early_stopping_generations_executed:
            self.IT.info_print(f"Generaciones que lleva activo el CrazyVariabilityExplossion: {self.total_early_stopping_generations_executed_counter}")
            self.IT.info_print(f"Generaciones desde ultima mejora: {self.early_stopping_generations_executed_counter}")