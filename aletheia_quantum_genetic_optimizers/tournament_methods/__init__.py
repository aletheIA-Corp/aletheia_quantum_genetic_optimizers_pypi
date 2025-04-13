from abc import ABC, abstractmethod
from aletheia_quantum_genetic_optimizers.individuals import Individual
from typing import List, Literal
import random
from info_tools import InfoTools


class Tournament(ABC):
    def __init__(self, podium_size: int = 3, problem_type: Literal["minimize", "maximize"] = "minimize", verbose: bool = False):
        self.podium_size: int = podium_size
        self.problem_type: Literal["minimize", "maximize"] = problem_type
        self.verbose: bool = verbose
        self.IT: InfoTools = InfoTools()

    @abstractmethod
    def run_tournament(self, individuals_list: List[Individual]):
        pass


class EaSimple(Tournament):
    def __init__(self, podium_size: int = 3, problem_type: Literal["minimize", "maximize"] = "minimize", verbose: bool = False):
        super().__init__(podium_size, problem_type, verbose)

    def run_tournament(self, individuals_list: List[Individual]) -> List[Individual]:

        """
        Metodo de torneo EaSimple
        :param individuals_list: Lista de individuos de una generación

        :return: La lista de los individuos ganadores
        """

        # -- Filtramos los individuos con malformaciones
        individuals_list: list = [ind for ind in individuals_list if not ind.get_individual_malformation()]

        # -- Definimos la lista de ganadores del torneo
        selected_individuals = []

        while individuals_list:

            # -- Seleccionamos hasta podium_size individuos aleatoriamente
            competitors = random.sample(individuals_list, min(self.podium_size, len(individuals_list)))

            # -- Seleccionamos el mejor según el tipo de problema
            winner = max(competitors, key=lambda ind: ind.get_individual_fitness()) if self.problem_type == "maximize" else min(competitors, key=lambda ind: ind.get_individual_fitness())
            selected_individuals.append(winner)

            # -- Eliminamos todos los competidores de la lista (no solo el ganador)
            individuals_list = [ind for ind in individuals_list if ind not in competitors]

        return selected_individuals


__all__ = ['Tournament', 'EaSimple']


