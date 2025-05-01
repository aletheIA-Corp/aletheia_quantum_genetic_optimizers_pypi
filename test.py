
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import datetime
from typing import Literal
from aletheia_quantum_genetic_optimizers import BoundCreator
from aletheia_quantum_genetic_optimizers.genetic_optimizer import GenethicOptimizer


# -- Definimos la función objetivo
def example_1_bounds_no_predefinidos():

    def objective_function(individual, dataset_loader: Literal["fetch_california_housing", "glass", "breast_cancer"] = "breast_cancer"):
        # Cargar dataset
        match dataset_loader:
            case "fetch_california_housing":
                data = fetch_california_housing()
                problem_type = 'regression'
                test_size = 0.2
            case "breast_cancer":
                data = load_breast_cancer()
                problem_type = 'binary'
                test_size = 0.7
            case "glass":
                data = fetch_openml(name="glass", version=1, as_frame=True)
                problem_type = 'multiclass'
                test_size = 0.2
            case _:
                data = fetch_california_housing()
                problem_type = 'regression'
                test_size = 0.2

        # print(f"PROBLEMA SKLEARN: {dataset_loader}")
        individual_dict = individual.get_individual_values()
        X, y = data.data, data.target

        # Dividir y normalizar
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=data.feature_names)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=data.feature_names)

        # Tiempo inicial
        start_time = datetime.datetime.now()

        # Entrenar y predecir
        if problem_type == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=int(individual_dict["n_estimators"]),
                max_depth=individual_dict["max_depth"],
                verbose=-1,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            # mae = mean_absolute_error(y_test, y_pred)
            # print(f"Mae: {mae} - R2: {score}")

        else:
            model = lgb.LGBMClassifier(
                n_estimators=int(individual_dict["n_estimators"]),
                max_depth=individual_dict["max_depth"],
                verbose=-1,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            # print(f"Acc: {score}")

        # Calcular tiempo
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

        # Penalización por tiempo
        # reference_time = 0.5  # segundos
        # max_penalty_ratio = 0.01  # 1%
        # time_penalty = max_penalty_ratio * score * (elapsed_time / reference_time)
        # time_penalty = min(time_penalty, max_penalty_ratio * score)
        # penalized_score = score - time_penalty
        # print(f"[{problem_type.upper()}] Score: {score:.4f} | Penalized: {penalized_score:.4f} | Time: {elapsed_time:.2f}s")

        return score

    # -- Creamos el diccionario de bounds
    bounds = BoundCreator()
    bounds.add_interval_bound("n_estimators", 50, 1000, 10, 1500, "int")
    bounds.add_predefined_bound("max_depth", (1, 2, 3, 4, 5, 6, 7, 8, 9), "int")

    return GenethicOptimizer(bounds_dict=bounds.get_bound(),
                             num_generations=5,
                             num_individuals=100,
                             max_qubits=20,
                             objective_function=objective_function,
                             metric_to_optimize="accuracy",
                             problem_restrictions="bound_restricted",
                             return_to_origin=None,
                             problem_type="maximize",
                             tournament_method="ea_simple",
                             podium_size=3,
                             mutate_probability=0.25,
                             mutate_gen_probability=0.25,
                             mutation_policy="normal",
                             verbose=False,
                             early_stopping_generations="gradient",
                             variability_explossion_mode="crazy",
                             variability_round_decimals=3,
                             randomness_quantum_technology="simulator",
                             randomness_service="aer",
                             optimization_quantum_technology="simulator",
                             optimization_service="aer",
                             qm_api_key="API_KEY",
                             qm_connection_service="ibm_quantum",
                             quantum_machine="least_busy"
                             )

def example_2_tsp():
    # Coordenadas de ciudades (ejemplo con 5 ciudades)

    original_cities = {
        0: (4.178515558765522, 3.8658505110962347),
        1: (9.404615166221248, 8.398020682045034),
        2: (0.3782334121284714, 8.295288013706802),
        3: (9.669161753695562, 5.593501025856912),
        4: (9.870966532678576, 4.756484445482374),
        5: (3.5045826424785007, 1.1043994011149494),
        6: (5.548867108083866, 5.842473649079045),
        7: (1.11377627026643, 1.304647970128091),
        8: (5.133591646349645, 3.8238217557909038),
        9: (7.074655346940579, 3.6554091142752734),
        10: (9.640123872995837, 1.3285594561699254),
        11: (0.021205320973052277, 7.018385604153457),
        12: (2.048903069073358, 2.562383464533476),
        13: (2.289964825687684, 4.325937821712228),
        14: (6.315627335092245, 3.7506598107821656),
        15: (1.0589427543395036, 6.2520630725232),
        16: (9.218474645470067, 4.106769373018785),
        17: (4.62163288328154, 9.583091224200263),
        18: (7.477615269848112, 7.597659062497909),
        19: (0.25092704950321565, 6.699275814039302),
    }

    def objective_function(individual):

        return_to_origin: bool = True

        original_cities = {
            0: (4.178515558765522, 3.8658505110962347),
            1: (9.404615166221248, 8.398020682045034),
            2: (0.3782334121284714, 8.295288013706802),
            3: (9.669161753695562, 5.593501025856912),
            4: (9.870966532678576, 4.756484445482374),
            5: (3.5045826424785007, 1.1043994011149494),
            6: (5.548867108083866, 5.842473649079045),
            7: (1.11377627026643, 1.304647970128091),
            8: (5.133591646349645, 3.8238217557909038),
            9: (7.074655346940579, 3.6554091142752734),
            10: (9.640123872995837, 1.3285594561699254),
            11: (0.021205320973052277, 7.018385604153457),
            12: (2.048903069073358, 2.562383464533476),
            13: (2.289964825687684, 4.325937821712228),
            14: (6.315627335092245, 3.7506598107821656),
            15: (1.0589427543395036, 6.2520630725232),
            16: (9.218474645470067, 4.106769373018785),
            17: (4.62163288328154, 9.583091224200263),
            18: (7.477615269848112, 7.597659062497909),
            19: (0.25092704950321565, 6.699275814039302),
        }

        # Obtener valores únicos en route y ordenarlos
        route = list(individual.get_individual_values().values())
        cities = original_cities.copy()

        if return_to_origin:
            route.append(route[0])
            cities[len([z for z in cities.keys()])] = cities[route[-1]]

        """if return_to_origin:
            cities = cities | route"""

        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))

        # Calcular la matriz de distancias con los nuevos índices
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    x1, y1 = cities[i]
                    x2, y2 = cities[j]
                    distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Calcular distancia total
        total_distance = sum(distance_matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

        return total_distance

    # -- Creamos el diccionario de bounds
    range_list: tuple = tuple([z for z in range(0, 20)])
    bounds = BoundCreator()
    bounds.add_predefined_bound("city_zero", range_list, "int")
    bounds.add_predefined_bound("city_first", range_list, "int")
    bounds.add_predefined_bound("city_second", range_list, "int")
    bounds.add_predefined_bound("city_third", range_list, "int")
    bounds.add_predefined_bound("city_fourth", range_list, "int")
    bounds.add_predefined_bound("city_fifth", range_list, "int")
    bounds.add_predefined_bound("city_sixth", range_list, "int")
    bounds.add_predefined_bound("city_seventh", range_list, "int")
    bounds.add_predefined_bound("city_eighth", range_list, "int")
    bounds.add_predefined_bound("city_ninth", range_list, "int")
    bounds.add_predefined_bound("city_tenth", range_list, "int")
    bounds.add_predefined_bound("city_eleventh", range_list, "int")
    bounds.add_predefined_bound("city_twelfth", range_list, "int")
    bounds.add_predefined_bound("city_thirteenth", range_list, "int")
    bounds.add_predefined_bound("city_fourteenth", range_list, "int")
    bounds.add_predefined_bound("city_fifteenth", range_list, "int")
    bounds.add_predefined_bound("city_sixteenth", range_list, "int")
    bounds.add_predefined_bound("city_seventeenth", range_list, "int")
    bounds.add_predefined_bound("city_eighteenth", range_list, "int")
    bounds.add_predefined_bound("city_nineteenth", range_list, "int")

    return GenethicOptimizer(bounds_dict=bounds.get_bound(),
                             num_generations=2,
                             num_individuals=3,
                             max_qubits=20,
                             objective_function=objective_function,
                             metric_to_optimize="other",
                             problem_restrictions="totally_restricted",
                             return_to_origin="return_to_origin",
                             problem_type="minimize",
                             tournament_method="ea_simple",
                             podium_size=3,
                             mutate_probability=0.25,
                             mutate_gen_probability=0.10,
                             mutation_policy="normal",
                             verbose=False,
                             early_stopping_generations="gradient",
                             variability_explossion_mode="crazy",
                             variability_round_decimals=3,
                             randomness_quantum_technology="simulator",
                             randomness_service="aer",
                             optimization_quantum_technology="simulator",
                             optimization_service="aer",
                             qm_api_key="API_KEY",
                             qm_connection_service="ibm_quantum",
                             quantum_machine="least_busy",
                             element_matrix=original_cities
                             )

# -- Instanciamos la función que ejecuta el AletheIA Quantum Genetic Optimizer
genetic_optimizer_object: GenethicOptimizer = example_1_bounds_no_predefinidos()

# -- Instanciamos el ploteo de resultados
genetic_optimizer_object.plot_generation_stats()
# genetic_optimizer_object.plot_evolution_animated()
genetic_optimizer_object.plot_evolution()

# -- Obtenemos los valores del mejor individuo
best_individual_value = genetic_optimizer_object.get_best_individual_values()

# -- Pintamos los valores del mejor individuo
print(best_individual_value)
