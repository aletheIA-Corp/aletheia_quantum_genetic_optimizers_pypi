# AletheIA Quantum Genetic Optimizers (AQGO)

## Equipo de desarrollo

| Nombre                  | Rol                         | Información de contacto  | Perfil de LinkedIn                                          |
|-------------------------|-----------------------------|--------------------------|-------------------------------------------------------------|
| Daniel Sarabia Torres   | Full Stack AI & Q Developer | dsarabiatorres@gmail.com | https://es.linkedin.com/in/danielsarabiatorres              |
| Luciano Ezequiel Bizin  | Full Stack AI & Q Developer | lucianobizin@gmail.com   | https://www.linkedin.com/in/luciano-ezequiel-bizin-81b85497 |

## Instalación

```pip install aqgo```

## Introducción

Este proyecto implementa un algoritmo genético cuántico diseñado para abordar problemas de optimización combinatoria mediante el uso de tecnologías cuánticas (AerSimulator y ordenadores cuánticos de IBM). 

La arquitectura del algoritmo fusiona principios clásicos de evolución con procesamiento de datos de manera cuántica, buscando así, ganar capacidad exploratoria de manera más eficiente en grandes espacios de soluciones.

El objetivo principal del AletheIA Quantum Genetic Optimizar es resolver dos clases generales de problemas:

- *`bounds_restricted`*: Problemas con límites o restricciones estructurales parciales, en los que se permite cierta flexibilidad pero dentro de márgenes definidos (por ejemplo, pensar en la búsqueda de hiperparámetros para un modelo de ML o DL).

- *`totally_restricted`*: Problemas con restricciones estrictas en los valores permitidos y en las configuraciones válidas, como el Traveling Salesman Problem (TSP) -único problema de este tipo que se puede resolver al momento con el optimizador-, donde solo se aceptan permutaciones válidas sin repeticiones.

## Uso de la computación cuántica

La computación cuántica se integra en la librería de dos maneras:

1. *Generación de aleatoriedad cuántica*: Para crear poblaciones iniciales o aplicar mutaciones. Se aprovecha la aleatoriedad genuina ofrecida por simuladores cuánticos o por hardware cuántico real, logrando una diversificación más natural de las soluciones.

2. *Reproducción de hijos mediante VQC*: Para generar los hijos en los problemas de tipo `bounds_restricted` se utiliza un Variational Quantum Circuit que une los circuitos cuánticos de ambos padres ganadores de los que se reproducirá el hijo.

3. *Solución de subproblemas mediante QAOA*: Para generar los hijos en los problemas de tipo `totally_restricted`, se emplea la clase `QAOA` de Qiskit para resolver sub-rutas óptimas que luego se concatenan en una única ruta. Es decir, con el QAOA se resuelven distintos subproblemas de optimización cuadrática binaria (QUBO), potenciando la búsqueda hacia regiones prometedoras del espacio de soluciones.

--------------------------------------------

## Explicación detallada de casos de uso 

Se explica de manera detallada ambos casos de uso: `bounds_restricted` (búsqueda de hiperparámetros para un modelo de ML) y `totally_restricted` (TSP)

## Caso de uso 1: 

En este apartado se describe el caso de uso para resolver el problema de encontrar los mejores hiperparámetros para un modelo de IA, en este caso, para un modelo básico de ML.

### Entendimiento del problema: El problema de encontrar los mejores hiperparámetros para un modelo de IA

El problema encontrar los mejores hiperparámetros para un modelo de IA puede ser abarcado sin ningún problema con los simuladores u ordenadores cuánticos de hoy en día. 

Imaginar que queremos entrenar una red neuronal o un modelo de ML, y para tal situación, necesitamos encontrar los mejores hiperparámetros para nuestro modelo.

El desafío radica en que, a medida que aumenta el número de hiperparámetros a optimizar (aunque también puede ser la estructura de la red neuronal o cantidad de nueronas si se ha categorizado estos valores), es necesario encontrar una muy buena combinación de hiperparámetros, lo que resulta muy costoso utilizando métodos tradicionales.

### Ejemplo de código

```

def example_1_bounds_no_predefinidos():

    # -- Definimos la función objetivo
    def objective_function(individual, dataset_loader: Literal["fetch_california_housing", "glass", "breast_cancer"] = "breast_cancer"):
    
        # Cargamos el dataset según el tipo de problema a resolver (claramente se puede utilizar un dataset propio)
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

        # -- Obtenemos el individuo a evaluar
        individual_dict = individual.get_individual_values()
        
        # -- Dividimos los datos y la target
        X, y = data.data, data.target

        # -- Dividimos los datos en train y test, y los normalizamos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=data.feature_names)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=data.feature_names)

        # -- Entrenamos y predecimos
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
            mae = mean_absolute_error(y_test, y_pred)

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

        return score

    # -- Creamos el diccionario de bounds
    bounds = BoundCreator()
    bounds.add_interval_bound("n_estimators", 50, 1000, 10, 1500, "int")
    bounds.add_predefined_bound("max_depth", (1, 2, 3, 4, 5, 6, 7, 8, 9), "int")

    # -- Instanciamos el optimizador
    return GenethicOptimizer(bounds_dict=bounds.get_bound(),
                             num_generations=50,
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
                             verbose=True,
                             early_stopping_generations="gradient",
                             variability_explossion_mode="crazy",
                             variability_round_decimals=3,
                             randomness_quantum_technology="simulator",
                             randomness_service="aer",
                             optimization_quantum_technology="simulator",
                             optimization_service="aer",
                             qm_api_key="IBMQUANTUM_API",
                             qm_connection_service="ibm_quantum",
                             quantum_machine="least_busy"
                             )
                             
# -- Instanciamos la función que ejecuta el AletheIA Quantum Genetic Optimizers
genetic_optimizer_object: GenethicOptimizer = example_1_bounds_no_predefinidos()

# -- Instanciamos el ploteo de resultados (opcional)
genetic_optimizer_object.plot_generation_stats()
genetic_optimizer_object.plot_evolution()

# -- Obtenemos los valores del mejor individuo
best_individual_value = genetic_optimizer_object.get_best_individual_values()

# -- Pintamos los valores del mejor individuo en consola
print(best_individual_value)

```

#### El enfoque genético cuántico

El código proporcionado implementa un algoritmo genético cuántico para abordar el problema de encotrar los mejores hiperparámetros de un modelo de IA. 

A continuación, se explica cómo funciona este algoritmo y sus potenciales ventajas:

1. *Introducción programática:* 

* El `individual` que recibe la función `objective_function` representa un conjunto de hiperparámetros (-por ejemplo: {max_depth: 2, n_estimators: 325-).
* El aspecto cuántico de este algoritmo genético se ejecuta al momento de generarse la primera generación (se utiliza un circuito para generar valores binarios aleatorios, que luego se reescalan entre -pi y pi, y sirven como ángulos de rotación del circuito cuántico que termina generando el individuo), y al momento de la reproducción de nuevos hijos a partir de la primera generación, combinando y ejecutando los circuitos de los mejores padres con algunas modificaciones cuánticas intermedias.
* ¿Cómo sucede este proceso de reproducción? Nos quedamos con los mejores padres de la generación previa (en este caso se utiliza el torneo EaSimple), se seleccionan aleatoriamente dos de los mejores padres, se aplica un circuito que combina mediante entrelazamiento y rotaciones ambos circuitos, se lo mide, y dicho número binario que se obtiene se reescala entre -pi y pi (ángulos) y, a su vez, este ángulo se reescala para el bound en cuestión que se quiere construir. Cada bound tiene su combinación y ejecución de circuitos cuánticos. Si tenemos 2 hiperparámetros se ejecutan 2 circuitos iguales, pero con mediciones distintas. 

2. *Función objetivo (`objective_function`):*

Esta función calcula el accuracy o el MAE dependiendo del problema, o cualquier otra métrica de interés.

* Toma un `individual` (un objeto que posee como valores la solución potencial) como entrada.
* Recupera los hiperparámetros del `individual` (reproducido y mutado en el algorítmo genético cuántico).
* Utiliza esos hiperparámetros para calcular el resultado de la función objetivo.
* Se devuelve el `score`, que el algoritmo genético busca minimizar/maximizar según el tipo de problema.

3. *Algoritmo genético (`GenethicOptimizer`):* Esta clase implementa la lógica central del algoritmo genético cuántico.

    * *Inicialización:* Comienza creando una población de soluciones potenciales (rutas). La naturaleza cuántica se utiliza para generar esta población inicial.
    * *Evaluación:* La función `objective_function` se utiliza para evaluar la "aptitud" de cada individuo en la población.
    * *Selección:* Los individuos con mejor aptitud tienen más probabilidades de ser seleccionados como "mejores padres" para crear la siguiente generación.
    * *Cruce por circuitos cuánticos (rotaciones y entrelazamientos):* El material genético (hiperparámetros) de dos individuos padres se combinan para crear nuevos individuos descendientes. 
    * *Mutación:* Se introducen pequeños cambios aleatorios en los valores de los hiperparámetros de los descendientes para mantener la diversidad en la población y evitar quedar atrapado en óptimos locales.
    * *Operaciones cuánticas:* Los parámetros `max_qubits`, `randomness_quantum_technology`, `optimization_quantum_technology` y los detalles de conexión (`qm_api_key`, `qm_connection_service`, `quantum_machine`, `optimization_service`) permiten que el algoritmo se pueda ejecutar o bien en simulador o en ordenadores cuánticos de IBM.
    * *Terminación:* El algoritmo continúa durante un número específico de generaciones (`num_generations`) o hasta que se cumple un criterio de detención (`early_stopping_generations`).

4*Límites (`BoundCreator`):* Los límites definen los posibles valores para cada "gen" en el individuo (en este caso, el índice de una ciudad).

### Posibles usos reales

Si bien el desarrollo de hardware cuántico todavía posee un largo camino por recorrer, eso no significa que hasta que alcancen un nivel más sofisticado, no se pueda hacer nada.

El Aletheia Quantum Genetic Optimizers está diseñado para resolver varios tipos de problemas de optimización. Uno de los escenarios para los que se ha programado, es encontrar los mejores hiperparámetros de modelos de IA, o problemas de tipo similar. 

A nivel real -al momento- puede ser utilizado para resolver cuantiosos problemas de este tipo:

| Industria             | Tipo de datos                                   | Ejemplos de modelos a usar   | Beneficio del enfoque cuántico-genético                                      |
|-----------------------|--------------------------------------------------|------------------------------------|--------------------------------------------------------------------------------|
| Finanzas              | Transacciones bancarias                          | LGBMClassifier, XGBoost            | Detección más precisa de fraude, menos falsos positivos, búsqueda más rápida  |
| Energía               | Variables climáticas, consumo histórico          | LGBMRegressor, Redes neuronales    | Mejor predicción de demanda, optimización eficiente de recursos energéticos   |
| Salud                 | Datos clínicos, imágenes médicas                 | RandomForest, CNNs, LGBMClassifier | Diagnósticos automáticos más precisos y rápidos                               |
| Marketing y Ventas    | Historial de usuarios, CRM                       | XGBoost, CatBoost, Árboles         | Mejor segmentación y predicción de abandono (churn)                           |
| Ciberseguridad        | Logs de red, sesiones de conexión                | RandomForest, SVM, LSTM            | Mejora en la detección de tráfico malicioso, reducción de falsos negativos    |
| Agricultura           | Condiciones ambientales, suelo, clima            | LGBMRegressor, Redes neuronales    | Predicción precisa del rendimiento, decisiones agronómicas más acertadas      |
| Industria 4.0         | Datos sensoriales de maquinaria                  | LSTM, RandomForest, SVM            | Mantenimiento predictivo, detección temprana de fallas                        |
| Educación             | Comportamiento de estudiantes, resultados        | Sistemas de recomendación, Árboles | Personalización del aprendizaje, mayor engagement y retención de alumnos      |
| Transporte y Logística| Rutas, tiempos de entrega, demanda               | Regresión, KNN, RandomForest       | Optimización del reparto, predicción de demanda, mejora en eficiencia logística|
| Recursos Humanos      | CVs, entrevistas, datos de rendimiento laboral   | NLP + árboles, XGBoost             | Mejora del proceso de selección, predicción de rendimiento de empleados       |
| Medioambiente         | Datos satelitales, sensores ambientales          | LGBM, CNN, RandomForest            | Monitoreo ambiental en tiempo real, predicción de fenómenos climáticos        |
| Telecomunicaciones    | Datos de llamadas, uso de apps, geolocalización  | CatBoost, Redes neuronales         | Predicción de churn, optimización de red, segmentación de usuarios            |
| eCommerce             | Historial de compras, comportamiento de navegación| XGBoost, Recommender Systems       | Mejora en sistemas de recomendación y personalización                         |
| Sector Legal          | Documentos legales, jurisprudencia, contratos    | NLP + Transformers, Decision Trees | Análisis automatizado de textos legales, detección de cláusulas de riesgo     |
| Juegos y Entretenimiento | Comportamiento de jugadores, patrones de uso  | RandomForest, Deep Q-Learning      | Optimización de diseño de niveles, detección de comportamiento irregular      |


### Ventajas de usar el AletheIA Quantum Genetic Optimizers para este tipo de problemas

* *Exploración mejorada del espacio de soluciones:* Los fenómenos cuánticos como la superposición y el entrelazamiento hacen que este algoritmo, tenga al menos en teoría, la capacidad de explorar una mayor cantidad de rutas de manera distinta a un optimizador clásico. Su potencialidad máxima la encontrará cuando no se deba dividir el problema en subgrupos, yse pueda evaluar la ruta completa o, al menos por la mitada, con el QAOA.
* *Operadores genéticos más eficientes:* Los operadores de cruce/reproducción y mutación basados en procesos cuánticos utilizados podrían ser más efectivos para combinar partes prometedoras de soluciones e introducir una diversidad beneficiosa.
* *Menor tendencia a quedar atascados en óptimos locales:* La naturaleza probabilística de las mediciones cuánticas podría ayudar al algoritmo a escapar de soluciones subóptimas.

#### Conclusiones luego de ver los resultados

Para este tipo de problemas, el AletheIA Quantum Genetic Optimizers es muy prometedor.

En general, en las pruebas realizadas para contrastar su potencialidad con respecto a otros algoritmos genético clásicos, a superado al AletheIA Classic Genetic Optimizers, que de por sí es muy bueno, en cantidad de generaciones requeridas para alcanzar la mejor métrica posible para el modelo y problema evaluado, tanto para problemas de clasificación como de regresión. Sin embargo, se debe tener en cuenta, que no converge a la manera que lo hace un optimizador clásico donde se puede notar que toda la población va acompañando la suba o baja de la métrica evaluada. En el caso del AletheIA Quantum Genetic Optimizers se aprecia un procedimiento más elitista, es decir, encuentra por capacidad explotaria máximos o mínimos globales muy buenos, pero lo hacen pocos individuos de cada población.

--------------------------------------------

## Caso de uso 2: 

En este apartado se describe el caso de uso para resolver el Problema del Viajero (TSP, por sus siglas en inglés) en sus variantes de ciclo cerrado y abierto.

### Entendimiento del problema: El problema del viajante de comercio (TSP)

El problema del viajante de comercio es un problema clásico de optimización combinatoria. 

Imaginar un viajante que necesita visitar un conjunto de ciudades, cuyo objetivo es encontrar la ruta más corta posible que visite cada ciudad exactamente una vez y, dependiendo de la variante:

* *Ciclo cerrado (con retorno al origen):* El viajante debe regresar a la ciudad de inicio después de visitar todas las demás ciudades, formando un tour o ciclo completo.
* *Ciclo abierto (sin retorno al origen):* El viajante visita cada ciudad exactamente una vez, pero no necesita regresar a la ciudad de inicio.

El desafío radica en que, a medida que aumenta el número de ciudades, la cantidad de posibles rutas crece factorialmente, lo que hace que encontrar la solución óptima sea computacionalmente muy costoso utilizando métodos tradicionales.

*NOTA: Actualmente como existen limitaciones de hardware (memoria RAM) o de costos (los precios para las iteraciones necesarias son elevados) se ideó dividir el problema de las rutas en distintos subproblemas. Por ejemplo, un problema de 20 ciudades se subdivide en 4 grupos de 5 ciudades, se aplica el QAOA a cada subgrupo para intentar encontrar la mejor combinación posible, para finalmente concatenarse en una sola ruta.*

### Ejemplo de código

```
def example_tsp():

    # Coordenadas de ciudades (ejemplo con 20 ciudades)
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

    # -- Creación de función objetivo que debe resolver el AletheIA Quantum Genetic Optimizers
    def objective_function(individual):
    
        # -- Definimos si es un problema de retorno al origen o no
        return_to_origin: bool = True

        # -- Definimos las ciudades (Dict --> {id_ciudad: (latitud y longitud)}
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

        # -- Obtenemos los valores únicos de las rutas y los ordenamos
        route = list(individual.get_individual_values().values())
        cities = original_cities.copy()

        # -- Si el problema es de retorno al origen (ciclo cerrado) agregamos la ruta de la última ciudad a la primera
        if return_to_origin:
            route.append(route[0])
            cities[len([z for z in cities.keys()])] = cities[route[-1]]

        # -- Creamos la matriz de distancias
        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    x1, y1 = cities[i]
                    x2, y2 = cities[j]
                    distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # -- Calculamos la distancia total
        total_distance = sum(distance_matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

        # -- Retornamos la distancia social para la ruta evaluada
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

    # -- Instanciamos el optimizador
    return GenethicOptimizer(bounds_dict=bounds.get_bound(),
                             num_generations=500,
                             num_individuals=5,
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
                             verbose=True,
                             early_stopping_generations="gradient",
                             variability_explossion_mode="crazy",
                             variability_round_decimals=3,
                             randomness_quantum_technology="simulator",
                             randomness_service="aer",
                             optimization_quantum_technology="simulator",
                             optimization_service="aer",
                             qm_api_key="IBMQUANTUM_IP",
                             qm_connection_service="ibm_quantum",
                             quantum_machine="least_busy",
                             element_matrix=original_cities
                             )
                             
# -- Instanciamos la función que ejecuta el AletheIA Quantum Genetic Optimizers
genetic_optimizer_object: GenethicOptimizer = example_2_tsp()

# -- Instanciamos el ploteo de resultados (opcional)
genetic_optimizer_object.plot_generation_stats()
genetic_optimizer_object.plot_evolution()

# -- Obtenemos los valores del mejor individuo
best_individual_value = genetic_optimizer_object.get_best_individual_values()

# -- Pintamos los valores del mejor individuo en consola
print(best_individual_value)

```

#### El enfoque genético cuántico

El código proporcionado implementa un algoritmo genético cuántico para abordar el problema del Traveling Salesman Problem (TSP). 

A continuación, se explica cómo funciona este algoritmo y sus potenciales ventajas:

1. *Introducción programática:* 

* El `individual` que recibe la función `objective_function` representa una ruta potencial (una permutación específica de las ciudades -por ejemplo: [2, 9, 4, 5, 7, ... 1]).
* El aspecto cuántico de este algoritmo genético se ejecuta al momento de generarse la primera generación (se utiliza un circuito para generar una semilla del random) y al momento de la reproducción de nuevos hijos a partir de la primera generación.
* ¿Cómo sucede este proceso de reproducción? Nos quedamos con los mejores padres de la generación previa (en este caso se utiliza el torneo EaSimple), se seleccionan aleatoriamente dos de los mejores padres, se aplica un cruce de tipo OX1, se divide en clústers o sublistas la ruta, se aplica un QAOA a cada ruta, y se unifican las subrutas en una ruta final. 

2. *Función objetivo (`objective_function`):*

Esta función calcula la distancia total de una ruta dada.

* Toma un `individual` (un objeto que posee como valores la solución potencial) como entrada.
* Recupera el orden de las ciudades visitadas del `individual` (reproducido y mutado en el algorítmo genético cuántico).
* Utiliza las coordenadas de `original_cities` para calcular las distancias entre ciudades consecutivas en la ruta.
* El parámetro `return_to_origin` controla si se incluye la distancia desde la última ciudad de vuelta a la primera (para un ciclo cerrado).
* Devuelve la `total_distance`, que el algoritmo genético busca minimizar.

3. *Algoritmo genético (`GenethicOptimizer`):* Esta clase implementa la lógica central del algoritmo genético cuántico.

    * *Inicialización:* Comienza creando una población de soluciones potenciales (rutas). La naturaleza cuántica se utiliza para generar esta población inicial. Se utiliza la superposición cuántica (puertas H) para generar semillas de un generador random.
    * *Evaluación:* La función `objective_function` se utiliza para evaluar la "aptitud" de cada individuo en la población (menor distancia = mayor aptitud, ya que se está minimizando).
    * *Selección:* Los individuos con mejor aptitud tienen más probabilidades de ser seleccionados como "mejores padres" para crear la siguiente generación.
    * *Cruce OX1 (recombinación):* El material genético (partes de las rutas) de dos individuos padres se combina para crear nuevos individuos descendientes. Una vez ejecutada la recombinación, se subdivide la ruta final en grupos de mínimo 2 y máximos 5 elementos, se aplica un QAOA para optimizar cada sublista y se vuelven a unificar en una sola ruta. 
    * *Mutación:* Se introducen pequeños cambios aleatorios en las rutas de los descendientes para mantener la diversidad en la población y evitar quedar atrapado en óptimos locales.
    * *Operaciones cuánticas:* Los parámetros `max_qubits`, `randomness_quantum_technology`, `optimization_quantum_technology` y los detalles de conexión (`qm_api_key`, `qm_connection_service`, `quantum_machine`, `optimization_service`) permiten que el algoritmo se pueda ejecutar o bien en simulador o en ordenadores cuánticos de IBM.
    * *Terminación:* El algoritmo continúa durante un número específico de generaciones (`num_generations`) o hasta que se cumple un criterio de detención (`early_stopping_generations`).

4. *Restricciones del problema (`problem_restrictions="totally_restricted"`):* Este parámetro indica que el algoritmo debe encontrar una ruta válida donde cada ciudad se visite exactamente una vez (y regrese al origen si `return_to_origin` es `True`). En el problema del TSP solo se revisa que estén todas las ciudades.

5. *Límites (`BoundCreator`):* Los límites definen los posibles valores para cada "gen" en el individuo (en este caso, el índice de una ciudad).

### Posibles usos reales

Si bien el desarrollo de hardware cuántico todavía posee un largo camino por recorrer, eso no significa que hasta que alcancen un nivel más sofisticado, no se pueda hacer nada.

El Aletheia Quantum Genetic Optimizers está diseñado para resolver varios tipos de problemas de optimización. Uno de los escenarios para los que se ha programado, es encontrar la ruta óptima o casi óptima en un TSP. 

A nivel real -al momento- puede ser utilizado para resolver problemas TSP del ámbito de:

* *Logística y servicios de entrega:*
    * *Optimización de rutas (en este caso solo euclidianas, por el momento):* Encontrar las rutas de entrega más eficientes para mensajeros, servicios postales o minoristas en línea para minimizar el tiempo de viaje, el consumo de combustible y los costos. Esto podría ser un ciclo cerrado si el conductor necesita regresar a un depósito o un ciclo abierto si termina en un destino final.

* *Fabricación y robótica:*
    * *Perforación de placas de circuito impreso:* Optimizar la trayectoria de una máquina perforadora para crear agujeros en una placa de circuito, minimizando el tiempo necesario para moverse entre los puntos de perforación.
    * *Soldadura/pintura robótica:* Encontrar la secuencia más eficiente de puntos para que un brazo robótico suelde o pinte componentes.
    * *Corte láser o fresado CNC:* Determinar la mejor secuencia de cortes en una pieza de material para reducir los movimientos innecesarios de la herramienta.

* *Transporte y viajes:*
    * *Planificación de tours:* Crear el itinerario más corto o eficiente para un turista que visita múltiples atracciones.
    * *Mantenimiento de infraestructuras:* Definir la ruta óptima para un equipo de mantenimiento que debe inspeccionar una serie de ubicaciones (por ejemplo, torres de telecomunicaciones, estaciones eléctricas o pozos petroleros).

* *Ciencias biológicas y medicina:* 
  * Reconstruir el orden de los fragmentos de ADN encontrando el camino más corto que visita cada fragmento basándose en la información de superposición.
  * Automatización en laboratorios: Optimizar el recorrido de brazos robóticos que deben dispensar líquidos o recolectar muestras en distintos puntos.

* *Agricultura de precisión*:
  * Monitoreo de cultivos: Determinar la mejor ruta para drones o tractores autónomos que deben visitar múltiples puntos de monitoreo o tratamiento dentro de un campo.

* *Astronomía:* 
  * Optimizar la secuencia de observaciones para telescopios para minimizar el tiempo dedicado a moverse entre objetos celestes.
  
* Mantenimiento de flotas aéreas o ferroviarias:*
  * Inspección de unidades: Encontrar la secuencia más eficiente de revisiones técnicas o inspecciones a realizar en una base o red distribuida de vehículos.

### Ventajas de usar el AletheIA Quantum Genetic Optimizers para este tipo de problemas

* *Exploración mejorada del espacio de soluciones:* Los fenómenos cuánticos como la superposición y el entrelazamiento hacen que este algoritmo, tenga al menos en teoría, la capacidad de explorar una mayor cantidad de rutas de manera distinta a un optimizador clásico. Su potencialidad máxima la encontrará cuando no se deba dividir el problema en subgrupos, yse pueda evaluar la ruta completa o, al menos por la mitada, con el QAOA.
* *Operadores genéticos más eficientes:* Los operadores de cruce/reproducción y mutación basados en procesos cuánticos utilizados podrían ser más efectivos para combinar partes prometedoras de soluciones e introducir una diversidad beneficiosa.
* *Menor tendencia a quedar atascados en óptimos locales:* La naturaleza probabilística de las mediciones cuánticas podría ayudar al algoritmo a escapar de soluciones subóptimas.

#### Ciclo cerrado vs. ciclo abierto

La variable `return_to_origin: bool = True` dentro de su función `objective_function` es clave para manejar ambas variantes del TSP:

* *Ciclo cerrado:* Cuando `return_to_origin` es `True` (como en su `objective_function`), la distancia desde la última ciudad de la ruta de vuelta a la primera ciudad se suma a la distancia total, cerrando efectivamente el ciclo.
* *Ciclo abierto:* Para resolver un TSP de ciclo abierto, simplemente se necesita establecer `return_to_origin` en `False`. En este caso, el algoritmo intentaría encontrar el camino más corto que visita todas las ciudades, partiendo del primer elemento (ciudad 0) sin requerir un retorno al punto de partida.

--------------------------------------------

## 1. Clase principal: GeneticOptimizer

La clase `GenethicOptimizer` implementa un algoritmo genético cuántico para resolver problemas de optimización combinatoria con restricciones específicas. 

Esta clase utiliza tanto computación cuántica o simuladores cuánticos para generar individuos y optimizar el espacio de soluciones mediante la evolución de una población.

### 1.1. Atributos

- **bounds_dict**: Diccionario de parámetros a optimizar y sus valores. Ejemplo: `{'learning_rate': (0.0001, 0.1)}`
- **num_generations**: Número de generaciones a ejecutar.
- **num_individuals**: Número de individuos iniciales a generar.
- **max_qubits**: Número máximo de qubits a emplear para reproducir individuos.
- **objective_function**: Función objetivo que se utiliza para puntuar a cada individuo. Debe retornar un valor numérico (float).
- **metric_to_optimize**: Métrica a optimizar (e.g., 'accuracy', 'recall', 'f1', 'mae').
- **problem_restrictions**: Restricciones aplicadas al problema ('bound_restricted' o 'totally_restricted').
- **return_to_origin**: Indica si el problema debe terminar en el origen (solo para problemas 'totally_restricted').
- **problem_type**: Tipo de optimización ('minimize' o 'maximize').
- **tournament_method**: Método de selección de individuos para reproducción (e.g., 'ea_simple').
- **podium_size**: Tamaño del grupo de individuos que competirá en el torneo para selección.
- **mutate_probability**: Probabilidad de mutación en cada gen.
- **mutate_gen_probability**: Probabilidad de mutar un gen.
- **mutation_policy**: Política de mutación ('soft', 'normal', 'hard').
- **verbose**: Indica si se deben mostrar mensajes detallados sobre la evolución.
- **early_stopping_generations**: Número de generaciones para considerar el "early stopping".
- **variability_explossion_mode**: Modo de explosión de variabilidad ('crazy').
- **variability_round_decimals**: Número de decimales para redondear estadísticas de la variabilidad.
- **randomness_quantum_technology**: Tecnología cuántica utilizada para los procesos random de generación de la población.
- **randomness_service**: Servicio de computación cuántica utilizado para los procesos random de generación de individuos.
- **optimization_quantum_technology**: Tecnología cuántica utilizada para la optimización de la población.
- **optimization_service**: Servicio de computación cuántica utilizado para la optimización.
- **qm_api_key**: Clave API para acceder a la computación cuántica.
- **qm_connection_service**: Servicio de conexión cuántica utilizado (e.g., 'ibm_quantum').
- **quantum_machine**: Nombre de la máquina cuántica a utilizar (e.g., 'ibm_brisbane').
- **element_matrix**: Matriz de distancias para problemas combinatorios (e.g., TSP).

### 1.2. Métodos

#### 1.2.2. `__init__()`
Este es el constructor de la clase `GenethicOptimizer`. 

Se encarga de inicializar todos los parámetros del algoritmo, como el tipo de optimización, las restricciones, las configuraciones de la mutación, entre otros. 

Además, se prepara el entorno de computación cuántica, si es necesario.


```
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
```

#### 1.2.1. Proceso de evolución

##### Creación y evaluación de la población inicial

1. *Inicialización de la población*: Se crea la población inicial de individuos, asignando valores aleatorios dentro de los límites definidos por `bounds_dict`.
   
2. *Evaluación de la función objetivo*: Cada individuo en la población inicial pasa por la función objetivo para obtener su fitness.

##### Evolución por generaciones

1. *Selección por torneo*: Cada generación selecciona a los mejores individuos utilizando un torneo. Los individuos ganadores serán los padres de la siguiente generación.

2. *Reproducción cuántica*: Los padres seleccionados se reproducen para crear nuevos individuos.

Ejemplo de circuito aplicado:

[./qc_reproduction_bound_restricted.png]

3. *Mutación*: 
    - *Mutación de los hijos*: Los nuevos individuos generados pasan por un proceso de mutación. Dependiendo de la probabilidad y política de mutación (`mutate_probability`, `mutate_gen_probability`, `mutation_policy`), se modifican algunos de los genes de los hijos.
    - *Actualización de la población*: Los individuos mutados se agregan a la población de la generación actual.

4. *Evaluación de la función objetivo*: Los nuevos individuos generados son evaluados nuevamente utilizando la función objetivo para obtener sus valores de fitness.

##### Estadísticas y seguimiento

1. *Estadísticas de la generación*: 
    - Después de cada generación, se muestra la información relevante sobre los mejores individuos y su fitness.
    - Se imprime la mejor solución obtenida de cada generación.
    - Si se activa el modo de "explosión de variabilidad", se muestra si se ha superado un mínimo local.

2. *Early Stopping y variabilidad*: 
    - Si se detecta que el modelo está atrapado en un mínimo local, se aplica una "explosión de variabilidad" para tratar de escapar de este mínimo, modificando las probabilidades de mutación y ajustando la política de mutación.

3. *Condición de parada*: 
    - El algoritmo puede detenerse antes de llegar al número máximo de generaciones si se cumplen ciertas condiciones de parada temprana (por ejemplo, si no se mejora el fitness en un número determinado de generaciones).

##### Resultado final

1. *Impresión de resultados finales*: 
    - Al finalizar el proceso, se muestran los resultados finales de cada generación y el mejor individuo encontrado, incluyendo su fitness y valores.

---

### 1.3. Flujo de trabajo

1. Inicialización de la población con parámetros aleatorios dentro de los límites de `bounds_dict`.
2. Evaluación de la función objetivo para cada individuo.
3. Selección de los mejores individuos mediante un torneo.
4. Reproducción cuántica y mutación de los individuos seleccionados.
5. Evaluación de la función objetivo nuevamente para los nuevos individuos.
6. Impresión de estadísticas de cada generación y el mejor individuo.
7. Aplicación de "explosión de variabilidad" si el algoritmo detecta un mínimo local.
8. Continuación hasta completar el número máximo de generaciones o alcanzar el criterio de parada.

## 2. Clase Population

### 2.1. Descripción general
La clase `Population` gestiona una población de individuos para algoritmos genéticos cuánticos de optimización. 

Permite la creación y evolución de poblaciones utilizando tecnología cuántica para la generación de parámetros y propiedades de los individuos.

### 2.1.1 Constructor

```
def __init__(self,
             bounds_dict: Dict[str, Tuple[Union[int, float]]],
             num_individuals: int,
             problem_restrictions: Literal['bound_restricted', 'totally_restricted'],
             round_decimals: int = 3):
```

#### 2.1.2. Parámetros
- *bounds_dict*: Diccionario que define las propiedades y límites de cada individuo
- *num_individuals*: Número de individuos en la población
- *problem_restrictions*: Tipo de restricciones del problema ('bound_restricted' o 'totally_restricted')
- *round_decimals*: Número de decimales para redondeo en comparaciones de similitud (predeterminado: 3)

#### 2.1.3. Atributos principales
- *IT*: Instancia de InfoTools para mostrar información
- *bounds_dict*: Diccionario con los límites de cada propiedad
- *num_individuals*: Cantidad de individuos en la población
- *problem_restrictions*: Tipo de restricción del problema
- *populuation_dict*: Diccionario que almacena los individuos por generación
- *hyperparameters*: Diccionario con los hiperparámetros extraídos de bounds_dict
- *num_qubits*: Número de qubits necesarios para el circuito cuántico
- *QT*: Instancia de QuantumTechnology para operaciones cuánticas
- *vqc_parameters*: Parámetros del circuito cuántico variacional

### 2.2.1 create_population()

```
def create_population(self, 
                     quantum_technology: Literal["simulator", "quantum_machine"], 
                     service: Literal["aer", "ibm"], 
                     qm_api_key: str | None, 
                     qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None, 
                     quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"], 
                     max_qubits: int, 
                     element_matrix: Dict[str, tuple] | None = None)
```

Crea la población inicial utilizando la tecnología cuántica especificada.

#### 2.2.2. Parámetros
- **quantum_technology**: Tipo de tecnología ("simulator" o "quantum_machine")
- **service**: Servicio a utilizar ("aer" o "ibm")
- **qm_api_key**: API Key para conectarse al servicio cuántico
- **qm_connection_service**: Servicio específico ("ibm_quantum" o "ibm_cloud")
- **quantum_machine**: Máquina cuántica a utilizar
- **max_qubits**: Número máximo de qubits disponibles
- **element_matrix**: Matriz de distancia para problemas tipo TSP (opcional)

### 2.3.1. add_generation_population()
```
def add_generation_population(self, children_list: List[Individual], generation: int) -> None
```
Añade una nueva generación de individuos a la población.

### 2.4.1. get_generation_fitness_statistics()
```
def get_generation_fitness_statistics(self, generation: int)
```
Calcula y devuelve estadísticas de fitness para una generación específica.

### 2.5.1. plot_generation_stats()
```
def plot_generation_stats(self, variability_explosion_starts_in_generation: int | None)
```
Genera un gráfico interactivo con la evolución de las estadísticas por generación.

### 2.6.1. plot_evolution_animated()
```
def plot_evolution_animated(self, problem_type: Literal['minimize', 'maximize'] = "maximize", transition_duration_ms: int = 50) -> None
```
Crea una visualización animada de la evolución de la población a lo largo de las generaciones.

### 2.7.1. plot_evolution()
```
def plot_evolution(self) -> None
```
Genera un gráfico estático de la evolución de la población.

## Comportamiento según el tipo de restricción

### Para "bound_restricted":
1. Crea un circuito cuántico que genera números aleatorios para los ángulos de los parámetros
2. Genera parámetros para cada individuo utilizando el circuito
3. Transforma y normaliza los valores dentro de los límites especificados
4. Crea individuos sin malformaciones hasta alcanzar el número requerido

### Para "totally_restricted":
1. Genera permutaciones utilizando un enfoque híbrido
2. Crea individuos y verifica que no tengan malformaciones ni sean duplicados

## 3. Clase Reproduction

### 3.1. Descripción General
La clase `Reproduction` gestiona el proceso de reproducción en algoritmos genéticos cuánticos de optimización. Es responsable de crear una nueva generación de individuos (hijos) a partir de individuos seleccionados (ganadores) mediante técnicas de optimización cuántica.

#### 3.1.1. Constructor

```
def __init__(self, 
             winners_list: List[Individual],
             number_of_children: int,
             problem_restrictions: Literal['bound_restricted', 'totally_restricted'],
             return_to_origin: Literal['return_to_origin', 'no_return'] | None,
             problem_type: Literal["minimize", "maximize"],
             metric_to_optimize: Literal['accuracy', 'recall', 'specificity', 'f1', 'aur_roc', 'precision', 'negative_precision', 'mae', 'mse', 'other', 'r2'],
             verbose: bool = True)
```

#### 3.1.2. Parámetros
- *winners_list*: Lista de individuos ganadores que serán los padres
- *number_of_children*: Cantidad de individuos hijos que se desean generar
- *problem_restrictions*: Tipo de restricciones del problema ('bound_restricted' o 'totally_restricted')
- *return_to_origin*: Para problemas 'totally_restricted', indica si el recorrido debe volver al origen
- *problem_type*: Indica si el problema busca minimizar o maximizar la función objetivo
- *metric_to_optimize*: Métrica específica que se desea optimizar en la función objetivo
- *verbose*: Determina si se muestran mensajes y gráficos durante el proceso (predeterminado: True)

#### 3.1.3. Atributos principales
- *winners_list*: Lista de individuos ganadores (padres)
- *number_of_children*: Número de hijos a generar
- *problem_restrictions*: Tipo de restricción del problema
- *return_to_origin*: Indicador de retorno al origen para problemas 'totally_restricted'
- *children_list*: Lista donde se almacenarán los individuos hijos generados
- *parents_generation*: Generación a la que pertenecen los padres
- *problem_type*: Tipo de problema (minimización o maximización)
- *metric_to_optimize*: Métrica específica a optimizar
- *verbose*: Indicador de verbosidad de la ejecución
- *IT*: Instancia de InfoTools para mostrar información
- *QT*: Instancia de QuantumTechnology para operaciones cuánticas

#### 3.1.4. Atributos configurables en ejecución
Estos atributos se definen posteriormente mediante el método `run_reproduction`:

- *optimization_quantum_technology*: Tecnología cuántica a utilizar ("simulator" o "quantum_machine")
- *optimization_service*: Servicio a utilizar ("aer" o "ibm")
- *qm_api_key*: API Key para conectarse al servicio cuántico
- *qm_connection_service*: Servicio específico de conexión ("ibm_quantum" o "ibm_cloud")
- *quantum_machine*: Máquina cuántica específica a utilizar
- *generations_fitness_statistics_df*: DataFrame con estadísticas de fitness por generación
- *max_qubits**: Número máximo de qubits disponibles
- *_reescaling_result_df*: DataFrame para almacenar resultados de reescalado entre representaciones binarias, valores de π y valores del diccionario bounds_dict

#### 3.1.5. Comportamiento según configuración
La clase Reproduction adapta su comportamiento según:

1. *Tipo de restricción* ('bound_restricted' o 'totally_restricted'):
   - Para problemas con límites restringidos, genera valores dentro de los rangos especificados
   - Para problemas totalmente restringidos, trabaja con permutaciones válidas

2. *Retorno al origen* (para 'totally_restricted'):
   - En problemas como TSP, puede exigir que el recorrido vuelva al punto inicial

3. *Tipo de problema* ('minimize' o 'maximize'):
   - Adapta el proceso de selección y evaluación según se busque minimizar o maximizar

4. *Métrica a optimizar*:
   - Ajusta la evaluación de individuos según la métrica específica indicada

### 3.2. Método run_reproduction

#### 3.2.1. Descripción
El método `run_reproduction` de la clase `Reproduction` inicia el proceso de reproducción para generar una nueva generación de individuos a partir de los ganadores de la generación anterior. Configura el entorno de ejecución y selecciona la estrategia de reproducción según el tipo de restricciones del problema.

#### 3.2.2. Firma del método

```
def run_reproduction(self, 
                     quantum_technology: Literal["simulator", "quantum_machine"], 
                     service: Literal["aer", "ibm"], 
                     qm_api_key: str | None, 
                     qm_connection_service: Literal["ibm_quantum", "ibm_cloud"] | None, 
                     quantum_machine: Literal["ibm_brisbane", "ibm_kyiv", "ibm_sherbrooke", "least_busy"], 
                     generations_fitness_statistics_df: pd.DataFrame, 
                     max_qubits: int) -> List[Individual]
```

#### 3.2.3 Parámetros
- *quantum_technology*: Tecnología cuántica a utilizar ("simulator" o "quantum_machine")
- *service*: Servicio tecnológico para ejecutar la lógica ("aer" o "ibm")
- *qm_api_key*: API KEY para conectarse con el servicio de computación cuántica
- *qm_connection_service*: Servicio específico de conexión ("ibm_quantum" o "ibm_cloud")
- *quantum_machine*: Nombre del ordenador cuántico a utilizar, o "least_busy" para seleccionar automáticamente el menos ocupado
- *generations_fitness_statistics_df*: DataFrame con información estadística de todas las generaciones
- *max_qubits*: Número máximo de qubits a utilizar

#### 3.2.4 Valor de retorno
- Lista de individuos (`List[Individual]`) que constituye la nueva generación

#### 3.2.5 Proceso de ejecución
1. *Configuración del entorno*
   - Almacena los parámetros recibidos en los atributos de la clase
   - Inicializa la tecnología cuántica con los parámetros proporcionados

2. *Selección de elitismo*
   - Selecciona el 15% de los mejores individuos de la generación anterior
   - Los mejores se determinan según el tipo de problema (maximización o minimización)
   - Estos individuos pasarán directamente a la siguiente generación (elitismo)

3. *Copia y actualización de individuos seleccionados*
   - Crea copias de los individuos seleccionados
   - Actualiza la generación de las copias para que pertenezcan a la nueva generación
   - Preserva todas las propiedades y parámetros de los individuos originales

4. *Selección de la estrategia de reproducción*
   - Según el tipo de restricciones del problema, invoca:
     - `bound_restricted_reproduction()` para problemas con límites restringidos
     - `totally_restricted_reproduction()` para problemas totalmente restringidos

#### 3.2.6. Notas importantes
- El porcentaje de elitismo está configurado al 15% (con un mínimo de 1 individuo)
- Los individuos seleccionados por elitismo mantienen todos sus atributos, incluyendo sus parámetros cuánticos
- El método actúa como un dispatcher que delega la reproducción específica a métodos especializados según el tipo de problema

### 3.3. Método `bound_restricted_reproduction`

```
def bound_restricted_reproduction(self):
```

Este método implementa la reproducción para problemas con restricciones de límites en las variables. Utiliza un enfoque basado en la computación cuántica para generar nuevos individuos combinando características de los padres.

#### 3.3.1. Funcionalidad

1.  *Obtención de límites:** Recupera el diccionario de límites (`bounds_dict`) del primer individuo en la lista de ganadores. Este diccionario define los rangos permitidos para cada parámetro del problema.

2.  *Generación de combinaciones de padres:* Crea todas las posibles parejas únicas de individuos ganadores para la reproducción utilizando `itertools.permutations`.

3.  *Iteración sobre combinaciones:* Para cada par de padres:
    * *Preparación de circuitos cuánticos:* Obtiene copias de los circuitos cuánticos variacionales (VQC) de ambos padres y elimina las mediciones finales para manipular los estados cuánticos.
    * *Adición de qubits para métricas:* Agrega un nuevo qubit a cada circuito padre. Estos qubits se parametrizan con el valor de la función objetivo normalizado del padre correspondiente. Esto permite que la información del rendimiento de los padres influya en la generación de los hijos.
    * *Obtención de parámetros:* Recopila los parámetros simbólicos de los circuitos de ambos padres.
    * *Creación de diccionario de parámetros:* Para cada padre, normaliza su valor de la función objetivo utilizando la información estadística de todas las generaciones y obtiene los valores de los parámetros theta de su VQC. Crea un diccionario que mapea los parámetros simbólicos a sus valores correspondientes (incluido el valor normalizado de la métrica).
    * *Asignación de parámetros:* Asigna los valores de los parámetros al circuito de cada padre.
    * *Creación de circuito combinado:* Utiliza la función `self.QT.create_parent_vqc` para combinar los circuitos de los dos padres en un nuevo circuito. Opcionalmente, podría incluirse la información del fitness de los padres en la creación del circuito combinado (línea comentada). Se añade una puerta CNOT al final del circuito combinado, lo que introduce entrelazamiento entre los qubits de los padres.
    * *Visualización (opcional):* Si `self.verbose` es `True`, se muestra una representación gráfica del circuito combinado.
    * *Adición de mediciones (procedimiento especial):* Se utiliza la función `self.QT.adding_measurements_parent_vqc` para añadir mediciones al circuito combinado. Este procedimiento está diseñado para obtener información sobre cómo las propiedades de los padres se combinan a nivel cuántico. Se generan réplicas del circuito combinado para cada propiedad del individuo, y las mediciones se aplican de manera específica para extraer la información relevante para cada propiedad.
    * *Almacenamiento de circuitos parentales:* El circuito con las mediciones para cada propiedad se guarda en el diccionario `combo_circuits`, junto con una referencia a los individuos padres que generaron este circuito.

4.  *Organización de circuitos por propiedad:* Se crea un nuevo diccionario (`prop_vqc_dict`) para organizar los circuitos. Las claves de este diccionario son los nombres de las propiedades de los individuos, y los valores son listas de los circuitos cuánticos (provenientes de las diferentes combinaciones de padres) que se utilizarán para generar los valores de esa propiedad en los hijos.

5.  *Reducción de circuitos (si es necesario):* Si la cantidad total de circuitos generados para alguna propiedad excede el número de hijos que se desean crear (teniendo en cuenta los individuos élite ya añadidos), se elimina aleatoriamente el exceso de circuitos para mantener el tamaño de la población.

6.  *Ejecución de circuitos cuánticos:* Se ejecutan todos los circuitos cuánticos almacenados en `prop_vqc_dict` utilizando el objeto de ejecución de `self.QT` con un número fijo de `shots` (20000 en este caso). Los resultados de las ejecuciones (distribuciones de probabilidad de los estados finales) se guardan en `results_dict`.

7.  *Interpretación de resultados:* Para cada propiedad y para cada circuito ejecutado:
    * Se selecciona aleatoriamente un resultado binario de la distribución de probabilidad obtenida de la ejecución del circuito cuántico. *(Nota: Hay una sección de código comentada que sugiere una estrategia alternativa de seleccionar los dos resultados binarios más frecuentes)*.
    * El valor binario se convierte a un valor flotante normalizado en el rango $[-\pi, \pi]$ utilizando `self.QT.binary_to_float_and_normalize_pi`.
    * El valor en el rango $[-\pi, \pi]$ se reescala al rango original definido por `bounds_dict` para la propiedad correspondiente utilizando `self.QT.rescaling_pi_to_integer_float`.
    * Los valores binarios, en el rango $[-\pi, \pi]$ y reescalados se almacenan en un diccionario (`individual_dict`), donde cada entrada representa un nuevo individuo potencial.

8.  *Creación de DataFrame de resultados:* Los resultados de la reescalado se organizan en un DataFrame (`_reescaling_result_df`). Se añade información sobre la generación de los padres y la generación de los hijos. Si el DataFrame aún no existe, se crea uno nuevo; de lo contrario, los nuevos resultados se concatenan con los existentes.

9.  *Visualización de DataFrame (opcional):* Si `self.verbose` es `True`, se muestra una tabla del DataFrame de resultados.

10. *Creación de nuevos individuos:* Se itera sobre las filas del DataFrame de resultados correspondientes a la generación actual de padres. Para cada fila, se crea un nuevo objeto `Individual` utilizando los valores reescalados como las propiedades del hijo, el VQC del padre (sin entrenar), los ángulos obtenidos (`pi`), la generación actualizada y las restricciones del problema. Estos nuevos individuos se añaden a la `children_list`.

11. *Impresión de información de los hijos (opcional):* Si `self.verbose` es `True`, se imprime información sobre el ID, las propiedades y los parámetros VQC de cada nuevo individuo creado.

12. *Retorno de la nueva generación:* Finalmente, el método devuelve la lista de los nuevos individuos (`self.children_list`).

### 3.4 Método `totally_restricted_reproduction`

```
def totally_restricted_reproduction(self):
```

Este método implementa la reproducción para problemas con restricciones complejas, como problemas de permutaciones (ej., TSP). Utiliza un enfoque híbrido que combina operadores genéticos clásicos (como OX1) con la optimización cuántica (QAOA) aplicada a subproblemas (clústeres).

#### 3.4.1. Funcionalidad

1.  *Inicialización:* Imprime un encabezado indicando el inicio del proceso de reproducción para problemas con restricciones totales. Obtiene el diccionario de límites (`bounds_dict`) del primer individuo ganador, que se utilizará para crear nuevos individuos.

2.  *Preparación para el QAOA:*
    * Determina el número total de elementos que componen un individuo (por ejemplo, el número de ciudades en un problema de TSP) a partir de la matriz de elementos del primer ganador.
    * Inicializa variables para el manejo de problemas con la restricción de no volver al origen (`zero_cluster`, `contains_zero`).

3.  *Bucle de generación de hijos:* El método itera hasta que se haya creado el número deseado de hijos (`self.number_of_children`).

4.  *Selección de padres:* En cada iteración, selecciona aleatoriamente dos individuos diferentes de la lista de ganadores para actuar como padres. Se imprime información sobre los padres seleccionados si `self.verbose` es `True`.

5.  *Cruce genético (OX1):* Aplica el operador de cruce Order Crossover (OX1) (`self.ox1`) a las representaciones de los padres (listas de valores de sus propiedades) para generar una nueva secuencia de elementos (`c1`), que representa un hijo potencial.

6.  *División en clústeres:* La secuencia del hijo (`c1`) se divide en clústeres de tamaño variable (entre `min_size=2` y `max_size=5`) utilizando el método `self.divide_ids_into_clusters_with_coords`. Esta división se realiza para hacer que la optimización con QAOA sea computacionalmente más viable al aplicarla a subproblemas más pequeños. Si `self.verbose` es `True`, se imprime el número de clústeres creados.

7.  *Optimización de clústeres con QAOA:* Para cada clúster generado:
    * Se imprime información sobre el clúster que se va a optimizar si `self.verbose` es `True`.
    * Si el problema tiene la restricción de no volver al origen (`self.return_to_origin == "no_return"`), se identifica el clúster que contiene el primer elemento (elemento con ID 0). Para este clúster, se establece la bandera `contains_zero` en `True`, lo que influirá en la forma en que QAOA intenta optimizar la ruta dentro de este clúster (asegurando que comience con el elemento 0).
    * Se intenta resolver el problema de optimización de la ruta dentro del clúster utilizando el algoritmo QAOA (`self.QT.solve_tsp_with_qaoa`). Se le proporciona la matriz de elementos del clúster, el tipo de problema (minimizar/maximizar), la bandera `contains_zero` y la restricción `return_to_origin`.
    * Si QAOA encuentra una combinación completa de elementos dentro del clúster (y la longitud de la combinación coincide con el número de elementos en el clúster), esta combinación se añade a la lista `cluster_combinations`.
    * Si QAOA no encuentra una combinación válida o la combinación no es completa, se genera una combinación aleatoria de los elementos del clúster y se añade a `cluster_combinations`. Se imprime una advertencia si esto ocurre.

8.  Combinación de rutas de clústeres: Después de optimizar (o generar aleatoriamente) la ruta para cada clúster, todas las rutas de los clústeres se concatenan en una única lista llamada `final_combination`. Se imprime la combinación final de elementos.

9.  Visualización de la ruta (opcional): Si `self.verbose` es `True`, se utiliza el método `self.plot_tsp_route` para generar y mostrar un gráfico de la ruta final, utilizando la matriz de elementos original y la combinación final.

10. Creación y validación del nuevo individuo:
    * Se verifica si la `final_combination` contiene todos los elementos esperados del problema.
    * Si la combinación es completa, se crea un nuevo objeto `Individual` con la combinación como sus valores (`child_values`), el diccionario de límites, información sobre la generación y las restricciones del problema. Los parámetros VQC se establecen en `None` ya que en problemas de permutaciones el individuo se define principalmente por el orden de sus elementos.
    * Se valida si el nuevo individuo es un duplicado de alguno ya presente en la `children_list` y si no tiene malformaciones (`new_child.get_individual_malformation()`). Si no es un duplicado y no tiene malformaciones, se añade a la `children_list` y se imprime un mensaje de éxito.

11. *Impresión de información de los hijos:* Una vez que se ha generado el número deseado de hijos, se imprime información sobre el ID, las propiedades (que en este caso representan la combinación de elementos) y la combinación de cada hijo en la `children_list`.

12. *Retorno de la nueva generación:* Finalmente, el método devuelve la lista de los nuevos individuos (`self.children_list`).

## 4. Clase CrazyVariabilityExplossion

La clase `CrazyVariabilityExplossion` hereda de `VariabilityExplossion` e implementa una estrategia específica para activar y gestionar la explosión de variabilidad.

#### 4.1. Herencia

`CrazyVariabilityExplossion` hereda de la clase base abstracta `VariabilityExplossion`.

#### 4.2. Inicialización (`__init__`)

```
def __init__(self, early_stopping_generations: int, problem_type: Literal['maximize', 'minimize'], round_decimals: int = 3, verbose: bool = False):
```

##### 4.2.1. Parámetros

Hereda los parámetros de la clase base `VariabilityExplossion`.

##### 4.2.2. Funcionalidad

Llama al constructor de la clase base (`super().__init__(...)`) para inicializar los atributos relacionados con el early stopping y la explosión de variabilidad.

#### 4.3. Métodos implementados

```
def evaluate_early_stopping(self, generations_fitness_statistics_df: pd.DataFrame | None) -> tuple:
```

#### 4.3.1. `evaluate_early_stopping`

Evalúa si se deben activar las condiciones para la explosión de variabilidad basándose en la repetición de la moda del fitness en las últimas generaciones.

* `generations_fitness_statistics_df` (`pd.DataFrame | None`): DataFrame con la información general de la generación.
* Retorna `tuple`:
    * `m_proba` (`float | None`): Probabilidad de mutación de un individuo.
    * `m_gen_proba` (`float | None`): Probabilidad de mutación de un gen.
    * `m_policy` (`Literal['soft', 'normal', 'hard'] | None`): Política de mutación.
    * `early_stopping_generations_execute` (`bool | None`): Indica si se debe ejecutar el early stopping. Retorna `None` si no se cumplen las condiciones.

##### 4.3.1.1. Funcionalidad

1.  Verifica si se proporciona un DataFrame de estadísticas de generaciones.
2.  Si el número de generaciones es al menos el doble de `early_stopping_generations`, analiza las últimas `early_stopping_generations`.
3.  Determina la moda de la columna 'min' (para problemas de minimización) o 'max' (para problemas de maximización) en las últimas generaciones.
4.  Si todos los valores de la moda son iguales, se considera que el algoritmo está estancado y se llama a `self.execute_variability_explossion()` para activar la explosión.
5.  Si la explosión de variabilidad ya está activa (`self.early_stopping_generations_executed`), se vuelve a ejecutar en cada iteración.
6.  Si no se cumplen las condiciones, retorna `None` para las probabilidades de mutación, la política y el flag de ejecución.

```
def execute_variability_explossion(self):
```

#### 4.3.2. `execute_variability_explossion`

Ejecuta la explosión de variabilidad aumentando las probabilidades de mutación y cambiando la política de mutación.

* Retorna `tuple`:
    * `mutate_probability` (`float`): Probabilidad de mutación de un individuo.
    * `mutate_gen_probability` (`float`): Probabilidad de mutación de un gen.
    * `mutation_policy` (`Literal['soft', 'normal', 'hard']`): Política de mutación.
    * `early_stopping_generations_executed` (`bool`): Indica si se debe ejecutar el early stopping (`True` en este caso).

##### 4.3.2.1. Funcionalidad

1.  Si la explosión de variabilidad ya está activa, restablece las probabilidades de mutación y la política a valores "suaves".
2.  Si la explosión de variabilidad no estaba activa, aumenta drásticamente las probabilidades de mutación (individuo y gen) y establece la política de mutación en 'hard' para introducir una gran cantidad de diversidad. También establece el flag `self.early_stopping_generations_executed` en `True`.
3.  Retorna las nuevas probabilidades de mutación, la política y el flag de ejecución.

```
def stop_genetic_iterations(self, generations_fitness_statistics_df: pd.DataFrame | None):
```

#### 4.3.3. `stop_genetic_iterations`

Evalúa si se debe detener el proceso de evolución después de que la explosión de variabilidad ha estado activa durante un cierto número de generaciones sin mejorar el mejor resultado encontrado hasta el momento.

* `generations_fitness_statistics_df` (`pd.DataFrame | None`): DataFrame con los resultados estadísticos de los individuos.
* Retorna `bool`: `True` si se debe detener la evolución, `False` si se debe continuar.

##### 4.3.3.1. Funcionalidad

1.  Verifica si la explosión de variabilidad está activa (`self.early_stopping_generations_executed`).
2.  Si está activa, incrementa los contadores de generaciones de early stopping.
3.  Cuando el contador de generaciones activas alcanza `self.early_stopping_generations`:
    * Compara el mejor valor de fitness global con el mejor valor de fitness en las últimas `self.early_stopping_generations` generaciones.
    * Si no ha habido mejora (o ha empeorado en el caso de maximización), se considera que la explosión de variabilidad no ha sido efectiva y se retorna `True` para detener la evolución.
    * Si ha habido mejora, se restablece el contador de generaciones activas para dar más margen al algoritmo.
4.  Si `self.verbose` es `True`, se llama a `self.print_variability_status()` para mostrar el estado de la explosión.
5.  Retorna `False` si la explosión de variabilidad no está activa o si aún no se ha alcanzado el límite de generaciones sin mejora.

```
def print_variability_status(self):
```

#### 4.3.4. `print_variability_status`

Imprime el estado actual de la explosión de variabilidad.

#### 4.3.4.1. Funcionalidad

1.  Imprime un encabezado con el resumen del estado de `CrazyVariabilityExplossion`.
2.  Indica si la explosión está activa (`self.early_stopping_generations_executed`) con un color diferente según el estado.
3.  Si la explosión está activa, muestra el número total de generaciones que lleva activa y el número de generaciones transcurridas desde la última mejora.


### 5. Clase `VariabilityExplossion` (Abstracta)

La clase abstracta `VariabilityExplossion` define la interfaz para implementar mecanismos de explosión de variabilidad en algoritmos genéticos. El objetivo de estos mecanismos es evitar la convergencia prematura y escapar de óptimos locales aumentando la diversidad de la población cuando se detecta estancamiento.

#### 5.1. Importaciones

```
from abc import ABC, abstractmethod
from info_tools import InfoTools
from typing import Literal

import pandas as pd
import numpy as np
```

#### 5.2. Descripción

Esta clase base proporciona la estructura para gestionar el early stopping y la activación de la explosión de variabilidad. Las subclases concretas deben implementar los métodos abstractos para definir la lógica específica de evaluación, ejecución y detención de la explosión de variabilidad.

#### 5.3. Inicialización (`__init__`)

```
def __init__(self, early_stopping_generations: int, problem_type: Literal['maximize', 'minimize'], round_decimals: int = 3, verbose: bool = False):
```

#### 5.4. Parámetros

* `early_stopping_generations` (`int`): Número de generaciones consecutivas con una moda de fitness similar que deben ocurrir para considerar la activación de la explosión de variabilidad.
* `problem_type` (`Literal['maximize', 'minimize']`): Indica si el problema es de maximización o minimización, lo que influye en la evaluación del estancamiento.
* `round_decimals` (`int`, opcional): Número de decimales a redondear al analizar las estadísticas de variabilidad. Por defecto es 3.
* `verbose` (`bool`, opcional): Flag para habilitar la impresión de mensajes detallados sobre el estado de la explosión de variabilidad. Por defecto es `False`.

#### 5.5. Atributos

* `early_stopping_generations` (`int`): Almacena el número de generaciones de espera para el early stopping.
* `problem_type` (`Literal['maximize', 'minimize']`): Almacena el tipo de problema.
* `IT` (`InfoTools`): Instancia de la clase `InfoTools` para facilitar la impresión informativa.
* `round_decimals` (`int`): Almacena el número de decimales para el redondeo.
* `verbose` (`int`): Almacena el valor del flag verbose.
* `early_stopping_generations_executed_counter` (`int`): Contador de las generaciones transcurridas desde la última explosión de variabilidad.
* `total_early_stopping_generations_executed_counter` (`int`): Contador total de generaciones transcurridas desde la primera explosión de variabilidad.
* `early_stopping_generations_executed` (`bool`): Flag que indica si la explosión de variabilidad está actualmente activa.
* `mutate_probability` (`float`): Probabilidad de mutación de un individuo (valor por defecto inicial).
* `mutate_gen_probability` (`float`): Probabilidad de mutación de un gen dentro de un individuo (valor por defecto inicial).
* `mutation_policy` (`Literal['soft', 'normal', 'hard']`): Política de mutación a aplicar (valor por defecto inicial).

#### 5.6. Métodos abstractos

```
@abstractmethod
def evaluate_early_stopping(self, generations_fitness_statistics_df: pd.DataFrame | None) -> None:
    pass
```

##### 5.6.1. `evaluate_early_stopping`

Método abstracto que debe implementar la lógica para determinar si se deben activar las condiciones de early stopping y la explosión de variabilidad. Por ejemplo, podría verificar si la moda de la aptitud se ha mantenido constante durante un cierto número de generaciones.

* `generations_fitness_statistics_df` (`pd.DataFrame | None`): DataFrame que contiene las estadísticas de fitness de las generaciones.

```
@abstractmethod
def execute_variability_explossion(self):
    pass
```

##### 5.6.2. `execute_variability_explossion`

Método abstracto que define cómo se ejecuta la explosión de variabilidad. Esto podría implicar el aumento de las probabilidades de mutación o la aplicación de otras estrategias para introducir mayor diversidad en la población.

```
@abstractmethod
def stop_genetic_iterations(self, generations_fitness_statistics_df: pd.DataFrame | None):
    pass
```

##### 5.6.3. `stop_genetic_iterations`

Método abstracto que evalúa si el proceso de evolución genética debe detenerse basándose en las condiciones de early stopping. Por ejemplo, si la explosión de variabilidad no ha logrado mejorar los resultados después de un cierto número de generaciones.

* `generations_fitness_statistics_df` (`pd.DataFrame | None`): DataFrame con las estadísticas de fitness de las generaciones.
* Retorna `bool`: `True` si se debe detener la evolución, `False` si debe continuar.

```
@abstractmethod
def print_variability_status(self):
    pass
```

##### 5.6.4. `print_variability_status`

Método abstracto que imprime el estado actual de la explosión de variabilidad, como si está activa, cuánto tiempo lleva activa y si se han observado mejoras.

