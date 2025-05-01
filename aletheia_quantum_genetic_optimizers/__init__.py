__version__ = "0.1.12"

# Importar las clases
from aletheia_quantum_genetic_optimizers.individuals import Individual
from aletheia_quantum_genetic_optimizers.bounds import BoundCreator
from aletheia_quantum_genetic_optimizers.genetic_optimizer import GenethicOptimizer

# Exportar las clases para que sean accesibles directamente desde el paquete
__all__ = ['Individual', 'BoundCreator', 'GenethicOptimizer']
