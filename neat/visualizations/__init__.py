"""
Módulo de visualizações para análise de redes NEAT e dados de Go-Around.

Este módulo fornece um conjunto abrangente de funções para visualizar:
- Topologia e evolução das redes NEAT
- Performance preditiva do modelo
- Análise genotípica
- Visualizações espaciais e meteorológicas
"""

from .network_visualization import plot_winner_net
from .evolution_visualization import (
    plot_fitness_history,
    plot_species_evolution,
    plot_genotype_embedding
)
from .performance_visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    generate_classification_report
)
from .spatial_visualization import (
    plot_airport_ga_map,
    plot_weather_vs_ga
)
from .utils import export_training_log

__all__ = [
    'plot_winner_net',
    'plot_fitness_history',
    'plot_species_evolution',
    'plot_genotype_embedding',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall',
    'generate_classification_report',
    'plot_airport_ga_map',
    'plot_weather_vs_ga',
    'export_training_log'
] 