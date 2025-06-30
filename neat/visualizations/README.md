# Módulo de Visualizações NEAT

Este módulo fornece um conjunto abrangente de funções para visualizar e analisar o treinamento de redes NEAT e dados de Go-Around.

## Estrutura

```
visualizations/
├── __init__.py              # Interface do módulo
├── network_visualization.py # Visualização de redes
├── evolution_visualization.py # Visualização da evolução
├── performance_visualization.py # Visualização de performance
├── spatial_visualization.py # Visualização espacial
├── utils.py                # Funções utilitárias
└── output/                 # Diretório para arquivos gerados
```

## Requisitos

```bash
pip install plotly networkx folium umap-learn scikit-learn pandas
```

## Uso

### 1. Visualização da Rede Campeã

```python
from neat.visualizations import plot_winner_net

# Após o treinamento
plot_winner_net(config, winner)
```

### 2. Histórico de Evolução

```python
from neat.visualizations import plot_fitness_history, plot_species_evolution

# Plotar histórico de fitness
plot_fitness_history(stats)

# Plotar evolução das espécies
plot_species_evolution(stats)
```

### 3. Performance do Modelo

```python
from neat.visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    generate_classification_report
)

# Matriz de confusão
plot_confusion_matrix(y_true, y_pred)

# Curva ROC
plot_roc_curve(y_true, y_score)

# Curva Precision-Recall
plot_precision_recall(y_true, y_score)

# Relatório de classificação
report = generate_classification_report(y_true, y_pred)
```

### 4. Análise Genotípica

```python
from neat.visualizations import plot_genotype_embedding

# Visualização dos genótipos
plot_genotype_embedding(stats)
```

### 5. Visualização Espacial e Meteorológica

```python
from neat.visualizations import plot_airport_ga_map, plot_weather_vs_ga

# Mapa de aeroportos
plot_airport_ga_map(airports_df)

# Análise meteorológica
plot_weather_vs_ga(weather_df)
```

### 6. Exportação de Logs

```python
from neat.visualizations import export_training_log

# Exportar logs do treinamento
json_path, csv_path = export_training_log(stats, config)
```

## Output

Todas as visualizações são salvas no diretório `neat/visualizations/output/` nos seguintes formatos:
- HTML (interativo)
- PNG (imagem)
- SVG (vetorial)

## Exemplos

### Visualização da Rede

```python
import neat
from neat.visualizations import plot_winner_net

# Carregar configuração e genoma vencedor
config = neat.Config(...)
with open('winner.pkl', 'rb') as f:
    winner = pickle.load(f)

# Visualizar rede
plot_winner_net(config, winner)
```

### Análise de Performance

```python
import numpy as np
from neat.visualizations import plot_confusion_matrix

# Gerar predições
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

# Plotar matriz de confusão
plot_confusion_matrix(y_true, y_pred)
```

## Contribuindo

Para adicionar novas visualizações:

1. Crie uma nova função no módulo apropriado
2. Adicione a função ao `__init__.py`
3. Atualize este README com exemplos de uso
4. Adicione testes se necessário 