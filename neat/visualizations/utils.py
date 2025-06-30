import os
import json
import pandas as pd
from datetime import datetime
import numpy as np

def export_training_log(stats, config, filename=None):
    if filename is None:
        filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs('neat/visualizations/output', exist_ok=True)
    fitness_stats = stats.get_fitness_stat(lambda x: x)
    generations = range(len(fitness_stats))
    species_sizes = stats.get_species_sizes()
    json_data = {
        'config': {
            'pop_size': config.pop_size,
            'fitness_threshold': config.fitness_threshold,
            'num_inputs': config.genome_config.num_inputs,
            'num_outputs': config.genome_config.num_outputs,
            'num_hidden': config.genome_config.num_hidden,
            'activation_default': config.genome_config.activation_default,
            'aggregation_default': config.genome_config.aggregation_default
        },
        'generations': []
    }
    csv_data = []
    for gen in generations:
        fitness_values = fitness_stats[gen]
        if hasattr(fitness_values, '__iter__') and len(fitness_values) > 0:
            fitness_array = np.array(list(fitness_values))
            best_fitness = float(np.max(fitness_array))
            avg_fitness = float(np.mean(fitness_array))
            std_fitness = float(np.std(fitness_array))
        else:
            best_fitness = float(fitness_values) if fitness_values is not None else 0.0
            avg_fitness = best_fitness
            std_fitness = 0.0
        gen_data = {
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'num_species': len(species_sizes[gen]) if gen < len(species_sizes) else 0,
            'avg_species_size': np.mean(species_sizes[gen]) if gen < len(species_sizes) and species_sizes[gen] else 0
        }
        json_data['generations'].append(gen_data)
        csv_data.append(gen_data)
    json_path = f'neat/visualizations/output/{filename}.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    csv_path = f'neat/visualizations/output/{filename}.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    return json_path, csv_path 