import os
import neat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from visualizations import (
    plot_winner_net,
    plot_fitness_history,
    plot_species_evolution,
    plot_genotype_embedding,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    generate_classification_report,
    export_training_log
)

X_PATH = "data/X.npy"
Y_PATH = "data/y.npy"
CONFIG_PATH = "neat/config_neat.txt"

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        correct = 0
        for xi, yi in zip(X_train, y_train):
            output = net.activate(xi)
            prediction = 1 if output[0] > 0.5 else 0
            if prediction == yi:
                correct += 1
        genome.fitness = correct / len(X_train)

if __name__ == "__main__":
    plot_winner_net(config, winner)
    plot_fitness_history(stats)
    plot_species_evolution(stats)
    plot_genotype_embedding(stats)
    plot_confusion_matrix(y_test, predictions)
    plot_roc_curve(y_test, scores)
    plot_precision_recall(y_test, scores)
    report = generate_classification_report(y_test, predictions)
    print("\nRelatório de Classificação:")
    print(report)
    export_training_log(stats, config)
