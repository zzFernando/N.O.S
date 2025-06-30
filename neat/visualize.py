import os
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import neat

# Create directory for visualization files if it doesn't exist
os.makedirs("neat/visualizations", exist_ok=True)

def plot_stats(statistics, ylog=False, view=False, filename="neat/visualizations/fitness.svg"):
    """ Plota a curva de fitness ao longo das gerações """
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [g.fitness for g in statistics.most_fit_genomes]

    plt.figure()
    plt.plot(generation, best_fitness, "b-", label="Melhor Fitness")
    plt.title("Fitness por Geração")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    if ylog:
        plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def plot_species(statistics, view=False, filename="neat/visualizations/species.svg"):
    species_sizes = statistics.get_species_sizes()
    plt.figure()
    plt.stackplot(
        range(len(species_sizes)),
        np.array(species_sizes).T,
        labels=["Espécie %d" % i for i in range(len(species_sizes[0]))]
    )
    plt.title("Tamanho das Espécies")
    plt.xlabel("Geração")
    plt.ylabel("Número de Genomas")
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def draw_net(config, genome, view=False, filename="neat/visualizations/network", node_names=None, show_disabled=True):
    from neat.graphs import feed_forward_layers
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }
    dot = graphviz.Digraph(format="svg", node_attr=node_attrs)
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, genome.connections)
    for n in inputs:
        name = node_names.get(n, str(n)) if node_names else str(n)
        dot.node(name, _attributes={"style": "filled", "fillcolor": "lightgray"})
    for n in outputs:
        name = node_names.get(n, str(n)) if node_names else str(n)
        dot.node(name, _attributes={"style": "filled", "fillcolor": "lightblue"})
    for conn_key, conn in genome.connections.items():
        if not show_disabled and not conn.enabled:
            continue
        input_node, output_node = conn_key
        a = node_names.get(input_node, str(input_node)) if node_names else str(input_node)
        b = node_names.get(output_node, str(output_node)) if node_names else str(output_node)
        style = "solid" if conn.enabled else "dotted"
        color = "green" if conn.weight > 0 else "red"
        width = str(0.1 + abs(conn.weight / 5.0))
        dot.edge(a, b, _attributes={"style": style, "color": color, "penwidth": width})
    dot.render(filename, view=view)
