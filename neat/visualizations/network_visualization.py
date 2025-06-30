"""
Módulo para visualização de redes neurais NEAT.
"""

import os
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import neat

def plot_winner_net(config, genome, view=True, filename="winner_network"):
    """
    Visualiza a rede neural do genoma vencedor usando plotly.
    
    Args:
        config (neat.Config): Configuração NEAT
        genome (neat.DefaultGenome): Genoma vencedor
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos (sem extensão)
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    G = nx.DiGraph()
    for key, conn in genome.connections.items():
        if conn.enabled:
            G.add_edge(key[0], key[1], weight=conn.weight)
    for node in genome.nodes:
        G.add_node(node)
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        color = 'green' if data['weight'] > 0 else 'red'
        edge_colors.append(color)
    fig = go.Figure()
    for i in range(len(edge_colors)):
        fig.add_trace(go.Scatter(
            x=edge_x[i*3:i*3+2],
            y=edge_y[i*3:i*3+2],
            mode='lines',
            line=dict(width=2, color=edge_colors[i]),
            hoverinfo='none',
            showlegend=False
        ))
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in genome.nodes:
            node_obj = genome.nodes[node]
            tooltip = f"""
            Node: {node}<br>
            Bias: {getattr(node_obj, 'bias', 0.0):.3f}<br>
            Response: {getattr(node_obj, 'response', 1.0):.3f}<br>
            Activation: {getattr(node_obj, 'activation', 'sigmoid')}<br>
            Aggregation: {getattr(node_obj, 'aggregation', 'sum')}
            """
        else:
            tooltip = f"Node: {node}<br>(sem atributos do genoma)"
        node_text.append(tooltip)
        node_sizes.append(10 + 5 * G.degree(node))
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                size=node_sizes,
                color='lightblue',
                line_width=2
            )
        )
    )
    fig.update_layout(
        title='Rede Neural Vencedora',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    os.makedirs('neat/visualizations/output', exist_ok=True)
    fig.write_html(f'neat/visualizations/output/{filename}.html')
    fig.write_image(f'neat/visualizations/output/{filename}.png')
    fig.write_image(f'neat/visualizations/output/{filename}.svg')
    if view:
        fig.show()
    return fig 