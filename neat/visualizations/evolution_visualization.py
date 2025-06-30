"""
Módulo para visualização da evolução das redes NEAT.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from sklearn.preprocessing import StandardScaler
import statistics
from sklearn.decomposition import PCA

def plot_fitness_history(stats, view=True, filename="fitness_history"):
    """
    Plota o histórico de fitness ao longo das gerações.
    
    Args:
        stats (neat.StatisticsReporter): Estatísticas do treinamento
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    try:
        # Extrair dados usando as funções estatísticas do NEAT
        fitness_mean = stats.get_fitness_mean()
        fitness_std = stats.get_fitness_stdev()
        
        # Converter range para lista
        generations = list(range(len(fitness_mean)))
        
        # Calcular max/min de forma segura
        fitness_max = []
        fitness_min = []
        for scores in stats.get_fitness_stat(lambda x: x):
            if scores:  # Verifica se a lista não está vazia
                fitness_max.append(max(scores))
                fitness_min.append(min(scores))
            else:
                fitness_max.append(0)
                fitness_min.append(0)
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar média com intervalo de confiança
        fig.add_trace(go.Scatter(
            x=generations,
            y=fitness_mean,
            name='Média',
            line=dict(color='blue')
        ))
        
        # Adicionar intervalo de confiança
        upper_bound = np.array(fitness_mean) + np.array(fitness_std)
        lower_bound = np.array(fitness_mean) - np.array(fitness_std)
        
        fig.add_trace(go.Scatter(
            x=generations + generations[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Desvio Padrão'
        ))
        
        # Adicionar máximo e mínimo
        fig.add_trace(go.Scatter(
            x=generations,
            y=fitness_max,
            name='Máximo',
            line=dict(color='green', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=fitness_min,
            name='Mínimo',
            line=dict(color='red', dash='dash')
        ))
        
        # Atualizar layout
        fig.update_layout(
            title='Evolução do Fitness',
            xaxis_title='Geração',
            yaxis_title='Fitness',
            hovermode='x unified',
            showlegend=True
        )
        
        # Criar diretório se não existir
        os.makedirs('neat/visualizations/output', exist_ok=True)
        
        # Salvar figuras
        fig.write_html(f'neat/visualizations/output/{filename}.html')
        fig.write_image(f'neat/visualizations/output/{filename}.png')
        fig.write_image(f'neat/visualizations/output/{filename}.svg')
        
        if view:
            fig.show()
        
        return fig
        
    except Exception as e:
        print(f"Erro ao plotar histórico de fitness: {str(e)}")
        print("Dados disponíveis:")
        print(f"Número de gerações: {len(fitness_mean)}")
        print(f"Média de fitness: {fitness_mean}")
        print(f"Desvio padrão: {fitness_std}")
        raise

def plot_species_evolution(stats, view=True, filename="species_evolution"):
    """
    Plota a evolução das espécies ao longo das gerações.
    
    Args:
        stats (neat.StatisticsReporter): Estatísticas do treinamento
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    try:
        # Extrair dados
        species_sizes = stats.get_species_sizes()
        generations = list(range(len(species_sizes)))
        num_species = [len(sizes) for sizes in species_sizes]
        avg_species_size = [np.mean(sizes) if sizes else 0 for sizes in species_sizes]
        
        # Criar figura com subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Número de Espécies', 'Tamanho Médio das Espécies')
        )
        
        # Plotar número de espécies
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=num_species,
                name='Número de Espécies',
                line=dict(color='purple')
            ),
            row=1, col=1
        )
        
        # Plotar tamanho médio
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=avg_species_size,
                name='Tamanho Médio',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # Atualizar layout
        fig.update_layout(
            title_text='Evolução das Espécies',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Geração', row=1, col=1)
        fig.update_xaxes(title_text='Geração', row=2, col=1)
        fig.update_yaxes(title_text='Número de Espécies', row=1, col=1)
        fig.update_yaxes(title_text='Tamanho Médio', row=2, col=1)
        
        # Criar diretório se não existir
        os.makedirs('neat/visualizations/output', exist_ok=True)
        
        # Salvar figuras
        fig.write_html(f'neat/visualizations/output/{filename}.html')
        fig.write_image(f'neat/visualizations/output/{filename}.png')
        fig.write_image(f'neat/visualizations/output/{filename}.svg')
        
        if view:
            fig.show()
        
        return fig
        
    except Exception as e:
        print(f"Erro ao plotar evolução das espécies: {str(e)}")
        print("Dados disponíveis:")
        print(f"Número de gerações: {len(species_sizes)}")
        print(f"Tamanhos das espécies: {species_sizes}")
        raise

def plot_genotype_embedding(stats, view=True, filename="genotype_embedding"):
    """
    Plota a evolução dos genótipos usando redução de dimensionalidade.
    
    Args:
        stats (neat.StatisticsReporter): Estatísticas do treinamento
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    try:
        # Extrair características dos genomas mais aptos
        features = []
        fitnesses = []
        generations = []
        
        # Usar apenas os genomas mais aptos de cada geração
        for gen, genome in enumerate(stats.most_fit_genomes):
            try:
                # Extrair características
                genome_features = [
                    len(genome.nodes),
                    len(genome.connections),
                    np.mean([c.weight for c in genome.connections.values()]) if genome.connections else 0,
                    np.std([c.weight for c in genome.connections.values()]) if genome.connections else 0,
                    np.mean([n.bias for n in genome.nodes.values()]) if genome.nodes else 0,
                    np.std([n.bias for n in genome.nodes.values()]) if genome.nodes else 0
                ]
                features.append(genome_features)
                fitnesses.append(genome.fitness)
                generations.append(gen)
            except Exception as e:
                print(f"Erro ao processar genoma da geração {gen}: {str(e)}")
                continue
        
        if not features:
            print("Nenhuma característica foi extraída dos genomas")
            return None
        
        # Verificar se temos dados suficientes para UMAP
        if len(features) < 3:
            print(f"Poucos dados para UMAP ({len(features)} genomas). Usando PCA em vez de UMAP.")
            # Usar PCA para poucos dados
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(features_scaled)
            method_name = "PCA"
        else:
            # Usar UMAP para muitos dados
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Ajustar parâmetros do UMAP para poucos dados
            n_neighbors = min(2, len(features) - 1)  # Máximo 2 vizinhos para poucos dados
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            embedding = reducer.fit_transform(features_scaled)
            method_name = "UMAP"
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar pontos
        fig.add_trace(go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(
                size=12,
                color=fitnesses,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Fitness')
            ),
            text=[f'Geração: {gen}<br>Fitness: {fit:.3f}<br>Nós: {len(stats.most_fit_genomes[gen].nodes)}<br>Conexões: {len(stats.most_fit_genomes[gen].connections)}' for gen, fit in zip(generations, fitnesses)],
            hoverinfo='text'
        ))
        
        # Atualizar layout
        fig.update_layout(
            title=f'Visualização dos Genótipos Mais Aptos ({method_name})',
            xaxis_title=f'{method_name} 1',
            yaxis_title=f'{method_name} 2',
            hovermode='closest'
        )
        
        # Criar diretório se não existir
        os.makedirs('neat/visualizations/output', exist_ok=True)
        
        # Salvar figuras
        fig.write_html(f'neat/visualizations/output/{filename}.html')
        fig.write_image(f'neat/visualizations/output/{filename}.png')
        fig.write_image(f'neat/visualizations/output/{filename}.svg')
        
        if view:
            fig.show()
        
        return fig
        
    except Exception as e:
        print(f"Erro ao plotar embedding dos genótipos: {str(e)}")
        print("Dados disponíveis:")
        print(f"Número de genomas mais aptos: {len(stats.most_fit_genomes)}")
        print(f"Tamanhos das espécies: {stats.get_species_sizes()}")
        raise 