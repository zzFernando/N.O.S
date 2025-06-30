"""
Módulo para visualização espacial e meteorológica.
"""

import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_airport_ga_map(airports_df, view=True, filename="airport_ga_map"):
    """
    Plota um mapa interativo dos aeroportos com taxas de Go-Around.
    
    Args:
        airports_df (pandas.DataFrame): DataFrame com colunas:
            - icao: código ICAO do aeroporto
            - lat: latitude
            - lon: longitude
            - ga_rate: taxa de Go-Around
            - n_landings: número de pousos
            - continent: continente
        view (bool): Se True, exibe o mapa interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        folium.Map: Mapa interativo
    """
    # Criar mapa base
    m = folium.Map(
        location=[airports_df['lat'].mean(), airports_df['lon'].mean()],
        zoom_start=4
    )
    
    # Criar clusters por continente
    clusters = {}
    for continent in airports_df['continent'].unique():
        clusters[continent] = MarkerCluster(name=continent)
    
    # Adicionar marcadores
    for _, row in airports_df.iterrows():
        # Calcular cor baseada na taxa de GA
        ga_rate = row['ga_rate']
        color = f'rgb({int(255 * ga_rate)}, {int(255 * (1 - ga_rate))}, 0)'
        
        # Calcular tamanho baseado no número de pousos
        size = 5 + 15 * (row['n_landings'] / airports_df['n_landings'].max())
        
        # Criar popup
        popup = f"""
        <b>Aeroporto:</b> {row['icao']}<br>
        <b>Taxa de GA:</b> {ga_rate:.2%}<br>
        <b>Pousos:</b> {row['n_landings']}<br>
        <b>Continente:</b> {row['continent']}
        """
        
        # Adicionar marcador ao cluster apropriado
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=size,
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(clusters[row['continent']])
    
    # Adicionar clusters ao mapa
    for cluster in clusters.values():
        cluster.add_to(m)
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    os.makedirs('neat/visualizations/output', exist_ok=True)
    m.save(f'neat/visualizations/output/{filename}.html')
    
    if view:
        import webbrowser
        webbrowser.open(f'neat/visualizations/output/{filename}.html')
    
    return m

def plot_weather_vs_ga(weather_df, view=True, filename="weather_vs_ga"):
    """
    Plota análise de condições meteorológicas vs Go-Around.
    
    Args:
        weather_df (pandas.DataFrame): DataFrame com colunas:
            - visibility: visibilidade
            - wind_speed: velocidade do vento
            - wind_direction: direção do vento
            - precipitation: precipitação
            - ga_occurred: ocorrência de GA (0/1)
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Visibilidade vs GA',
            'Velocidade do Vento vs GA',
            'Direção do Vento vs GA',
            'Precipitação vs GA'
        )
    )
    
    # Plotar visibilidade
    fig.add_trace(
        go.Box(
            x=weather_df['ga_occurred'],
            y=weather_df['visibility'],
            name='Visibilidade',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ),
        row=1, col=1
    )
    
    # Plotar velocidade do vento
    fig.add_trace(
        go.Box(
            x=weather_df['ga_occurred'],
            y=weather_df['wind_speed'],
            name='Velocidade do Vento',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ),
        row=1, col=2
    )
    
    # Plotar direção do vento
    fig.add_trace(
        go.Box(
            x=weather_df['ga_occurred'],
            y=weather_df['wind_direction'],
            name='Direção do Vento',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ),
        row=2, col=1
    )
    
    # Plotar precipitação
    fig.add_trace(
        go.Box(
            x=weather_df['ga_occurred'],
            y=weather_df['precipitation'],
            name='Precipitação',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ),
        row=2, col=2
    )
    
    # Atualizar layout
    fig.update_layout(
        title_text='Análise de Condições Meteorológicas vs Go-Around',
        height=800,
        showlegend=False
    )
    
    # Atualizar eixos
    fig.update_xaxes(title_text='Go-Around Ocorreu', row=1, col=1)
    fig.update_xaxes(title_text='Go-Around Ocorreu', row=1, col=2)
    fig.update_xaxes(title_text='Go-Around Ocorreu', row=2, col=1)
    fig.update_xaxes(title_text='Go-Around Ocorreu', row=2, col=2)
    
    fig.update_yaxes(title_text='Visibilidade (m)', row=1, col=1)
    fig.update_yaxes(title_text='Velocidade do Vento (kt)', row=1, col=2)
    fig.update_yaxes(title_text='Direção do Vento (°)', row=2, col=1)
    fig.update_yaxes(title_text='Precipitação (mm)', row=2, col=2)
    
    # Salvar figuras
    os.makedirs('neat/visualizations/output', exist_ok=True)
    fig.write_html(f'neat/visualizations/output/{filename}.html')
    fig.write_image(f'neat/visualizations/output/{filename}.png')
    fig.write_image(f'neat/visualizations/output/{filename}.svg')
    
    if view:
        fig.show()
    
    return fig 