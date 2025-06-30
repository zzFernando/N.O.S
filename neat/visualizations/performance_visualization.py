"""
Módulo para visualização da performance do modelo.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, view=True, filename="confusion_matrix"):
    """
    Plota a matriz de confusão.
    
    Args:
        y_true (array-like): Valores reais
        y_pred (array-like): Valores preditos
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Negativo", "Positivo"],
        y=["Negativo", "Positivo"],
        colorscale="Blues",
        showscale=True
    ))
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real"
    )
    fig.write_html(f'neat/visualizations/output/{filename}.html')
    fig.write_image(f'neat/visualizations/output/{filename}.png')
    fig.write_image(f'neat/visualizations/output/{filename}.svg')
    if view:
        fig.show()
    return fig

def plot_roc_curve(y_true, y_score, view=True, filename="roc_curve"):
    """
    Plota a curva ROC.
    
    Args:
        y_true (array-like): Valores reais
        y_score (array-like): Scores de probabilidade
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar curva ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='blue')
    ))
    
    # Adicionar linha de referência
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Aleatório',
        line=dict(color='red', dash='dash')
    ))
    
    # Atualizar layout
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        hovermode='x unified'
    )
    
    # Salvar figuras
    os.makedirs('neat/visualizations/output', exist_ok=True)
    fig.write_html(f'neat/visualizations/output/{filename}.html')
    fig.write_image(f'neat/visualizations/output/{filename}.png')
    fig.write_image(f'neat/visualizations/output/{filename}.svg')
    
    if view:
        fig.show()
    
    return fig

def plot_precision_recall(y_true, y_score, view=True, filename="precision_recall"):
    """
    Plota a curva Precision-Recall.
    
    Args:
        y_true (array-like): Valores reais
        y_score (array-like): Scores de probabilidade
        view (bool): Se True, exibe o gráfico interativamente
        filename (str): Nome base para salvar os arquivos
    
    Returns:
        plotly.graph_objects.Figure: Figura do plotly
    """
    # Calcular curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar curva Precision-Recall
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        name=f'Precision-Recall (AUC = {pr_auc:.3f})',
        line=dict(color='green')
    ))
    
    # Atualizar layout
    fig.update_layout(
        title='Curva Precision-Recall',
        xaxis_title='Recall',
        yaxis_title='Precision',
        hovermode='x unified'
    )
    
    # Salvar figuras
    os.makedirs('neat/visualizations/output', exist_ok=True)
    fig.write_html(f'neat/visualizations/output/{filename}.html')
    fig.write_image(f'neat/visualizations/output/{filename}.png')
    fig.write_image(f'neat/visualizations/output/{filename}.svg')
    
    if view:
        fig.show()
    
    return fig

def generate_classification_report(y_true, y_pred, y_score=None, filename="classification_report"):
    """
    Gera e salva o relatório de classificação.
    
    Args:
        y_true (array-like): Valores reais
        y_pred (array-like): Valores preditos
        y_score (array-like, optional): Scores de probabilidade
        filename (str): Nome base para salvar o arquivo
    
    Returns:
        str: Relatório de classificação formatado
    """
    # Gerar relatório
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Converter para DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Salvar como CSV
    os.makedirs('neat/visualizations/output', exist_ok=True)
    df.to_csv(f'neat/visualizations/output/{filename}.csv')
    
    # Retornar relatório formatado
    return classification_report(y_true, y_pred) 