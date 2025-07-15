# 🧠 N.O.S Visualization

**Visualização Interativa da Trajetória Evolutiva de Redes Neurais**

Interface D3.js para análise da evolução de genomas NEAT em problemas aeronáuticos.

## 🚀 Inicialização Rápida

### Opção 1: Inicialização Automática
```bash
python launch.py
```

### Opção 2: Inicialização Manual
```bash
# 1. Gerar dados
python generate_data.py

# 2. Iniciar servidor
python -m http.server 8000

# 3. Abrir navegador
# http://localhost:8000
```

## 📁 Estrutura

```
viz/
├── assets/              # Dados de projeção
├── scripts/             # Scripts JS modulares
│   ├── main.js         # Lógica principal
│   ├── draw_scatter.js # Visualização de pontos
│   ├── draw_trajectories.js # Trajetórias
│   └── draw_vector_field.js # Campo vetorial
├── styles/              # CSS customizado
├── index.html           # Interface principal
├── generate_data.py     # Gerador de dados
├── launch.py           # Script de inicialização
└── test_visualization.html # Página de teste
```

## 🎛️ Controles

- **Geração**: Filtro por período evolutivo
- **Fitness**: Threshold mínimo de performance
- **Visualizações**: Toggle de elementos
- **Configurações**: Cores, tamanhos, estilos

## 📊 Funcionalidades

- Projeção PCA/UMAP/TSNE
- Trajetórias evolutivas
- Campo vetorial de deslocamento
- Estatísticas em tempo real
- Interface responsiva

---

**Parte do projeto N.O.S - Neuroevolution Analysis**
