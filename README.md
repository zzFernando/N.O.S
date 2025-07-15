# 🧠 N.O.S - Neuroevolution Analysis

**Análise da Trajetória Evolutiva de Redes Neurais em Problemas Aeronáuticos**

Este projeto implementa e analisa algoritmos de Neuroevolution (NEAT) aplicados à predição de go-arounds em aviação, fornecendo visualizações avançadas da trajetória evolutiva das redes neurais.

## 🚀 Funcionalidades

### Visualizações Principais

1. **🔄 Evolução no Espaço Latente**
   - Projeção PCA/UMAP/TSNE dos genomas por geração
   - Análise da distribuição populacional no espaço latente
   - Identificação de padrões evolutivos

2. **🌊 Campo Vetorial de Deslocamento**
   - Visualização da direção e intensidade do movimento evolutivo
   - Análise da trajetória dos melhores indivíduos
   - Identificação de convergência ou divergência

3. **🕸️ Arquitetura Agregada (Grafo)**
   - Grafo de frequência de conexões dos melhores indivíduos
   - Análise da topologia neural emergente
   - Identificação de padrões arquiteturais estáveis

## 📁 Estrutura do Projeto

```
N.O.S/
├── core/                   # Lógica de treinamento
│   ├── train.py           # Script de treinamento NEAT
│   └── neat_config_*.txt  # Configurações NEAT
├── data/                   # Datasets
│   ├── go_arounds_sample.csv
│   └── go_arounds_augmented.csv
├── outputs/                # Resultados
│   ├── genomes/           # Genomas por geração
│   └── logs/              # Logs de treinamento
├── viz/                    # Visualização D3.js
│   ├── assets/            # Dados de projeção
│   ├── scripts/           # Scripts JS modulares
│   ├── styles/            # CSS customizado
│   ├── index.html         # Interface principal
│   ├── generate_data.py   # Gerador de dados
│   └── launch.py          # Script de inicialização
├── utils/                  # Utilitários
│   └── feature_engineering.py
└── requirements.txt
```

## 🛠️ Instalação

1. **Clone o repositório:**
   ```bash
   git clone <repository-url>
   cd N.O.S
   ```

2. **Crie um ambiente virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # ou
   .venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Como Usar

### Inicialização Rápida

Execute o script principal para acessar todas as funcionalidades:

```bash
python run.py
```

### 1. Treinamento

Execute o treinamento NEAT:

```bash
python core/train.py
```

O script irá:
- Carregar os dados de go-arounds
- Executar o algoritmo NEAT
- Salvar genomas por geração em `outputs/genomes/`
- Gerar logs de fitness em `outputs/logs/`

### 2. Visualização

**Opção 1: Inicialização Automática**
```bash
cd viz
python launch.py
```

**Opção 2: Inicialização Manual**
```bash
cd viz
python generate_data.py
python -m http.server 8000
```

Acesse `http://localhost:8000` no navegador.

### 3. Análise

Use os controles interativos para:

- **Filtrar por Geração**: Analise períodos específicos da evolução
- **Ajustar Fitness**: Foque em indivíduos com melhor performance
- **Toggle de Elementos**: Mostre/oculte trajetórias e campo vetorial
- **Estatísticas**: Monitore métricas em tempo real

## 📊 Configurações

### Métodos de Redução de Dimensionalidade

- **PCA**: Análise de Componentes Principais (rápido)
- **UMAP**: Uniform Manifold Approximation and Projection (preserva estrutura local)
- **TSNE**: t-Distributed Stochastic Neighbor Embedding (preserva distâncias)

### Parâmetros NEAT

Edite `core/neat_config_*.txt` para ajustar:
- Tamanho da população
- Taxa de mutação
- Critérios de fitness
- Parâmetros de espécies

## 🔬 Análise Científica

### Insights Esperados

1. **Convergência Evolutiva**: Identificação de regiões estáveis no espaço latente
2. **Diversidade Populacional**: Análise da manutenção da diversidade genética
3. **Arquitetura Emergente**: Padrões topológicos que surgem naturalmente
4. **Trajetória de Otimização**: Direção e velocidade da evolução

### Métricas Disponíveis

- **Deslocamento Médio**: Velocidade média da evolução
- **Densidade do Grafo**: Complexidade da arquitetura neural
- **Diversidade Populacional**: Variabilidade genética
- **Convergência**: Estabilidade da solução

## 📈 Exemplos de Uso

### Análise de Convergência
1. Execute treinamento por 50+ gerações
2. Use o campo vetorial para identificar convergência
3. Analise as trajetórias para padrões estáveis

### Comparação de Configurações
1. Treine com diferentes parâmetros NEAT
2. Compare visualizações do espaço latente
3. Analise diferenças na trajetória evolutiva

### Análise de Robustez
1. Execute múltiplos treinamentos
2. Compare as visualizações geradas
3. Identifique padrões consistentes

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📚 Referências

- Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies.
- Clune, J., Mouret, J. B., & Lipson, H. (2013). The evolutionary origins of modularity.
- Real, E., et al. (2019). Regularized evolution for image classifier architecture search.

---

**Desenvolvido para pesquisa em Neuroevolution aplicada à Aviação**
