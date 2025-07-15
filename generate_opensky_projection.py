import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA

# 1. Ler CSV
csv_path = 'data/go_arounds_sample.csv'
df = pd.read_csv(csv_path)

# 2. Selecionar features numéricas relevantes (exclui ids, datas, label)
ignore_cols = ['time','icao24','callsign','airport','runway','registration','typecode','icaoaircrafttype','wtc','airport_country','airport_region','operator_country','operator_region','weather_intensity','weather_precipitation','weather_desc','weather_obscuration','weather_other','has_ga']
num_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

# 3. Projeção PCA 2D
X = df[num_cols].fillna(0).values
pca = PCA(n_components=2)
proj = pca.fit_transform(X)

# 4. Montar lista de genomas
label_map = {True: 'go_around', False: 'normal', 'True': 'go_around', 'False': 'normal', 1: 'go_around', 0: 'normal', '1': 'go_around', '0': 'normal'}
genomes = []
labels_found = set()
for i, row in df.iterrows():
    raw_label = row['has_ga']
    label = label_map.get(raw_label, str(raw_label))
    labels_found.add(label)
    genomes.append({
        'id': str(row['icao24']) if not pd.isna(row['icao24']) else f'flight_{i}',
        'generation': 0,
        'label': label,
        'fitness': None,
        'projection': [float(proj[i,0]), float(proj[i,1])],
        'trajectory': [[float(proj[i,0]), float(proj[i,1])]],
        'vector': [0,0]
    })

print(f'Classes encontradas em has_ga: {labels_found}')

# 5. Campo vetorial sintético (malha 10x10)
vector_field = []
x_min, x_max = float(np.min(proj[:,0])), float(np.max(proj[:,0]))
y_min, y_max = float(np.min(proj[:,1])), float(np.max(proj[:,1]))
for xi in np.linspace(x_min, x_max, 10):
    for yi in np.linspace(y_min, y_max, 10):
        vector_field.append({
            'x': float(xi),
            'y': float(yi),
            'dx': float(np.random.uniform(-0.2, 0.2)),
            'dy': float(np.random.uniform(-0.2, 0.2))
        })

# 6. Salvar JSON
out_path = 'frontend/assets/projections.json'
with open(out_path, 'w') as f:
    json.dump({'genomes': genomes, 'vector_field': vector_field}, f, indent=2)

print(f'Arquivo salvo em {out_path} com {len(genomes)} genomas.') 