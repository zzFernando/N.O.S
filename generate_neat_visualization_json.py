import os
import glob
import json
import numpy as np
from sklearn.decomposition import PCA

# 1. Encontrar todos os arquivos de genomas por geração
files = sorted(glob.glob('outputs/genomes/gen_*.json'))
all_genomes = []
genomes_by_id = {}
gen_per_file = []

# 2. Ler todos os genomas e registrar histórico
for gen_idx, f in enumerate(files):
    with open(f) as fh:
        gen_list = json.load(fh)
        for g in gen_list:
            gid = str(g.get('id', g.get('genome_id', f"gen_{gen_idx}")))
            vec = g.get('vector', [])
            fitness = g.get('fitness', None)
            label = str(g.get('species', g.get('label', 'unknown')))
            if gid not in genomes_by_id:
                genomes_by_id[gid] = {'id': gid, 'label': label, 'fitness': fitness, 'generations': [], 'vectors': []}
            genomes_by_id[gid]['generations'].append(gen_idx)
            genomes_by_id[gid]['vectors'].append(vec)
    gen_per_file.append(len(gen_list))

# 3. Concatenar todos os vetores para projeção
all_vectors = []
max_len = 0
for g in genomes_by_id.values():
    for v in g['vectors']:
        max_len = max(max_len, len(v))
for g in genomes_by_id.values():
    for v in g['vectors']:
        padded = list(v) + [0.0] * (max_len - len(v))
        all_vectors.append(padded)
all_vectors = np.array(all_vectors)

# 4. Projeção PCA 2D
pca = PCA(n_components=2)
proj_2d = pca.fit_transform(all_vectors)

# 5. Montar JSON unificado
out_genomes = []
idx = 0
for g in genomes_by_id.values():
    traj = []
    for _ in g['vectors']:
        traj.append([float(proj_2d[idx,0]), float(proj_2d[idx,1])])
        idx += 1
    # Ajuste: fitness nunca None, label padrão
    fitness = g['fitness'] if g['fitness'] is not None else 0.0
    label = g['label'] if g['label'] not in [None, '', 'unknown'] else f"genome_{g['id']}"
    out_genomes.append({
        'id': g['id'],
        'generation': g['generations'][-1],
        'label': label,
        'fitness': fitness,
        'projection': traj[-1],
        'trajectory': traj,
        'vector': [traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1]]
    })

# 6. Campo vetorial sintético (malha 10x10)
all_proj = np.array([g['projection'] for g in out_genomes])
x_min, x_max = float(np.min(all_proj[:,0])), float(np.max(all_proj[:,0]))
y_min, y_max = float(np.min(all_proj[:,1])), float(np.max(all_proj[:,1]))
vector_field = []
for xi in np.linspace(x_min, x_max, 10):
    for yi in np.linspace(y_min, y_max, 10):
        vector_field.append({
            'x': float(xi),
            'y': float(yi),
            'dx': float(np.random.uniform(-0.2, 0.2)),
            'dy': float(np.random.uniform(-0.2, 0.2))
        })

# 7. Salvar JSON
out_path = 'frontend/assets/projections.json'
with open(out_path, 'w') as f:
    json.dump({'genomes': out_genomes, 'vector_field': vector_field}, f, indent=2)

print(f'Arquivo salvo em {out_path} com {len(out_genomes)} genomas.') 