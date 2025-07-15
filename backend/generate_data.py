#!/usr/bin/env python3
"""
Schema do JSON unificado para visualização N.O.S:

{
  "genomes": [
    {
      "id": "gen_0_1",
      "generation": 0,
      "label": "go_around",
      "fitness": 0.45,
      "projection": [0.12, -0.44],
      "trajectory": [[0.12, -0.44], [0.20, -0.35], [0.32, -0.30]],
      "vector": [0.08, 0.09]
    },
    ...
  ],
  "vector_field": [
    { "x": -1.0, "y": -1.0, "dx": 0.1, "dy": 0.05 },
    ...
  ]
}

Exemplo realista:
{
  "genomes": [
    {
      "id": "gen_0_1",
      "generation": 0,
      "label": "go_around",
      "fitness": 0.45,
      "projection": [0.12, -0.44],
      "trajectory": [[0.12, -0.44], [0.20, -0.35], [0.32, -0.30]],
      "vector": [0.08, 0.09]
    },
    {
      "id": "gen_0_2",
      "generation": 0,
      "label": "landing",
      "fitness": 0.32,
      "projection": [-0.22, 0.14],
      "trajectory": [[-0.22, 0.14], [-0.18, 0.20], [-0.10, 0.25]],
      "vector": [0.12, 0.11]
    }
  ],
  "vector_field": [
    { "x": -1.0, "y": -1.0, "dx": 0.1, "dy": 0.05 },
    { "x": 0.0, "y": 0.0, "dx": -0.02, "dy": 0.03 }
  ]
}

Generate data for D3.js neuroevolution visualization from genome files.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Tuple, Optional
import glob
import os

class DataGenerator:
    def __init__(self, genomes_dir: str = "../outputs/genomes"):
        self.genomes_dir = Path(genomes_dir)

    def extract_genome_vector(self, genome_data: Dict) -> np.ndarray:
        vector = genome_data.get('vector', [])
        if isinstance(vector, list):
            return np.array(vector)
        else:
            return np.array([])

    def load_all_genomes(self) -> List[Tuple[int, str, Dict]]:
        genomes = []
        if not self.genomes_dir.exists():
            print(f"Warning: Genomes directory {self.genomes_dir} not found")
            return genomes
        pattern = str(self.genomes_dir / "gen_*.json")
        files = glob.glob(pattern)
        for file_path in sorted(files):
            try:
                filename = Path(file_path).stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    generation = int(parts[1])
                    genome_id = parts[2]
                    with open(file_path, 'r') as f:
                        genome_data = json.load(f)
                        # Se for lista, itere; se for dict, coloque em lista
                        if isinstance(genome_data, list):
                            for g in genome_data:
                                genomes.append((generation, genome_id, g))
                        elif isinstance(genome_data, dict):
                            genomes.append((generation, genome_id, genome_data))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        return sorted(genomes, key=lambda x: (x[0], x[1]))

    def build_genomes_json(self, genomes: List[Tuple[int, str, Dict]], method: str = "PCA") -> Dict:
        # 1. Coletar todos os vetores e metadados
        vectors = []
        meta = []
        for generation, genome_id, genome_data in genomes:
            vector = self.extract_genome_vector(genome_data)
            if len(vector) > 0:
                vectors.append(vector)
                meta.append({
                    'generation': generation,
                    'id': genome_data.get('id', f'gen_{generation}_{genome_id}'),
                    'label': genome_data.get('label', ''),
                    'fitness': genome_data.get('fitness', 0.0),
                    'vector': vector.tolist()
                })
        if len(vectors) < 2:
            print("Warning: Not enough vectors for projection")
            return {'genomes': [], 'vector_field': []}
        # 2. Projeção 2D
        max_len = max(len(v) for v in vectors)
        padded_vectors = [np.pad(v, (0, max_len - len(v)), 'constant') if len(v) < max_len else v for v in vectors]
        X = np.array(padded_vectors)
        if method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "UMAP":
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)-1))
        elif method == "TSNE":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        else:
            raise ValueError(f"Unknown projection method: {method}")
        X_proj = reducer.fit_transform(X)
        # 3. Mapear projeções para cada genoma
        for i, (x, y) in enumerate(X_proj):
            meta[i]['projection'] = [float(x), float(y)]
        # 4. Trajetórias por id base
        id_to_traj = {}
        for m in meta:
            base_id = m['id']
            if base_id not in id_to_traj:
                id_to_traj[base_id] = []
            id_to_traj[base_id].append((m['generation'], m['projection']))
        for m in meta:
            traj = sorted(id_to_traj[m['id']], key=lambda x: x[0])
            m['trajectory'] = [p for _, p in traj]
        # 5. Vetor de deslocamento (entre gerações consecutivas)
        for m in meta:
            traj = m['trajectory']
            if len(traj) > 1:
                m['vector'] = [traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1]]
            else:
                m['vector'] = [0.0, 0.0]
        # 6. Campo vetorial em malha 2D
        grid_size = 20
        x_vals = [p[0] for m in meta for p in m['trajectory']]
        y_vals = [p[1] for m in meta for p in m['trajectory']]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        vector_field = []
        for xi in x_grid:
            for yi in y_grid:
                dxs, dys, count = 0.0, 0.0, 0
                for m in meta:
                    traj = m['trajectory']
                    for i in range(1, len(traj)):
                        x0, y0 = traj[i-1]
                        x1, y1 = traj[i]
                        if abs(x0 - xi) < (x_max-x_min)/grid_size and abs(y0 - yi) < (y_max-y_min)/grid_size:
                            dxs += x1 - x0
                            dys += y1 - y0
                            count += 1
                if count > 0:
                    vector_field.append({
                        'x': float(xi), 'y': float(yi),
                        'dx': dxs/count, 'dy': dys/count
                    })
        # 7. Montar saída final
        return {'genomes': meta, 'vector_field': vector_field}

    def save_data(self, data: Dict, output_file: str = "assets/projections.json"):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {output_file}")

    def generate_data(self, method: str = "PCA") -> Dict:
        genomes = self.load_all_genomes()
        if not genomes:
            print("No genomes found, aborting.")
            return {'genomes': [], 'vector_field': []}
        data = self.build_genomes_json(genomes, method)
        self.save_data(data, "assets/projections.json")
        return data

def generate_mock_unified_json(output_file="assets/projections.json"):
    import numpy as np
    import json
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    genomes = []
    n_genomes = 10
    n_generations = 8
    labels = ["go_around", "landing"]
    for i in range(n_genomes):
        label = labels[i % len(labels)]
        trajectory = []
        x0, y0 = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        for g in range(n_generations):
            dx, dy = np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)
            x0 += dx
            y0 += dy
            trajectory.append([x0, y0])
        projection = trajectory[-1]
        vector = [trajectory[-1][0] - trajectory[0][0], trajectory[-1][1] - trajectory[0][1]]
        genomes.append({
            "id": f"gen_{i}",
            "generation": n_generations-1,
            "label": label,
            "fitness": float(np.random.uniform(0.2, 1.0)),
            "projection": projection,
            "trajectory": trajectory,
            "vector": vector
        })
    # Campo vetorial sintético
    vector_field = []
    grid_size = 10
    for xi in np.linspace(-1, 1, grid_size):
        for yi in np.linspace(-1, 1, grid_size):
            vector_field.append({
                "x": float(xi), "y": float(yi),
                "dx": float(np.random.uniform(-0.1, 0.1)),
                "dy": float(np.random.uniform(-0.1, 0.1))
            })
    with open(output_file, "w") as f:
        json.dump({"genomes": genomes, "vector_field": vector_field}, f, indent=2)
    print(f"Mock unified JSON saved to {output_file}")

def main():
    generator = DataGenerator()
    data = generator.generate_data(method="PCA")
    generate_mock_unified_json()

if __name__ == "__main__":
    main() 