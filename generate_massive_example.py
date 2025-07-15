import json
import numpy as np

N_GENOMES = 20  # 10 de cada classe
N_GENERATIONS = 10
LABELS = ["A", "B"]  # Apenas 2 classes

# Definir centros fixos para cada label
LABEL_CENTERS = {
    "A": (-1.5, -1.5),
    "B": (1.5, 1.5)
}

INIT_STD = 0.2

genomes = []
for i in range(N_GENOMES):
    label = LABELS[i % len(LABELS)]
    fitness = float(np.clip(np.random.normal(0.7 if label == "A" else 0.5, 0.1), 0, 1))
    cx, cy = LABEL_CENTERS[label]
    x0, y0 = np.random.normal(cx, INIT_STD), np.random.normal(cy, INIT_STD)
    trajectory = []
    for g in range(N_GENERATIONS):
        dx, dy = np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)
        x0 += dx
        y0 += dy
        trajectory.append([x0, y0])
    projection = trajectory[-1]
    vector = [trajectory[-1][0] - trajectory[0][0], trajectory[-1][1] - trajectory[0][1]]
    genomes.append({
        "id": f"gen_{i}",
        "generation": N_GENERATIONS-1,
        "label": label,
        "fitness": fitness,
        "projection": projection,
        "trajectory": trajectory,
        "vector": vector
    })

# Campo vetorial sintético (malha 10x10)
vector_field = []
for xi in np.linspace(-2, 2, 10):
    for yi in np.linspace(-2, 2, 10):
        vector_field.append({
            "x": float(xi),
            "y": float(yi),
            "dx": float(np.random.uniform(-0.2, 0.2)),
            "dy": float(np.random.uniform(-0.2, 0.2))
        })

with open("frontend/assets/projections.json", "w") as f:
    json.dump({"genomes": genomes, "vector_field": vector_field}, f, indent=2)

print("Exemplo sintético salvo em frontend/assets/projections.json (2 classes)") 