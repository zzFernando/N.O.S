import glob
import json

for path in glob.glob('outputs/genomes/gen_*.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    # Se não for lista, transforma em lista
    if isinstance(data, dict):
        with open(path, 'w') as f:
            json.dump([data], f, indent=2)
        print(f"Corrigido: {path}")
    else:
        print(f"OK: {path}") 