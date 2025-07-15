import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/go_arounds_augmented.csv"
X_OUTPUT_PATH = "data/X.npy"
Y_OUTPUT_PATH = "data/y.npy"

SELECTED_COLUMNS = [
    "wind_speed_knts",
    "wind_dir_deg",
    "temperature_deg",
    "press_p",
    "visibility_m",
    "rwy_length",
    "glide_slope_angle",
    "n_approaches"
]

def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df.dropna(subset=SELECTED_COLUMNS + ['has_ga'])
    X = df[SELECTED_COLUMNS].astype(float).to_numpy()
    y = df['has_ga'].apply(lambda x: 1 if str(x).lower() == 'true' else 0).to_numpy()
    np.save(X_OUTPUT_PATH, X)
    np.save(Y_OUTPUT_PATH, y)

if __name__ == "__main__":
    main()
