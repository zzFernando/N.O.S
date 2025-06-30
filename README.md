## 🧠 N.O.S – Neuroevolution OpenSky

**N.O.S** (Neuroevolution OpenSky) is a research-oriented tool designed to explore the use of evolutionary algorithms—specifically NEAT (NeuroEvolution of Augmenting Topologies)—applied to open aviation data from the [OpenSky Network](https://opensky-network.org).

The initial focus is on predicting **go-arounds**, which are critical events in terminal airspace that affect flight safety, traffic efficiency, and controller workload.

---

### 🎯 Objectives

* Apply neuroevolution to detect patterns in real aircraft trajectories.
* Compare performance against traditional machine learning baselines (e.g., Random Forest).
* Promote interpretability through the visualization of evolved neural networks.
* Deliver a reproducible tool for air traffic behavior analysis.

---

### 📁 Project Structure

```bash
.
├── baseline/             # Traditional ML models (e.g., Random Forest)
├── data/                 # Preprocessed datasets (CSV format)
├── neat/                 # NEAT implementation and visualizations
├── notebooks/            # Exploratory data analysis and experiments
├── utils/                # Feature engineering and data utilities
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── LICENSE
```

---

### 🚀 Getting Started

#### 1. Clone the repository

```bash
git clone https://github.com/zzFernando/N.O.S.git
cd N.O.S
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You also need to install **Graphviz** on your system:

* **macOS:** `brew install graphviz`
* **Linux (Ubuntu):** `sudo apt install graphviz`

#### 3. Run the baseline model

```bash
python baseline/rf_classifier.py
```

#### 4. Run the NEAT model

```bash
python neat/train.py
```

Visual outputs (fitness evolution, network topologies, species diversity) are saved in `neat/` as `.svg` files.

---

### 📊 About the Dataset

* Derived from OpenSky Network's ADS-B historical data.
* Includes labeled go-around events and features such as:

  * Altitude
  * Vertical rate
  * Distance to runway
  * Groundspeed
  * Heading
  * Timestamp, airport metadata, and more

---

### 📚 References

* Stanley, K. O., & Miikkulainen, R. (2002). *Evolving neural networks through augmenting topologies (NEAT)*.
* OpenSky Network publications: [https://opensky-network.org/about/publications](https://opensky-network.org/about/publications)
* Related works:

  * [Predicting Airplane Go-Arounds using Machine Learning](https://opensky-network.org/files/publications/ga-ml.pdf)
  * [Trajectory Clustering in Terminal Airspace](https://opensky-network.org/files/publications/trajectory-clustering.pdf)

---

### 👨‍💻 Author

Fernando Kavinsky
MSc Student in Artificial Intelligence (UFRGS, Brazil)
[LinkedIn](https://www.linkedin.com/in/zzFernando) • [GitHub](https://github.com/zzFernando)
