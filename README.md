# 🌱 AgroIntel Crop Advisor

A Streamlit application that recommends the optimal crop for a plot of land based on soil nutrients (N, P, K, pH) and environmental conditions (temperature, humidity, rainfall). It trains and compares classic machine learning models (Random Forest, KNN, SVM) on the [Crop Recommendation dataset](data/Crop_recommendation.csv) and exposes the results through an interactive UI for data exploration, model evaluation, and live predictions.

---

## Features

The full application (`simple_app.py`) is organized into the following pages:

| Page | What it does |
|---|---|
| **Home** | Dataset overview, summary statistics, and crop-distribution chart. |
| **Data Analysis** | Feature distributions, correlation heatmap, and soil-nutrient breakdown by crop. |
| **Model Performance** | Accuracy/precision/recall/F1 for Random Forest, KNN, and SVM, plus feature importance. |
| **Prediction Tool** | Enter soil/environmental values and get a recommended crop with suitability scores. |
| **Feature Engineering** | Polynomial features, ratio features, PCA, and automated feature selection. |
| **Clustering Analysis** | K-Means/DBSCAN clustering, optimal-k analysis, and cluster profiling. |
| **Advanced Analysis** | Model comparison, confusion-matrix analysis, ROC/PR curves, and learning/validation curves. |

A second, simplified entry point (`app.py`) implements the Home, Data Analysis, Model Performance, and Prediction Tool pages as a smaller, modular reference (its Advanced Analysis, Advanced Modeling, and Seasonal Analysis pages are present in the UI but currently stubbed out).

---

## Tech Stack

- **Language**: Python 3.11
- **UI**: [Streamlit](https://streamlit.io/)
- **ML**: scikit-learn (Random Forest, KNN, SVM, K-Means, DBSCAN, PCA)
- **Data**: pandas, numpy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model persistence**: joblib
- **Packaging**: [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`)
- **Containerization**: Docker / Docker Compose

---

## Project Structure

```
.
├── app.py                   # Modular entry point (imports from src/)
├── simple_app.py            # Full-featured, self-contained entry point
├── src/                     # Shared analysis/modeling modules used by app.py
│   ├── data_preprocessing.py
│   ├── exploratory_analysis.py
│   ├── modeling.py
│   ├── utils.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── advanced_modeling.py
│   └── seasonal_analysis.py
├── data/
│   └── Crop_recommendation.csv
├── models/                  # Trained model artifacts (.joblib), created on first run
├── assets/                  # App icon and screenshots
├── .streamlit/config.toml   # Streamlit server config (headless, port 5000)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml / uv.lock # Python dependencies
└── .replit / replit.nix     # Replit deployment config
```

---

## Running with Docker (recommended)

This spins up the full application — dependencies, data, and pretrained models — with a single command.

**Requirements**: [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

```bash
git clone <repo-url>
cd AgroIntel_CropAdvisor
docker compose up --build
```

Then open **http://localhost:5000** in your browser.

- The container runs `simple_app.py` by default.
- The `models/` directory is mounted into the container, so any model retraining persists back to your host instead of being lost when the container stops.
- Stop the app with `docker compose down`.

---

## Running locally (without Docker)

**Requirements**: Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or `pip`).

```bash
git clone <repo-url>
cd AgroIntel_CropAdvisor

# Using uv (matches the committed lockfile)
uv run streamlit run simple_app.py

# ...or with pip
pip install -e .   # or: pip install joblib matplotlib numpy pandas plotly scikit-learn seaborn streamlit
streamlit run simple_app.py
```

Then open **http://localhost:5000** (configured in `.streamlit/config.toml`).

To run the modular version instead:

```bash
streamlit run app.py
```

---

## Dataset

[`data/Crop_recommendation.csv`](data/Crop_recommendation.csv) contains 2,200 samples across 22 crop labels, with the following features:

| Feature | Description |
|---|---|
| `N` | Nitrogen content in soil (kg/ha) |
| `P` | Phosphorus content in soil (kg/ha) |
| `K` | Potassium content in soil (kg/ha) |
| `temperature` | Temperature (°C) |
| `humidity` | Relative humidity (%) |
| `ph` | Soil pH value |
| `rainfall` | Rainfall (mm) |
| `label` | Target crop |

---

## Upcoming Enhancements

- Pest risk alerts based on environmental and historical data.
- Irrigation planning recommendations tailored to crop type and soil moisture.
- Market-driven crop selection based on demand trends.

---

## License

No license has been specified for this project.
