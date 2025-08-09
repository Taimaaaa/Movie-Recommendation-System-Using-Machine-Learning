# Movie Recommendation System (KNN, Decision Tree, Naïve Bayes) – Flask Web App

**Authors:** Taima Nasser, Ahmad Hamdan, Laith Nimer  
**Course:** Artificial Intelligence – Birzeit University

## Overview
A full-stack movie recommendation system with a real web interface. Users select liked movies and receive top recommendations ranked by similarity. The backend compares three ML approaches—**K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Naïve Bayes**—on the TMDB dataset with robust preprocessing and class-imbalance handling. The web app is built with **Flask** and a responsive **HTML/CSS/JavaScript** frontend.

**Highlights**
- End-to-end pipeline: data cleaning, feature engineering, training, evaluation, and UI.
- Cosine-similarity ranking on model-filtered candidates.
- Class imbalance addressed with **SMOTE**.
- Fuzzy search for movie titles (RapidFuzz).
- Returns **top 6** recommendations; users can select up to **10** liked movies.

---

## Features
- Add up to 10 liked movies (exact or fuzzy matches).
- Run recommendations via KNN, Decision Tree, or Naïve Bayes.
- View accuracy, precision, recall, F1, and confusion matrix.
- Poster rendering using TMDB poster paths.

---

## Architecture
- **Frontend:** HTML, CSS, JavaScript (Jinja templates).
- **Backend:** Flask (`app.py`), session state for liked movies.
- **ML stack:** scikit-learn (KNN / Decision Tree / Naïve Bayes), imbalanced-learn (SMOTE).
- **Similarity:** cosine similarity for ranking recommended items.
- **Data:** TMDB movies CSV (Kaggle). Place as `movies.csv` in repo root.

---

## Dataset
- **Source:** TMDB Movies dataset (Kaggle).  
- **Expected columns (examples):**  
  `id, title, genres, popularity, runtime, original_language, production_companies, production_countries, adult, poster_path`
- **Note:** The dataset is **not** included. Add `movies.csv` at the project root.

---

## Installation

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/movie-recommender.git
cd movie-recommender

# (Optional) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install flask pandas scikit-learn imbalanced-learn rapidfuzz
```
## Quick Start

```bash
# Ensure movies.csv is in the project root
python app.py
# Then open http://127.0.0.1:5000 in your browser
```
## Methodology

### Data Import & Preprocessing
- Load `movies.csv` with pandas.
- Genres: `MultiLabelBinarizer` on comma-separated `genres`.
- Categoricals: `LabelEncoder` for `original_language`, `production_companies`, `production_countries`, `adult`.
- Numericals: `MinMaxScaler` for `popularity`, `runtime`.
- Build feature matrix **X**; target **y** is `liked` (1 if user-selected, else 0).

### Models
- **KNN:** `KNeighborsClassifier` (Euclidean). Predicts `liked`, then ranks candidates by **cosine similarity** to user-liked set; returns **top 6**.
- **Decision Tree:** `DecisionTreeClassifier` with restricted depth; exports feature importances.
- **Naïve Bayes:** `GaussianNB` baseline.

### Class Imbalance
- **SMOTE** oversampling during training (config differs per route).

### Evaluation
- `train_test_split` with configurable `test_split`.
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1**, and confusion matrix.
- Tested across multiple dataset sizes and splits; scenarios include “similar liked movies” vs. “diverse liked movies”.

## Results Summary

- **KNN:** Strong and stable across dataset sizes; best when liked movies share themes.
- **Decision Tree:** Competitive on mixed datasets; interpretable via feature importances.
- **Naïve Bayes:** Fast baseline; drops when features are interdependent.

See the report for full metric tables and discussion.
## Technologies Used

- Language: Python
- Libraries: scikit-learn, pandas, imbalanced-learn (SMOTE), rapidfuzz
- Web: Flask, HTML, CSS, JavaScript
- Similarity: cosine similarity
- Metrics: Accuracy, Precision, Recall, F1
