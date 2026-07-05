# Hotel Booking Cancellation Prediction

A machine learning pipeline that predicts whether a hotel booking will be cancelled, built for the Faculty of Computers and Artificial Intelligence.

**Authors:** Youssef Essam · Mostafa Mokhtar · Mohaned Ezzat · Youssef Morad · Mohamed Mahmoud

---

## Overview

This project implements a complete, seven-stage ML pipeline over a dataset of **6,792 hotel bookings** (33 original features) to predict cancellations. It covers preprocessing, exploratory data analysis, feature engineering, genetic-algorithm-based feature selection, model training, and evaluation.

**Key findings:**
- Cancellation rate: **51.8%** (naturally balanced classes — no SMOTE/undersampling needed)
- Strongest predictor: **country frequency encoding** (correlation ≈ 0.625)
- Lead time has only a weak negative correlation (≈ -0.007) with cancellations, despite common industry assumptions
- Feature engineering expanded the dataset from 33 → 53 features

## Pipeline Architecture

The pipeline is implemented as a `HotelBookingPredictor` class that orchestrates seven stages:

| Stage | Component | Description |
|---|---|---|
| A | `DataPreprocessor` | Missing value imputation, type fixes, duplicate removal, outlier capping, categorical encoding, date feature extraction |
| B | `EDAVisualizer` | Cancellation distribution, booking trends, correlation heatmap |
| C | `DataBalancer` | Checks class balance; applies SMOTE/undersampling only if needed |
| D | `FeatureEngineer` | Derives new features (total guests, stay length, season, ADR per guest, etc.) |
| E | `GeneticAlgorithmFeatureSelector` | Evolves an optimal feature subset (population=30, generations=15) |
| F | `ModelBuilder` | Trains KNN, Decision Tree, and Neural Network with `GridSearchCV` |
| G | `ModelEvaluator` | Compares models on accuracy, precision, recall, F1, ROC-AUC |

## Dataset

| Attribute | Value |
|---|---|
| Total bookings | 6,792 |
| Original features | 33 |
| Features after preprocessing | 53 |
| Cancellation rate | 51.8% (Cancelled) / 48.2% (Not Cancelled) |
| Hotel types | City Hotel, Resort Hotel |
| Time range | 2017–2019 |
| Missing values remaining | 0% |
| Duplicate rows | 0 |

### Preprocessing steps
1. **Missing values:** median imputation (numerical), mode imputation (categorical) — e.g. `company` (6,394 missing), `agent` (1,120 missing)
2. **Encoding:**
   - Binary columns → Label Encoding
   - Multi-category (< 10 unique values) → One-Hot Encoding
   - High cardinality (≥ 10 unique values) → Frequency Encoding
3. **Dates:** extracted year, month, day, day-of-week, quarter, and weekend indicators
4. **Outliers:** capped using the IQR method

### Engineered features
`total_guests`, `total_stay_nights`, `season`, `booking_flexibility`, `lead_time_category`, `adr_per_guest`

## Feature Selection (Genetic Algorithm)

| Parameter | Value |
|---|---|
| Population size | 30 |
| Generations | 15 |
| Crossover rate | 0.8 |
| Mutation rate | 0.1 |
| Tournament size | 3 |
| Elitism count | 2 |
| Fitness function | 3-fold CV F1 score |
| Chromosome length | 53 |

Of 39 candidate engineered features, the GA selected **18 (46.15%)**, converging after ~10 generations. Selected features that stood out in interpretation include lead time, ADR, total nights, previous cancellations, booking changes, and deposit type.

## Models

Three classifiers are tuned via `GridSearchCV` (3-fold CV):

| Model | Hyperparameters searched |
|---|---|
| K-Nearest Neighbors | `n_neighbors`: [3,5,7,9,11], `weights`: [uniform, distance], `metric`: [euclidean, manhattan] |
| Decision Tree | `max_depth`: [3,5,7,10,None], `min_samples_split`: [2,5,10], `min_samples_leaf`: [1,2,4], `criterion`: [gini, entropy] |
| Neural Network (MLP) | `hidden_layer_sizes`: [(50,), (100,), (50,50), (100,50)], `activation`: [relu, tanh], `alpha`: [0.0001,0.001,0.01], `learning_rate`: [constant, adaptive] |

**Data split:** 60% train / 20% validation / 20% test (4,074 / 1,358 / 1,360 samples)

### Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| K-Nearest Neighbors | 0.5325 | 0.3872 | 0.3152 | 0.3475 | 0.5098 |
| Decision Tree | 0.5580 | 0.3907 | 0.2127 | 0.2754 | 0.5094 |
| Neural Network | 0.6050 | 0.0000 | 0.0000 | 0.0000 | 0.5065 |

**Note:** the Neural Network achieved the highest accuracy but a precision/recall of 0 — it collapsed to predicting a single class, so it is not actually a usable classifier despite the accuracy figure. Decision Tree offers the best balance of interpretability and modest predictive signal; overall performance across all three models is close to chance level (ROC-AUC ≈ 0.50–0.51), indicating the current feature set has limited discriminative power for this task.

## Repository Contents

- `AI.ipynb` — full pipeline implementation (preprocessing, EDA, feature engineering, GA feature selection, model building, evaluation)
- `AI_Document.pdf` — project report with methodology, charts, and business insights

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

```python
from pipeline import HotelBookingPredictor

predictor = HotelBookingPredictor()
results = predictor.run_pipeline('hotel_bookings.csv')
```

The pipeline expects a CSV of hotel booking records with an `is_canceled` target column (or will generate synthetic sample data if no file is provided).

## Business Insights

| Factor | Correlation | Implication |
|---|---|---|
| Country patterns | 0.625 | High impact — tailor marketing/deposit policies by country |
| Room assignment | 0.194 | Moderate impact — improve room allocation to match preferences |
| Lead time | -0.007 | Low impact alone — combine with other signals |
| Stay duration | 0.016 | Low impact — consider weekend-specific policies |
| Temporal patterns | 0.005–0.013 | Seasonal effects worth season-specific strategies |

## Limitations & Future Work

- **Dataset size:** 6,792 bookings may not capture all patterns — consider combining with external data
- **Time period:** data covers only 2017–2019 — retrain regularly with newer data
- **Missing external factors:** weather, local events, and economic conditions aren't included
- **Batch-only:** no real-time/streaming prediction capability yet
- **Model performance:** current models are close to chance level; further work is needed on feature quality, class-imbalance-aware modeling (e.g., addressing the Neural Network's collapse to one class), and possibly ensemble methods before deployment
