# Industrial Energy Consumption Analysis

This project provides a professional-grade solution for forecasting energy consumption in an industrial setting. By leveraging machine learning models, we analyze and predict `Usage_kWh` based on temporal patterns, reactive power, and power factors. This project is designed to aid operational efficiency, load balancing, and anomaly detection.

## Key Features

* **Data Preprocessing:** Robust cleaning and conversion of timestamp and categorical data, including `dayfirst` formatting.
* **Feature Engineering:** Advanced cyclical encoding of time features (Sine/Cosine transformation for hours) to ensure the model understands chronological continuity.
* **Benchmarking:** Comparative analysis of three industry-standard models: **XGBoost, Random Forest, and LightGBM**.
* **High Performance:** Achieved an R-squared score of up to **0.9990**.
* **Business Insights:** Visualization of energy consumption heatmaps and feature importance analysis to drive operational decisions.

## Project Structure

```
Steel Industry Energy Consumption Analysis/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/               # Original, immutable CSV/data files 
│   └── processed/         # Cleaned/engineered data here
│
├── notebooks/
│   └── explore.ipynb      # For EDA & Jupyter-based experiments
│
├── src/                   # Reusable production code
│   ├── config.py
│   ├── preprocess.py
│   └── model.py
│
└── tests/
    └── __init__.py
```

## Setup Instructions

1. Clone this repository.
2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

```


3. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn

```


4. Run the analysis in `explore.ipynb`.

## Methodology

This project adheres to the standard 10-step Machine Learning Lifecycle:

1. **Define Problem:** Forecasting industrial energy consumption.
2. **Collect/Prepare Data:** Cleaning and feature extraction.
3. **Exploratory Data Analysis (EDA):** Identifying temporal "hot zones" via heatmaps.
4. **Feature Engineering:** Creating cyclical time features and label encoding.
5. **Split Data:** 80/20 train-test split.
6. **Choose Model:** Benchmarking boosting vs. bagging regressors.
7. **Train Model:** XGBoost/RF/LightGBM implementation.
8. **Evaluate:** Using R-squared and residual plots.
9. **Improve:** Validating against overfitting using learning curves.
10. **Deploy (Proposed):** Using the model for anomaly detection.

## License

This project is for educational and professional internship purposes.