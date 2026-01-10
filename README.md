# Predictive Analytics Engine

A modular machine learning pipeline built with Python, Pandas, and Scikit-learn to predict property prices based on various features.

## 🧱 Architecture

The project follows a modular architecture:
- **Data Ingestion**: Loads data from CSV.
- **Preprocessing**: Handles missing values and outliers.
- **Feature Engineering**: Scales numeric features and encodes categorical ones.
- **Model Training**: Trains multiple models (Linear Regression, Random Forest, Gradient Boosting) using Pipelines and GridSearchCV.
- **Evaluation**: Compares models using RMSE, MAE, and R².

## 📂 Structure

```
predictive_analytics_engine/
├── data/                   # Dataset directory
├── src/                    # Source code
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── main.py                 # Main execution script
├── generate_data.py        # Synthetic data generator
└── requirements.txt        # Dependencies
```

## 🚀 How to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data**
    (Pre-generated, but you can regenerate)
    ```bash
    python generate_data.py
    ```

3.  **Run Pipeline**
    Trains models and verifies performance.
    ```bash
    python main.py
    ```

## 📊 Sample Output

The pipeline will output evaluation metrics for each model and select the best one:
```
--- Processing random_forest ---
Best params: {'regressor__n_estimators': 100, ...}
RMSE: 20500.23
R²:   0.85
```
It also prints sample predictions vs actual values.
