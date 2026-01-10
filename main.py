import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_ingestion import load_data
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
import warnings

warnings.filterwarnings('ignore')

def main():
    print("starting Predictive Analytics Pipeline...")
    
    # 1. Load Data
    data_path = 'data/synthetic_data.csv'
    df = load_data(data_path)
    
    # 2. Preprocessing
    preprocessor = Preprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Define features and target
    target_column = 'price'
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Handle Outliers (on X only or before split? usually before split for easier cleaning, but rigorous would be fit on train)
    # For simplicity in this demo, we'll handle outliers on the dataset or skipping it to keep it simple as the pipeline handles scaling.
    # But let's use the preprocessor's outlier handler on specific columns
    outlier_cols = ['square_feet', 'num_bedrooms', 'num_bathrooms']
    df_clean = preprocessor.handle_outliers(df_clean, outlier_cols)
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # 3. Train-Test Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Feature Engineering
    # Define columns
    numeric_features = ['square_feet', 'num_bedrooms', 'num_bathrooms', 'year_built', 'location_score']
    categorical_features = ['neighborhood_type', 'has_pool']
    
    feat_eng = FeatureEngineer()
    pipeline_preprocessor = feat_eng.get_preprocessor_pipeline(numeric_features, categorical_features)
    
    # 5. Model Training & Tuning
    trainer = ModelTrainer(pipeline_preprocessor)
    
    models_to_train = ['linear_regression', 'random_forest', 'gradient_boosting']
    best_results = {}
    best_model_overall = None
    best_r2 = -float('inf')
    
    for model_name in models_to_train:
        print(f"\n--- Processing {model_name} ---")
        model = trainer.tune_and_train(X_train, y_train, model_type=model_name)
        
        # 6. Evaluation
        metrics = ModelEvaluator.evaluate(model, X_test, y_test)
        best_results[model_name] = metrics
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_overall = model
            
    print(f"\n=== Best Model: {best_model_overall.steps[-1][1].__class__.__name__} with R2: {best_r2:.4f} ===")
    
    # 7. Sample Prediction
    print("\nGenerating Sample Predictions with Best Model...")
    sample_data = X_test.iloc[:5].copy()
    predictions = best_model_overall.predict(sample_data)
    
    results_df = sample_data.copy()
    results_df['Actual Price'] = y_test.iloc[:5]
    results_df['Predicted Price'] = predictions
    
    print(results_df[['square_feet', 'neighborhood_type', 'Actual Price', 'Predicted Price']])

if __name__ == "__main__":
    main()
