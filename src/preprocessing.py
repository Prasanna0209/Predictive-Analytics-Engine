import pandas as pd
import numpy as np

class Preprocessor:
    """
    Handles data preprocessing steps:
    - Missing value imputation
    - Outlier handling
    """
    
    def __init__(self):
        pass
        
    def handle_missing_values(self, df):
        """
        Imputes missing values.
        - Numerical: Median
        - Categorical: Mode
        """
        print("Handling missing values...")
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        
        # Simple imputation for demonstration
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                
        return df_clean

    def handle_outliers(self, df, columns, method='iqr'):
        """
        Handles outliers using IQR method.
        Caps outliers to the lower/upper bounds.
        """
        print(f"Handling outliers in {columns} using {method}...")
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap values
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
        return df_clean
