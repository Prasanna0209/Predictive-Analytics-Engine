from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineer:
    """
    Handles feature engineering steps:
    - Scaling numeric features
    - Encoding categorical features
    """
    
    def __init__(self):
        self.preprocessor = None

    def get_preprocessor_pipeline(self, numeric_features, categorical_features):
        """
        Creates a ColumnTransformer for preprocessing.
        
        Args:
            numeric_features (list): List of numeric column names.
            categorical_features (list): List of categorical column names.
            
        Returns:
            ColumnTransformer: The preprocessing pipeline.
        """
        print("Creating Feature Engineering Pipeline...")
        
        numeric_transformer = StandardScaler()
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return self.preprocessor
