from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelTrainer:
    """
    Handles model training and tuning.
    """
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.pipeline = None
        self.best_model = None

    def create_pipeline(self, model_type='random_forest'):
        """
        Creates a Scikit-learn pipeline.
        """
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError("Invalid model type")
            
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])
        
        return self.pipeline

    def tune_and_train(self, X_train, y_train, model_type='random_forest'):
        """
        Performs GridSearchCV to find best hyperparameters and trains the model.
        """
        print(f"Training and tuning {model_type}...")
        self.create_pipeline(model_type)
        
        param_grid = {}
        
        if model_type == 'random_forest':
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5]
            }
        elif model_type == 'gradient_boosting':
             param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__learning_rate': [0.01, 0.1]
            }
        elif model_type == 'ridge':
            param_grid = {
                'regressor__alpha': [0.1, 1.0, 10.0]
            }
            
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        print(f"Best params for {model_type}: {grid_search.best_params_}")
        
        return self.best_model
