from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class ModelEvaluator:
    """
    Handles model evaluation.
    """
    
    @staticmethod
    def evaluate(model, X_test, y_test):
        """
        Calculates and prints metrics.
        """
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Evaluation Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
