import pandas as pd
import os

def load_data(filepath):
    """
    Loads data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e
