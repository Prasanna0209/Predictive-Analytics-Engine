import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_records=12000, output_file='data/synthetic_data.csv'):
    """
    Generates a synthetic dataset for predictive analytics logic.
    Target variable: 'price' (Regression task)
    Features:
        - square_feet (Numeric): Size of the property
        - num_bedrooms (Numeric): Number of bedrooms
        - num_bathrooms (Numeric): Number of bathrooms
        - year_built (Numeric): Year of construction
        - location_score (Numeric): Quality of location
        - neighborhood_type (Categorical): 'Rural', 'Suburban', 'Urban'
        - has_pool (Categorical): 'Yes', 'No'
    
    Introduces:
        - Missing values
        - Outliers
    """
    
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {num_records} records...")
    
    # Generate base features
    square_feet = np.random.normal(2000, 500, num_records).astype(int)
    num_bedrooms = np.random.randint(1, 6, num_records)
    num_bathrooms = np.random.randint(1, 4, num_records)
    year_built = np.random.randint(1950, 2024, num_records)
    location_score = np.random.uniform(1, 10, num_records)
    
    neighborhood_types = ['Rural', 'Suburban', 'Urban']
    neighborhood = np.random.choice(neighborhood_types, num_records, p=[0.2, 0.5, 0.3])
    
    has_pool = np.random.choice(['Yes', 'No'], num_records, p=[0.3, 0.7])
    
    # Generate Target (Price) based on a formula + noise
    # Base price calculation
    price = (
        square_feet * 150 +
        num_bedrooms * 10000 + 
        num_bathrooms * 5000 + 
        (year_built - 1950) * 1000 + 
        location_score * 5000 + 
        np.where(neighborhood == 'Urban', 50000, 0) + 
        np.where(neighborhood == 'Suburban', 20000, 0) + 
        np.where(has_pool == 'Yes', 15000, 0)
    )
    
    # Add random noise
    noise = np.random.normal(0, 20000, num_records)
    price = price + noise
    
    df = pd.DataFrame({
        'square_feet': square_feet,
        'num_bedrooms': num_bedrooms,
        'num_bathrooms': num_bathrooms,
        'year_built': year_built,
        'location_score': location_score,
        'neighborhood_type': neighborhood,
        'has_pool': has_pool,
        'price': price
    })
    
    # --- Introduce Missing Values (~5% in some columns) ---
    for col in ['num_bedrooms', 'location_score', 'has_pool']:
        mask = np.random.rand(num_records) < 0.05
        df.loc[mask, col] = np.nan
        
    # --- Introduce Outliers ---
    # Create some mansions with huge square footage but low price (error?) or high price
    outlier_indices = np.random.choice(df.index, size=int(num_records * 0.01), replace=False)
    df.loc[outlier_indices, 'square_feet'] = df.loc[outlier_indices, 'square_feet'] * 3
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    print("Sample data:")
    print(df.head())
    print("\nData info:")
    print(df.info())

if __name__ == "__main__":
    generate_synthetic_data()
