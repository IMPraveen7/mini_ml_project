import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
from io import StringIO

# User-defined functions
def load_wine_data(url):
    """Download and load wine data from a given URL."""
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data, sep=';')
    return df

def calculate_stats(df, columns):
    """Calculate basic statistics for specified columns."""
    stats = {}
    for col in columns:
        stats[col] = {
            'mean': np.mean(df[col]),
            'median': np.median(df[col]),
            'std': np.std(df[col])
        }
    return stats

def engineer_features(df):
    """Create new features from existing ones."""
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    df['sugar_to_acid_ratio'] = df['residual sugar'] / df['total_acidity']
    return df

# Higher-order function for applying transformations
def apply_transformations(df, transformations):
    for column, transform_func in transformations.items():
        df[column] = df[column].apply(transform_func)
    return df

# Main workflow
def main():
    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = load_wine_data(url)

    # Data cleaning (using built-in function)
    df = df.dropna()

    # Feature engineering
    df = engineer_features(df)

    # Apply transformations using lambda functions
    transformations = {
        'alcohol': lambda x: x / 100,  # Convert to decimal
        'pH': lambda x: 10**(-x)  # Convert pH to hydrogen ion concentration
    }
    df = apply_transformations(df, transformations)

    # Calculate statistics for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    stats = calculate_stats(df, numeric_columns)
    print("Basic statistics:")
    print(pd.DataFrame(stats))

    # Prepare features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nMean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Print feature importances (using lambda function)
    feature_importances = sorted(
        zip(model.feature_importances_, X.columns),
        key=lambda x: x[0],
        reverse=True
    )
    print("\nTop 5 Feature Importances:")
    for importance, feature in feature_importances[:5]:
        print(f"{feature}: {importance:.4f}")

# Call the main workflow
if __name__ == "__main__":
    main()