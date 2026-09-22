import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model():
    """
    Trains a time-series forecasting model to predict day-ahead electricity prices.
    Identifies key drivers of market volatility using XGBoost.
    """
    print("Connecting to database...")
    engine = create_engine('sqlite:///energy_market.db')
    
    # Query data directly from the SQL database
    print("Querying historical energy and weather data...")
    query = "SELECT * FROM energy_weather_data ORDER BY time ASC;"
    df = pd.read_sql(query, engine)
    
    # Set time as datetime index for time-series operations
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    print("Performing feature engineering...")
    # Extract temporal features to capture seasonal and daily demand cycles
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Create rolling averages to capture momentum and supply trends in the energy mix
    df['price_roll_mean_24'] = df['price actual'].shift(1).rolling(window=24).mean()
    df['wind_roll_mean_24'] = df['generation wind onshore'].rolling(window=24).mean()
    df['solar_roll_mean_24'] = df['generation solar'].rolling(window=24).mean()
    df['gas_roll_mean_24'] = df['generation fossil gas'].rolling(window=24).mean()
    
    # Drop rows with NaNs resulting from rolling windows and shifts
    df.dropna(inplace=True)
    
    # Define target variable: the actual clearing price in the market
    target_col = 'price actual'
    
    # Select numerical features and avoid data leakage
    # We exclude 'price day ahead' because we are building our own day-ahead forecaster
    features = [col for col in df.columns if col not in [target_col, 'price day ahead'] 
                and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[features]
    y = df[target_col]
    
    # Chronological Split (80% train, 20% test)
    # NEVER use a random split for time-series to prevent looking into the future.
    print("Splitting data chronologically into train and test sets...")
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train XGBoost Model
    print("Training XGBoost Regressor for day-ahead electricity price...")
    model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and Evaluation
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\n--- Model Evaluation ---")
    print(f"RMSE: €{rmse:.2f} / MWh")
    print(f"MAE:  €{mae:.2f} / MWh")
    
    # Feature Importances: What drives market volatility?
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 10 Drivers of Market Volatility (Feature Importances) ---")
    for feat, imp in importances.head(10).items():
        print(f"{feat}: {imp:.4f}")
    
    # Output predictions for visualization in Power BI
    results_df = pd.DataFrame({
        'time': y_test.index,
        'actual_price': y_test.values,
        'predicted_price': preds
    })
    
    # Reformat time for Power BI compatibility if necessary
    results_df.to_csv('predictions.csv', index=False)
    print("\nPipeline Complete: Predictions saved to 'predictions.csv' for Power BI dashboarding.")

if __name__ == "__main__":
    train_model()
