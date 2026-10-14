import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model():
    """
    Day-Ahead Electricity Price Forecasting Model

    Author: Personal Portfolio Project
    Description:
        I built this model to forecast day-ahead wholesale electricity spot prices (€/MWh).
        I pull historical hourly generation, demand, and weather indicators from my SQLite
        database, engineer lag and 24-hour rolling features, and train a tuned XGBoost Regressor.
        I use a strict chronological train/test split (2015-2017 train, 2018 test) to evaluate
        real-world out-of-sample performance and analyze the key market drivers.
    """
    print("Connecting to my local database...")
    engine = create_engine('sqlite:///energy_market.db')
    
    # Query data directly from my SQLite database
    print("Loading historical generation, load, and weather records...")
    query = "SELECT * FROM energy_weather_data ORDER BY time ASC;"
    df = pd.read_sql(query, engine)
    
    # Set time as datetime index for time-series operations
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    print("Engineering features (temporal cycles and rolling grid momentum)...")
    # I extract calendar features to capture daily peak hours and weekend demand dips
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # 24-hour rolling windows capture supply trends and intraday momentum
    # Note: I shift price by 1 hour to prevent lookahead leakage
    df['price_roll_mean_24'] = df['price actual'].shift(1).rolling(window=24).mean()
    df['wind_roll_mean_24'] = df['generation wind onshore'].rolling(window=24).mean()
    df['solar_roll_mean_24'] = df['generation solar'].rolling(window=24).mean()
    df['gas_roll_mean_24'] = df['generation fossil gas'].rolling(window=24).mean()
    
    # Drop initial NaN rows created by the rolling windows
    df.dropna(inplace=True)
    
    # Target: actual market clearing spot price
    target_col = 'price actual'
    
    # I exclude 'price day ahead' (the TSO's published price) so my model learns purely from fundamentals
    features = [col for col in df.columns if col not in [target_col, 'price day ahead'] 
                and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[features]
    y = df[target_col]
    
    # Chronological Split (80% train ~2015-2017, 20% test ~2018)
    # I strictly split chronologically because random shuffling leaks future weather and grid states into the past
    print("Splitting data chronologically (first 80% for training, final 20% for testing)...")
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train XGBoost Model
    print("Training XGBoost Regressor for day-ahead electricity prices...")
    model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and Evaluation
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\n--- Model Test Performance ---")
    print(f"Root Mean Squared Error (RMSE): €{rmse:.2f} / MWh")
    print(f"Mean Absolute Error (MAE):     €{mae:.2f} / MWh")
    
    # Feature Importances: What drivers matter most in the forecast?
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 10 Drivers of Spot Price Forecasts ---")
    for feat, imp in importances.head(10).items():
        print(f"{feat}: {imp:.4f}")
    
    # Export predictions for visualization in Power BI and portfolio notebook
    results_df = pd.DataFrame({
        'time': y_test.index,
        'actual_price': y_test.values,
        'predicted_price': preds
    })
    
    results_df.to_csv('predictions.csv', index=False)
    print("\nForecast completed! Predictions saved to 'predictions.csv' for my Power BI dashboard.")

if __name__ == "__main__":
    train_model()
