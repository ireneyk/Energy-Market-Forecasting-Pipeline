import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model():
    """
    Genuine 24-Hour-Ahead Electricity Price Forecasting Model

    Author: Personal Portfolio Project
    Description:
        I built this model to perform a genuine day-ahead (24 hours ahead) wholesale
        electricity spot price forecast (€/MWh).
        To prevent any data leakage, this model STRICTLY uses:
          1. Genuinely available day-ahead TSO forecasts for hour t (solar, wind, load, net load).
          2. Cyclical and calendar features for hour t (hour, day of week, weekend, month).
          3. Historical realized price, generation, load, and weather features lagged by AT LEAST 24 hours (t-24, t-48, t-168).
        All realized generation, weather observations, and demand at hour t are strictly excluded.
        The model is evaluated chronologically (first 80% train, final 20% unseen test)
        and benchmarked against multiple persistence baselines.
    """
    print("Connecting to my local database...")
    engine = create_engine('sqlite:///energy_market.db')
    
    # Query data directly from my SQLite database
    print("Loading historical generation, load, and weather records...")
    query = "SELECT * FROM energy_weather_data ORDER BY time ASC;"
    raw_df = pd.read_sql(query, engine)
    
    # Parse timestamps to UTC and sort chronologically
    raw_df['time'] = pd.to_datetime(raw_df['time'], utc=True)
    raw_df = raw_df.sort_values('time').set_index('time')
    
    print("Engineering genuine 24-hour-ahead features...")
    # Target variable: wholesale electricity market clearing price at delivery hour t
    target_col = 'price actual'
    
    # Feature DataFrame constructed with ZERO lookahead leakage
    f_df = pd.DataFrame(index=raw_df.index)
    f_df['target'] = raw_df[target_col]
    
    # 1. Day-Ahead TSO Forecasts for delivery hour t (available prior to day-ahead auction close)
    f_df['forecast_solar_da'] = raw_df['forecast solar day ahead']
    f_df['forecast_wind_da'] = raw_df['forecast wind onshore day ahead']
    f_df['total_load_forecast'] = raw_df['total load forecast']
    
    # Engineered fundamental market indicator: Net load (residual thermal demand)
    f_df['net_load_forecast'] = f_df['total_load_forecast'] - (f_df['forecast_solar_da'] + f_df['forecast_wind_da'])
    f_df['renewable_share_forecast'] = (f_df['forecast_solar_da'] + f_df['forecast_wind_da']) / f_df['total_load_forecast'].replace(0, np.nan)
    
    # 2. Deterministic Calendar & Diurnal Features for hour t
    f_df['hour'] = raw_df.index.hour
    f_df['dayofweek'] = raw_df.index.dayofweek
    f_df['is_weekend'] = (raw_df.index.dayofweek >= 5).astype(int)
    f_df['month'] = raw_df.index.month
    f_df['sin_hour'] = np.sin(2 * np.pi * f_df['hour'] / 24.0)
    f_df['cos_hour'] = np.cos(2 * np.pi * f_df['hour'] / 24.0)
    
    # 3. Historical Price Lags (Strictly >= 24 hours prior to prediction hour)
    f_df['price_lag_24'] = raw_df[target_col].shift(24)        # Same hour yesterday (Day-Ahead Persistence)
    f_df['price_lag_48'] = raw_df[target_col].shift(24 * 2)    # 2 days prior
    f_df['price_lag_168'] = raw_df[target_col].shift(24 * 7)   # Same hour last week
    f_df['price_roll_mean_24'] = raw_df[target_col].shift(24).rolling(window=24).mean()
    f_df['price_roll_std_24'] = raw_df[target_col].shift(24).rolling(window=24).std()
    f_df['price_roll_max_24'] = raw_df[target_col].shift(24).rolling(window=24).max()
    f_df['price_roll_min_24'] = raw_df[target_col].shift(24).rolling(window=24).min()
    
    # 4. Historical Realized Generation & Load (Shifted by >= 24 hours)
    f_df['gas_lag_24'] = raw_df['generation fossil gas'].shift(24)
    f_df['gas_roll_mean_24'] = raw_df['generation fossil gas'].shift(24).rolling(window=24).mean()
    f_df['wind_lag_24'] = raw_df['generation wind onshore'].shift(24)
    f_df['wind_roll_mean_24'] = raw_df['generation wind onshore'].shift(24).rolling(window=24).mean()
    f_df['solar_lag_24'] = raw_df['generation solar'].shift(24)
    f_df['solar_roll_mean_24'] = raw_df['generation solar'].shift(24).rolling(window=24).mean()
    f_df['load_actual_lag_24'] = raw_df['total load actual'].shift(24)
    
    # 5. Historical Weather Observations (Shifted by >= 24 hours)
    f_df['temp_lag_24'] = raw_df['temp'].shift(24)
    f_df['wind_speed_lag_24'] = raw_df['wind_speed'].shift(24)
    f_df['temp_roll_mean_24'] = raw_df['temp'].shift(24).rolling(window=24).mean()
    
    # Drop initial rows with NaNs resulting from the 168-hour lag window
    f_df.dropna(inplace=True)
    
    feature_cols = [col for col in f_df.columns if col != 'target']
    X = f_df[feature_cols]
    y = f_df['target']
    
    # Chronological 80/20 train/test split
    print("\nSplitting data chronologically into train and test sets...")
    split_idx = int(len(f_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training period: {X_train.index.min()} to {X_train.index.max()} ({len(X_train):,} samples)")
    print(f"Testing period:  {X_test.index.min()} to {X_test.index.max()} ({len(X_test):,} samples)")
    
    # Train XGBoost Model
    print("\nTraining XGBoost Regressor on genuine 24-hour-ahead feature set...")
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    # ---------------------------------------------------------
    # Baseline Comparisons: Benchmark Against Persistence
    # ---------------------------------------------------------
    # 1. Day-Ahead Persistence Baseline (Lag-24: Same hour yesterday)
    b_lag24 = X_test['price_lag_24']
    b_lag24_mae = mean_absolute_error(y_test, b_lag24)
    b_lag24_rmse = np.sqrt(mean_squared_error(y_test, b_lag24))

    # 2. Weekly Persistence Baseline (Lag-168: Same hour last week)
    b_lag168 = X_test['price_lag_168']
    b_lag168_mae = mean_absolute_error(y_test, b_lag168)
    b_lag168_rmse = np.sqrt(mean_squared_error(y_test, b_lag168))

    # 3. 24-Hour Rolling Historical Mean Baseline
    b_roll = X_test['price_roll_mean_24']
    b_roll_mae = mean_absolute_error(y_test, b_roll)
    b_roll_rmse = np.sqrt(mean_squared_error(y_test, b_roll))

    print(f"\n--- Model Test Performance vs Baselines (Identical Test Period) ---")
    print(f"XGBoost 24h Model:              RMSE: €{rmse:.2f} / MWh  |  MAE: €{mae:.2f} / MWh")
    print(f"Day-Ahead Baseline (Lag-24):    RMSE: €{b_lag24_rmse:.2f} / MWh  |  MAE: €{b_lag24_mae:.2f} / MWh")
    print(f"Weekly Baseline (Lag-168):      RMSE: €{b_lag168_rmse:.2f} / MWh  |  MAE: €{b_lag168_mae:.2f} / MWh")
    print(f"24h Rolling Mean Baseline:      RMSE: €{b_roll_rmse:.2f} / MWh  |  MAE: €{b_roll_mae:.2f} / MWh")
    
    imp_lag24 = ((b_lag24_mae - mae) / b_lag24_mae) * 100
    print(f"Outperformance vs Day-Ahead Baseline (Lag-24): {imp_lag24:.1f}% lower MAE")
    
    # Feature Importances: Top drivers of 24h forecast
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Top 10 Drivers of 24h Day-Ahead Forecast (Feature Importances) ---")
    for feat, imp in importances.head(10).items():
        print(f"{feat}: {imp:.4f} ({imp*100:.1f}%)")
    
    # Export predictions for visualization in Power BI and portfolio notebook
    results_df = pd.DataFrame({
        'time': y_test.index.astype(str),
        'actual_price': y_test.values,
        'predicted_price': preds
    })
    
    results_df.to_csv('predictions.csv', index=False)
    results_df.to_sql('predictions', engine, if_exists='replace', index=False)

    # Recreate/Refresh SQLite energy_dashboard table
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS energy_dashboard;")
        conn.exec_driver_sql("""
            CREATE TABLE energy_dashboard AS
            SELECT d.*, p.predicted_price
            FROM energy_weather_data d
            LEFT JOIN predictions p ON d.time = p.time;
        """)

    print("\nPredictions exported to 'predictions.csv' and SQLite 'energy_dashboard' updated.")

    # Also update power_bi_data_fixed.xlsx so Power BI Desktop stays synchronized
    print("Updating 'power_bi_data_fixed.xlsx' for Power BI...")
    dashboard_df = pd.read_sql("SELECT * FROM energy_dashboard", engine)
    dashboard_df.to_excel('power_bi_data_fixed.xlsx', index=False)
    print(f"Power BI dataset updated with {len(dashboard_df):,} rows and {dashboard_df['predicted_price'].notnull().sum():,} predictions.")

if __name__ == "__main__":
    train_model()
