import pandas as pd
import numpy as np

def prep_data():
    """
    Data Ingestion & Cleaning Pipeline

    Author: Personal Portfolio Project
    Description:
        I built this script to merge 4 years of hourly Iberian electricity grid data
        (generation by fuel type, system load, spot prices) with regional weather features.
        I clean up timestamp misalignments, aggregate multi-city weather data to a national
        level, and resolve missing telemetry gaps using forward-filling for time-series continuity.
    """
    print("Loading raw energy and weather datasets...")
    energy_df = pd.read_csv('energy_dataset.csv')
    weather_df = pd.read_csv('weather_features.csv')

    # Both datasets have timestamps, but need to be aligned to UTC
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True)
    
    if 'dt_iso' in weather_df.columns:
        weather_df['time'] = pd.to_datetime(weather_df['dt_iso'], utc=True)
        weather_df.drop(columns=['dt_iso'], inplace=True)
    elif 'time' in weather_df.columns:
        weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)

    print("Averaging multi-city weather records to national hourly baseline...")
    # The weather dataset records observations across 5 key cities (Madrid, Barcelona, Valencia, Seville, Bilbao).
    # Since power prices clear at the bidding zone (national) level, I calculate the hourly mean
    # across meteorological stations to match the grid-level resolution.
    numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
    weather_agg = weather_df.groupby('time')[numeric_cols].mean().reset_index()

    print("Merging grid generation, demand, and price data with weather indicators...")
    merged_df = pd.merge(energy_df, weather_agg, on='time', how='inner')

    print("Cleaning nulls and handling missing sensor data...")
    # Drop columns that are mostly empty (e.g., offshore wind which has zero Spanish capacity in this period)
    threshold = len(merged_df) * 0.5
    merged_df.dropna(thresh=threshold, axis=1, inplace=True)

    # For time-series consistency, I use forward-fill to handle minor telemetry dropouts,
    # followed by back-fill for any residual initial NaNs, avoiding lookahead leakage.
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)

    output_filename = 'cleaned_energy_weather.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"Data cleaning complete! Processed dataset saved to '{output_filename}'.")

if __name__ == "__main__":
    prep_data()
