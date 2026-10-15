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
    merged_df = merged_df.sort_values('time').reset_index(drop=True)

    print("Cleaning nulls and handling missing sensor data...")
    # Drop columns that are mostly empty (e.g., offshore wind which has zero Spanish capacity in this period)
    threshold = len(merged_df) * 0.5
    merged_df.dropna(thresh=threshold, axis=1, inplace=True)

    # For time-series consistency, I use forward-fill only (ffill).
    # I strictly avoid back-fill (bfill) to prevent information from future observations leaking backward.
    merged_df.ffill(inplace=True)

    # Inspect if any missing values remain after forward-filling
    remaining_nulls = merged_df.isnull().sum()
    if remaining_nulls.sum() > 0:
        print("\nRemaining missing values after forward-filling:")
        print(remaining_nulls[remaining_nulls > 0].head(10))
        # Drop any remaining unfillable leading rows to ensure zero lookahead contamination
        merged_df.dropna(inplace=True)
    else:
        print("Verified: 0 missing values remain after forward-fill. No future back-filling needed.")

    # ---------------------------------------------------------
    # Pipeline Data Validation Checks
    # ---------------------------------------------------------
    is_monotonic = merged_df['time'].is_monotonic_increasing
    duplicate_count = merged_df['time'].duplicated().sum()
    null_count = merged_df.isnull().sum().sum()
    retained_pct = (len(merged_df) / len(energy_df)) * 100

    print("\n--- Pipeline Data Validation ---")
    print(f"Dataset shape: {merged_df.shape}")
    print(f"Chronologically sorted: {is_monotonic}")
    print(f"Duplicate timestamps: {duplicate_count}")
    print(f"Missing values after cleaning: {null_count}")
    print(f"Date range: {merged_df['time'].min()} to {merged_df['time'].max()}")
    print(f"Rows retained from energy dataset: {len(merged_df)} / {len(energy_df)} ({retained_pct:.1f}% matched)")

    output_filename = 'cleaned_energy_weather.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"\nData cleaning complete! Processed dataset saved to '{output_filename}'.")

if __name__ == "__main__":
    prep_data()
