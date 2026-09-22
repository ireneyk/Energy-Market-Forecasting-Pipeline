import pandas as pd
import numpy as np

def prep_data():
    """
    Loads, cleans, and merges energy and weather datasets.
    Prepares data for downstream market analysis and pricing models.
    """
    print("Loading datasets...")
    # Assuming datasets are in the current working directory
    energy_df = pd.read_csv('energy_dataset.csv')
    weather_df = pd.read_csv('weather_features.csv')

    # Parse time columns into proper datetime objects
    # The energy dataset uses 'time' with UTC offset
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True)
    
    # The weather dataset uses 'dt_iso' or 'time'
    if 'dt_iso' in weather_df.columns:
        weather_df['time'] = pd.to_datetime(weather_df['dt_iso'], utc=True)
        weather_df.drop(columns=['dt_iso'], inplace=True)
    elif 'time' in weather_df.columns:
        weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)

    print("Aggregating regional weather data to national level...")
    # Weather data contains multiple cities. For a macro-level energy mix analysis,
    # we aggregate (mean) the meteorological features to a single national hourly timeline.
    numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
    weather_agg = weather_df.groupby('time')[numeric_cols].mean().reset_index()

    print("Merging energy generation/pricing data with weather data...")
    # Merge datasets on the aligned datetime column
    merged_df = pd.merge(energy_df, weather_agg, on='time', how='inner')

    print("Handling missing values...")
    # Drop columns that have more than 50% missing values
    threshold = len(merged_df) * 0.5
    merged_df.dropna(thresh=threshold, axis=1, inplace=True)

    # Time-series continuity is critical for energy forecasting. 
    # Forward-fill gaps (e.g., sensor downtime), then back-fill any remaining leading NaNs.
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)

    # Save the cleaned and merged dataset
    output_filename = 'cleaned_energy_weather.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"Data cleaning and merging complete. Saved to '{output_filename}'.")

if __name__ == "__main__":
    prep_data()
