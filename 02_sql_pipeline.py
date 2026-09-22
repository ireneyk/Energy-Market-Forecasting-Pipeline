import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def build_pipeline():
    """
    Ingests cleaned energy/weather data into a SQLite database.
    Establishes the foundation for scalable commodity querying.
    """
    print("Loading cleaned data...")
    df = pd.read_csv('cleaned_energy_weather.csv')
    
    # Connect to local SQLite database (this simulates a production PostgreSQL/Snowflake environment)
    db_name = 'sqlite:///energy_market.db'
    engine = create_engine(db_name)
    
    print("Creating optimized table schema and ingesting data...")
    # Ingest data into the database
    # In a fully scaled scenario, chunksize and explicit dtypes would be defined.
    df.to_sql('energy_weather_data', engine, if_exists='replace', index=False)
    print("Ingestion into SQLite database complete.")
    
    # ---------------------------------------------------------
    # Proving Database Integrity: Querying the Energy Transition
    # ---------------------------------------------------------
    print("\n--- Executing Sample Query: Fossil vs Renewables & Pricing ---")
    
    # We query specific generation types to analyze the grid's transition 
    # from fossil fuels to renewables alongside actual spot prices.
    query = """
    SELECT 
        time,
        "generation fossil gas",
        "generation wind onshore",
        "generation solar",
        "price actual"
    FROM energy_weather_data
    ORDER BY time ASC
    LIMIT 10;
    """
    
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
        print(result.to_string(index=False))

if __name__ == "__main__":
    build_pipeline()
