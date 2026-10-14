import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def build_pipeline():
    """
    Database Ingestion & Validation Script

    Author: Personal Portfolio Project
    Description:
        To keep data access fast, reliable, and decoupled from flat CSV files,
        I store my cleaned time-series data into a local SQLite relational database (energy_market.db).
        This script writes the master 'energy_weather_data' table and runs a quick sanity
        check query inspecting fossil vs. renewable generation and spot prices.
    """
    print("Loading cleaned dataset from CSV...")
    df = pd.read_csv('cleaned_energy_weather.csv')
    
    # I use SQLite as my lightweight local data warehouse for fast querying
    db_name = 'sqlite:///energy_market.db'
    engine = create_engine(db_name)
    
    print("Writing records into 'energy_weather_data' table...")
    df.to_sql('energy_weather_data', engine, if_exists='replace', index=False)
    print("Database write complete.")
    
    # ---------------------------------------------------------
    # Sanity Check: Inspecting Generation Mix & Spot Prices
    # ---------------------------------------------------------
    print("\n--- Running Sanity Check Query: Fossil vs Renewables & Pricing ---")
    
    # I run a quick query to verify generation columns and price alignment
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
