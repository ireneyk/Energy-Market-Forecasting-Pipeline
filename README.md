# Iberian Power Market: AI Day-Ahead Forecasting & Grid Transition

## Overview
This project is an end-to-end Data Engineering and Data Science pipeline built for a portfolio at Rystad Energy. It takes raw, hourly energy generation, pricing, and weather data from Spain (Kaggle), cleans it, stores it in a relational database, and applies a Machine Learning model to forecast electricity prices. Finally, the insights are visualized in both a Jupyter Notebook and Power BI.

## Pipeline Architecture
1. **Data Preparation (`01_data_prep.py`)**: Merges complex Kaggle datasets, handles missing values via forward-filling for time-series continuity, and aggregates regional weather data into a national hourly timeline.
2. **Database Ingestion (`02_sql_pipeline.py`)**: Uses SQLAlchemy to build an optimized SQLite database (`energy_market.db`). It simulates a production data warehouse environment by creating a clean, queryable master table.
3. **Machine Learning (`03_forecasting_model.py`)**: Queries the SQLite database directly using Pandas and trains an XGBoost Regressor. It performs feature engineering (rolling averages, temporal features) to predict the day-ahead electricity spot price, evaluating drivers of market volatility.
4. **Visualization (`Rystad_Portfolio.ipynb` & Power BI)**: Connects directly to the SQL database to visualize three key market insights:
   * **The Grid's Transition**: A stacked area chart showing the structural shift from fossil fuel baseload (gas/coal) to intermittent renewables (solar/wind).
   * **Forecasting Accuracy**: A line chart comparing the actual spot price against the XGBoost AI's predicted price.
   * **The Cannibalization Effect**: A scatter plot proving that high renewable generation drives the commodity spot price down (the merit-order effect).

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data prep: `python 01_data_prep.py`
3. Build the database: `python 02_sql_pipeline.py`
4. Train the model: `python 03_forecasting_model.py`
5. Open `Rystad_Portfolio.ipynb` or connect Power BI to `energy_market.db` to view the dashboards.
