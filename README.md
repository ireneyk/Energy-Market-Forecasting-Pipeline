# Iberian Power Market: Day-Ahead Electricity Price Forecasting & Grid Transition
**Author:** Personal Portfolio Project | Energy Data Analytics & Quantitative Market Analysis

---

## Why I Built This Project

I built this end-to-end data pipeline and predictive modeling project to analyze how the aggressive rollout of solar and wind generation has reshaped wholesale electricity price dynamics across Spain and the Iberian electricity market (MIBEL/OMIE).

When trading power or evaluating merchant renewable investments, two core questions always come up:
1. **The Merit-Order Cannibalization Effect:** When renewable output surges during the midday solar peak or during windy winter fronts, how severely does it depress wholesale spot prices?
2. **Day-Ahead Price Forecasting:** Can I build an accurate day-ahead price forecaster using purely fundamental generation by technology type, grid demand, and meteorological variables, without leaking future information?

To answer these, I pulled 4 full years (2015–2018) of hourly generation by fuel type, system demand, actual clearing spot prices, and multi-city weather data, built a clean SQL database pipeline, engineered predictive features, and trained an XGBoost regression model.

---

## Architecture & My Engineering Decisions

```
[Raw Datasets] (Hourly Generation, System Load, Spot Prices + 5-City Weather)
       │
       ▼
[01_data_prep.py]
  • Aligned timestamps to UTC
  • Aggregated 5 weather stations (Madrid, Barcelona, Valencia, Seville, Bilbao) to national baseline
  • Handled sensor dropouts with forward-fill (strictly avoiding lookahead leakage)
       │
       ▼
[02_sql_pipeline.py]
  • Stored clean time-series records in local SQLite database (energy_market.db)
  • Wrote sanity check SQL queries verifying fuel-type mixes and price spreads
       │
       ▼
[03_forecasting_model.py]
  • Feature engineering: diurnal temporal flags + 24-hour rolling averages (wind, solar, gas, price)
  • Strict 80/20 chronological train/test split (2015–2017 train, 2018 out-of-sample test)
  • XGBoost Regressor for non-linear spot price dynamics
       │
       ▼
[Visualization & Analytics]
  • Rystad_Portfolio.ipynb (Deep-dive walkthrough with market commentary & charts)
  • Power BI Executive Dashboard (Interactive dark-theme trading view with KPI cards & slicers)
```

### 1. Data Cleaning & Spatial Aggregation (`01_data_prep.py`)
* **Multi-City Weather Alignment:** The weather data had independent observations from 5 major Spanish cities (Madrid, Barcelona, Valencia, Seville, Bilbao). Since Spain trades under a single national bidding zone clearing price, I averaged these city readings into an hourly national meteorological profile.
* **Preserving Time-Series Continuity:** Rather than dropping rows with temporary sensor dropouts (which would break rolling calculations and lag structures), I forward-filled missing telemetry points and back-filled residual leading NaNs.

### 2. Relational Storage Layer (`02_sql_pipeline.py`)
* Instead of repeatedly loading memory-heavy CSV files during model iteration, I loaded the processed 35,040+ hourly records into SQLite (`energy_market.db`). This gave me an indexed, queryable data store to inspect generation profiles, run aggregations, and join model forecasts with actuals.

### 3. Feature Engineering & Model Development (`03_forecasting_model.py`)
* **Why XGBoost:** Power prices exhibit non-linearities: when demand spikes near capacity limits or when gas turbines set the marginal price, the relationship between generation and price is not linear. Tree-based gradient boosting handles these regime shifts better than simple OLS or ARIMA models.
* **Temporal & Momentum Features:** I engineered calendar features (`hour`, `dayofweek`, `month`) to model the typical Iberian diurnal double-peak (morning industrial ramp and evening domestic peak), alongside 24-hour rolling averages for wind, solar, combined-cycle gas (CCGT), and prior prices.
* **Preventing Data Leakage:** I enforced a strict chronological 80/20 train/test split (2015–2017 for training, all of 2018 as an untouched test set). I deliberately excluded the TSO's published `price day ahead` so my model relied purely on generation and weather fundamentals.

---

## Key Insights & Market Takeaways

1. **The Merit-Order Effect is Clear:** Every surge in zero-marginal-cost renewable generation (solar and wind) directly displaces higher-marginal-cost combined-cycle gas turbines (CCGT) and coal, pushing clearing spot prices down noticeably toward €20–€30/MWh.
2. **Intraday Volatility & Duck Curve Emergence:** During midday hours, high solar irradiance suppresses wholesale prices, while the evening ramp (19:00–21:00) when solar disappears and system load peaks creates the highest price spikes and volatility.
3. **Model Performance:** On the out-of-sample 2018 test set, the XGBoost model achieved solid tracking against actual market clearing prices, capturing both seasonal baselines and daily peak swings with strong feature reliance on prior 24h price momentum, gas dispatch levels, and onshore wind generation.

---

## Project Structure

* `01_data_prep.py` — Cleans raw Kaggle data, aggregates multi-city weather, handles missing values.
* `02_sql_pipeline.py` — Ingests cleaned data into `energy_market.db` with validation queries.
* `03_forecasting_model.py` — Extracts rolling features, splits chronologically, trains XGBoost model, outputs `predictions.csv`.
* `Rystad_Portfolio.ipynb` — My interactive notebook walkthrough with dark-theme visual analytics and market commentary.
* `Energy Market Forecasting.pbix` — Interactive Power BI dashboard layout built with custom dark palette (`Theme.json`).
* `power bi dashboard.png` — Screenshot preview of my finished Power BI dashboard.
* `energy_market.db` — SQLite database housing the processed tables and prediction views.

---

## How to Run My Code

```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Run data preparation
python 01_data_prep.py

# 3. Ingest into SQLite database
python 02_sql_pipeline.py

# 4. Train the forecasting model & generate predictions
python 03_forecasting_model.py

# 5. Open the analysis notebook or Power BI report
jupyter notebook Rystad_Portfolio.ipynb
# Or open 'Energy Market Forecasting.pbix' in Power BI Desktop
```
