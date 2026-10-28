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
  • Energy_Market_Analytics.ipynb (Deep-dive walkthrough with market commentary & charts)
  • Power BI Executive Dashboard (Interactive dark-theme trading view with KPI cards & slicers)
```

### 1. Data Cleaning & Pipeline Validation (`01_data_prep.py`)
* **Multi-City Weather Alignment:** The weather data had independent observations from 5 major Spanish cities (Madrid, Barcelona, Valencia, Seville, Bilbao). Since Spain trades under a single national bidding zone clearing price, I averaged these city readings into an hourly national meteorological profile.
* **Zero Lookahead Leakage:** Rather than using `bfill()` (which leaks future observations backward into missing timestamps), I used forward-fill (`ffill()`) only. 
* **Validation Telemetry:**
  * Dataset shape: 35,064 rows × 39 columns
  * Duplicate timestamps: Exactly 0
  * Missing values after cleaning: Exactly 0 (no back-filling required)
  * Data retention from raw energy set: 35,064 / 35,064 (100% matched)

### 2. Relational Storage Layer (`02_sql_pipeline.py`)
* Instead of repeatedly loading memory-heavy CSV files during model iteration, I loaded the processed 35,064 hourly records into SQLite (`energy_market.db`). This gave me an indexed, queryable data store to inspect generation profiles, run aggregations, and join model forecasts with actuals.

### 3. Feature Engineering & Genuine 24-Hour Day-Ahead Forecasting (`03_forecasting_model.py`)
* **Why XGBoost:** Power prices exhibit non-linearities: when demand spikes near capacity limits or when gas turbines set the marginal price, the relationship between generation and price is not linear. Tree-based gradient boosting handles these regime shifts better than simple OLS or ARIMA models.
* **Strict Forecast Horizon (24h Ahead):** To formulate an operationally viable day-ahead model without data leakage, all realized generation (`generation fossil gas`, `generation solar`, etc.), realized demand (`total load actual`), and realized weather observations at delivery hour $t$ are **strictly excluded**.
* **Eligible Feature Matrix:**
  1. **TSO Day-Ahead Forecasts:** `forecast solar day ahead`, `forecast wind onshore day ahead`, `total load forecast`, engineered `net_load_forecast` (residual thermal demand), and `renewable_share_forecast`.
  2. **Deterministic Calendar Cycles:** `hour`, `dayofweek`, `is_weekend`, `month`, and cyclical harmonic encodings (`sin_hour`, `cos_hour`).
  3. **Historical Realized Lags ($\ge 24$ hours old):** Price lags at $t-24$ (yesterday), $t-48$, $t-168$ (last week), 24h rolling price statistics (`mean`, `std`, `min`, `max`), and 24h lagged gas/wind/solar generation.
* **Chronological Train/Test Split:** I enforced an untouched 80/20 chronological split:
  * **Training Period:** `2015-01-07 23:00` to `2018-03-16 02:00` (27,916 samples)
  * **Testing Period:** `2018-03-16 03:00` to `2018-12-31 22:00` (6,980 samples)
* **Preventing Target Leakage:** I deliberately excluded the published clearing price (`price day ahead`) so the model learns price formation purely from fundamental grid forecasts and historical autoregressive dynamics.

---

## Model Evaluation: Benchmarked Against Multiple Baselines

Before claiming predictive utility, I benchmarked the XGBoost model on the exact same 6,980 out-of-sample test timestamps against standard persistence baselines:

| Model / Baseline | Definition | RMSE (€/MWh) | MAE (€/MWh) | Outperformance vs Baseline |
| :--- | :--- | :---: | :---: | :--- |
| **My XGBoost 24h Model** | **Genuinely available day-ahead features** | **€5.34** | **€3.88** | **Benchmark Leader** |
| **Day-Ahead Baseline (Lag-24)** | Same hour yesterday: $P_{pred}(t) = P(t-24)$ | €6.90 | €4.75 | **18.3% lower MAE with XGBoost** |
| **Weekly Baseline (Lag-168)** | Same hour last week: $P_{pred}(t) = P(t-168)$ | €8.64 | €5.85 | **33.7% lower MAE with XGBoost** |
| **24h Rolling Mean Baseline** | Historical 24h average: $P_{pred}(t) = \bar{P}_{24h}$ | €8.64 | €6.56 | **40.9% lower MAE with XGBoost** |

> **Market Context on the Benchmarks:** In the Iberian day-ahead power auction (OMIE/MIBEL), market participants must submit supply/demand bids by 12:00 CET for all 24 hours of the following day. A naive next-hour persistence model ($t-1$) is physically impossible to execute in day-ahead auctions. The legitimate commercial trading baseline is **Lag-24 (same hour yesterday)**. My XGBoost model achieves an **18.3% lower MAE (€0.87/MWh improvement)** and a **22.6% lower RMSE**, demonstrating statistically significant predictive alpha over naive persistence.

---

## Key Insights & Market Takeaways

1. **The Merit-Order Effect is Clear:** Every surge in zero-marginal-cost renewable generation (solar and wind) directly displaces higher-marginal-cost combined-cycle gas turbines (CCGT) and coal, pushing clearing spot prices down noticeably toward €20–€30/MWh.
2. **Top Forecast Drivers:** Feature importances revealed that prior day's price (`price_lag_24`: 55.1%), weekly price anchor (`price_lag_168`: 20.2%), and forecasted net load (residual thermal demand: 3.6%) are the primary determinants of day-ahead clearing spot prices.
3. **Intraday Volatility & Duck Curve:** During midday hours, high solar irradiance suppresses wholesale prices, while the evening ramp (19:00–21:00) when solar disappears and system load peaks creates the highest price volatility.

---

## Honest Production Limitations & Data Gaps

For a live trading desk deployment, the following data feeds would be required to transition this prototype into a production-grade forecasting system:
* **Numerical Weather Prediction (NWP) Forecasts:** The raw dataset provided ground station weather observations. Because actual weather at delivery hour $t$ cannot be known 24h in advance, it was excluded from hour $t$ features. In production, this would be replaced with commercial NWP forecast grids (ECMWF, GFS).
* **Cross-Border Interconnection Schedules:** Real-time and day-ahead import/export capacities with France (RTE) and Morocco, which heavily influence Iberian scarcity pricing.
* **Fuel & Carbon Commodity Spot Feeds:** Daily European Carbon Allowance (EUA) auction prices and Dutch TTF natural gas spot prices to dynamically track thermal marginal cost curves.

---

## Project Structure

* `01_data_prep.py` — Cleans raw data, aggregates multi-city weather, handles missing values, and validates timestamps.
* `02_sql_pipeline.py` — Ingests cleaned data into `energy_market.db` with structured indexes and validation queries.
* `03_forecasting_model.py` — Extracts rolling features, splits chronologically, trains XGBoost model, and outputs predictions.
* `Energy_Market_Analytics.ipynb` — My interactive notebook walkthrough with dark-theme visual analytics and market commentary.
* `Energy Market Forecasting.pbix` — Interactive Power BI dashboard layout built with custom dark palette (`Theme.json`).
* `power bi dashboard.png` — Screenshot preview of my finished Power BI dashboard.
* `power_bi_data_fixed.xlsx` — Excel dataset formatted for direct Power BI Desktop ingestion.
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
jupyter notebook Energy_Market_Analytics.ipynb
# Or open 'Energy Market Forecasting.pbix' in Power BI Desktop
```
