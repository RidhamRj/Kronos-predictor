# Kronos Predictor (Indian Exchanges & GOLDM Live Tracking)

This repository serves as an experimental testbed and live tracking dashboard for evaluating the [Kronos Foundation Model](https://github.com/shiyu-coder/Kronos) on Indian financial markets, specifically NIFTY 50 and MCX Gold (GOLDM).

## Project Overview

The primary goal of this project was to rigorously backtest and test the out-of-sample predictive capabilities of the Kronos model (a foundation model pre-trained on global financial candlesticks) on the Indian Stock Market and Commodities.

### Key Features
- **Live Price Tracking:** Real-time data pipeline fetching live candlestick data for NIFTY 50 and MCX GOLDM via MoneyControl.
- **Web Dashboard:** A responsive, dynamic web UI showing real-time price feeds, synthesized AI forecasts, and volatility gauges.
- **Walk-Forward Validation Harness:** A rigorous quantitative backtesting engine built to stress-test the model's predictive edge across multiple isolated historical windows (2018-2026).
- **Vercel Serverless Ready:** Configured to deploy the dashboard and basic model variants via Vercel's Python Serverless functions.

## Experimental Findings

Through rigorous multi-window out-of-sample testing, we documented two critical negative results regarding the base Kronos model's edge on NIFTY 50:
1. **Directional Prediction Failure:** The model exhibited a severe **mean-reversion bias** (~56-58% of the time, it predicted tomorrow's direction would be the exact opposite of today's return), causing it to severely underperform simple Buy & Hold strategies in trending markets.
2. **Volatility Forecasting Failure:** When pivoted to predict absolute return magnitudes (volatility), the model failed to beat simple 20-day Exponentially Weighted Moving Average (EWMA) baselines across all tested market regimes.

This repository preserves the live tracking dashboard and the rigorous backtesting harness as a template for testing future foundation models or basic EWMA/statistical strategies on Indian market data.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the backend server:
   ```bash
   python server.py
   ```
3. Open `http://127.0.0.1:8000` in your browser.
