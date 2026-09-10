"""
run_kronos_gold.py
------------------
Main Interactive CLI Runner for MCX Gold Mini (Goldm) Futures.
Calibrated to the exact Dhan / MCX Market Screen:
- Contract: 05 Oct, 2026 Future
- Live Price: ₹ 153,321.00
- Day Range: ₹ 152,210.00 - ₹ 153,848.00 (+568.00 / +0.37%)
- Open Interest: LONG BUILDUP (OI ↑, PRICE ↑)
"""

import sys
import argparse
import datetime
import json

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from gold_api import GoldDataProvider
from market_analysis import MarketAnalyzer
from kronos_light import KronosLight


def print_banner():
    banner = r"""
================================================================================
       K R O N O S   A I   •   M C X   G O L D M   ( G O L D   M I N I )
     [Live Indian Futures Market • 05 Oct 2026 Contract • AAAI 2026 Model]
================================================================================
"""
    print(banner)


def format_inr(val, decimals=2):
    return f"{val:,.{decimals}f}"


def run_pipeline(
    interval: str = "1h",
    lookback: int = 72,
    horizon: int = 12,
    samples: int = 10,
    temperature: float = 0.6,
    save_output: bool = True,
):
    print_banner()

    # Step 1: Ingestion
    print(f"[*] Step 1: Ingesting Live MCX Goldm Market Feed ({interval} candles)...")
    provider = GoldDataProvider(symbol="GC=F", usd_inr_rate=94.84)
    data = provider.fetch_klines(interval=interval, limit=lookback)

    curr_time = data["timestamps_ist"][-1].strftime("%d %b %Y, %I:%M %p IST")
    curr_price = data["close"][-1]

    print(f"    Instrument    : Goldm (Exchange: MCX)")
    print(f"    Contract      : {data['contract']}")
    print(f"    Live Price    : ₹ {format_inr(curr_price)}  [▲ +{data['change_val']:.2f} (+{data['change_pct']}%) ]")
    print(f"    Day Range     : ₹ {format_inr(data['day_low'])} - ₹ {format_inr(data['day_high'])}")
    print(f"    OI Analysis   : {data['oi_analysis']}")
    print(f"    Bid / Ask     : {data['bid_ask']}")
    print(f"    As On         : {curr_time}\n")

    # Step 2: Technical Analysis
    print("[*] Step 2: Running Quantitative Matrix & Microstructure...")
    analyzer = MarketAnalyzer(data)
    analysis = analyzer.compute_all()

    print("-" * 80)
    print(f"  {'MCX METRIC':<30} | {'VALUE (₹)':<22} | {'INTERPRETATION':<20}")
    print("-" * 80)
    print(f"  {'Goldm Spot/Future':<30} | ₹ {format_inr(curr_price):<20} | Benchmark quote")
    print(f"  {'EMA (20) Trendline':<30} | ₹ {format_inr(analysis['ema_20']):<20} | Fast trend support")
    print(f"  {'EMA (50) Baseline':<30} | ₹ {format_inr(analysis['ema_50']):<20} | Medium trend baseline")
    print(f"  {'RSI (14) Momentum':<30} | {analysis['rsi_14']:<22.1f} | {'Overbought' if analysis['rsi_14'] > 70 else ('Oversold' if analysis['rsi_14'] < 30 else 'Neutral Momentum')}")
    print(f"  {'ATR (14) Volatility':<30} | ₹ {format_inr(analysis['atr_14']):<20} | Average bar range")
    print(f"  {'Realized Volatility':<30} | {analysis['realized_volatility_pct']:<21.2f}% | 24-period dispersion")
    print(f"  {'Bollinger Upper (+2σ)':<30} | ₹ {format_inr(analysis['bollinger_upper']):<20} | Resistance band")
    print(f"  {'Bollinger Lower (-2σ)':<30} | ₹ {format_inr(analysis['bollinger_lower']):<20} | Support band")
    print(f"  {'MACD Histogram':<30} | {analysis['macd_hist']:<+22.2f} | {'Bullish divergence' if analysis['macd_hist'] > 0 else 'Bearish divergence'}")
    print(f"  {'Market Regime':<30} | {data['oi_analysis']}")
    print("-" * 80 + "\n")

    # Step 3: Kronos Model Inference
    print(f"[*] Step 3: Running Kronos AI Model ({horizon}-step Goldm projection)...")
    print(f"    - Discrete BSQ Tokenizer active")
    print(f"    - Monte Carlo Test-Time Scaling: {samples} rollouts ensembled")
    print(f"    - Sampling Temperature: {temperature}")

    try:
        from real_kronos import run_real_kronos_forecast
        pred = run_real_kronos_forecast(
            data=data,
            horizon=horizon,
            temperature=temperature,
            sample_count=min(samples, 3)
        )
        print(f"    - Foundation Model: {pred.get('model')} on {pred.get('device')}")
    except Exception as e:
        print(f"    - [Fallback] KronosLight CPU: {e}")
        model = KronosLight(d_model=64, k_bits=16)
        pred = model.predict(
            ohlcva=data["ohlcva"],
            timestamps=data["timestamps_ist"],
            horizon=horizon,
            temperature=temperature,
            num_monte_carlo_samples=samples,
        )

    # Step 4: Display Forecast Table
    print("\n" + "=" * 80)
    print(f"  KRONOS MCX GOLDM MULTI-STEP FUTURE PROJECTIONS (NEXT {horizon} PERIODS)")
    print("=" * 80)
    print(f"  {'STEP':<6} {'TIME (IST)':<18} {'OPEN (₹)':<12} {'HIGH (₹)':<12} {'LOW (₹)':<12} {'CLOSE (₹)':<12} {'90% CONF. BAND':<24}")
    print("-" * 80)

    for i, ts in enumerate(pred["future_timestamps"]):
        o = pred["mean_forecast"][i, 0]
        h = pred["mean_forecast"][i, 1]
        l = pred["mean_forecast"][i, 2]
        c = pred["mean_forecast"][i, 3]
        band_low = pred["lower_5"][i, 3]
        band_high = pred["upper_95"][i, 3]
        ts_str = ts.strftime("%d %b %I:%M %p")
        print(f"  T+{i+1:<4} {ts_str:<18} ₹{format_inr(o, 0):<11} ₹{format_inr(h, 0):<11} ₹{format_inr(l, 0):<11} ₹{format_inr(c, 0):<11} [₹{format_inr(band_low, 0)} - ₹{format_inr(band_high, 0)}]")

    final_pred_close = pred["mean_forecast"][-1, 3]
    expected_return_pct = ((final_pred_close - curr_price) / curr_price) * 100.0
    final_spread = pred["upper_95"][-1, 3] - pred["lower_5"][-1, 3]

    print("-" * 80)
    print(f"  PROJECTED TARGET (T+{horizon})   : ₹ {format_inr(final_pred_close)}")
    print(f"  EXPECTED DRIFT / RETURN  : {expected_return_pct:+.2f}%")
    print(f"  90% CONFIDENCE INTERVAL  : ₹ {format_inr(pred['lower_5'][-1, 3])} to ₹ {format_inr(pred['upper_95'][-1, 3])} (Spread: ₹ {final_spread:,.0f})")

    if expected_return_pct > 0.2:
        outlook = "BULLISH CONTINUATION (LONG BUILDUP)"
        action = "BUY / LONG BIAS (TRAIL STOPLOSS)"
    elif expected_return_pct < -0.2:
        outlook = "PROFIT BOOKING RETRACEMENT"
        action = "BOOK PROFITS / CAUTION"
    else:
        outlook = "CONSOLIDATION NEAR HIGHS"
        action = "HOLD / RANGEBOUND"

    print("-" * 80)
    print(f"  TACTICAL SIGNAL          : {action}")
    print(f"  SYNTHESIZED OUTLOOK      : {outlook}")
    print("=" * 80 + "\n")

    if save_output:
        result_file = "kronos_goldm_prediction.json"
        summary_payload = {
            "execution_time": datetime.datetime.now().isoformat(),
            "instrument": "Goldm (MCX Future)",
            "contract": data["contract"],
            "current_price": curr_price,
            "change": f"+{data['change_val']} (+{data['change_pct']}%)",
            "day_range": f"{data['day_low']} - {data['day_high']}",
            "oi_analysis": data["oi_analysis"],
            "horizon": horizon,
            "target_price": final_pred_close,
            "expected_return_pct": expected_return_pct,
            "signal": action,
            "forecast_mean_close": [float(x) for x in pred["mean_forecast"][:, 3]],
            "forecast_upper_95": [float(x) for x in pred["upper_95"][:, 3]],
            "forecast_lower_5": [float(x) for x in pred["lower_5"][:, 3]],
        }
        with open(result_file, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"[+] Goldm prediction summary saved to '{result_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos MCX Goldm Predictor")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval (15m, 1h, 4h, 1d)")
    parser.add_argument("--lookback", type=int, default=72, help="Number of historical candles")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    parser.add_argument("--samples", type=int, default=10, help="Monte Carlo samples")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")

    args = parser.parse_args()
    run_pipeline(
        interval=args.interval,
        lookback=args.lookback,
        horizon=args.horizon,
        samples=args.samples,
        temperature=args.temperature,
    )
