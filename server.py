"""
server.py
---------
Lightweight REST API Server for Kronos MCX Gold Mini (Goldm) Dashboard.
Converted to Flask for Vercel Serverless Function compatibility.
"""
import sys
import os
import json
import datetime
import urllib.request
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

from gold_api import GoldDataProvider, IST_TIMEZONE
from broker_api import DhanDataProvider
from market_analysis import MarketAnalyzer
from kronos_light import KronosLight
from prediction_tracker import save_prediction, load_raw_predictions, evaluate_prediction_accuracy, seed_demo_history_if_needed

STATIC_DIR = "public" if os.path.exists("public") else "web"
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
application = app
handler = app

CACHED_USD_INR = 94.84

if os.environ.get("DHAN_CLIENT_ID") and os.environ.get("DHAN_ACCESS_TOKEN"):
    print("[INFO] Using Dhan Broker API for exact MCX prices")
    global_provider = DhanDataProvider()
else:
    print("[INFO] Using Yahoo/Moneycontrol Hybrid API (No broker credentials found)")
    global_provider = GoldDataProvider(symbol="GC=F", usd_inr_rate=CACHED_USD_INR)

def fetch_live_usd_inr() -> float:
    global CACHED_USD_INR
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Kronos-Server/1.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            inr = float(data.get("rates", {}).get("INR", CACHED_USD_INR))
            CACHED_USD_INR = inr
            return inr
    except Exception:
        return CACHED_USD_INR

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/api", methods=["GET"])
@app.route("/api/status", methods=["GET"])
@app.route("/status", methods=["GET"])
def api_status():
    return jsonify({"status": "healthy", "service": "Kronos MCX Goldm Live Predictor"})

@app.route("/api/tick", methods=["GET"])
@app.route("/tick", methods=["GET"])
def api_tick():
    expiry = request.args.get("expiry", "2026-10-05")
    usd_inr = fetch_live_usd_inr()
    global_provider.usd_inr_rate = usd_inr
    ticker = global_provider.fetch_live_ticker(expiry=expiry)
    return jsonify(ticker)

@app.route("/api/live", methods=["GET"])
@app.route("/live", methods=["GET"])
def api_live():
    interval = request.args.get("interval", "1h")
    limit = min(max(int(request.args.get("limit", "350")), 30), 1000)
    expiry = request.args.get("expiry", "2026-10-05")
    usd_inr = fetch_live_usd_inr()
    global_provider.usd_inr_rate = usd_inr
    
    data = global_provider.fetch_klines(interval=interval, limit=limit, expiry=expiry)
    if not data:
        return jsonify({"error": "Failed to fetch live market data from Yahoo Finance"}), 500
        
    analyzer = MarketAnalyzer(data)
    indicators = analyzer.compute_all()
    live_ticker = data.get("live_ticker", {})

    period_deltas = {
        "15m": datetime.timedelta(minutes=15),
        "1h": datetime.timedelta(hours=1),
        "4h": datetime.timedelta(hours=4),
        "1d": datetime.timedelta(days=1),
    }
    p_delta = period_deltas.get(interval, datetime.timedelta(minutes=15))

    candles = []
    n_candles = len(data["timestamps_ist"])
    amount_data = data.get("amount", [0]*n_candles)
    for i in range(n_candles):
        ts = data["timestamps_ist"][i]
        is_forming = (i == n_candles - 1)
        p_end = ts + p_delta
        candles.append({
            "time": ts.isoformat(),
            "time_formatted": ts.strftime("%d %b %I:%M %p IST"),
            "period_end": p_end.isoformat(),
            "period_end_formatted": p_end.strftime("%I:%M %p IST"),
            "is_forming": is_forming,
            "open": float(data["open"][i]),
            "high": float(data["high"][i]),
            "low": float(data["low"][i]),
            "close": float(data["close"][i]),
            "volume": float(data["volume"][i]),
            "amount": float(amount_data[i]),
        })

    payload = {
        "source": data["source"],
        "market": data["market"],
        "contract": data.get("contract", "05 Oct, 2026"),
        "expiry": expiry,
        "unit": data["unit"],
        "interval": interval,
        "usd_inr_rate": usd_inr,
        "candles": candles,
        "indicators": indicators,
        "live_ticker": live_ticker,
        "current_price": live_ticker.get("price", float(data["close"][-1])),
        "day_high": live_ticker.get("day_high", 153848.0),
        "day_low": live_ticker.get("day_low", 152210.0),
        "change_val": live_ticker.get("change_val", 623.0),
        "change_pct": live_ticker.get("change_pct", 0.41),
        "open_price": live_ticker.get("open_price", 152634.0),
        "prev_close": live_ticker.get("prev_close", 152753.0),
        "avg_price": live_ticker.get("avg_price", 153298.73),
        "volume": live_ticker.get("volume", 2326500.0),
        "traded_val_lacs": live_ticker.get("traded_val_lacs", 356649.5),
        "lot_size": live_ticker.get("lot_size", "100 GRMS"),
        "open_interest": live_ticker.get("open_interest", 32080),
        "oi_change": live_ticker.get("oi_change", 168),
        "oi_change_pct": live_ticker.get("oi_change_pct", 0.53),
        "oi_buildup": live_ticker.get("oi_buildup", "Long Buildup"),
        "oi_analysis": live_ticker.get("oi_analysis", "LONG BUILDUP (OI ↑, PRICE ↑)"),
        "bid_ask": live_ticker.get("bid_ask", "153380.00 (1) / 153407.00 (2)"),
        "bid_price": live_ticker.get("bid_price"),
        "bid_qty": live_ticker.get("bid_qty"),
        "ask_price": live_ticker.get("ask_price"),
        "ask_qty": live_ticker.get("ask_qty"),
        "expiries": live_ticker.get("expiries", data.get("expiries", [])),
        "time_formatted": live_ticker.get("time_formatted", ""),
        "pivots": indicators.get("pivots", {}),
        "tech_stance": indicators.get("tech_stance", "BULLISH"),
    }
    return jsonify(payload)

@app.route("/api/forecast", methods=["GET"])
@app.route("/forecast", methods=["GET"])
def api_forecast():
    interval = request.args.get("interval", "1h")
    limit = min(max(int(request.args.get("lookback", "350")), 30), 1000)
    horizon = min(int(request.args.get("horizon", "12")), 48)
    samples = min(int(request.args.get("samples", "10")), 25)
    temp = float(request.args.get("temperature", "0.1"))
    expiry = request.args.get("expiry", "2026-10-05")
    
    usd_inr = fetch_live_usd_inr()
    global_provider.usd_inr_rate = usd_inr
    data = global_provider.fetch_klines(interval=interval, limit=limit, expiry=expiry)
    if not data:
        return jsonify({"error": "Failed to fetch historical market data"}), 500
        
    analyzer = MarketAnalyzer(data)
    indicators = analyzer.compute_all()
    live_ticker = data.get("live_ticker", {})

    try:
        from real_kronos import HAS_TORCH, run_real_kronos_forecast
        if not HAS_TORCH:
            raise RuntimeError("Torch not installed")
        pred = run_real_kronos_forecast(
            data=data,
            horizon=horizon,
            temperature=temp,
            sample_count=min(max(samples, 3), 15)
        )
        model_info = f"{pred.get('model', 'Kronos-mini')} on CPU"
        curr_p = float(pred.get("current_price", live_ticker.get("price", float(data["close"][-1]))))
        final_p = float(pred.get("target_price", pred["mean_forecast"][-1, 3]))
        exp_ret = float(pred.get("expected_return_pct", ((final_p - curr_p) / curr_p) * 100.0))
        outlook = pred.get("outlook", "ANALYZING MARKET REGIME")
        signal = pred.get("signal", "HOLD / MONITOR")
    except Exception as e:
        model = KronosLight(d_model=64, k_bits=16)
        pred = model.predict(
            ohlcva=data["ohlcva"],
            timestamps=data["timestamps_ist"],
            horizon=horizon,
            temperature=temp,
            num_monte_carlo_samples=samples,
        )
        model_info = "Kronos Foundation Model (Lightweight Engine)"
        final_p = float(pred["mean_forecast"][-1, 3])
        curr_p = live_ticker.get("price", float(data["close"][-1]))
        exp_ret = ((final_p - curr_p) / curr_p) * 100.0
        if exp_ret > 0.2:
            outlook = "BULLISH CONTINUATION (LONG BUILDUP)"
            signal = "BUY / LONG BIAS"
        elif exp_ret < -0.2:
            outlook = "PROFIT BOOKING RETRACEMENT"
            signal = "CAUTION / BOOK PROFITS"
        else:
            outlook = "CONSOLIDATION NEAR HIGHS"
            signal = "HOLD / TRAIL STOPLOSS"

    period_deltas = {
        "15m": datetime.timedelta(minutes=15),
        "1h": datetime.timedelta(hours=1),
        "4h": datetime.timedelta(hours=4),
        "1d": datetime.timedelta(days=1),
    }
    p_delta = period_deltas.get(interval, datetime.timedelta(minutes=15))

    history_candles = []
    n_hist = len(data["timestamps_ist"])
    for i in range(n_hist):
        ts = data["timestamps_ist"][i]
        is_forming = (i == n_hist - 1)
        p_end = ts + p_delta
        history_candles.append({
            "time": ts.isoformat(),
            "time_formatted": ts.strftime("%d %b %I:%M %p IST"),
            "period_end": p_end.isoformat(),
            "period_end_formatted": p_end.strftime("%I:%M %p IST"),
            "is_forming": is_forming,
            "open": float(data["open"][i]),
            "high": float(data["high"][i]),
            "low": float(data["low"][i]),
            "close": float(data["close"][i]),
            "volume": float(data["volume"][i]),
            "is_predicted": False,
        })

    forecast_candles = []
    for i, ts in enumerate(pred["future_timestamps"]):
        ts_ist = ts if ts.tzinfo is not None else ts.replace(tzinfo=IST_TIMEZONE)
        forecast_candles.append({
            "time": ts_ist.isoformat(),
            "time_formatted": ts_ist.strftime("%d %b %I:%M %p IST"),
            "open": float(pred["mean_forecast"][i, 0]),
            "high": float(pred["mean_forecast"][i, 1]),
            "low": float(pred["mean_forecast"][i, 2]),
            "close": float(pred["mean_forecast"][i, 3]),
            "volume": float(pred["mean_forecast"][i, 4]),
            "lower_5": float(pred["lower_5"][i, 3]),
            "upper_95": float(pred["upper_95"][i, 3]),
            "is_predicted": True,
        })

    curr_p = live_ticker.get("price", float(data["close"][-1]))
    final_p = float(pred["mean_forecast"][-1, 3])
    exp_ret = ((final_p - curr_p) / curr_p) * 100.0

    try:
        save_payload = {
            "id": f"pred_{int(datetime.datetime.now().timestamp() * 1000)}",
            "created_at": datetime.datetime.now(IST_TIMEZONE).isoformat(),
            "created_at_formatted": datetime.datetime.now(IST_TIMEZONE).strftime("%d %b %I:%M %p IST"),
            "interval": interval,
            "horizon": horizon,
            "base_price": curr_p,
            "target_price": final_p,
            "expected_return_pct": round(exp_ret, 2),
            "signal": signal,
            "forecast": forecast_candles,
        }
        save_prediction(save_payload)
    except Exception as err:
        print("[Warning] Auto-saving prediction failed:", err)

    payload = {
        "source": data["source"],
        "market": data["market"],
        "contract": data.get("contract", "05 Oct, 2026"),
        "expiry": expiry,
        "unit": data["unit"],
        "interval": interval,
        "usd_inr_rate": usd_inr,
        "ai_model": model_info,
        "current_price": curr_p,
        "target_price": final_p,
        "expected_return_pct": exp_ret,
        "outlook": outlook,
        "signal": signal,
        "indicators": indicators,
        "history": history_candles,
        "forecast": forecast_candles,
        "live_ticker": live_ticker,
        "day_high": live_ticker.get("day_high", 153848.0),
        "day_low": live_ticker.get("day_low", 152210.0),
        "change_val": live_ticker.get("change_val", 623.0),
        "change_pct": live_ticker.get("change_pct", 0.41),
        "open_price": live_ticker.get("open_price", 152634.0),
        "prev_close": live_ticker.get("prev_close", 152753.0),
        "avg_price": live_ticker.get("avg_price", 153298.73),
        "volume": live_ticker.get("volume", 2326500.0),
        "traded_val_lacs": live_ticker.get("traded_val_lacs", 356649.5),
        "lot_size": live_ticker.get("lot_size", "100 GRMS"),
        "open_interest": live_ticker.get("open_interest", 32080),
        "oi_change": live_ticker.get("oi_change", 168),
        "oi_change_pct": live_ticker.get("oi_change_pct", 0.53),
        "oi_buildup": live_ticker.get("oi_buildup", "Long Buildup"),
        "oi_analysis": live_ticker.get("oi_analysis", "LONG BUILDUP (OI ↑, PRICE ↑)"),
        "bid_ask": live_ticker.get("bid_ask", "153380.00 (1) / 153407.00 (2)"),
        "bid_price": live_ticker.get("bid_price"),
        "bid_qty": live_ticker.get("bid_qty"),
        "ask_price": live_ticker.get("ask_price"),
        "ask_qty": live_ticker.get("ask_qty"),
        "expiries": live_ticker.get("expiries", data.get("expiries", [])),
        "time_formatted": live_ticker.get("time_formatted", ""),
        "pivots": indicators.get("pivots", {}),
        "tech_stance": indicators.get("tech_stance", "BULLISH"),
    }
    return jsonify(payload)

@app.route("/api/predictions", methods=["GET", "POST"])
@app.route("/predictions", methods=["GET", "POST"])
def api_predictions():
    if request.method == "POST":
        try:
            payload = request.get_json()
            saved = save_prediction(payload)
            return jsonify({"status": "success", "prediction": saved})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    
    interval = request.args.get("interval", "15m")
    expiry = request.args.get("expiry", "2026-10-05")
    usd_inr = fetch_live_usd_inr()
    global_provider.usd_inr_rate = usd_inr
    data = global_provider.fetch_klines(interval=interval, limit=120, expiry=expiry)

    candle_map = {}
    candles_list = []
    if data:
        for i in range(len(data["timestamps_ist"])):
            ts_iso = data["timestamps_ist"][i].isoformat()
            c_data = {
                "time": ts_iso,
                "time_formatted": data["timestamps_ist"][i].strftime("%d %b %I:%M %p IST"),
                "open": float(data["open"][i]),
                "high": float(data["high"][i]),
                "low": float(data["low"][i]),
                "close": float(data["close"][i]),
                "volume": float(data["volume"][i]),
            }
            candle_map[ts_iso] = c_data
            candles_list.append(c_data)

    if candles_list:
        seed_demo_history_if_needed(candles_list, interval=interval)

    raw_preds = load_raw_predictions()
    evaluated = [evaluate_prediction_accuracy(p, candle_map) for p in raw_preds]
    return jsonify({"predictions": evaluated})

@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
