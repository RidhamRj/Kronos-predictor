"""
server.py
---------
Lightweight HTTP & REST API Server for Kronos MCX Gold Mini (Goldm) Dashboard.
Includes fast `/api/tick` endpoint for real-time live ticker streaming every 1-2 seconds.
"""

import sys
import http.server
import socketserver
import urllib.parse
import urllib.request
import json
import os
import mimetypes
import datetime
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from gold_api import GoldDataProvider, IST_TIMEZONE
from broker_api import DhanDataProvider
from market_analysis import MarketAnalyzer
from kronos_light import KronosLight

PORT = 8000

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CACHED_USD_INR = 94.84

# Global provider to persist cache and dynamic basis shifts across HTTP requests
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


class KronosServerHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/tick":
            self._handle_api_tick(query)
        elif path == "/api/live":
            self._handle_api_live(query)
        elif path == "/api/forecast":
            self._handle_api_forecast(query)
        elif path == "/api/predictions":
            self._handle_api_predictions_get(query)
        elif path == "/api/status":
            self._send_json({"status": "healthy", "service": "Kronos MCX Goldm Live Predictor"})
        else:
            self._serve_static(path)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path == "/api/predictions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                from prediction_tracker import save_prediction
                saved = save_prediction(payload)
                self._send_json({"status": "success", "prediction": saved})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=400)
        else:
            self.send_error(404, "Not Found")

    def _handle_api_tick(self, query):
        expiry = query.get("expiry", ["2026-10-05"])[0]
        usd_inr = fetch_live_usd_inr()
        global_provider.usd_inr_rate = usd_inr
        ticker = global_provider.fetch_live_ticker(expiry=expiry)
        self._send_json(ticker)

    def _handle_api_live(self, query):
        interval = query.get("interval", ["1h"])[0]
        limit = min(max(int(query.get("limit", ["350"])[0]), 30), 1000)
        expiry = query.get("expiry", ["2026-10-05"])[0]
        usd_inr = fetch_live_usd_inr()

        global_provider.usd_inr_rate = usd_inr
        data = global_provider.fetch_klines(interval=interval, limit=limit, expiry=expiry)
        if not data:
            self.send_error(500, "Failed to fetch live market data from Yahoo Finance")
            return
        
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
                "amount": float(data["amount"][i]),
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
        self._send_json(payload)

    def _handle_api_forecast(self, query):
        interval = query.get("interval", ["1h"])[0]
        limit = min(max(int(query.get("lookback", ["350"])[0]), 30), 1000)
        horizon = min(int(query.get("horizon", ["12"])[0]), 48)
        samples = min(int(query.get("samples", ["10"])[0]), 25)
        temp = float(query.get("temperature", ["0.1"])[0])
        expiry = query.get("expiry", ["2026-10-05"])[0]
        usd_inr = fetch_live_usd_inr()

        global_provider.usd_inr_rate = usd_inr
        data = global_provider.fetch_klines(interval=interval, limit=limit, expiry=expiry)
        if not data:
            self.send_error(500, "Failed to fetch historical market data for forecast")
            return
            
        analyzer = MarketAnalyzer(data)
        indicators = analyzer.compute_all()
        live_ticker = data.get("live_ticker", {})

        try:
            from real_kronos import run_real_kronos_forecast
            pred = run_real_kronos_forecast(
                data=data,
                horizon=horizon,
                temperature=temp,
                sample_count=min(max(samples, 3), 15)
            )
            model_info = f"{pred.get('model', 'Kronos-small')} on {pred.get('device', 'GPU')}"
            curr_p = float(pred.get("current_price", live_ticker.get("price", float(data["close"][-1]))))
            final_p = float(pred.get("target_price", pred["mean_forecast"][-1, 3]))
            exp_ret = float(pred.get("expected_return_pct", ((final_p - curr_p) / curr_p) * 100.0))
            outlook = pred.get("outlook", "ANALYZING MARKET REGIME")
            signal = pred.get("signal", "HOLD / MONITOR")
        except Exception as e:
            print("[Warning] Real Kronos GPU model error, falling back to KronosLight:", e)
            model = KronosLight(d_model=64, k_bits=16)
            pred = model.predict(
                ohlcva=data["ohlcva"],
                timestamps=data["timestamps_ist"],
                horizon=horizon,
                temperature=temp,
                num_monte_carlo_samples=samples,
            )
            model_info = "KronosLight (Fallback)"
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

        # Auto-record prediction to prediction tracker
        try:
            from prediction_tracker import save_prediction
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
        self._send_json(payload)

    def _handle_api_predictions_get(self, query):
        interval = query.get("interval", ["15m"])[0]
        expiry = query.get("expiry", ["2026-10-05"])[0]
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

        from prediction_tracker import load_raw_predictions, evaluate_prediction_accuracy, seed_demo_history_if_needed
        if candles_list:
            seed_demo_history_if_needed(candles_list, interval=interval)

        raw_preds = load_raw_predictions()
        evaluated = [evaluate_prediction_accuracy(p, candle_map) for p in raw_preds]
        self._send_json({"predictions": evaluated})

    def _serve_static(self, path):
        if path == "/" or path == "":
            path = "/index.html"

        safe_path = os.path.normpath(path.lstrip("/"))
        file_path = os.path.join(WEB_DIR, safe_path)

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.send_error(404, "File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Error: {e}")

    def _send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server(port=PORT):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", port), KronosServerHandler) as httpd:
        print(f"[+] Kronos MCX Goldm Server running at http://127.0.0.1:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Shutting down server.")


if __name__ == "__main__":
    start_server()
