import os
import json
import urllib.request
import urllib.error
import time
import datetime

IST_TIMEZONE = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

class DhanDataProvider:
    def __init__(self, client_id=None, access_token=None):
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID")
        self.access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        self.base_url = "https://api.dhan.co"
        
        # Dhan mapping for MCX GOLDM
        self.instruments = {
            "2026-10-05": "569003",
            "2026-11-05": "571445"
        }
        
    def _get_headers(self):
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
    def _fetch_quote(self, security_id):
        url = f"{self.base_url}/marketfeed/quote"
        payload = {
            "MCX": [security_id]
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=self._get_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                return res.get("data", {}).get("MCX", {}).get(security_id, {})
        except Exception as e:
            print("[Warning] Dhan Quote fetch failed:", e)
            return None

    def fetch_live_ticker(self, expiry: str = "2026-10-05"):
        if not self.client_id or not self.access_token:
            return {"error": "Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN in .env file"}
            
        now_ist = datetime.datetime.now(IST_TIMEZONE)
        security_id = self.instruments.get(expiry, "569003")
        
        dhan_data = self._fetch_quote(security_id)
        if not dhan_data:
            return {"error": "Failed to fetch data from Dhan API"}
            
        last_price = float(dhan_data.get("last_price", 0))
        prev_close = float(dhan_data.get("previous_close", last_price))
        change_val = last_price - prev_close
        
        return {
            "price": last_price,
            "change_val": change_val,
            "change_pct": (change_val / prev_close * 100) if prev_close else 0,
            "day_high": float(dhan_data.get("high_price", last_price)),
            "day_low": float(dhan_data.get("low_price", last_price)),
            "open_price": float(dhan_data.get("open_price", last_price)),
            "prev_close": prev_close,
            "avg_price": float(dhan_data.get("avg_price", last_price)),
            "volume": float(dhan_data.get("volume", 0)),
            "traded_val_lacs": 0,
            "lot_size": "100 GRMS",
            "open_interest": int(dhan_data.get("oi", 0)),
            "oi_change": 0,
            "oi_change_pct": 0,
            "oi_buildup": "Live",
            "oi_analysis": "LIVE",
            "bid_price": float(dhan_data.get("buy_price", last_price)),
            "bid_qty": str(dhan_data.get("buy_quantity", "1")),
            "ask_price": float(dhan_data.get("sell_price", last_price)),
            "ask_qty": str(dhan_data.get("sell_quantity", "1")),
            "bid_ask": f"{float(dhan_data.get('buy_price', last_price)):.2f} / {float(dhan_data.get('sell_price', last_price)):.2f}",
            "expiry": expiry,
            "contract_month": "OCT2026",
            "expiries": [["2026-10-05", "05 Oct, 2026"]],
            "timestamp": now_ist.isoformat(),
            "time_formatted": now_ist.strftime("%d %b, %Y | %I:%M:%S %p IST"),
            "source": "Dhan API (Exact MCX Live)"
        }

    def fetch_klines(self, interval: str = "1h", limit: int = 120, expiry: str = "2026-10-05"):
        # Since getting historical data from Dhan requires different historical API endpoints (Intraday vs Daily)
        # We will use Yahoo for history but adjust the last close to match Dhan's exact live price!
        from gold_api import GoldDataProvider
        fallback = GoldDataProvider(symbol="GC=F", usd_inr_rate=94.84)
        
        # 1. Fetch live exact from Dhan
        dhan_live = self.fetch_live_ticker(expiry=expiry)
        
        # 2. Fetch history from Yahoo
        hist = fallback.fetch_klines(interval=interval, limit=limit, expiry=expiry)
        
        if hist and "error" not in dhan_live:
            # Shift the entire history up/down so the LAST close exactly matches Dhan!
            dhan_price = dhan_live["price"]
            yahoo_last_price = hist["closes"][-1]
            diff = dhan_price - yahoo_last_price
            
            # Apply uniform shift to make history connect perfectly to live price
            hist["closes"] = [c + diff for c in hist["closes"]]
            hist["opens"] = [o + diff for o in hist["opens"]]
            hist["highs"] = [h + diff for h in hist["highs"]]
            hist["lows"] = [l + diff for l in hist["lows"]]
            hist["last_close"] = dhan_price
            
        return hist
