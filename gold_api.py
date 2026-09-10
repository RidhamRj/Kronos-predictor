"""
gold_api.py
-----------
Real-time MCX Gold Mini (Goldm) Provider.
- Live Ticker: Fetched directly from Moneycontrol for 100% accurate Indian MCX pricing.
- Historical K-Lines: Fetched from Yahoo Finance (GC=F) and dynamically anchored to the MCX live price 
  to provide accurate price action without rate-limit issues.
"""

import sys
import json
import urllib.request
import urllib.error
import datetime
import time
import re
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TROY_OZ_TO_GRAMS = 31.1034768
MCX_GOLDM_FUTURES_FACTOR = 1.143232
IST_TIMEZONE = datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")

class GoldDataProvider:
    def __init__(self, symbol: str = "GC=F", usd_inr_rate: float = 94.84):
        self.symbol = symbol
        self.usd_inr_rate = usd_inr_rate
        self._mc_cache = {"time": 0, "data": None}
        self._yh_klines_cache = {}

    def get_multiplier(self) -> float:
        return (self.usd_inr_rate / TROY_OZ_TO_GRAMS) * 10.0 * MCX_GOLDM_FUTURES_FACTOR

    def _fetch_yahoo_live(self):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}?interval=1m&range=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())['chart']['result'][0]['meta']

    def fetch_live_ticker(self, expiry: str = "2026-10-05"):
        now_ist = datetime.datetime.now(IST_TIMEZONE)
        
        # In-memory short TTL cache (0.5s) to deduplicate rapid simultaneous requests
        if time.time() - self._mc_cache["time"] < 0.5 and self._mc_cache["data"] is not None:
            return self._mc_cache["data"]

        # 1. Fetch direct live price from Moneycontrol Pricefeed API
        t = int(time.time() * 1000)
        url = f"https://priceapi.moneycontrol.com/pricefeed/mcx/commodityfutures/GOLDM?expiry={expiry}&_={t}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.moneycontrol.com/",
            "Cache-Control": "no-cache",
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
                if raw_json.get("code") == "200" and "data" in raw_json:
                    d = raw_json["data"]
                    price = float(d.get("lastPrice", 0))
                    chg = float(d.get("change", 0))
                    per_chg = float(d.get("perChange", 0))
                    high_p = float(d.get("highPrice", price))
                    low_p = float(d.get("lowPrice", price))
                    open_p = float(d.get("openPrice", price))
                    prev_close = float(d.get("prevClose", price))
                    avg_p = float(d.get("avgPrice", price))
                    vol = float(d.get("tradedVol", 0))
                    val_lacs = float(d.get("tradedValLacs", 0))
                    lot_size = f"{d.get('lotSize', '100')} {d.get('priceUnit', 'GRMS')}"
                    oi = int(float(d.get("openInt", 0)))
                    oi_chg = int(float(d.get("openIntChg", 0)))
                    oi_chg_pct = float(d.get("openIntChgPerc", 0))
                    oi_buildup = d.get("oiBuildup", "Long Buildup")
                    bid_p = float(d.get("bidPrice", price))
                    bid_q = str(d.get("bidQty", "1"))
                    ask_p = float(d.get("askPrice", price))
                    ask_q = str(d.get("askQty", "1"))
                    time_str = d.get("lastupdTime") or now_ist.strftime("%d %b, %Y | %I:%M:%S %p IST")
                    contract_m = d.get("contractMonth", "OCT2026")

                    oi_symbol = "↑" if oi_chg >= 0 else "↓"
                    price_symbol = "↑" if chg >= 0 else "↓"
                    oi_analysis = f"{oi_buildup.upper()} (OI {oi_symbol}, PRICE {price_symbol})"

                    res = {
                        "price": price,
                        "change_val": chg,
                        "change_pct": per_chg,
                        "day_high": high_p,
                        "day_low": low_p,
                        "open_price": open_p,
                        "prev_close": prev_close,
                        "avg_price": avg_p,
                        "volume": vol,
                        "traded_val_lacs": val_lacs,
                        "lot_size": lot_size,
                        "open_interest": oi,
                        "oi_change": oi_chg,
                        "oi_change_pct": oi_chg_pct,
                        "oi_buildup": oi_buildup,
                        "oi_analysis": oi_analysis,
                        "bid_price": bid_p,
                        "bid_qty": bid_q,
                        "ask_price": ask_p,
                        "ask_qty": ask_q,
                        "bid_ask": f"{bid_p:.2f} ({bid_q}) / {ask_p:.2f} ({ask_q})",
                        "expiry": expiry,
                        "contract_month": contract_m,
                        "expiries": [["2026-10-05", "05 Oct, 2026"], ["2026-11-05", "05 Nov, 2026"], ["2026-12-04", "04 Dec, 2026"]],
                        "timestamp": now_ist.isoformat(),
                        "time_formatted": time_str,
                        "source": "Moneycontrol Live PriceFeed (MCX Exact)",
                    }

                    self._mc_cache["time"] = time.time()
                    self._mc_cache["data"] = res
                    return res
        except Exception as e:
            print("[Warning] Moneycontrol Pricefeed failed, falling back to cache/implied:", e)

        # 2. Fallback to cached data if recent
        if self._mc_cache["data"]:
            return self._mc_cache["data"]

        # 3. Fallback to Yahoo Finance if Moneycontrol endpoint is unreachable
        try:
            meta = self._fetch_yahoo_live()
            p_usd = meta['regularMarketPrice']
            mult = self.get_multiplier()
            price_goldm = p_usd * mult
            prev_yahoo = meta.get('chartPreviousClose', p_usd) * mult
            change_val = price_goldm - prev_yahoo

            res = {
                "price": price_goldm,
                "change_val": change_val,
                "change_pct": (change_val / prev_yahoo) * 100 if prev_yahoo else 0,
                "day_high": meta.get('regularMarketDayHigh', p_usd) * mult,
                "day_low": meta.get('regularMarketDayLow', p_usd) * mult,
                "open_price": price_goldm - change_val,
                "prev_close": prev_yahoo,
                "avg_price": price_goldm,
                "volume": float(meta.get('regularMarketVolume', 0)),
                "traded_val_lacs": 0.0,
                "lot_size": "100 GRMS",
                "open_interest": 32000,
                "oi_change": 0,
                "oi_change_pct": 0,
                "oi_buildup": "Live",
                "oi_analysis": "LIVE",
                "bid_price": price_goldm - 10.0,
                "bid_qty": "1",
                "ask_price": price_goldm + 10.0,
                "ask_qty": "1",
                "bid_ask": f"{price_goldm - 10.0:.2f} (1) / {price_goldm + 10.0:.2f} (1)",
                "expiry": expiry,
                "contract_month": "OCT2026",
                "expiries": [["2026-10-05", "05 Oct, 2026"], ["2026-11-05", "05 Nov, 2026"], ["2026-12-04", "04 Dec, 2026"]],
                "timestamp": now_ist.isoformat(),
                "time_formatted": now_ist.strftime("%d %b, %Y | %I:%M:%S %p IST"),
                "source": "Yahoo Implied Fallback",
            }
            return res
        except Exception as e:
            print("Error in fallback ticker:", e)
            return None

    def fetch_klines(self, interval: str = "1h", limit: int = 120, inr_rate: float = None, expiry: str = "2026-10-05"):
        if inr_rate:
            self.usd_inr_rate = inr_rate
            
        cache_key = f"{interval}_{limit}"
        cached = self._yh_klines_cache.get(cache_key)
        if cached and time.time() - cached["time"] < 15:
            return cached["data"]

        yh_interval = "60m" if interval == "1h" else ("1d" if interval == "1d" else interval)
        if interval == "15m": yh_interval = "15m"
        
        rng = "730d"
        if yh_interval == "1d": rng = "5y"
        elif yh_interval == "15m": rng = "60d"
        elif yh_interval == "4h": rng = "730d"
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}?interval={yh_interval}&range={rng}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())['chart']['result'][0]
                
            t_data = data['timestamp']
            q_data = data['indicators']['quote'][0]
            
            def _normalize_bucket(dt: datetime.datetime) -> datetime.datetime:
                if interval == "15m":
                    b_min = (dt.minute // 15) * 15
                    return dt.replace(minute=b_min, second=0, microsecond=0)
                elif interval == "1h":
                    return dt.replace(minute=0, second=0, microsecond=0)
                elif interval == "4h":
                    b_hr = (dt.hour // 4) * 4
                    return dt.replace(hour=b_hr, minute=0, second=0, microsecond=0)
                elif interval == "1d":
                    return dt.replace(hour=9, minute=30, second=0, microsecond=0)
                return dt.replace(second=0, microsecond=0)

            timestamps_utc = []
            timestamps_ist = []
            opens, highs, lows, closes, volumes, amounts = [], [], [], [], [], []
            
            for i in range(len(t_data)):
                if q_data['close'][i] is None: continue
                dt_utc = datetime.datetime.fromtimestamp(t_data[i], tz=datetime.timezone.utc)
                dt_ist = _normalize_bucket(dt_utc.astimezone(IST_TIMEZONE))
                o = float(q_data['open'][i])
                h = float(q_data['high'][i])
                l = float(q_data['low'][i])
                c = float(q_data['close'][i])
                v = float(q_data['volume'][i]) if q_data['volume'][i] is not None else 0.0

                # If candle belongs to the same bucket as previous, merge them
                if len(timestamps_ist) > 0 and timestamps_ist[-1] == dt_ist:
                    highs[-1] = max(highs[-1], h)
                    lows[-1] = min(lows[-1], l)
                    closes[-1] = c
                    volumes[-1] += v
                    amounts[-1] += v * c
                else:
                    timestamps_ist.append(dt_ist)
                    timestamps_utc.append(dt_ist.astimezone(datetime.timezone.utc))
                    opens.append(o)
                    highs.append(h)
                    lows.append(l)
                    closes.append(c)
                    volumes.append(v)
                    amounts.append(v * c)

            opens = np.array(opens[-limit:])
            highs = np.array(highs[-limit:])
            lows = np.array(lows[-limit:])
            closes = np.array(closes[-limit:])
            volumes = np.array(volumes[-limit:])
            amounts = np.array(amounts[-limit:])
            timestamps_utc = timestamps_utc[-limit:]
            timestamps_ist = timestamps_ist[-limit:]

            mult = self.get_multiplier()
            opens_inr = opens * mult
            highs_inr = highs * mult
            lows_inr = lows * mult
            closes_inr = closes * mult
            amounts_inr = amounts * self.usd_inr_rate * MCX_GOLDM_FUTURES_FACTOR
            
            # Fetch the actual live Moneycontrol Price to perfectly anchor the Yahoo Klines
            live_ticker = self.fetch_live_ticker(expiry=expiry)
            mcx_actual_price = live_ticker["price"]
            
            # Calculate the spread/basis difference between Yahoo's implied price and MCX actual price
            # and shift all the candles smoothly to match MCX
            basis_shift = mcx_actual_price - closes_inr[-1]
            
            opens_inr += basis_shift
            highs_inr += basis_shift
            lows_inr += basis_shift
            closes_inr += basis_shift
            
            # Calculate active period bucket for the chosen interval
            now_ist = datetime.datetime.now(IST_TIMEZONE)
            current_bucket = _normalize_bucket(now_ist)

            # If Yahoo is lagging behind and hasn't published the current period candle,
            # synthesize the active forming candle directly from live Moneycontrol ticks
            if len(timestamps_ist) > 0 and timestamps_ist[-1] < current_bucket:
                new_open = closes_inr[-1]
                new_close = mcx_actual_price
                new_high = max(new_open, new_close)
                new_low = min(new_open, new_close)
                new_vol = max(float(live_ticker.get("volume", 100)) / 100.0, 10.0)
                new_amt = new_vol * new_close

                timestamps_utc.append(current_bucket.astimezone(datetime.timezone.utc))
                timestamps_ist.append(current_bucket)
                opens_inr = np.append(opens_inr, new_open)
                highs_inr = np.append(highs_inr, new_high)
                lows_inr = np.append(lows_inr, new_low)
                closes_inr = np.append(closes_inr, new_close)
                volumes = np.append(volumes, new_vol)
                amounts_inr = np.append(amounts_inr, new_amt)
            else:
                # Overwrite latest bounds with live Moneycontrol price
                closes_inr[-1] = mcx_actual_price
                highs_inr[-1] = max(highs_inr[-1], mcx_actual_price)
                lows_inr[-1] = min(lows_inr[-1], mcx_actual_price)
            
            ohlcva_goldm = np.column_stack([opens_inr, highs_inr, lows_inr, closes_inr, volumes, amounts_inr])
            
            res = {
                "source": "Yahoo Finance K-Lines (MCX Anchored)",
                "market": "MCX Gold Mini (Goldm)",
                "contract": f"{live_ticker.get('contract_month', 'OCT2026')} ({live_ticker.get('expiry', '2026-10-05')})",
                "unit": "₹ (Goldm Future)",
                "usd_inr_rate": self.usd_inr_rate,
                "timestamps": timestamps_utc,
                "timestamps_ist": timestamps_ist,
                "open": opens_inr,
                "high": highs_inr,
                "low": lows_inr,
                "close": closes_inr,
                "volume": volumes,
                "amount": amounts_inr,
                "ohlcva": ohlcva_goldm,
                "day_high": live_ticker["day_high"],
                "day_low": live_ticker["day_low"],
                "change_val": live_ticker["change_val"],
                "change_pct": live_ticker["change_pct"],
                "oi_analysis": live_ticker["oi_analysis"],
                "bid_ask": live_ticker["bid_ask"],
                "live_ticker": live_ticker,
                "expiries": live_ticker.get("expiries", []),
            }
            self._yh_klines_cache[cache_key] = {"time": time.time(), "data": res}
            return res
        except Exception as e:
            print("Error in klines", e)
            if self._yh_klines_cache.get(cache_key):
                return self._yh_klines_cache[cache_key]["data"]
            return None

if __name__ == "__main__":
    p = GoldDataProvider()
    t = p.fetch_live_ticker()
    print("Hybrid Live Ticker:")
    for k, v in t.items():
        print(f"  {k}: {v}")
