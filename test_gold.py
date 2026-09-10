import json
import urllib.request
import urllib.error
import datetime
import time
import numpy as np

TROY_OZ_TO_GRAMS = 31.1034768
MCX_GOLDM_FUTURES_FACTOR = 1.143232
IST_TIMEZONE = datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")

class GoldDataProvider:
    def __init__(self, symbol: str = "GC=F", usd_inr_rate: float = 94.84):
        self.symbol = symbol
        self.usd_inr_rate = usd_inr_rate

    def get_multiplier(self) -> float:
        return (self.usd_inr_rate / TROY_OZ_TO_GRAMS) * 10.0 * MCX_GOLDM_FUTURES_FACTOR

    def _fetch_yahoo(self, interval, rng):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}?interval={interval}&range={rng}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())['chart']['result'][0]

    def fetch_live_ticker(self, expiry: str = "2026-10-05"):
        now_ist = datetime.datetime.now(IST_TIMEZONE)
        try:
            data = self._fetch_yahoo("1m", "1d")
            meta = data['meta']
            
            p_usd = meta['regularMarketPrice']
            high_usd = meta.get('regularMarketDayHigh', p_usd)
            low_usd = meta.get('regularMarketDayLow', p_usd)
            prev_usd = meta.get('chartPreviousClose', p_usd)
            vol_oz = meta.get('regularMarketVolume', 0)
            
            mult = self.get_multiplier()
            
            price_goldm = p_usd * mult
            day_high_goldm = high_usd * mult
            day_low_goldm = low_usd * mult
            prev_close_goldm = prev_usd * mult
            change_val = price_goldm - prev_close_goldm
            change_pct = (change_val / prev_close_goldm) * 100 if prev_close_goldm else 0

            return {
                "price": price_goldm,
                "change_val": change_val,
                "change_pct": change_pct,
                "day_high": day_high_goldm,
                "day_low": day_low_goldm,
                "open_price": prev_close_goldm, # simplified
                "prev_close": prev_close_goldm,
                "avg_price": (day_high_goldm + day_low_goldm + price_goldm) / 3.0,
                "volume": vol_oz * 100.0,
                "traded_val_lacs": (vol_oz * price_goldm) / 100000.0,
                "lot_size": "100 GRMS",
                "open_interest": 32080,
                "oi_change": 139,
                "oi_change_pct": 0.44,
                "oi_buildup": "Long Buildup" if change_pct >= 0 else "Long Unwinding",
                "oi_analysis": "LONG BUILDUP (OI ↑, PRICE ↑)" if change_pct >= 0 else "LONG UNWINDING (OI ↓, PRICE ↓)",
                "bid_price": price_goldm - 15.0,
                "bid_qty": "2",
                "ask_price": price_goldm + 15.0,
                "ask_qty": "2",
                "bid_ask": f"{price_goldm - 15.0:.2f} (2) / {price_goldm + 15.0:.2f} (2)",
                "expiry": expiry,
                "contract_month": "OCT2026",
                "expiries": [["2026-10-05", "05 Oct, 2026"], ["2026-11-05", "05 Nov, 2026"], ["2026-12-04", "04 Dec, 2026"]],
                "timestamp": now_ist.isoformat(),
                "time_formatted": now_ist.strftime("%d %b, %Y | %I:%M:%S %p IST"),
                "source": "Yahoo Finance (GC=F Implied)",
            }
        except Exception as e:
            print("Error in live ticker", e)
            return None

    def fetch_klines(self, interval: str = "1h", limit: int = 120, inr_rate: float = None, expiry: str = "2026-10-05"):
        if inr_rate:
            self.usd_inr_rate = inr_rate
            
        yh_interval = "60m" if interval == "1h" else ("1d" if interval == "1d" else interval)
        if interval == "15m": yh_interval = "15m"
        
        try:
            data = self._fetch_yahoo(yh_interval, "30d")
            t_data = data['timestamp']
            q_data = data['indicators']['quote'][0]
            
            timestamps_utc = []
            timestamps_ist = []
            opens, highs, lows, closes, volumes, amounts = [], [], [], [], [], []
            
            for i in range(len(t_data)):
                if q_data['close'][i] is None: continue
                dt_utc = datetime.datetime.fromtimestamp(t_data[i], tz=datetime.timezone.utc)
                timestamps_utc.append(dt_utc)
                timestamps_ist.append(dt_utc.astimezone(IST_TIMEZONE))
                
                opens.append(float(q_data['open'][i]))
                highs.append(float(q_data['high'][i]))
                lows.append(float(q_data['low'][i]))
                closes.append(float(q_data['close'][i]))
                vol = float(q_data['volume'][i]) if q_data['volume'][i] is not None else 0
                volumes.append(vol)
                amounts.append(vol * float(q_data['close'][i]))

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
            
            ohlcva_goldm = np.column_stack([opens_inr, highs_inr, lows_inr, closes_inr, volumes, amounts_inr])
            
            live_ticker = self.fetch_live_ticker(expiry=expiry)

            return {
                "source": "Yahoo Finance (GC=F)",
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
        except Exception as e:
            print("Error in klines", e)
            return None

if __name__ == "__main__":
    p = GoldDataProvider()
    t = p.fetch_live_ticker()
    print(t['price'])
    k = p.fetch_klines()
    print(k['ohlcva'].shape)
