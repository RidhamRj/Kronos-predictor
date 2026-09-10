"""
market_analysis.py
------------------
Standard Financial & Quantitative Market Analysis Engine.
Calculates technical indicators, momentum, volatility metrics,
and candlestick microstructure properties from live OHLCVA data.
"""

import numpy as np


class MarketAnalyzer:
    def __init__(self, data: dict):
        self.data = data
        self.open = data["open"]
        self.high = data["high"]
        self.low = data["low"]
        self.close = data["close"]
        self.volume = data["volume"]
        self.amount = data["amount"]
        self.n = len(self.close)

    def compute_all(self) -> dict:
        """Runs a complete battery of quantitative indicators and market diagnostics."""
        ema20 = self._ema(self.close, 20)
        ema50 = self._ema(self.close, 50)
        sma20 = self._sma(self.close, 20)
        rsi14 = self._rsi(self.close, 14)
        atr14 = self._atr(14)
        volatility = self._realized_volatility(24)
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(self.close, 20, 2.0)
        macd, macd_signal, macd_hist = self._macd(self.close)

        curr_close = self.close[-1]
        prev_close = self.close[-2] if self.n > 1 else curr_close
        pct_change_1 = ((curr_close - prev_close) / prev_close) * 100.0
        pct_change_24 = ((curr_close - self.close[-24]) / self.close[-24]) * 100.0 if self.n >= 24 else 0.0

        # Microstructure Analysis on latest candle
        curr_open = self.open[-1]
        curr_high = self.high[-1]
        curr_low = self.low[-1]
        candle_range = max(curr_high - curr_low, 1e-6)
        body = abs(curr_close - curr_open)
        upper_wick = curr_high - max(curr_close, curr_open)
        lower_wick = min(curr_close, curr_open) - curr_low
        body_ratio = (body / candle_range) * 100.0
        upper_wick_ratio = (upper_wick / candle_range) * 100.0
        lower_wick_ratio = (lower_wick / candle_range) * 100.0

        # Volume Flow
        avg_vol20 = np.mean(self.volume[-20:]) if self.n >= 20 else np.mean(self.volume)
        vol_surge_ratio = self.volume[-1] / max(avg_vol20, 1e-6)

        # Diagnose Market Regime
        regime = self._diagnose_regime(curr_close, ema20[-1], ema50[-1], rsi14[-1], volatility)

        # Classic MCX Floor Trader Pivot Points (matching Moneycontrol)
        pivot_p = (curr_high + curr_low + curr_close) / 3.0
        r1 = 2.0 * pivot_p - curr_low
        s1 = 2.0 * pivot_p - curr_high
        r2 = pivot_p + (curr_high - curr_low)
        s2 = pivot_p - (curr_high - curr_low)
        r3 = curr_high + 2.0 * (pivot_p - curr_low)
        s3 = curr_low - 2.0 * (curr_high - pivot_p)

        # Moneycontrol Technical Summary Stance
        if rsi14[-1] >= 65 and curr_close > ema20[-1]:
            tech_stance = "VERY BULLISH"
        elif rsi14[-1] >= 55:
            tech_stance = "BULLISH"
        elif rsi14[-1] <= 35 and curr_close < ema20[-1]:
            tech_stance = "VERY BEARISH"
        elif rsi14[-1] <= 45:
            tech_stance = "BEARISH"
        else:
            tech_stance = "NEUTRAL"

        return {
            "current_price": curr_close,
            "pct_change_1_period": pct_change_1,
            "pct_change_24_periods": pct_change_24,
            "ema_20": ema20[-1],
            "ema_50": ema50[-1],
            "sma_20": sma20[-1],
            "rsi_14": rsi14[-1],
            "atr_14": atr14[-1],
            "realized_volatility_pct": volatility * 100.0,
            "bollinger_upper": bb_upper[-1],
            "bollinger_mid": bb_mid[-1],
            "bollinger_lower": bb_lower[-1],
            "macd": macd[-1],
            "macd_signal": macd_signal[-1],
            "macd_hist": macd_hist[-1],
            "candle_body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "volume_surge_ratio": vol_surge_ratio,
            "market_regime": regime,
            "tech_stance": tech_stance,
            "pivots": {
                "pivot": pivot_p,
                "r1": r1, "r2": r2, "r3": r3,
                "s1": s1, "s2": s2, "s3": s3
            }
        }

    def _sma(self, series: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(series, fill_value=np.nan)
        eff_period = min(period, len(series))
        if eff_period <= 0:
            return series
        for i in range(eff_period - 1, len(series)):
            result[i] = np.mean(series[max(0, i - eff_period + 1) : i + 1])
        first_valid = result[eff_period - 1]
        result[: eff_period - 1] = first_valid
        return result

    def _ema(self, series: np.ndarray, period: int) -> np.ndarray:
        if len(series) == 0:
            return series
        result = np.zeros_like(series)
        alpha = 2.0 / (period + 1.0)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1.0 - alpha) * result[i - 1]
        return result

    def _rsi(self, series: np.ndarray, period: int = 14) -> np.ndarray:
        if len(series) < 2:
            return np.full_like(series, 50.0)
        deltas = np.diff(series)
        eff_period = min(period, len(deltas))
        seed = deltas[:eff_period]
        up = seed[seed >= 0].sum() / eff_period if len(seed) > 0 else 0
        down = -seed[seed < 0].sum() / eff_period if len(seed) > 0 else 0
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(series)
        rsi[:eff_period] = 100.0 - 100.0 / (1.0 + rs)

        up_val = up
        down_val = down
        for i in range(eff_period, len(series)):
            delta = deltas[i - 1]
            if delta > 0:
                up_val = (up_val * (period - 1) + delta) / period
                down_val = (down_val * (period - 1)) / period
            else:
                up_val = (up_val * (period - 1)) / period
                down_val = (down_val * (period - 1) - delta) / period
            rs = up_val / down_val if down_val != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    def _atr(self, period: int = 14) -> np.ndarray:
        if self.n == 0:
            return np.array([])
        tr = np.zeros(self.n)
        tr[0] = self.high[0] - self.low[0]
        for i in range(1, self.n):
            hl = self.high[i] - self.low[i]
            hc = abs(self.high[i] - self.close[i - 1])
            lc = abs(self.low[i] - self.close[i - 1])
            tr[i] = max(hl, hc, lc)
        return self._ema(tr, period)

    def _realized_volatility(self, window: int = 24) -> float:
        """Standard deviation of log returns over rolling window."""
        if self.n < 2:
            return 0.01
        effective_w = min(window, self.n - 1)
        sub_closes = self.close[-effective_w - 1 :]
        log_ret = np.diff(np.log(np.maximum(sub_closes, 1e-6)))
        return float(np.std(log_ret))

    def _bollinger_bands(self, series: np.ndarray, period: int = 20, num_std: float = 2.0):
        mid = self._sma(series, period)
        upper = np.zeros_like(series)
        lower = np.zeros_like(series)
        eff_period = min(period, len(series))
        for i in range(eff_period - 1, len(series)):
            std = np.std(series[max(0, i - eff_period + 1) : i + 1])
            upper[i] = mid[i] + num_std * std
            lower[i] = mid[i] - num_std * std
        upper[: eff_period - 1] = upper[eff_period - 1]
        lower[: eff_period - 1] = lower[eff_period - 1]
        return upper, mid, lower

    def _macd(self, series: np.ndarray):
        ema12 = self._ema(series, 12)
        ema26 = self._ema(series, 26)
        macd = ema12 - ema26
        signal = self._ema(macd, 9)
        hist = macd - signal
        return macd, signal, hist

    def _diagnose_regime(self, price, ema20, ema50, rsi, vol) -> str:
        bullish_ma = price > ema20 > ema50
        bearish_ma = price < ema20 < ema50

        if bullish_ma and rsi > 55:
            return "Strong Bullish Trend (Momentum Expansion)"
        elif bullish_ma:
            return "Moderate Bullish Consolidation"
        elif bearish_ma and rsi < 45:
            return "Strong Bearish Trend (Selling Pressure)"
        elif bearish_ma:
            return "Moderate Bearish Consolidation"
        elif rsi > 70:
            return "Overbought Squeeze"
        elif rsi < 30:
            return "Oversold Exhaustion"
        else:
            return "Range-Bound Equilibrium / Choppy"


if __name__ == "__main__":
    from gold_api import GoldDataProvider
    data = GoldDataProvider().fetch_klines(interval="1h", limit=60)
    analyzer = MarketAnalyzer(data)
    analysis = analyzer.compute_all()
    print("Market Analysis Results:")
    for k, v in analysis.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:,.4f}")
        else:
            print(f"  {k:25s}: {v}")
