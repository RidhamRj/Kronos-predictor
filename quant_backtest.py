"""
quant_backtest.py
-----------------
Rigorous Walk-Forward Out-Of-Sample Validation Engine for Kronos-base (AAAI 2026).
VOLATILITY FORECASTING PIVOT
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np
import pandas as pd
import scipy.stats as stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure Kronos_official is in search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.path.join(CURRENT_DIR, "Kronos_official")
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)

import torch
from model import Kronos, KronosTokenizer, KronosPredictor

# --------------------------------------------------------------------------
# 1. Data Fetching & Cleaning
# --------------------------------------------------------------------------
def fetch_nifty_data(range_str="10y") -> pd.DataFrame:
    print(f"[Data Loader] Fetching NIFTY 50 (^NSEI) daily data for {range_str}...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range={range_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode())
        chart = d["chart"]["result"][0]
        timestamps = chart["timestamp"]
        quote = chart["indicators"]["quote"][0]
        
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata")
    df = pd.DataFrame({
        "open": np.array(quote["open"], dtype=np.float32),
        "high": np.array(quote["high"], dtype=np.float32),
        "low": np.array(quote["low"], dtype=np.float32),
        "close": np.array(quote["close"], dtype=np.float32),
        "volume": np.array([v if v is not None else 0 for v in quote["volume"]], dtype=np.float32),
    }, index=dates)
    
    df = df.dropna().reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    df["amount"] = df["volume"] * df["close"]
    
    # Precalculate baseline metrics
    df['ret'] = df['close'].pct_change()
    df['abs_ret'] = df['ret'].abs()
    df['roll_std_20'] = df['ret'].rolling(20).std()
    df['roll_abs_20'] = df['abs_ret'].rolling(20).mean()
    df['ewma_abs_20'] = df['abs_ret'].ewm(span=20).mean()
    
    # Shift them by 1 so they are causally sound (representing the state *before* the next day)
    df['roll_std_20'] = df['roll_std_20'].shift(1)
    df['roll_abs_20'] = df['roll_abs_20'].shift(1)
    df['ewma_abs_20'] = df['ewma_abs_20'].shift(1)
    
    df = df.dropna(subset=['roll_std_20', 'ewma_abs_20']).reset_index(drop=True)
    
    print(f"[Data Loader] Successfully loaded {len(df)} clean trading days.")
    print(f"              Date Range: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    return df

# --------------------------------------------------------------------------
# 2. Main Walk-Forward Validation Engine (Multi-Window)
# --------------------------------------------------------------------------
def run_multi_window_validation(
    df: pd.DataFrame,
    context_len: int = 250,
    batch_size: int = 20,
    sample_count: int = 5
):
    print("\n" + "=" * 85)
    print("      VOLATILITY FORECASTING WALK-FORWARD VALIDATION AUDIT")
    print("=" * 85)
    print(f"Target Asset          : NIFTY 50 Index (^NSEI)")
    print(f"Target Variable       : Continuous Absolute Return |return|")
    print(f"Binary Threshold      : 1 Sigma of trailing 20-day realized volatility")
    print(f"Rolling In-Sample Ctx : {context_len} Days (1 Calendar Year)")
    print(f"Monte Carlo Paths     : {sample_count} Paths per Day")
    print("=" * 85)

    # Initialize GPU Model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Model Setup] Initializing NeoQuasar/Kronos-base (102.3M) on {device}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", local_files_only=True).to(device)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base", local_files_only=True).to(device)
    model.eval()
    predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)
    print("[Model Setup] Model initialized in VRAM successfully.")

    windows = [
        ("2018-01", "2019-12"),
        ("2020-01", "2021-12"),
        ("2022-01", "2023-12"),
        ("2024-01", "2026-12")
    ]

    all_metrics = []
    
    # Audit log setup
    with open("audit_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*85}\n")
        log_file.write(f"VOLATILITY MULTI-WINDOW RUN: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Config: context={context_len}, batch={batch_size}, samples={sample_count}\n")
        log_file.write(f"{'='*85}\n")

    for w_start, w_end in windows:
        print(f"\n>>> Running Window: {w_start} to {w_end}")
        
        # Filter dataframe for dates
        test_start_date = pd.to_datetime(f"{w_start}-01").tz_localize("Asia/Kolkata")
        end_year, end_month = map(int, w_end.split('-'))
        if end_month == 12:
            test_end_date = pd.to_datetime(f"{end_year+1}-01-01").tz_localize("Asia/Kolkata") - pd.Timedelta(days=1)
        else:
            test_end_date = pd.to_datetime(f"{end_year}-{end_month+1:02d}-01").tz_localize("Asia/Kolkata") - pd.Timedelta(days=1)
            
        test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]
        if len(test_df) == 0:
            print("No data for this window. Skipping.")
            continue
            
        test_start_idx = test_df.index[0]
        if test_start_idx < context_len:
            print("Not enough context data before this window. Skipping.")
            continue
            
        window_full_df = df.iloc[test_start_idx - context_len : test_df.index[-1] + 1].copy().reset_index(drop=True)
        
        # Run Walk Forward for this window
        results_df = run_walk_forward_for_window(window_full_df, predictor, context_len, batch_size, sample_count)
        results_df.to_csv(f"walk_forward_volatility_{w_start}_{w_end}.csv", index=False)
        
        metrics = compute_metrics(results_df)
        metrics["Window"] = f"{w_start} to {w_end}"
        all_metrics.append(metrics)
        
        print(f"\n[Window {w_start} to {w_end} Result]")
        print(f"MAE (Kronos): {metrics['mae_k']*100:.3f}% | MAE (EWMA): {metrics['mae_ewma']*100:.3f}% | IC (Kronos vs Realized): {metrics['ic_k']:+.4f}")
        
    print("\n" + "=" * 95)
    print("                            VOLATILITY SUMMARY TABLE")
    print("=" * 95)
    print(f"{'Window':<20} | {'MAE Kronos':<12} | {'MAE EWMA':<12} | {'MAE Roll20':<12} | {'Hit Rate (>1σ)':<16} | {'Maj Class Base'}")
    print("-" * 95)
    
    with open("audit_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'Window':<20} | {'MAE Kronos':<12} | {'MAE EWMA':<12} | {'MAE Roll20':<12} | {'Hit Rate (>1σ)':<16} | {'Maj Class Base'}\n")
        log_file.write("-" * 95 + "\n")
        for m in all_metrics:
            row_str = f"{m['Window']:<20} | {m['mae_k']*100:>11.3f}% | {m['mae_ewma']*100:>11.3f}% | {m['mae_roll']*100:>11.3f}% | {m['hit_k']:>15.2f}% | {m['hit_base0']:>13.2f}%"
            print(row_str)
            log_file.write(row_str + "\n")

def run_walk_forward_for_window(df, predictor, context_len, batch_size, sample_count):
    total_days = len(df)
    test_start_idx = context_len
    total_test_days = total_days - test_start_idx
    
    records = []
    t0 = time.time()
    
    for start_i in range(test_start_idx, total_days, batch_size):
        end_i = min(start_i + batch_size, total_days)
        cur_batch_size = end_i - start_i
        
        batch_dfs = []
        batch_x_stamps = []
        batch_y_stamps = []
        batch_meta = []
        
        for idx in range(start_i, end_i):
            window_df = df.iloc[idx - context_len : idx].copy()
            next_row = df.iloc[idx]
            
            x_stamp = pd.Series(pd.to_datetime(window_df["date"]))
            y_stamp = pd.Series(pd.to_datetime([next_row["date"]]))
            
            batch_dfs.append(window_df)
            batch_x_stamps.append(x_stamp)
            batch_y_stamps.append(y_stamp)
            
            c_curr = float(window_df["close"].iloc[-1])
            c_next = float(next_row["close"])
            
            real_return = (c_next - c_curr) / c_curr
            abs_real_return = abs(real_return)
            
            # Using precalculated baseline metrics aligned via index lookup
            roll_std = float(next_row["roll_std_20"])
            ewma_abs = float(next_row["ewma_abs_20"])
            roll_abs = float(next_row["roll_abs_20"])
            
            real_high_vol = 1 if abs_real_return > roll_std else 0
            
            batch_meta.append({
                "date": next_row["date"].strftime("%Y-%m-%d"),
                "curr_close": c_curr,
                "real_close": c_next,
                "real_return": real_return,
                "abs_real_return": abs_real_return,
                "roll_std": roll_std,
                "ewma_abs": ewma_abs,
                "roll_abs": roll_abs,
                "real_high_vol": real_high_vol,
            })

        with torch.no_grad():
            preds = predictor.predict_batch(
                df_list=batch_dfs,
                x_timestamp_list=batch_x_stamps,
                y_timestamp_list=batch_y_stamps,
                pred_len=1,
                T=0.2,
                top_p=0.9,
                sample_count=sample_count,
                verbose=False
            )
            
        for b_idx in range(cur_batch_size):
            pred_df = preds[b_idx]
            pred_close = float(pred_df["close"].iloc[0])
            c_curr = batch_meta[b_idx]["curr_close"]
            
            pred_return = (pred_close - c_curr) / c_curr
            abs_pred_return = abs(pred_return)
            
            roll_std = batch_meta[b_idx]["roll_std"]
            pred_high_vol = 1 if abs_pred_return > roll_std else 0
            
            entry = batch_meta[b_idx]
            entry["pred_close"] = pred_close
            entry["pred_return"] = pred_return
            entry["abs_pred_return"] = abs_pred_return
            entry["pred_high_vol"] = pred_high_vol
            records.append(entry)
            
        done_count = len(records)
        elapsed = time.time() - t0
        print(f"  Processed {done_count}/{total_test_days} days ({done_count/total_test_days*100:.1f}%) | Elapsed: {elapsed:.1f}s", end="\r")
    print()
    return pd.DataFrame(records)

def compute_metrics(df):
    n = len(df)
    
    # MAE of absolute returns
    mae_kronos = (df["abs_pred_return"] - df["abs_real_return"]).abs().mean()
    mae_ewma = (df["ewma_abs"] - df["abs_real_return"]).abs().mean()
    mae_roll = (df["roll_abs"] - df["abs_real_return"]).abs().mean()
    
    # Info coefficient (rank correlation)
    ic_kronos, _ = stats.spearmanr(df["abs_pred_return"], df["abs_real_return"])
    ic_ewma, _ = stats.spearmanr(df["ewma_abs"], df["abs_real_return"])
    
    # Binary Hit Rate (Predicting > 1 Sigma Volatility Spike)
    k_vol_correct = (df["pred_high_vol"] == df["real_high_vol"]).sum()
    k_vol_hit_rate = (k_vol_correct / n) * 100.0
    
    # Majority class baseline (usually 0, i.e., "no spike")
    base_0_hit_rate = (df["real_high_vol"] == 0).mean() * 100.0
    
    return {
        "mae_k": mae_kronos,
        "mae_ewma": mae_ewma,
        "mae_roll": mae_roll,
        "ic_k": ic_kronos,
        "ic_ewma": ic_ewma,
        "hit_k": k_vol_hit_rate,
        "hit_base0": base_0_hit_rate
    }

if __name__ == "__main__":
    data = fetch_nifty_data(range_str="10y")
    run_multi_window_validation(data, context_len=250, batch_size=25, sample_count=5)
