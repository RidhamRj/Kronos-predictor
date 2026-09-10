"""
prediction_tracker.py
---------------------
Persists Kronos AI market predictions and evaluates their accuracy
against realized market candles (Directional Accuracy, MAE error in ₹,
and 90% confidence band coverage).
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")


def _ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_raw_predictions() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[Warning] Could not load predictions.json:", e)
        return []


def save_raw_predictions(preds: List[Dict[str, Any]]):
    _ensure_data_dir()
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)


def save_prediction(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves a newly generated Kronos prediction snapshot.
    """
    preds = load_raw_predictions()
    
    # Avoid duplicate timestamps
    pred_id = prediction_data.get("id") or f"pred_{int(datetime.datetime.now().timestamp() * 1000)}"
    prediction_data["id"] = pred_id
    if "created_at" not in prediction_data:
        prediction_data["created_at"] = datetime.datetime.now().isoformat()

    # Prepend new prediction (newest first)
    # Check if already exists
    filtered = [p for p in preds if p.get("id") != pred_id]
    filtered.insert(0, prediction_data)
    
    # Keep up to 50 predictions
    save_raw_predictions(filtered[:50])
    return prediction_data


def evaluate_prediction_accuracy(pred: Dict[str, Any], candle_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares a prediction's forecast steps against real market candles.
    candle_map is keyed by ISO timestamp or formatted date string.
    """
    forecast_steps = pred.get("forecast", [])
    base_price = pred.get("base_price", pred.get("current_price", 0.0))

    evaluated_steps = []
    correct_directions = 0
    in_band_count = 0
    total_abs_error = 0.0
    total_pct_error = 0.0

    for step_item in forecast_steps:
        t_key = step_item.get("time")
        pred_close = float(step_item.get("close", step_item.get("predicted_close", 0.0)))
        lower_5 = float(step_item.get("lower_5", pred_close * 0.998))
        upper_95 = float(step_item.get("upper_95", pred_close * 1.002))

        # Check if actual candle is available for this timestamp
        actual_candle = candle_map.get(t_key)
        # Also try matching prefix (e.g. YYYY-MM-DDTHH:MM)
        if not actual_candle and t_key:
            short_key = t_key[:16]
            for ck, cv in candle_map.items():
                if ck[:16] == short_key:
                    actual_candle = cv
                    break

        if actual_candle:
            actual_close = float(actual_candle.get("close", 0.0))
            err_inr = actual_close - pred_close
            abs_err_inr = abs(err_inr)
            pct_err = (abs_err_inr / actual_close) * 100.0 if actual_close else 0.0

            pred_dir_up = pred_close >= base_price
            actual_dir_up = actual_close >= base_price
            dir_correct = (pred_dir_up == actual_dir_up)

            in_band = (actual_close >= lower_5 and actual_close <= upper_95)

            if dir_correct:
                correct_directions += 1
            if in_band:
                in_band_count += 1
            total_abs_error += abs_err_inr
            total_pct_error += pct_err

            evaluated_steps.append({
                "step": step_item.get("step", len(evaluated_steps) + 1),
                "time": t_key,
                "time_formatted": step_item.get("time_formatted", ""),
                "predicted_close": pred_close,
                "actual_close": actual_close,
                "lower_5": lower_5,
                "upper_95": upper_95,
                "error_inr": err_inr,
                "abs_error_inr": abs_err_inr,
                "pct_error": pct_err,
                "dir_correct": dir_correct,
                "in_band": in_band,
                "status": "EVALUATED"
            })
        else:
            evaluated_steps.append({
                "step": step_item.get("step", len(evaluated_steps) + 1),
                "time": t_key,
                "time_formatted": step_item.get("time_formatted", ""),
                "predicted_close": pred_close,
                "actual_close": None,
                "lower_5": lower_5,
                "upper_95": upper_95,
                "error_inr": None,
                "abs_error_inr": None,
                "pct_error": None,
                "dir_correct": None,
                "in_band": None,
                "status": "PENDING"
            })

    eval_count = len([s for s in evaluated_steps if s["status"] == "EVALUATED"])
    total_steps = len(forecast_steps)

    directional_accuracy = (correct_directions / eval_count * 100.0) if eval_count > 0 else 0.0
    band_coverage = (in_band_count / eval_count * 100.0) if eval_count > 0 else 0.0
    mae_inr = (total_abs_error / eval_count) if eval_count > 0 else 0.0
    mae_pct = (total_pct_error / eval_count) if eval_count > 0 else 0.0

    is_complete = (eval_count == total_steps and total_steps > 0)
    status_label = "COMPLETED" if is_complete else (f"IN PROGRESS ({eval_count}/{total_steps})" if eval_count > 0 else "PENDING")

    return {
        "id": pred.get("id"),
        "created_at": pred.get("created_at"),
        "created_at_formatted": pred.get("created_at_formatted", pred.get("created_at")),
        "interval": pred.get("interval", "15m"),
        "horizon": total_steps,
        "base_price": base_price,
        "target_price": pred.get("target_price"),
        "signal": pred.get("signal", "BUY / LONG BIAS"),
        "expected_return_pct": pred.get("expected_return_pct", 0.0),
        "steps_evaluated": eval_count,
        "total_steps": total_steps,
        "is_complete": is_complete,
        "status": status_label,
        "metrics": {
            "directional_accuracy_pct": round(directional_accuracy, 1),
            "band_coverage_pct": round(band_coverage, 1),
            "mae_inr": round(mae_inr, 2),
            "mae_pct": round(mae_pct, 3),
            "evaluated_steps": eval_count,
            "total_steps": total_steps,
        },
        "comparison_steps": evaluated_steps,
        "original_forecast": forecast_steps,
    }


def seed_demo_history_if_needed(candles: List[Dict[str, Any]], interval: str = "15m"):
    """
    If no predictions exist, seeds realistic past forecasts from earlier today
    against actual candles so the user can immediately test comparison mode.
    """
    preds = load_raw_predictions()
    if preds:
        return

    if len(candles) < 16:
        return

    # Create a prediction from 6 bars ago
    origin_idx = len(candles) - 7
    origin_candle = candles[origin_idx]
    horizon = 6

    forecast_items = []
    base_p = origin_candle["close"]
    for step_i in range(1, horizon + 1):
        target_candle = candles[origin_idx + step_i]
        c_p = target_candle["close"]
        spread = 45.0 * (step_i ** 0.5)
        forecast_items.append({
            "step": step_i,
            "time": target_candle["time"],
            "time_formatted": target_candle.get("time_formatted", target_candle["time"]),
            "open": target_candle["open"] - 5.0,
            "high": target_candle["high"] + spread * 0.4,
            "low": target_candle["low"] - spread * 0.4,
            "close": c_p + (12.0 if step_i % 2 == 0 else -8.0),
            "volume": target_candle.get("volume", 100),
            "lower_5": c_p - spread,
            "upper_95": c_p + spread,
        })

    target_p = forecast_items[-1]["close"]
    exp_ret = ((target_p - base_p) / base_p) * 100.0

    demo_pred = {
        "id": f"pred_demo_{origin_candle['time'][:16].replace(':', '').replace('-', '')}",
        "created_at": origin_candle["time"],
        "created_at_formatted": origin_candle.get("time_formatted", origin_candle["time"]),
        "interval": interval,
        "horizon": horizon,
        "base_price": base_p,
        "target_price": target_p,
        "expected_return_pct": round(exp_ret, 2),
        "signal": "BUY / LONG BIAS" if exp_ret >= 0 else "BEARISH BIAS",
        "forecast": forecast_items,
        "is_demo": True,
    }

    save_prediction(demo_pred)
    print(f"[Tracker] Seeded benchmark prediction from {demo_pred['created_at_formatted']}")
