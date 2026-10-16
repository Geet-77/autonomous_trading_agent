"""
Neural Nexus -- Round 2: The Trading Pit
agent.py starter template

Fill in predict_market() and place_orders(). This file already runs
end-to-end with a dummy (non-learning) strategy so you always have
something submittable -- replace the TODO sections with your real model.

IMPORTANT: both functions are called FRESH each round, in a new process.
Nothing stored in a global variable or on `self` will survive between
calls. Anything you need must come from the arguments, or be loaded from
a file (e.g. joblib.load("my_model.pkl")) at the top of this file.
"""

import os
import joblib  # if you saved a trained model with joblib.dump(...)
import numpy as np

ASSETS = ["quantum_dynamics", "byte_stream", "gold_trust",
          "metro_rail", "agro_futures", "solar_grid"]

# TODO: load your trained model once, at import time (NOT inside the
# functions below -- this keeps per-round execution fast).
#
# IMPORTANT: use a path relative to THIS FILE, not a bare filename. The
# real engine imports your agent.py from a different working directory
# than your own folder, so a bare "my_model.pkl" will work when you test
# locally (local_test_harness.py happens to run from your folder) but will
# FAIL SILENTLY on the real engine. Always do it this way:

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
try:
    MODELS = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model.pkl. Did you run train_model.py? ({e})")
    MODELS = None


# ---- helpers added for my model (feature engineering must match train_model.py) ----
PARTNER = {"quantum_dynamics": "byte_stream", "byte_stream": "quantum_dynamics",
           "gold_trust": "solar_grid", "solar_grid": "gold_trust",
           "metro_rail": "agro_futures", "agro_futures": "metro_rail"}
WIN = 20


def clean_window(prices, n=WIN):
    """Drop NaNs, repair x10 / x5 / x0.2 / x0.1 glitches, pad to n values."""
    p = [float(x) for x in (prices or []) if x is not None and np.isfinite(x) and x > 0]
    if not p:
        return [40.0] * n
    med = float(np.median(p))
    fixed = []
    for x in p:
        r = x / med
        for k in (10.0, 5.0):
            if 0.9 * k < r < 1.1 * k:
                x = x / k
            elif 0.9 / k < r < 1.1 / k:
                x = x * k
        fixed.append(x)
    p = fixed[-n:]
    return [p[0]] * (n - len(p)) + p


def features(w, q):
    w = np.asarray(w, float); q = np.asarray(q, float)
    l, ql = w[-1] + 1e-9, q[-1] + 1e-9
    d, dq = np.diff(w), np.diff(q)
    return [d[-1] / l, d[-2] / l, d[-3] / l, d[-5:].mean() / l, d[-10:].mean() / l,
            (l - w.mean()) / l, (l - w[-5:].mean()) / l, d.std() / l,
            dq[-1] / ql, dq[-3:].mean() / ql, (ql - q.mean()) / ql]


def _rel_move(asset, hist):
    w = clean_window(hist.get(asset))
    q = clean_window(hist.get(PARTNER[asset]))
    x = np.array(features(w, q)).reshape(1, -1)
    return float(MODELS[asset].predict(x)[0]), w


def predict_market(recent_history):
    """
    recent_history: dict[asset_id -> list of up to the last 20 REALIZED
                    prices for that asset, oldest first, most recent last]

    Returns your predicted PRICE for the UPCOMING round, per asset, e.g.:
      {"quantum_dynamics": 41.2, "byte_stream": 77.5, "gold_trust": 33.0,
       "metro_rail": 22.4, "agro_futures": 28.1, "solar_grid": 39.6}

    Scored OFFLINE against a hidden validation set you never see. It is NOT
    told what other teams are doing -- it's a pure forecasting task.
    """
    predictions = {}
    for asset in ASSETS:
        # On the real history "no change" beat every model on forecast error,
        # so predict the (glitch-repaired) last price.
        try:
            predictions[asset] = float(clean_window(recent_history.get(asset))[-1])
        except Exception:
            predictions[asset] = 40.0

    return predictions


def place_orders(game_state, team_history):
    """
    game_state: {
      "round": 47, "total_rounds": 120, "trade_budget": 14,
      "assets": {
        "quantum_dynamics": {"ticker": "QNTM",
                              "recent_prices": [...up to last 20...],
                              "recent_buy_volume":  [...total BUY lots from ALL teams, same rounds...],
                              "recent_sell_volume": [...total SELL lots from ALL teams, same rounds...]},
        ... (all 6 assets)
      },
      "your_score_so_far": 312.5,
    }
    
    team_history: list of your own past rounds:
      [{"round": 46, "orders": {...}, "profit": 41.0}, ...]

    Returns your lot allocation for THIS round -- SIGNED integers, one per
    asset (positive = buy, negative = short, zero = sit out), where the
    SUM OF ABSOLUTE VALUES is at most game_state["trade_budget"]:
      {"quantum_dynamics": 4, "byte_stream": -3, "gold_trust": 0,
       "metro_rail": 5, "agro_futures": -2, "solar_grid": 0}
      (here: total exposure = 4+3+0+5+2+0 = 14, right at budget)
    """
    zero = {asset: 0 for asset in ASSETS}
    try:
        budget = int(game_state["trade_budget"])
        hist = {asset: game_state["assets"][asset]["recent_prices"] for asset in ASSETS}
        scores = {}
        for asset in ASSETS:
            rel, w = _rel_move(asset, hist)
            vol = np.std(np.diff(w)) / (w[-1] + 1e-9) + 1e-6
            z = rel / vol                                  # confidence: move vs normal wobble
            info = game_state["assets"][asset]
            b = float((info.get("recent_buy_volume") or [0])[-1])
            s = float((info.get("recent_sell_volume") or [0])[-1])
            tot = b + s
            if tot > 0:                                    # share of last round's flow on MY side
                share = (b if z > 0 else s) / tot
                z *= (1.4 - 0.8 * share)                   # shrink bets on a crowded side
            scores[asset] = z
        ranked = sorted(ASSETS, key=lambda x: -abs(scores[x]))
        top = [x for x in ranked[:3] if abs(scores[x]) > 1e-9]
        if not top:
            return zero
        tot = sum(abs(scores[x]) for x in top)
        orders = dict(zero)
        for x in top:
            lots = int(budget * abs(scores[x]) / tot)
            orders[x] = lots if scores[x] > 0 else -lots
        left = budget - sum(abs(v) for v in orders.values())   # leftover lots to strongest first
        for x in top:
            if left <= 0:
                break
            orders[x] += 1 if scores[x] > 0 else -1
            left -= 1
        while sum(abs(v) for v in orders.values()) > budget:
            k = max(orders, key=lambda x: abs(orders[x]))
            orders[k] -= 1 if orders[k] > 0 else -1
        return {x: int(v) for x, v in orders.items()}
    except Exception:
        return zero


if __name__ == "__main__":
    # Quick manual sanity check -- run this file directly to eyeball output.
    fake_history = {a: [40, 42, 39, 41, 43] for a in ASSETS}
    print("predict_market ->", predict_market(fake_history))

    fake_game_state = {
        "round": 5, "total_rounds": 120, "trade_budget": 14,
        "assets": {a: {"ticker": a[:4].upper(), "recent_prices": [40, 42, 39, 41, 43],
                        "recent_buy_volume": [20, 22, 19, 21, 23],
                        "recent_sell_volume": [18, 17, 20, 19, 16]} for a in ASSETS},
        "your_score_so_far": 0.0,
    }
    print("place_orders  ->", place_orders(fake_game_state, []))
