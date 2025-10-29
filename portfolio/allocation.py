from __future__ import annotations
from math import floor
from typing import Dict
import pandas as pd


def _round_cash(x: float, step: float = 0.01) -> float:
    return round(round(x / step) * step, 2)


def split_cash(amount: float, weights: Dict[str, float], step: float = 0.01) -> Dict[str, float]:
    """
    Allocate cash across tickers according to target weights with currency-step rounding.
    Any rounding remainder is assigned to the ticker with the largest target weight.
    """
    raw = {t: amount * w for t, w in weights.items()}
    cash = {t: _round_cash(v, step) for t, v in raw.items()}
    delta = round(amount - sum(cash.values()), 2)
    if abs(delta) >= step / 2:
        top = max(weights, key=weights.get)
        cash[top] = _round_cash(cash[top] + delta, step)
    return cash


def _shares_for_cash(cash: float, price: float, fractional: bool, min_shares: int) -> float:
    if fractional:
        return round(cash / price, 4)
    whole = floor(cash / price)
    return max(0, (whole // min_shares) * min_shares)


def to_share_plan(
    cash: Dict[str, float],
    last_prices: pd.Series,
    fractional: bool = False,
    min_shares: int = 1,
) -> Dict[str, dict]:
    plan: Dict[str, dict] = {}
    for t, c in cash.items():
        p = float(last_prices[t])
        if p <= 0 or not pd.notna(p):
            raise ValueError(f"Invalid price for {t}: {p}")
        q = _shares_for_cash(c, p, fractional, min_shares)
        spend = round(q * p, 2)
        plan[t] = {"price": p, "shares": q, "spend": spend}
    return plan


def apply_fees(plan: Dict[str, dict], fee_bps: float = 0.0, min_fee: float = 0.0) -> Dict[str, dict]:
    for t, row in plan.items():
        spend = row["spend"]
        fee = max(min_fee, round(spend * fee_bps / 10000.0, 2)) if spend > 0 else 0.0
        row["fee"] = fee
        row["total"] = round(spend + fee, 2)
    return plan


def residual_allocate(
    plan: Dict[str, dict],
    weights: Dict[str, float],
    amount: float,
    strategy: str,
    last_prices: pd.Series,
    fractional: bool,
    min_shares: int,
):
    if fractional:
        # No residual pass needed when fractional shares are enabled.
        return plan

    def try_buy(ticker: str) -> bool:
        p = float(last_prices[ticker])
        q_new = plan[ticker]["shares"] + min_shares
        add_cost = round(min_shares * p, 2)
        budget_left = round(amount - sum(x["spend"] for x in plan.values()), 2)
        if add_cost <= budget_left + 1e-6:
            plan[ticker]["shares"] = q_new
            plan[ticker]["spend"] = round(q_new * p, 2)
            return True
        return False

    changed = True
    while changed:
        changed = False
        if strategy == "topweight":
            t = max(weights, key=weights.get)
            changed = try_buy(t)
        else:
            total_spend = sum(p["spend"] for p in plan.values()) or 1.0
            actual = {t: plan[t]["spend"] / total_spend for t in plan}
            diff = {t: weights[t] - actual.get(t, 0.0) for t in plan}
            t = max(diff, key=diff.get)
            if diff[t] > 0:
                changed = try_buy(t)
    return plan


def summarize_plan(plan: Dict[str, dict], amount: float) -> dict:
    total_spend = round(sum(x["spend"] for x in plan.values()), 2)
    total_fee = round(sum(x.get("fee", 0.0) for x in plan.values()), 2)
    total = round(total_spend + total_fee, 2)
    leftover = round(amount - total, 2)
    return {"total_spend": total_spend, "total_fee": total_fee, "total": total, "leftover": leftover}


