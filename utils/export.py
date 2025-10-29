from __future__ import annotations
import pandas as pd
from typing import Dict


def plan_to_df(weights: Dict[str, float], plan: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for t, w in weights.items():
        row = plan[t]
        rows.append({
            "ticker": t,
            "target_weight": w,
            "price": row.get("price"),
            "shares": row.get("shares"),
            "spend": row.get("spend"),
            "fee": row.get("fee", 0.0),
            "total": row.get("total", row.get("spend")),
        })
    df = pd.DataFrame(rows)
    return df


def export_plan(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
