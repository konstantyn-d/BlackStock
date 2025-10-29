from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

# Absolute imports from the local project structure
from config import CONFIG
from utils.logging import setup_logging, get_logger
from data.loader import load_prices
from data.pricing import last_close_from_window, live_last_prices
from portfolio.allocation import split_cash, to_share_plan, summarize_plan, apply_fees, residual_allocate
from utils.export import plan_to_df, export_plan
from utils.pdf_export import export_analysis_pdf
from portfolio.metrics import to_log_returns, annualize_returns_daily, annualize_cov_daily, portfolio_stats
from portfolio.optimizer import min_variance, max_sharpe, target_volatility
from ml.predictor import MeanReturnPredictor
from risk.monte_carlo import simulate_gbm, simulate_bootstrap, mc_summary
from utils.console import render_results, is_rich_available

log = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="BlackStock",
        description="Portfolio Intelligence Engine (console MVP)"
    )
    p.add_argument("--tickers", type=str, required=False, default=None,
                   help="Список тикеров через запятую (например: AAPL,MSFT,GOOGL)")
    p.add_argument("--start", type=str, default=CONFIG.default_start,
                   help=f"Дата начала (YYYY-MM-DD), default {CONFIG.default_start}")
    p.add_argument("--end", type=str, default=CONFIG.default_end,
                   help=f"Дата конца (YYYY-MM-DD), default {CONFIG.default_end} (=по сегодня)")
    p.add_argument("--rf", type=float, default=CONFIG.default_risk_free,
                   help=f"Годовая безрисковая ставка, default {CONFIG.default_risk_free}")
    p.add_argument("--target-vol", type=float, default=None,
                   help="Целевая волатильность для сбалансированного портфеля, например 0.15 (15%)")
    p.add_argument("--no-cache", action="store_true",
                   help="Не использовать кэш (перекачать данные)")
    p.add_argument("--amount", type=float, required=False,
                   help="Сумма инвестиций (например, 10000)")
    p.add_argument("--use-last", action="store_true",
                   help="Взять рыночную последнюю цену (иначе last close из окна)")
    p.add_argument("--fractional", action="store_true",
                   help="Разрешить дробные акции")
    p.add_argument("--cash-step", type=float, default=0.01,
                   help="Шаг валюты для округления")
    p.add_argument("--min-shares", type=int, default=1,
                   help="Минимальный лот (обычно 1)")
    p.add_argument("--residual", type=str, default="topweight", choices=["topweight", "min_error"],
                   help="Стратегия раздачи остатка для целочисленных акций")
    p.add_argument("--fee-bps", type=float, default=0.0,
                   help="Комиссия в bps (0.10% = 10)")
    p.add_argument("--min-fee", type=float, default=0.0,
                   help="Минимальная комиссия на тикер")
    p.add_argument("--export", type=str, default=None,
                   help="Путь для экспорта CSV (или .xlsx)")
    p.add_argument("--export-pdf", type=str, default=None,
                   help="Путь для экспорта PDF-отчёта")
    p.add_argument("--plan-only", action="store_true",
                   help="Печатать только план покупок без метрик")

    p.add_argument("--mc-years", type=int, default=0,
                   help="Лет для Монте-Карло (0 = выкл)")
    p.add_argument("--mc-n", type=int, default=10000,
                   help="Число симуляций для Монте-Карло")
    p.add_argument("--mc-method", type=str, default="gbm", choices=["gbm", "bootstrap"],
                   help="Метод Монте-Карло")

    return p.parse_args()


def pretty_weights(names, w):
    return {n: round(float(x), 4) for n, x in zip(names, w)}

def main():
    setup_logging()
    args = parse_args()

    # Interactive input flow when CLI args are not provided
    if args.tickers is None or not args.tickers.strip():
        try:
            entered = input("Enter tickers comma-separated (min 2): ").strip()
        except Exception:
            entered = ""
        tickers = [t.strip().upper() for t in entered.split(",") if t.strip()]
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        raise ValueError("Нужно минимум 2 тикера.")

    log.info(f"Тикеры: {tickers}")
    prices = load_prices(tickers, args.start, args.end, use_cache=not args.no_cache)

    rets = to_log_returns(prices)  # daily log returns
    mu_daily = rets.mean().values
    cov_daily = rets.cov().values

    mu = annualize_returns_daily(mu_daily)      # annualized mean returns
    cov = annualize_cov_daily(cov_daily)        # annualized covariance

    # ML placeholder: use historical means for prediction for now
    ml = MeanReturnPredictor().fit(None, None)
    mu_pred = ml.predict(mu)

    # Portfolio optimizations
    w_minv = min_variance(mu_pred, cov)
    w_shrp = max_sharpe(mu_pred, cov, rf=args.rf)

    if args.target_vol is not None:
        w_bal = target_volatility(mu_pred, cov, target_vol=args.target_vol, rf=args.rf)
    else:
        # default: average between min-var and max-sharpe, then project onto simplex
        w_bal = (w_minv + w_shrp) / 2
        w_bal = np.maximum(w_bal, 0)
        w_bal = w_bal / w_bal.sum()

    # Metrics helper: compute (return, vol, sharpe)
    def stats(w):
        r, v, s = portfolio_stats(w, mu_pred, cov, rf=args.rf)
        return r, v, s

    # Percent printer for consistent formatting
    def fmt_pct(x: float, digits: int = 2) -> str:
        return f"{x*100:.{digits}f}%"

    # Console output header (rich renderer will override layout later)
    print("\n=== BlackStock — Рекомендованные портфели ===")

    def as_weights_dict(names, w):
        return {n: float(x) for n, x in zip(names, w)}

    blocks = [
        ("Conservative (min-vol)", w_minv),
        ("Balanced", w_bal),
        ("Aggressive (max-sharpe)", w_shrp),
    ]

    # Price source to value purchase plans
    amount = args.amount
    if amount is None:
        try:
            entered_amount = input("Enter investment amount (blank = no plan): ").strip()
            if entered_amount:
                amount = float(entered_amount)
        except Exception:
            amount = None

    # Ask for fractional shares if not specified explicitly
    fractional = args.fractional
    if amount is not None and amount > 0 and not args.fractional:
        try:
            ans_frac = input("Allow fractional shares? (y/N): ").strip().lower()
            fractional = (ans_frac == "y")
        except Exception:
            pass
    if args.use_last:
        last_prices = live_last_prices(tickers)
        if last_prices.isna().any():
            last_prices = last_close_from_window(prices)
    else:
        last_prices = last_close_from_window(prices)

    portfolio_results = []
    for name, w in blocks:
        w_dict = as_weights_dict(tickers, w)
        r, v, s = stats(w)
        stats_dict = {"return": r, "vol": v, "sharpe": s}
        # When rich is available, suppress plain text to avoid duplicated output
        if (not args.plan_only) and (not is_rich_available()):
            print(f"\n[{name}]")
            print("Веса:")
            cash_inline = None
            if amount is not None and amount > 0:
                cash_inline = split_cash(amount, w_dict, step=args.cash_step)
            for t, ww in w_dict.items():
                if cash_inline is not None:
                    print(f"  {t:<6} : {fmt_pct(ww, 2)} ({cash_inline[t]:,.2f})")
                else:
                    print(f"  {t:<6} : {fmt_pct(ww, 2)}")
            print(f"Метрики: ann_return={fmt_pct(r,2)}, ann_vol={fmt_pct(v,2)}, sharpe={s:.3f}")

        plan = None
        summary = None
        if amount is not None and amount > 0:
            cash = split_cash(amount, w_dict, step=args.cash_step)
            plan = to_share_plan(cash, last_prices, fractional=fractional, min_shares=args.min_shares)
            plan = residual_allocate(plan, w_dict, amount, strategy=args.residual, last_prices=last_prices,
                                     fractional=fractional, min_shares=args.min_shares)
            plan = apply_fees(plan, fee_bps=args.fee_bps, min_fee=args.min_fee)
            summary = summarize_plan(plan, amount)
            if not is_rich_available():
                print("План покупок:")
                for t in w_dict.keys():
                    row = plan[t]
                    if fractional:
                        shares_str = f"{row['shares']:>8.4f}"
                    else:
                        shares_str = f"{int(row['shares']):>8d}"
                    print(f"  {t:<6} @ {row['price']:>8.2f} → {shares_str} шт.  ≈ {row['spend']:>10.2f}  fee {row.get('fee',0):>6.2f}  total {row.get('total',row['spend']):>10.2f}")
                print(f"Итого к покупке: {summary['total']:,.2f}  (спенд {summary['total_spend']:,.2f} + комиссии {summary['total_fee']:,.2f})")
                print(f"Остаток кэша  : {summary['leftover']:,.2f}")

            if args.export:
                df = plan_to_df(w_dict, plan)
                export_plan(df, args.export)
                print(f"Экспортирован план → {args.export}")

        # Monte Carlo (optional)
        mc_out = None
        if args.mc_years and args.mc_years > 0:
            ret_hist = prices.pct_change().dropna()
            # portfolio daily returns from history
            port_daily = ret_hist.mul([w_dict[t] for t in ret_hist.columns], axis=1).sum(axis=1)
            if args.mc_method == "gbm":
                paths = simulate_gbm(port_daily, years=args.mc_years, n_paths=args.mc_n, dt_per_year=252, seed=42)
            else:
                paths = simulate_bootstrap(w_dict, ret_hist, years=args.mc_years, n_paths=args.mc_n, dt_per_year=252, block=21, seed=42)
            summary_mc = mc_summary(paths)
            mc_out = summary_mc
            print(f"Монте-Карло — {name} ({args.mc_years} лет, {args.mc_n} путей, {args.mc_method}): P5/P50/P95 = {summary_mc['p5']:.2f}/{summary_mc['p50']:.2f}/{summary_mc['p95']:.2f}; MDD≈{summary_mc['mdd_med']:.1%}; P(capital<start)≈{summary_mc['prob_below_1']:.1%}")

        portfolio_results.append({
            "name": name,
            "weights": w_dict,
            "stats": stats_dict,
            "plan": plan or {},
            "summary": summary or {"total": 0.0, "leftover": 0.0, "total_spend": 0.0, "total_fee": 0.0},
            "mc": mc_out,
        })

    # Render final results via rich (with fallback to plain output above)
    try:
        render_results(tickers, amount, portfolio_results)
    except Exception:
        # Fallback to plain output printed above (or quietly fail)
        pass

    # Interactive export prompts when flags are not provided
    if args.export is None and amount:
        try:
            want_csv = input("Export plan to CSV/XLSX? (y/N): ").strip().lower()
            if want_csv == "y":
                path = input("Enter file path (.csv or .xlsx): ").strip() or "plan.csv"
                # Export per-portfolio files for simplicity (suffix by portfolio name)
                for res in portfolio_results:
                    if res["plan"]:
                        df = plan_to_df(res["weights"], res["plan"])
                        suffix = res["name"].replace(" ", "_")
                        export_plan(df, path.replace(".csv", f"_{suffix}.csv").replace(".xlsx", f"_{suffix}.xlsx"))
                print("CSV/XLSX export done.")
        except Exception:
            pass

    if args.export_pdf is None:
        try:
            want_pdf = input("Create PDF report? (y/N): ").strip().lower()
            if want_pdf == "y":
                pdf_path = input("Enter PDF path (default blackstock_report.pdf): ").strip() or "blackstock_report.pdf"
                try:
                    export_analysis_pdf(pdf_path, tickers, portfolio_results, meta={"period": f"{args.start}..{args.end}", "rf": args.rf})
                    print(f"PDF report saved → {pdf_path}")
                except ImportError as e:
                    print(f"[WARN] PDF not created: {e}")
        except Exception:
            pass
    else:
        try:
            export_analysis_pdf(args.export_pdf, tickers, portfolio_results, meta={"period": f"{args.start}..{args.end}", "rf": args.rf})
            print(f"PDF report saved → {args.export_pdf}")
        except ImportError as e:
            print(f"[WARN] PDF not created: {e}")

if __name__ == "__main__":
    main()
