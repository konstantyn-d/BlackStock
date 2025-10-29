from __future__ import annotations
from typing import Dict, List, Optional


def _get_rich_console():
    try:
        from rich.console import Console
        from rich.theme import Theme
        theme = Theme({
            "title": "bold cyan",
            "subtitle": "bold white",
            "ok": "bold green",
            "warn": "bold yellow",
            "err": "bold red",
            "metric": "bold magenta",
            "label": "bold white",
            "money": "bold bright_white",
            "weight": "bold bright_cyan",
        })
        return Console(theme=theme)
    except Exception:
        return None


def is_rich_available() -> bool:
    return _get_rich_console() is not None


def render_results(
    tickers: List[str],
    amount: Optional[float],
    portfolios: List[Dict],  # [{name, weights, stats, plan, summary, mc}]
) -> None:
    console = _get_rich_console()
    if console is None:
        # Fallback to plain stdout rendering (rich not installed)
        print("\n=== BlackStock — Рекомендованные портфели ===")
        for p in portfolios:
            print(f"\n[{p['name']}]")
            print("Веса:")
            for t, w in p["weights"].items():
                print(f"  {t:<6} : {w*100:5.2f}%")
            s = p["stats"]
            print(f"Метрики: ann_return={s['return']*100:.2f}%, ann_vol={s['vol']*100:.2f}%, sharpe={s['sharpe']:.3f}")
            if p.get("plan"):
                print("План покупок:")
                for t in p["weights"].keys():
                    row = p["plan"][t]
                    print(f"  {t:<6} @ {row['price']:>8.2f} → {row['shares']:>10} шт.  ≈ {row['spend']:>10.2f}  fee {row.get('fee',0):>6.2f}  total {row.get('total',row['spend']):>10.2f}")
                summ = p["summary"]
                print(f"Итого к покупке: {summ['total']:,.2f}  (спенд {summ['total_spend']:,.2f} + комиссии {summ['total_fee']:,.2f})")
                print(f"Остаток кэша  : {summ['leftover']:,.2f}")
            if p.get("mc"):
                mc = p["mc"]
                print(f"Монте-Карло: P5/P50/P95 = {mc['p5']:.2f}/{mc['p50']:.2f}/{mc['p95']:.2f}; MDD≈{mc['mdd_med']:.1%}; P(<start)≈{mc['prob_below_1']:.1%}")
        return

    # Rich-based rendering
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns

    console.clear()
    console.rule(Text("BlackStock — Recommended Portfolios", style="title"))

    for p in portfolios:
        header = Text(p["name"], style="subtitle")

        # Weights table
        tbl_w = Table(title="Weights", expand=True)
        tbl_w.add_column("Ticker", style="label", no_wrap=True)
        tbl_w.add_column("Weight", style="weight", justify="right", no_wrap=True)
        for t, w in p["weights"].items():
            tbl_w.add_row(t, f"[bold]{w*100:.2f}%[/bold]")

        # Metrics table
        s = p["stats"]
        tbl_m = Table(title="Metrics", expand=True)
        tbl_m.add_column("ann_return", style="metric", justify="right")
        tbl_m.add_column("ann_vol", style="metric", justify="right")
        tbl_m.add_column("sharpe", style="metric", justify="right")
        tbl_m.add_row(f"[bold]{s['return']*100:.2f}%[/bold]", f"[bold]{s['vol']*100:.2f}%[/bold]", f"[bold]{s['sharpe']:.3f}[/bold]")

        top = Columns([tbl_w, tbl_m])
        blocks = [top]

        # Plan table (optional)
        if p.get("plan"):
            tbl_p = Table(title="Purchase plan", expand=True)
            tbl_p.add_column("Ticker", style="label", no_wrap=True)
            tbl_p.add_column("Price", justify="right")
            tbl_p.add_column("Shares", justify="right")
            tbl_p.add_column("Spend", justify="right")
            tbl_p.add_column("Fee", justify="right")
            tbl_p.add_column("Total", justify="right")
            for t in p["weights"].keys():
                row = p["plan"][t]
                shares_fmt = f"{row['shares']:.4f}" if isinstance(row['shares'], float) and (row['shares'] % 1 != 0) else f"{int(row['shares'])}"
                tbl_p.add_row(
                    t,
                    f"{row['price']:.2f}",
                    f"[bold]{shares_fmt}[/bold]",
                    f"{row['spend']:.2f}",
                    f"{row.get('fee',0):.2f}",
                    f"[bold]{row.get('total',row['spend']):.2f}[/bold]",
                )
            summ = p["summary"]
            tbl_p.add_row("", "", "", "", "", "")
            tbl_p.add_row("", "", "", "", "Total:", f"[bold]{summ['total']:.2f}[/bold]")
            tbl_p.add_row("", "", "", "", "Leftover:", f"[bold]{summ['leftover']:.2f}[/bold]")
            blocks.append(tbl_p)

        # MC summary (optional)
        if p.get("mc"):
            mc = p["mc"]
            mc_text = Text(
                f"P5/P50/P95 = {mc['p5']:.2f}/{mc['p50']:.2f}/{mc['p95']:.2f}  •  "
                f"MDD≈{mc['mdd_med']:.1%}  •  P(<start)≈{mc['prob_below_1']:.1%}",
                style="ok",
            )
            blocks.append(Panel(Align.left(mc_text), title="Monte Carlo", border_style="green"))

        panel = Panel.fit(
            Align.left(Columns(blocks)),
            title=header,
            border_style="cyan",
        )
        console.print(panel)
    console.rule()


