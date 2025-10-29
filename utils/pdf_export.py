from __future__ import annotations
from typing import Dict, List, Optional


def _ascii(text: str) -> str:
    try:
        return text.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return "".join(ch for ch in text if ord(ch) < 128)


def _draw_wrapped(c, text: str, x: float, y: float, max_width: float, line_height: float = 14):
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test) <= max_width:
            line = test
        else:
            c.drawString(x, y, _ascii(line))
            y -= line_height
            line = w
    if line:
        c.drawString(x, y, _ascii(line))
        y -= line_height
    return y


def export_analysis_pdf(
    path: str,
    tickers: List[str],
    portfolios: List[Dict],  # [{name, weights, stats, plan, summary, mc(optional)}]
    meta: Optional[Dict] = None,
) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib import colors
    except Exception as e:
        raise ImportError("ReportLab is not installed. Install 'reportlab' or run: pip install -r requirements.txt") from e

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    x0 = margin
    y = height - margin

    title = "BlackStock - Portfolio Intelligence Engine (Report)"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x0, y, _ascii(title))
    y -= 18

    c.setFont("Helvetica", 10)
    y = _draw_wrapped(c, _ascii(f"Tickers: {', '.join(tickers)}"), x0, y, width - 2 * margin)
    if meta:
        if meta.get("period"):
            y = _draw_wrapped(c, _ascii(f"Period: {meta['period']}"), x0, y, width - 2 * margin)
        if meta.get("rf") is not None:
            y = _draw_wrapped(c, _ascii(f"Risk-free rate: {meta['rf']:.2%}"), x0, y, width - 2 * margin)

    y -= 6
    c.setStrokeColor(colors.black)
    c.line(x0, y, width - margin, y)
    y -= 14

    for p in portfolios:
        name = _ascii(p.get("name", "Portfolio"))
        weights: Dict[str, float] = p.get("weights", {})
        stats = p.get("stats", {})
        plan: Dict[str, dict] = p.get("plan", {})
        summary: Dict[str, float] = p.get("summary", {})
        mc: Optional[Dict] = p.get("mc")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x0, y, name)
        y -= 16

        c.setFont("Helvetica", 10)
        # Weights + stats line
        ws = ", ".join([f"{t}: {w*100:.2f}%" for t, w in weights.items()])
        y = _draw_wrapped(c, _ascii(f"Weights: {ws}"), x0, y, width - 2 * margin)
        y = _draw_wrapped(c, _ascii(f"Metrics: Return {stats.get('return',0):.2%}; Volatility {stats.get('vol',0):.2%}; Sharpe {stats.get('sharpe',0):.3f}"), x0, y, width - 2 * margin)

        # Plan table header
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x0, y, _ascii("Purchase plan:"))
        y -= 14
        c.setFont("Helvetica", 9)
        c.drawString(x0 + 0*cm, y, _ascii("Ticker"))
        c.drawString(x0 + 3*cm, y, _ascii("Price"))
        c.drawString(x0 + 6*cm, y, _ascii("Shares"))
        c.drawString(x0 + 9*cm, y, _ascii("Spend"))
        c.drawString(x0 + 12*cm, y, _ascii("Fee"))
        c.drawString(x0 + 15*cm, y, _ascii("Total"))
        y -= 12
        for t, w in weights.items():
            row = plan.get(t, {})
            c.drawString(x0 + 0*cm, y, f"{t}")
            c.drawRightString(x0 + 5.5*cm, y, f"{row.get('price', 0):.2f}")
            c.drawRightString(x0 + 8.5*cm, y, f"{row.get('shares', 0)}")
            c.drawRightString(x0 + 11.5*cm, y, f"{row.get('spend', 0):.2f}")
            c.drawRightString(x0 + 14.5*cm, y, f"{row.get('fee', 0):.2f}")
            c.drawRightString(x0 + 18*cm - margin, y, f"{row.get('total', row.get('spend', 0)):.2f}")
            y -= 12
            if y < margin + 120:
                c.showPage(); width, height = A4; y = height - margin; x0 = margin
        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(x0 + 18*cm - margin, y, _ascii(f"Total: {summary.get('total', 0):.2f}"))
        y -= 12
        c.setFont("Helvetica", 10)
        c.drawRightString(x0 + 18*cm - margin, y, _ascii(f"Cash leftover: {summary.get('leftover', 0):.2f}"))
        y -= 14

        if mc:
            c.setFont("Helvetica", 10)
            y = _draw_wrapped(c,
                _ascii(f"Monte Carlo: P5/P50/P95 = {mc['p5']:.2f}/{mc['p50']:.2f}/{mc['p95']:.2f}; MDD~{mc['mdd_med']:.1%}; P(< start)~{mc['prob_below_1']:.1%}"),
                x0, y, width - 2 * margin)
            y -= 6

        c.setStrokeColor(colors.grey)
        c.line(x0, y, width - margin, y)
        y -= 16
        if y < margin + 140:
            c.showPage(); width, height = A4; y = height - margin; x0 = margin

    c.save()


