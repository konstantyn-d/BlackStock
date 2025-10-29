# 📈 BlackStock — Portfolio Intelligence Engine (CLI MVP)

BlackStock turns a simple list of tickers into three institution-grade portfolios — Conservative, Balanced, and Risk-Seeking — with clear metrics, an executable purchase plan (incl. fractional shares), and optional Monte Carlo risk scenarios.
Launch via cli.py and press Start — no terminal commands needed.


# ✨ Why BlackStock

Most tools show charts. BlackStock delivers decisions: target weights, expected risk/return, and exact quantities to buy. It’s fast, explainable, and reproducible — the kind of rigor hiring managers appreciate.


# 🧮 Methodology (in plain English)

**Modern Portfolio Theory (Markowitz)**

- Estimate daily returns & covariances from adjusted prices (Yahoo Finance), annualize.
- Solve three long-only, fully invested problems:

    - Min-Vol — minimize 
    - Max-Sharpe — maximize 
    - Balanced — target-volatility point on the efficient frontier.

**Monte Carlo (optional)**

  - Simulate thousands of future paths (GBM or historical bootstrap).

  - Report P5/P50/P95, median max drawdown, and probability of finishing below start.
    Translation: not just “average return,” but realistic scenarios.

**Purchase Plan**

  - Convert weights → cash split → share quantities (fractional or whole).
  - Export PDF/XLSX/CSV for audit and execution.


# 🎯 What you get today

- Three optimized portfolios from clean inputs (tickers, dates, amount).
- Transparent metrics (annual return, volatility, Sharpe) and an actionable purchase plan.
- Optional scenario analysis via Monte Carlo.
- One-click run: open cli.py, press Start.

# 🗺️ Roadmap

- Desktop App (WPF/Streamlit): interactive sliders, rebalancing, report builder.
- Mobile App: on-the-go “what-if” checks.
- Robust covariance (Ledoit–Wolf), sector/asset caps, tolerance bands, walk-forward validation.
- Commission & tax modeling (calibration coming next).
- Optional ML overlays (Ridge/LightGBM) to inform μ/σ/regime estimates for mixed-scenario Monte Carlo.

# 🧱 Tech Snapshot

- Python
- NumPy/Pandas/SciPy
- Yahoo Finance (yfinance)
- ReportLab (PDF)
- Clean CLI entrypoint (cli.py).

# ⚖️ Status & Note

- Core algorithm: productionized for CLI, actively evolving.
- Commission calibration: not included yet; planned in upcoming updates.
- This project is educational/analytical and not investment advice.

**BlackStock turns financial theory into a disciplined plan you can execute — fast, explainable, and scenario-aware.**
