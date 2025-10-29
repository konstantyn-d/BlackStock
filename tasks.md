* ** Реализуй инпуты, чтобы пользователь сам мог вводть тикеры "выбирать какие акции ему нужны и какие анализировать" *

* ** Реализуй вывод результатов в % а не в "0.2988, 0.0456 и тд" *

* ** добавляем в консольный MVP «план покупок»: по введённой сумме рассчитываем деньги по тикерам, количество акций, общую сумму и остаток кэша. Ниже — точечные изменения, которые можно просто вставить. 

##portfolio/allocation.py


from __future__ import annotations
from math import floor
from typing import Dict
import pandas as pd

def _round_cash(x: float, step: float = 0.01) -> float:
    # округление до шага валюты (0.01 для EUR/USD)
    return round(round(x / step) * step, 2)

def split_cash(amount: float, weights: Dict[str, float], step: float = 0.01) -> Dict[str, float]:
    """
    Делим сумму по весам с аккуратным округлением до денежного шага.
    Остаток после округления докидываем в тикер с максимальным весом.
    """
    raw = {t: amount * w for t, w in weights.items()}
    cash = {t: _round_cash(v, step) for t, v in raw.items()}
    # Подтягиваем до точной суммы (± шаг)
    delta = round(amount - sum(cash.values()), 2)
    if abs(delta) >= step / 2:
        # Наибольший вес получает корректировку
        top = max(weights, key=weights.get)
        cash[top] = _round_cash(cash[top] + delta, step)
    return cash

def to_share_plan(
    cash: Dict[str, float],
    last_prices: pd.Series,
    fractional: bool = False,
    min_shares: int = 1,
) -> Dict[str, dict]:
    """
    Считает количество акций по каждому тикеру.
    Возвращает dict: {ticker: {"price": p, "shares": q, "spend": s}}
    """
    plan: Dict[str, dict] = {}
    for t, c in cash.items():
        price = float(last_prices[t])
        if price <= 0:
            raise ValueError(f"Invalid price for {t}: {price}")

        if fractional:
            # Дробные акции (до 4 знаков)
            shares = round(c / price, 4)
        else:
            # Только целые акции
            shares = max(0, floor(c / price // min_shares) * min_shares)

        spend = round(shares * price, 2)
        plan[t] = {"price": price, "shares": shares, "spend": spend}
    return plan

def summarize_plan(plan: Dict[str, dict], amount: float) -> dict:
    total_spend = round(sum(x["spend"] for x in plan.values()), 2)
    leftover = round(amount - total_spend, 2)
    return {"total_spend": total_spend, "leftover": leftover}


*



** * 2) Обновления cli.py

Добавим аргументы --amount, --fractional, --cash-step, --min-shares

После расчёта весов получим последние цены из prices.iloc[-1]

Сформируем план покупок для каждого из трёх портфелей

Добавь импорты:

from portfolio.allocation import split_cash, to_share_plan, summarize_plan


Расширь парсер аргументов:

p.add_argument("--amount", type=float, required=True,
               help="Сумма инвестиций (например, 10000)")
p.add_argument("--fractional", action="store_true",
               help="Разрешить дробные акции (если брокер поддерживает)")
p.add_argument("--cash-step", type=float, default=0.01,
               help="Шаг валюты для округления (0.01 для EUR/USD)")
p.add_argument("--min-shares", type=int, default=1,
               help="Минимальный лот (обычно 1)")


После того, как ты уже получил prices и посчитал три набора весов
w_minvol, w_balanced, w_maxsharpe (у тебя это уже есть), вставь:

amount = args.amount
last_prices = prices.iloc[-1]   # последние доступные цены из нашего окна

def print_plan(title: str, weights_dict: dict):
    print(f"\n[{title}]")
    # 1) Красивый вывод весов
    for t, w in weights_dict.items():
        print(f"  вес {t:<6}: {w:0.4f}")

    # 2) Денежная разбивка
    cash = split_cash(amount, weights_dict, step=args.cash_step)
    # 3) Кол-во акций и траты
    plan = to_share_plan(cash, last_prices, fractional=args.fractional, min_shares=args.min_shares)
    summary = summarize_plan(plan, amount)

    print("План покупок:")
    for t in weights_dict.keys():
        row = plan[t]
        print(f"  {t:<6} @ {row['price']:>8.2f} → {row['shares']:>8} шт.  ≈ {row['spend']:>10.2f}")

    print(f"Итого к покупке: {summary['total_spend']:,.2f}")
    print(f"Остаток кэша  : {summary['leftover']:,.2f}")

# Выводим для трёх портфелей
print_plan("Консервативный (min-vol)", w_minvol)
print_plan("Сбалансированный",         w_balanced)
print_plan("Рискованный (max-sharpe)", w_maxsharpe) 

*


* ** --use-close / --use-last — какую цену брать для расчёта плана

Зачем: чтобы план покупок опирался либо на последнюю историческую цену из нашего окна (Close), либо на текущую рыночную (Last/Market).
Как работает:

--use-close — берём цену последнего дня из скачанных данных. План устойчивый и воспроизводимый.

--use-last — пытаемся подтянуть актуальную цену (если доступно). План ближе к реальности «прямо сейчас».
Что видит пользователь: те же веса, но другое количество акций и итоговая сумма/остаток. Это помогает не «промахнуться» из-за движения рынка в день покупки. *


* ** Стратегия раздачи остатка кэша

Зачем: когда запрещены дробные акции, почти всегда остаётся мелкий хвост денег. Его можно распределить умно.
Как работает (варианты):

В топ-веса — докидываем остаток в тикер с наибольшим целевым весом. Простая эвристика.

По наименьшей ошибке — считаем, где текущая доля сильнее всего отстаёт от целевой, и докупаем туда (если хватает на ещё одну акцию).
Что видит пользователь: чуть более «прилизанный» план, где фактические доли ближе к целевым, остаток меньше. * 

* ** Учёт комиссии: --fee-bps и минимальная комиссия на тикер

Зачем: комиссии реально «едят» доходность, а иногда меняют оптимальный план (невыгодно разбивать мелко).
Как работает:

--fee-bps 10 = комиссия 0.10% на оборот (покупка).

Дополнительно можно задать минимальную комиссию на тикер (например, 1€).
Мы учитываем комиссию при расчёте «сколько акций берём», чтобы не уйти в минус из-за округлений и фи.
Что видит пользователь: в плане появляется строка «Комиссия», итог «к покупке с комиссией», остаток after-fee. План реалистичнее. * 

* ** Экспорт: CSV/XLSX и «сухая печать» --plan-only

Зачем: чтобы план можно было сохранить/переслать/загрузить в Google Sheets или брокера.
Как работает:

--export plan.csv или --export plan.xlsx — пишем файл с колонками: тикер, цена, штук, сумма, целевой вес, фактическая доля, комиссия, итого.

--plan-only — выводим только итоговую таблицу без метрик и длинных текстов. Удобно копировать.
Что видит пользователь: аккуратный файл для отчёта или мгновенный «чек-лист покупок» в консоли. * 


* ** Монте-Карло симуляция на заданный горизонт

Зачем: веса и Шарп — это средние. Реальная жизнь — разброс. Симуляция показывает диапазон будущих исходов: медиану, «хвосты» убытков, максимальную просадку.
Как работает:

Из исторических дневных доходностей строим модель (простая i.i.d., блок-бутстрэп или GBM),

Генерим, скажем, 10 000 траекторий портфеля на N лет,

Считаем распределение итогового капитала, p5/p50/p95, максимальную просадку (max drawdown), вероятность уйти ниже стартовой суммы и т. п.
Что видит пользователь: * 


* ** 


--amount 10000                    # сумма инвестиций
--use-last                        # брать «рыночные» последние цены (иначе Close из окна)
--fractional                      # разрешить дробные акции
--cash-step 0.01                  # шаг валюты
--min-shares 1                    # минимум лота
--residual {topweight,min_error}  # стратегия раздачи остатка
--fee-bps 10                      # комиссия в bps (0.10%)
--min-fee 1.0                     # минимальная комиссия на тикер
--export plan.csv|.xlsx           # сохраняем план
--plan-only                       # печатаем только план покупок
--mc-years 5                      # горизонт Монте-Карло (лет)
--mc-n 10000                      # число симуляций
--mc-method {gbm,bootstrap}       # метод симуляции


*




* ** ## 1) Источник цен: data/pricing.py

# data/pricing.py
from __future__ import annotations
import pandas as pd
import yfinance as yf

def last_close_from_window(prices: pd.DataFrame) -> pd.Series:
    """Последние 'Close' из нашего окна данных."""
    return prices.iloc[-1].astype(float)

def live_last_prices(tickers: list[str]) -> pd.Series:
    """
    Пытаемся получить «рыночные» последние цены.
    1) intraday 1m последний бар; fallback -> fast_info.last_price; fallback -> close (history 1d)
    """
    tickers = [t.strip().upper() for t in tickers]
    try:
        df = yf.download(" ".join(tickers), period="1d", interval="1m", progress=False, auto_adjust=True, threads=True)
        # yfinance может вернуть многоуровневой столбец при нескольких тикерах
        if isinstance(df.columns, pd.MultiIndex):
            last = df["Close"].iloc[-1]
            last.name = None
            return last.astype(float)
        else:
            # один тикер
            return pd.Series({tickers[0]: float(df["Close"].iloc[-1])})
    except Exception:
        pass

    # fast_info fallback
    data = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            p = tk.fast_info.last_price
            if p is None:
                # ещё fallback: дневная close
                p = tk.history(period="1d")["Close"].iloc[-1]
            data[t] = float(p)
        except Exception:
            data[t] = float("nan")
    return pd.Series(data)


*



* ** 

## 2) Аллокация денег, лоты, комиссии: portfolio/allocation.py (расширенная)

# portfolio/allocation.py
from __future__ import annotations
from math import floor
from typing import Dict
import pandas as pd

def _round_cash(x: float, step: float = 0.01) -> float:
    return round(round(x / step) * step, 2)

def split_cash(amount: float, weights: Dict[str, float], step: float = 0.01) -> Dict[str, float]:
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
    """Добавляем комиссию на тикер (макс(минимальная, bps*spend))."""
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
    """
    Раздаём остаток для целочисленных акций.
    strategy: 'topweight' или 'min_error'
    """
    if fractional:
        return plan  # остаток неважен
    # фактические доли
    spend_sum = sum(p["spend"] for p in plan.values())
    if spend_sum <= 0:
        return plan
    actual = {t: (plan[t]["spend"] / spend_sum) if spend_sum else 0.0 for t in plan}
    target = weights

    def try_buy(ticker: str):
        p = float(last_prices[ticker])
        q_new = plan[ticker]["shares"] + min_shares
        add_cost = round(min_shares * p, 2)
        if add_cost <= (amount - sum(x["spend"] for x in plan.values())) + 1e-6:
            plan[ticker]["shares"] = q_new
            plan[ticker]["spend"] = round(q_new * p, 2)
            return True
        return False

    changed = True
    while changed:
        changed = False
        if strategy == "topweight":
            t = max(target, key=target.get)
            changed = try_buy(t)
        else:
            # min_error: выбираем тикер с максимальным (target - actual)
            spend_sum = sum(p["spend"] for p in plan.values()) or 1.0
            actual = {t: plan[t]["spend"] / spend_sum for t in plan}
            diff = {t: target[t] - actual.get(t, 0.0) for t in plan}
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


*



* ** 

## Монте-Карло: risk/monte_carlo.py


# risk/monte_carlo.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = (series / peak - 1.0).min()
    return float(dd)

def simulate_gbm(port_ret: pd.Series, years: int, n_paths: int, dt_per_year: int = 252, seed: int | None = None) -> pd.DataFrame:
    """
    Строим GBM из эмпирических μ и σ портфельной доходности.
    Возвращаем финальные множители капитала для каждой траектории (и при желании можно отдать весь путь).
    """
    rng = np.random.default_rng(seed)
    mu = port_ret.mean() * dt_per_year
    sigma = port_ret.std(ddof=1) * np.sqrt(dt_per_year)
    steps = years * dt_per_year
    # дискретизация GBM: log-returns ~ N((μ - σ²/2)*dt, σ*sqrt(dt))
    drift = (mu - 0.5 * sigma ** 2) / dt_per_year
    vol = sigma / np.sqrt(dt_per_year)
    z = rng.standard_normal((steps, n_paths))
    log_paths = drift + vol * z
    cum_log = log_paths.cumsum(axis=0)
    factors = np.exp(cum_log)
    # превращаем в DataFrame (1.0 — стартовый капитал)
    df = pd.DataFrame(factors, index=range(steps))
    df = pd.concat([pd.Series(1.0, index=[-1]), (1.0 * df).iloc[:, :]], axis=0)
    return df  # шаги множителей, можно брать df.iloc[-1] для финала

def simulate_bootstrap(weights: dict, returns: pd.DataFrame, years: int, n_paths: int, dt_per_year: int = 252, block: int = 21, seed: int | None = None) -> pd.DataFrame:
    """
    Исторический блок-бустрэп: тасуем блоки дневных доходностей, сохраняем корреляции.
    """
    rng = np.random.default_rng(seed)
    steps = years * dt_per_year
    blocks = []
    R = returns.values
    n = len(returns)
    for _ in range(int(np.ceil(steps / block))):
        i = rng.integers(0, max(1, n - block))
        blocks.append(R[i:i+block])
    sim = np.vstack(blocks)[:steps]
    # портфельная доходность по весам
    w = np.array([weights[t] for t in returns.columns], dtype=float)
    port_daily = sim @ w
    factors = (1.0 + port_daily).cumprod()
    df = pd.DataFrame(factors, index=range(steps))
    df = pd.concat([pd.Series(1.0, index=[-1]), df], axis=0)
    return df

def mc_summary(factors_df: pd.DataFrame) -> dict:
    final = factors_df.iloc[-1].values
    p5, p50, p95 = np.percentile(final, [5, 50, 95])
    # оценим медианную max drawdown по траекториям
    mdds = []
    for col in factors_df.columns:
        ser = factors_df[col]
        mdds.append(max_drawdown(ser))
    mdd_med = float(np.median(mdds)) if mdds else float("nan")
    prob_below_1 = float((final < 1.0).mean())
    return {"p5": p5, "p50": p50, "p95": p95, "mdd_med": mdd_med, "prob_below_1": prob_below_1}


*
## 5) Правки в cli.py (импорты + флаги + вызовы)


##Расширь парсер аргументов:


p.add_argument("--amount", type=float, required=True, help="Сумма инвестиций")
p.add_argument("--use-last", action="store_true", help="Взять рыночную последнюю цену (иначе last close из окна)")
p.add_argument("--fractional", action="store_true", help="Разрешить дробные акции")
p.add_argument("--cash-step", type=float, default=0.01, help="Шаг валюты")
p.add_argument("--min-shares", type=int, default=1, help="Лот (обычно 1)")
p.add_argument("--residual", type=str, default="topweight", choices=["topweight", "min_error"], help="Стратегия раздачи остатка")
p.add_argument("--fee-bps", type=float, default=0.0, help="Комиссия (bps)")
p.add_argument("--min-fee", type=float, default=0.0, help="Мин. комиссия на тикер")
p.add_argument("--export", type=str, default=None, help="Путь для экспорта CSV/XLSX")
p.add_argument("--plan-only", action="store_true", help="Печатать только план без метрик")

p.add_argument("--mc-years", type=int, default=0, help="Лет для Монте-Карло (0 = выкл)")
p.add_argument("--mc-n", type=int, default=10000, help="Число симуляций")
p.add_argument("--mc-method", type=str, default="gbm", choices=["gbm", "bootstrap"], help="Метод Монте-Карло")



##После расчёта цен и весов добавь блок выбора цен:


tickers = [t.strip().upper() for t in args.tickers.split(",")]
prices = load_prices(tickers, args.start, args.end, use_cache=not args.no_cache)

# ... твой расчёт лог-доходностей, ковариации и весов: w_minvol, w_balanced, w_maxsharpe

if args.use_last:
    last_prices = live_last_prices(tickers)
    # если где-то NaN, падём на last_close
    if last_prices.isna().any():
        last_prices = last_close_from_window(prices)
else:
    last_prices = last_close_from_window(prices)



##Хелпер печати плана (с учётом комиссий и стратегии остатка):


def print_and_optionally_export(title: str, weights_dict: dict):
    amount = args.amount
    cash = split_cash(amount, weights_dict, step=args.cash_step)
    plan = to_share_plan(cash, last_prices, fractional=args.fractional, min_shares=args.min_shares)
    plan = residual_allocate(plan, weights_dict, amount, strategy=args.residual, last_prices=last_prices,
                             fractional=args.fractional, min_shares=args.min_shares)
    plan = apply_fees(plan, fee_bps=args.fee_bps, min_fee=args.min_fee)
    summary = summarize_plan(plan, amount)

    if not args.plan_only:
        print(f"\n[{title}]")
        for t, w in weights_dict.items():
            print(f"  вес {t:<6}: {w:0.4f}")

    print("План покупок:")
    for t in weights_dict.keys():
        row = plan[t]
        print(f"  {t:<6} @ {row['price']:>8.2f} → {row['shares']:>8} шт.  "
              f"≈ {row['spend']:>10.2f}  fee {row.get('fee',0):>6.2f}  total {row.get('total',row['spend']):>10.2f}")

    print(f"Итого к покупке: {summary['total']:,.2f}  (спенд {summary['total_spend']:,.2f} + комиссии {summary['total_fee']:,.2f})")
    print(f"Остаток кэша  : {summary['leftover']:,.2f}")

    # экспорт
    if args.export:
        df = plan_to_df(weights_dict, plan)
        export_plan(df, args.export)
        print(f"Экспортирован план → {args.export}")

    return plan



### 6) Монте-Карло вызов в cli.py

После расчёта портфельных весов и наличия returns (дневные доходности, prices.pct_change().dropna() или лог-доходности) добавь:



if args.mc_years > 0:
    # Возьмём простые дневные доходности под GBM/Bootstrap
    ret = prices.pct_change().dropna()
    dt_per_year = 252

    def mc_for(weights_dict: dict, title: str):
        # портфельная дневная доходность из истории (для GBM оценки μ,σ и/или для bootstrap)
        w = ret[ret.columns].mul([weights_dict[t] for t in ret.columns], axis=1)
        port_ret = w.sum(axis=1)

        if args.mc_method == "gbm":
            paths = simulate_gbm(port_ret, years=args.mc_years, n_paths=args.mc_n, dt_per_year=dt_per_year, seed=42)
        else:
            paths = simulate_bootstrap(weights_dict, ret, years=args.mc_years, n_paths=args.mc_n, dt_per_year=dt_per_year, block=21, seed=42)
        summary = mc_summary(paths)
        print(f"\nМонте-Карло — {title} (горизонт {args.mc_years} лет, {args.mc_n} траекторий, {args.mc_method}):")
        print(f"  P5 / P50 / P95 множителя капитала: {summary['p5']:.2f} / {summary['p50']:.2f} / {summary['p95']:.2f}")
        print(f"  Медианный max drawdown: {summary['mdd_med']:.1%}")
        print(f"  Вероятность быть ниже стартовой суммы: {summary['prob_below_1']:.1%}")

    mc_for(w_minvol,  "Консервативный")
    mc_for(w_balanced,"Сбалансированный")
    mc_for(w_maxsharpe,"Рискованный (max-sharpe)")



*


