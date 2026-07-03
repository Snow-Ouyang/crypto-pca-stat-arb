from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "s_score_audit" / "s_score_panel_window360_pc3.csv",
    PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "final_strategy" / "w360_pc3_signal_panel_with_beta.csv",
]
PC1_RET_PATH = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "pca_diagnostics" / "rolling_pc_factor_returns.csv"
PRICE_PATH = PROJECT_ROOT / "data" / "raw" / "coin_all_prices_full.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "bad_trade_diagnostics"
REPORT_DIR = PROJECT_ROOT / "reports" / "final_report"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures" / "bad_trades"

FEE_BPS_LIST = [0, 5, 10]
DD_START = pd.Timestamp("2021-10-28 10:00:00+00:00")
DD_END = pd.Timestamp("2021-11-26 11:00:00+00:00")

LONG_ENTRY = {
    "hl_0_9": 1.00,
    "hl_9_18": 1.00,
    "hl_18_36": 1.00,
    "hl_36_60": 1.25,
    "hl_60_90": 1.25,
}
LONG_EXIT = {
    "hl_0_9": 0.25,
    "hl_9_18": 0.25,
    "hl_18_36": 0.25,
    "hl_36_60": 0.25,
    "hl_60_90": 1.00,
}
SHORT_ENTRY = 1.00
SHORT_EXIT = 0.25


def finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(x))
    except TypeError:
        return False


def hl_bucket(half_life: Any) -> str:
    if not finite(half_life) or float(half_life) <= 0:
        return "missing"
    hl = float(half_life)
    if hl <= 9:
        return "hl_0_9"
    if hl <= 18:
        return "hl_9_18"
    if hl <= 36:
        return "hl_18_36"
    if hl <= 60:
        return "hl_36_60"
    if hl <= 90:
        return "hl_60_90"
    return "hl_gt_90"


def quality_ok(row: Any) -> bool:
    return (
        finite(row.ou_b)
        and 0 < float(row.ou_b) < 1
        and finite(row.ou_half_life_hours)
        and 0 < float(row.ou_half_life_hours) <= 90
        and finite(row.sigma_eq_percentile_at_t)
        and float(row.sigma_eq_percentile_at_t) <= 0.95
        and finite(row.regression_r2)
        and float(row.regression_r2) >= 0.50
        and finite(row.s_score)
        and finite(row.raw_log_return)
        and finite(row.price)
        and float(row.price) > 0
        and bool(row.in_no_lookahead_universe)
    )


def load_panel() -> pd.DataFrame:
    for path in PANEL_CANDIDATES:
        if path.exists():
            panel = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
            break
    else:
        raise FileNotFoundError("No W360/PC3 signal panel found.")
    required = [
        "timestamp",
        "ticker",
        "price",
        "raw_log_return",
        "s_score",
        "ou_b",
        "ou_half_life_hours",
        "ou_sigma_eq",
        "sigma_eq_percentile_at_t",
        "regression_r2",
        "in_no_lookahead_universe",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Signal panel missing required columns: {missing}")
    panel["hl_bucket_final"] = panel["ou_half_life_hours"].map(hl_bucket)
    return panel.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def load_raw_prices(tickers: list[str]) -> pd.DataFrame:
    usecols = ["startTime", *[t for t in tickers if t]]
    available = pd.read_csv(PRICE_PATH, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in available]
    prices = pd.read_csv(PRICE_PATH, usecols=usecols)
    prices["startTime"] = pd.to_datetime(prices["startTime"], utc=True)
    prices = prices.set_index("startTime").sort_index()
    return prices.apply(pd.to_numeric, errors="coerce")


@dataclass
class OpenTrade:
    trade_id: int
    side: str
    ticker: str
    entry_row: Any
    hl_bucket: str


def close_trade(open_trade: OpenTrade, exit_row: Any, reason: str) -> dict[str, Any]:
    entry = open_trade.entry_row
    entry_price = float(entry.price)
    exit_price = float(exit_row.price)
    if open_trade.side == "long":
        gross_pnl = exit_price / entry_price - 1.0
    else:
        gross_pnl = 1.0 - exit_price / entry_price
    exit_abs_exposure = exit_price / entry_price
    row = {
        "trade_id": open_trade.trade_id,
        "side": open_trade.side,
        "ticker": open_trade.ticker,
        "entry_time": entry.timestamp,
        "exit_time": exit_row.timestamp,
        "exit_reason": reason,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_s_score": float(entry.s_score),
        "exit_s_score": float(exit_row.s_score) if finite(exit_row.s_score) else np.nan,
        "entry_half_life": float(entry.ou_half_life_hours),
        "exit_half_life": float(exit_row.ou_half_life_hours) if finite(exit_row.ou_half_life_hours) else np.nan,
        "entry_sigma_eq": float(entry.ou_sigma_eq),
        "exit_sigma_eq": float(exit_row.ou_sigma_eq) if finite(exit_row.ou_sigma_eq) else np.nan,
        "entry_sigma_pct": float(entry.sigma_eq_percentile_at_t),
        "exit_sigma_pct": float(exit_row.sigma_eq_percentile_at_t) if finite(exit_row.sigma_eq_percentile_at_t) else np.nan,
        "entry_r2": float(entry.regression_r2),
        "exit_r2": float(exit_row.regression_r2) if finite(exit_row.regression_r2) else np.nan,
        "entry_ou_b": float(entry.ou_b),
        "exit_ou_b": float(exit_row.ou_b) if finite(exit_row.ou_b) else np.nan,
        "hl_bucket": open_trade.hl_bucket,
        "holding_hours": (exit_row.timestamp - entry.timestamp).total_seconds() / 3600.0,
        "gross_pnl": gross_pnl,
        "gross_return": gross_pnl,
    }
    for fee_bps in FEE_BPS_LIST:
        fees = fee_bps / 10000.0 * (1.0 + exit_abs_exposure)
        row[f"net_pnl_{fee_bps}bps"] = gross_pnl - fees
    row["net_return_5bps"] = row["net_pnl_5bps"]
    return row


def make_missing_exit_row(timestamp: pd.Timestamp, price: float) -> Any:
    return SimpleNamespace(
        timestamp=timestamp,
        price=price,
        s_score=np.nan,
        ou_half_life_hours=np.nan,
        ou_sigma_eq=np.nan,
        sigma_eq_percentile_at_t=np.nan,
        regression_r2=np.nan,
        ou_b=np.nan,
    )


def make_last_price_exit_row(timestamp: pd.Timestamp, price: float) -> Any:
    return SimpleNamespace(
        timestamp=timestamp,
        price=price,
        s_score=np.nan,
        ou_half_life_hours=np.nan,
        ou_sigma_eq=np.nan,
        sigma_eq_percentile_at_t=np.nan,
        regression_r2=np.nan,
        ou_b=np.nan,
    )


def run_naive_trades(panel: pd.DataFrame, raw_prices: pd.DataFrame) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    trade_id = 1
    all_times = pd.Index(sorted(panel["timestamp"].drop_duplicates()))
    for side in ["long", "short"]:
        for ticker, group in panel.groupby("ticker", sort=False):
            if ticker not in raw_prices.columns:
                continue
            signal = group.sort_values("timestamp").copy()
            signal["has_signal_row"] = True
            timeline = signal.set_index("timestamp").reindex(all_times)
            timeline["timestamp"] = timeline.index
            timeline["has_signal_row"] = timeline["has_signal_row"].fillna(False).astype(bool)
            timeline["ticker"] = ticker
            # Use the signal-panel price when present. Missing signal rows intentionally
            # keep signal fields missing; raw prices are only used to maintain the last
            # valid settlement price before a disappearance.
            timeline["raw_price_for_last_valid"] = raw_prices[ticker].reindex(all_times).to_numpy()
            active: OpenTrade | None = None
            last_valid_price = np.nan
            exited_this_bar: set[pd.Timestamp] = set()
            for row in timeline.itertuples(index=False):
                has_signal_row = bool(row.has_signal_row)
                current_price_valid = finite(row.price) and float(row.price) > 0
                qok = quality_ok(row)
                if active is not None:
                    reason = None
                    exit_row = row
                    if not has_signal_row:
                        if finite(last_valid_price) and float(last_valid_price) > 0:
                            exit_row = make_last_price_exit_row(row.timestamp, float(last_valid_price))
                            reason = "missing_signal_exit_last_price"
                        else:
                            reason = "missing_exit_no_last_price"
                    elif not finite(row.price) or not finite(row.s_score):
                        if finite(last_valid_price) and float(last_valid_price) > 0:
                            exit_row = make_last_price_exit_row(row.timestamp, float(last_valid_price))
                            reason = "missing_exit_last_price"
                        else:
                            reason = "missing_exit_no_last_price"
                    elif not qok:
                        reason = "eligibility_lost"
                    elif side == "long" and float(row.s_score) > -LONG_EXIT[active.hl_bucket]:
                        reason = "threshold_exit"
                    elif side == "short" and float(row.s_score) < SHORT_EXIT:
                        reason = "threshold_exit"
                    if reason is not None:
                        trades.append(close_trade(active, exit_row, reason))
                        exited_this_bar.add(row.timestamp)
                        active = None
                        trade_id += 1
                        if current_price_valid:
                            last_valid_price = float(row.price)
                        elif finite(row.raw_price_for_last_valid) and float(row.raw_price_for_last_valid) > 0:
                            last_valid_price = float(row.raw_price_for_last_valid)
                        continue
                if active is None and has_signal_row and row.timestamp not in exited_this_bar and qok:
                    bucket = hl_bucket(row.ou_half_life_hours)
                    if bucket not in LONG_ENTRY:
                        continue
                    if side == "long" and float(row.s_score) < -LONG_ENTRY[bucket]:
                        active = OpenTrade(trade_id, side, ticker, row, bucket)
                    elif side == "short" and float(row.s_score) > SHORT_ENTRY:
                        active = OpenTrade(trade_id, side, ticker, row, bucket)
                if current_price_valid:
                    last_valid_price = float(row.price)
                elif finite(row.raw_price_for_last_valid) and float(row.raw_price_for_last_valid) > 0:
                    last_valid_price = float(row.raw_price_for_last_valid)
            if active is not None:
                last_ts = all_times[-1]
                last_price = raw_prices.at[last_ts, ticker] if last_ts in raw_prices.index else np.nan
                if last_ts > active.entry_row.timestamp and finite(last_price) and float(last_price) > 0:
                    last = make_missing_exit_row(last_ts, float(last_price))
                    trades.append(close_trade(active, last, "end_of_sample"))
                    trade_id += 1
    trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    trades_df["trade_id"] = np.arange(1, len(trades_df) + 1)
    return trades_df


def select_large_loss_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bottom_n = max(50, int(np.ceil(len(trades) * 0.01)))
    groups = {
        "worst_all_trades": trades.nsmallest(bottom_n, "net_pnl_5bps").assign(selection_reason="bottom_1pct_or_50_all"),
        "worst_long_trades": trades[trades["side"] == "long"].nsmallest(max(1, int(np.ceil((trades["side"] == "long").sum() * 0.01))), "net_pnl_5bps").assign(selection_reason="bottom_1pct_long"),
        "worst_short_trades": trades[trades["side"] == "short"].nsmallest(max(1, int(np.ceil((trades["side"] == "short").sum() * 0.01))), "net_pnl_5bps").assign(selection_reason="bottom_1pct_short"),
    }
    overlap = trades[(trades["entry_time"] <= DD_END) & (trades["exit_time"] >= DD_START)].copy()
    groups["worst_drawdown_window_trades"] = overlap.nsmallest(50, "net_pnl_5bps").assign(selection_reason="overlaps_final_max_dd_window")
    for name, df in groups.items():
        part = df[["trade_id", "side", "ticker", "entry_time", "exit_time", "net_pnl_5bps", "holding_hours", "hl_bucket", "selection_reason"]].copy()
        part["selection_group"] = name
        rows.append(part)
    selection = pd.concat(rows, ignore_index=True).drop_duplicates(["trade_id", "selection_group"])
    return selection[["trade_id", "side", "ticker", "entry_time", "exit_time", "net_pnl_5bps", "holding_hours", "hl_bucket", "selection_group", "selection_reason"]]


def load_pc1_and_market(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PC1_RET_PATH.exists():
        pc1 = pd.read_csv(PC1_RET_PATH, parse_dates=["timestamp"])
        pc1 = pc1[["timestamp", "PC1"]].rename(columns={"PC1": "pc1_return"}).sort_values("timestamp")
    else:
        pc1 = pd.DataFrame(columns=["timestamp", "pc1_return"])
    market = (
        panel[panel["ticker"].isin(["BTC", "ETH"])][["timestamp", "ticker", "price"]]
        .pivot_table(index="timestamp", columns="ticker", values="price", aggfunc="last")
        .sort_index()
    )
    return pc1, market


def compute_path_diagnostics(panel: pd.DataFrame, trades: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    pc1, market = load_pc1_and_market(panel)
    pc1 = pc1.set_index("timestamp") if not pc1.empty else pc1
    by_ticker = {ticker: g.sort_values("timestamp").reset_index(drop=True) for ticker, g in panel.groupby("ticker")}
    selected_ids = sorted(selection["trade_id"].unique())
    trade_lookup = trades.set_index("trade_id")
    rows = []
    for trade_id in selected_ids:
        tr = trade_lookup.loc[trade_id]
        g = by_ticker.get(tr.ticker)
        if g is None:
            continue
        path = g[(g["timestamp"] >= tr.entry_time) & (g["timestamp"] <= tr.exit_time)].copy()
        if path.empty:
            continue
        price_rel = path["price"] / float(tr.entry_price) - 1.0
        if tr.side == "long":
            max_adv = float(price_rel.min())
            max_fav = float(price_rel.max())
            adv_s = float(tr.entry_s_score - path["s_score"].min())
            fav_s = float(path["s_score"].max() - tr.entry_s_score)
        else:
            max_adv = float((-price_rel).min())
            max_fav = float((-price_rel).max())
            adv_s = float(path["s_score"].max() - tr.entry_s_score)
            fav_s = float(tr.entry_s_score - path["s_score"].min())
        pc1_ret = np.nan
        pc1_cum = np.nan
        if not pc1.empty:
            pc1_slice = pc1[(pc1.index > tr.entry_time) & (pc1.index <= tr.exit_time)]
            if not pc1_slice.empty:
                pc1_ret = float(pc1_slice["pc1_return"].sum())
                pc1_cum = pc1_ret
        btc_ret = np.nan
        eth_ret = np.nan
        for coin, dest in [("BTC", "btc_ret"), ("ETH", "eth_ret")]:
            if coin in market.columns:
                m = market[(market.index >= tr.entry_time) & (market.index <= tr.exit_time)][coin].dropna()
                if len(m) >= 2:
                    if coin == "BTC":
                        btc_ret = float(m.iloc[-1] / m.iloc[0] - 1.0)
                    else:
                        eth_ret = float(m.iloc[-1] / m.iloc[0] - 1.0)
        rows.append(
            {
                "trade_id": trade_id,
                "side": tr.side,
                "ticker": tr.ticker,
                "entry_time": tr.entry_time,
                "exit_time": tr.exit_time,
                "exit_reason": tr.exit_reason,
                "net_pnl_5bps": tr.net_pnl_5bps,
                "holding_hours": tr.holding_hours,
                "hl_bucket": tr.hl_bucket,
                "max_adverse_price_move": max_adv,
                "max_favorable_price_move": max_fav,
                "price_return_during_trade": float(tr.exit_price / tr.entry_price - 1.0),
                "entry_s_score": tr.entry_s_score,
                "exit_s_score": tr.exit_s_score,
                "max_s_score_during_trade": float(path["s_score"].max()),
                "min_s_score_during_trade": float(path["s_score"].min()),
                "adverse_s_score_continuation": adv_s,
                "favorable_s_score_reversal": fav_s,
                "entry_half_life": tr.entry_half_life,
                "exit_half_life": tr.exit_half_life,
                "max_half_life_during_trade": float(path["ou_half_life_hours"].max()),
                "median_half_life_during_trade": float(path["ou_half_life_hours"].median()),
                "half_life_change": tr.exit_half_life - tr.entry_half_life,
                "holding_to_entry_half_life": tr.holding_hours / tr.entry_half_life if tr.entry_half_life > 0 else np.nan,
                "entry_sigma_eq": tr.entry_sigma_eq,
                "exit_sigma_eq": tr.exit_sigma_eq,
                "max_sigma_eq_during_trade": float(path["ou_sigma_eq"].max()),
                "entry_sigma_pct": tr.entry_sigma_pct,
                "exit_sigma_pct": tr.exit_sigma_pct,
                "max_sigma_pct_during_trade": float(path["sigma_eq_percentile_at_t"].max()),
                "sigma_expansion_ratio": float(path["ou_sigma_eq"].max() / tr.entry_sigma_eq) if tr.entry_sigma_eq > 0 else np.nan,
                "entry_r2": tr.entry_r2,
                "min_r2_during_trade": float(path["regression_r2"].min()),
                "exit_r2": tr.exit_r2,
                "entry_ou_b": tr.entry_ou_b,
                "max_ou_b_during_trade": float(path["ou_b"].max()),
                "min_ou_b_during_trade": float(path["ou_b"].min()),
                "pc1_return_during_trade": pc1_ret,
                "pc1_cumulative_move_during_trade": pc1_cum,
                "btc_return_during_trade": btc_ret,
                "eth_return_during_trade": eth_ret,
            }
        )
    return pd.DataFrame(rows)


def select_cases(diag: pd.DataFrame) -> pd.DataFrame:
    cases = []
    dd_ids = set()
    dd_sel = pd.read_csv(TABLE_DIR / "bad_trade_selection.csv", parse_dates=["entry_time", "exit_time"])
    dd_ids.update(dd_sel.loc[dd_sel["selection_group"] == "worst_drawdown_window_trades", "trade_id"].tolist())
    d = diag.copy()
    d["in_dd_selection"] = d["trade_id"].isin(dd_ids)
    candidates = d[d["side"] == "short"].copy()
    if candidates.empty:
        candidates = d.copy()
    candidates["score"] = (
        -candidates["net_pnl_5bps"].rank(pct=True)
        + candidates["adverse_s_score_continuation"].rank(pct=True)
        + candidates["price_return_during_trade"].rank(pct=True)
        + candidates["in_dd_selection"].astype(float)
    )
    row = candidates.sort_values("score", ascending=False).iloc[0]
    cases.append((1, "short_squeeze_adverse_continuation", row, "short loss with adverse s-score continuation and rising price"))

    candidates = d.copy()
    candidates["score"] = -candidates["net_pnl_5bps"].rank(pct=True) + candidates["holding_to_entry_half_life"].rank(pct=True) * 2
    row = candidates.sort_values("score", ascending=False).iloc[0]
    cases.append((2, "slow_mean_reversion_holding_exceeds_half_life", row, "high holding-to-half-life ratio among large-loss trades"))

    candidates = d.copy()
    candidates["score"] = (
        -candidates["net_pnl_5bps"].rank(pct=True)
        + candidates["sigma_expansion_ratio"].replace([np.inf, -np.inf], np.nan).rank(pct=True)
        + candidates["max_sigma_pct_during_trade"].rank(pct=True)
        - candidates["min_r2_during_trade"].rank(pct=True)
    )
    row = candidates.sort_values("score", ascending=False).iloc[0]
    cases.append((3, "model_instability_sigma_expansion", row, "sigma expansion or quality deterioration among large-loss trades"))

    out = []
    used = set()
    for case_id, mechanism, row, reason in cases:
        if int(row.trade_id) in used:
            fallback = d[~d["trade_id"].isin(used)].sort_values("net_pnl_5bps").iloc[0]
            row = fallback
            reason = "fallback to avoid duplicate representative trade"
        used.add(int(row.trade_id))
        out.append(
            {
                "case_id": case_id,
                "mechanism": mechanism,
                "trade_id": int(row.trade_id),
                "side": row.side,
                "ticker": row.ticker,
                "entry_time": row.entry_time,
                "exit_time": row.exit_time,
                "net_pnl_5bps": row.net_pnl_5bps,
                "holding_hours": row.holding_hours,
                "entry_half_life": row.entry_half_life,
                "holding_to_entry_half_life": row.holding_to_entry_half_life,
                "adverse_s_score_continuation": row.adverse_s_score_continuation,
                "sigma_expansion_ratio": row.sigma_expansion_ratio,
                "max_sigma_pct_during_trade": row.max_sigma_pct_during_trade,
                "reason_selected": reason,
            }
        )
    return pd.DataFrame(out)


def build_case_path(panel: pd.DataFrame, raw_prices: pd.DataFrame, case: Any, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    signal = panel[panel["ticker"] == case.ticker].sort_values("timestamp").set_index("timestamp")
    idx = raw_prices.loc[(raw_prices.index >= start) & (raw_prices.index <= end)].index
    path = pd.DataFrame(index=idx)
    if case.ticker in raw_prices.columns:
        path["price"] = raw_prices.loc[idx, case.ticker]
    for col in ["s_score", "ou_half_life_hours", "ou_sigma_eq", "sigma_eq_percentile_at_t", "regression_r2", "ou_b"]:
        if col in signal.columns:
            path[col] = signal[col].reindex(idx)
    return path.reset_index().rename(columns={"startTime": "timestamp", "index": "timestamp"})


def plot_case(panel: pd.DataFrame, raw_prices: pd.DataFrame, case: Any, filename: str) -> None:
    entry = pd.Timestamp(case.entry_time)
    exit_time = pd.Timestamp(case.exit_time)
    start = entry - pd.Timedelta(hours=24)
    end = exit_time + pd.Timedelta(hours=12)
    path = build_case_path(panel, raw_prices, case, start, end)
    if path.empty:
        return
    entry_price = path.loc[path["timestamp"] >= entry, "price"].iloc[0]
    norm_price = path["price"] / entry_price * 100
    entry_bucket = str(case.hl_bucket) if hasattr(case, "hl_bucket") else hl_bucket(case.entry_half_life)
    entry_thr = LONG_ENTRY.get(entry_bucket, 1.0) if case.side == "long" else SHORT_ENTRY
    exit_thr = LONG_EXIT.get(entry_bucket, 0.25) if case.side == "long" else SHORT_EXIT
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    title = f"{case.mechanism} | {case.ticker} {str(case.side).upper()} | net={case.net_pnl_5bps:.4f} | hold={case.holding_hours:.1f}h | {entry_bucket}"
    axes[0].plot(path["timestamp"], norm_price, color="#2f6fdd")
    axes[0].set_ylabel("Price=100")
    axes[0].set_title(title)
    axes[1].plot(path["timestamp"], path["s_score"], color="#6b4fb3")
    if case.side == "long":
        axes[1].axhline(-entry_thr, color="red", linestyle="--", linewidth=0.9, label="entry")
        axes[1].axhline(-exit_thr, color="green", linestyle="--", linewidth=0.9, label="exit")
    else:
        axes[1].axhline(entry_thr, color="red", linestyle="--", linewidth=0.9, label="entry")
        axes[1].axhline(exit_thr, color="green", linestyle="--", linewidth=0.9, label="exit")
    axes[1].set_ylabel("s-score")
    axes[1].legend(loc="upper left")
    axes[2].plot(path["timestamp"], path["ou_half_life_hours"], color="#d05a42")
    axes[2].axhline(case.entry_half_life, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_ylim(bottom=0, top=min(120, max(10, np.nanmax(path["ou_half_life_hours"]) * 1.1)))
    axes[2].set_ylabel("half-life")
    axes[3].plot(path["timestamp"], path["ou_sigma_eq"], label="sigma_eq", color="#00806a")
    ax2 = axes[3].twinx()
    ax2.plot(path["timestamp"], path["sigma_eq_percentile_at_t"], label="sigma pct", color="#8a8a8a", alpha=0.7)
    axes[3].set_ylabel("sigma")
    ax2.set_ylabel("sigma pct")
    axes[4].plot(path["timestamp"], path["regression_r2"], label="R2", color="#222222")
    axes[4].plot(path["timestamp"], path["ou_b"], label="ou_b", color="#cc7a00", alpha=0.8)
    axes[4].set_ylabel("R2 / b")
    axes[4].legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.axvline(entry, color="red", linestyle=":", linewidth=1.0)
        ax.axvline(exit_time, color="black", linestyle=":", linewidth=1.0)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


def plot_case_trade_lifecycle(panel: pd.DataFrame, raw_prices: pd.DataFrame, case: Any, filename: str) -> None:
    entry = pd.Timestamp(case.entry_time)
    exit_time = pd.Timestamp(case.exit_time)
    path = build_case_path(panel, raw_prices, case, entry, exit_time)
    if path.empty:
        return
    entry_price = float(path["price"].iloc[0])
    norm_price = path["price"] / entry_price * 100.0
    entry_bucket = str(case.hl_bucket) if hasattr(case, "hl_bucket") else hl_bucket(case.entry_half_life)
    entry_thr = LONG_ENTRY.get(entry_bucket, 1.0) if case.side == "long" else SHORT_ENTRY
    exit_thr = LONG_EXIT.get(entry_bucket, 0.25) if case.side == "long" else SHORT_EXIT

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"{case.mechanism} | {case.ticker} {str(case.side).upper()} | "
        f"net={case.net_pnl_5bps:.4f} | hold={case.holding_hours:.1f}h | {entry_bucket}"
    )

    axes[0].plot(path["timestamp"], norm_price, color="#2f6fdd", linewidth=1.2)
    axes[0].set_ylabel("Price=100")

    axes[1].plot(path["timestamp"], path["s_score"], color="#6b4fb3", linewidth=1.2)
    if case.side == "long":
        axes[1].axhline(-entry_thr, color="red", linestyle="--", linewidth=0.9, label="entry threshold")
        axes[1].axhline(-exit_thr, color="green", linestyle="--", linewidth=0.9, label="exit threshold")
    else:
        axes[1].axhline(entry_thr, color="red", linestyle="--", linewidth=0.9, label="entry threshold")
        axes[1].axhline(exit_thr, color="green", linestyle="--", linewidth=0.9, label="exit threshold")
    axes[1].set_ylabel("s-score")
    axes[1].legend(loc="upper left")

    axes[2].plot(path["timestamp"], path["ou_half_life_hours"], color="#d05a42", linewidth=1.2)
    axes[2].axhline(case.entry_half_life, color="black", linestyle="--", linewidth=0.8, label="entry half-life")
    axes[2].set_ylabel("Half-life hours")
    axes[2].legend(loc="upper left")

    axes[3].plot(path["timestamp"], path["ou_sigma_eq"], color="#00806a", linewidth=1.2, label="sigma_eq")
    ax_pct = axes[3].twinx()
    ax_pct.plot(path["timestamp"], path["sigma_eq_percentile_at_t"], color="#8a8a8a", linewidth=1.0, alpha=0.75, label="sigma percentile")
    axes[3].set_ylabel("sigma_eq")
    ax_pct.set_ylabel("sigma percentile")
    lines, labels = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax_pct.get_legend_handles_labels()
    axes[3].legend(lines + lines2, labels + labels2, loc="upper left")

    for ax in axes:
        ax.axvline(entry, color="red", linestyle=":", linewidth=1.0)
        ax.axvline(exit_time, color="black", linestyle=":", linewidth=1.0)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


def summarize_bad_trades(diag: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bad_ids = set(diag["trade_id"])
    bad = trades[trades["trade_id"].isin(bad_ids)].merge(
        diag[["trade_id", "adverse_s_score_continuation", "sigma_expansion_ratio", "max_sigma_pct_during_trade", "holding_to_entry_half_life", "min_r2_during_trade"]],
        on="trade_id",
        how="left",
    )

    def agg(df: pd.DataFrame) -> pd.Series:
        long_share = (df["side"] == "long").mean() if "side" in df.columns else np.nan
        short_share = (df["side"] == "short").mean() if "side" in df.columns else np.nan
        return pd.Series(
            {
                "count": len(df),
                "avg_net_pnl_5bps": df["net_pnl_5bps"].mean(),
                "median_net_pnl_5bps": df["net_pnl_5bps"].median(),
                "avg_holding_hours": df["holding_hours"].mean(),
                "median_holding_hours": df["holding_hours"].median(),
                "avg_entry_half_life": df["entry_half_life"].mean(),
                "avg_holding_to_half_life": df["holding_to_entry_half_life"].mean(),
                "avg_adverse_s_score_continuation": df["adverse_s_score_continuation"].mean(),
                "avg_sigma_expansion_ratio": df["sigma_expansion_ratio"].mean(),
                "avg_max_sigma_pct": df["max_sigma_pct_during_trade"].mean(),
                "avg_entry_r2": df["entry_r2"].mean(),
                "avg_min_r2_during_trade": df["min_r2_during_trade"].mean(),
                "threshold_exit_share": (df["exit_reason"] == "threshold_exit").mean(),
                "eligibility_lost_share": (df["exit_reason"] == "eligibility_lost").mean(),
                "missing_signal_exit_last_price_share": (df["exit_reason"] == "missing_signal_exit_last_price").mean(),
                "missing_exit_share": df["exit_reason"].astype(str).str.startswith("missing").mean(),
                "long_share": long_share,
                "short_share": short_share,
            }
        )

    overall = agg(bad).to_frame().T
    by_side = bad.groupby("side").apply(agg).reset_index()
    by_hl = bad.groupby(["side", "hl_bucket"]).apply(agg).reset_index()
    by_ticker = (
        bad.groupby(["ticker", "side"])
        .agg(
            count=("trade_id", "count"),
            total_net_pnl_5bps=("net_pnl_5bps", "sum"),
            avg_net_pnl_5bps=("net_pnl_5bps", "mean"),
            max_loss=("net_pnl_5bps", "min"),
            avg_adverse_s_score_continuation=("adverse_s_score_continuation", "mean"),
            avg_sigma_expansion_ratio=("sigma_expansion_ratio", "mean"),
            avg_holding_to_half_life=("holding_to_entry_half_life", "mean"),
        )
        .reset_index()
        .sort_values("total_net_pnl_5bps")
    )
    bad["year_month"] = bad["entry_time"].dt.to_period("M").astype(str)
    by_month = bad.groupby(["year_month", "side"]).apply(agg).reset_index()
    return overall, by_side, by_hl, by_ticker, by_month


def plot_aggregate(diag: pd.DataFrame, trades: pd.DataFrame) -> None:
    bad = trades[trades["trade_id"].isin(diag["trade_id"])].merge(diag, on=["trade_id", "side", "ticker", "entry_time", "exit_time", "holding_hours", "hl_bucket", "net_pnl_5bps"], how="left")
    loss_side = bad.groupby("side")["net_pnl_5bps"].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    loss_side.plot(kind="bar", ax=ax, color=["#2f6fdd", "#d05a42"])
    ax.set_title("Large-loss trade net PnL by side")
    ax.set_ylabel("Net PnL, 5bps")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bad_trade_loss_by_side.png", dpi=160)
    plt.close(fig)

    hl = bad.groupby(["side", "hl_bucket"])["net_pnl_5bps"].sum().unstack(0).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    hl.plot(kind="bar", ax=ax)
    ax.set_title("Large-loss trade net PnL by half-life bucket")
    ax.set_ylabel("Net PnL, 5bps")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bad_trade_loss_by_hl_bucket.png", dpi=160)
    plt.close(fig)

    for col, filename, title in [
        ("adverse_s_score_continuation", "adverse_s_score_continuation_distribution.png", "Adverse s-score continuation"),
        ("holding_to_entry_half_life", "holding_to_half_life_distribution.png", "Holding / entry half-life"),
        ("sigma_expansion_ratio", "sigma_expansion_distribution.png", "Sigma expansion ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        bad[col].replace([np.inf, -np.inf], np.nan).dropna().hist(ax=ax, bins=30)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=160)
        plt.close(fig)

    bad["year_month"] = bad["entry_time"].dt.to_period("M").astype(str)
    counts = bad.groupby(["year_month", "side"]).size().unstack(1).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Large-loss trade monthly counts")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bad_trade_monthly_counts.png", dpi=160)
    plt.close(fig)


def validate(trades: pd.DataFrame, cases: pd.DataFrame) -> dict[str, bool]:
    short = trades[trades["side"] == "short"].copy()
    short_ok = bool(((short["exit_price"] < short["entry_price"]) == (short["gross_pnl"] > 0)).all())
    fee0_ok = bool(np.allclose(trades["net_pnl_0bps"], trades["gross_pnl"]))
    same_bar = trades[["trade_id", "side", "ticker", "entry_time"]].merge(
        trades[["trade_id", "side", "ticker", "exit_time"]].rename(columns={"exit_time": "entry_time", "trade_id": "exit_trade_id"}),
        on=["side", "ticker", "entry_time"],
        how="inner",
    )
    same_bar = same_bar[same_bar["trade_id"] != same_bar["exit_trade_id"]] if "exit_trade_id" in same_bar.columns else same_bar
    plots = [
        FIG_DIR / "case_1_short_squeeze.png",
        FIG_DIR / "case_2_slow_mean_reversion.png",
        FIG_DIR / "case_3_model_instability.png",
    ]
    return {
        "short_accounting_ok": short_ok,
        "fee0_net_equals_gross": fee0_ok,
        "no_same_timestamp_exit_and_reentry": same_bar.empty,
        "representative_cases_in_trade_table": set(cases["trade_id"]).issubset(set(trades["trade_id"])),
        "all_case_plots_generated": all(p.exists() for p in plots),
    }


def write_report(overall: pd.DataFrame, by_side: pd.DataFrame, by_hl: pd.DataFrame, by_ticker: pd.DataFrame, cases: pd.DataFrame, validation: dict[str, bool]) -> None:
    lines = [
        "# Bad Trade Mechanism Diagnostics",
        "",
        "## 1. Purpose",
        "",
        "This diagnostic returns to the naive 1-dollar-per-position stage to study why individual residual signals fail before matched-sleeve sizing, dollar neutrality, gross caps, or soft PC1 optimization affect the outcome.",
        "",
        "## 2. Large Loss Trade Definition",
        "",
        "Large-loss trades combine the worst bottom 1% or 50 trades overall, trades overlapping the final maximum drawdown window, and side-specific bottom 1% long and short trades.",
        "",
        "## 3. Representative Case Studies",
        "",
        cases.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![Case 1](figures/bad_trades/case_1_short_squeeze.png)",
        "",
        "![Case 2](figures/bad_trades/case_2_slow_mean_reversion.png)",
        "",
        "![Case 3](figures/bad_trades/case_3_model_instability.png)",
        "",
        "## 4. Aggregate Bad Trade Statistics",
        "",
        "Overall:",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "By side:",
        "",
        by_side.to_markdown(index=False, floatfmt=".4f"),
        "",
        "By half-life bucket:",
        "",
        by_hl.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Worst ticker-side contributors:",
        "",
        by_ticker.head(15).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. Implications for Future Risk Control",
        "",
        "- Short-side regime filtering is worth testing if large short losses cluster in squeeze windows.",
        "- A time stop based on holding-to-entry-half-life is worth testing if bad trades persist far beyond estimated half-life.",
        "- Sigma expansion and quality deterioration stops are worth testing when losses coincide with rising sigma or falling R2.",
        "- Bucket-specific de-risking can be considered for half-life buckets that dominate large losses.",
        "",
        "## Validation",
        "",
        pd.DataFrame([validation]).to_markdown(index=False),
    ]
    (REPORT_DIR / "bad_trade_mechanism_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    raw_prices = load_raw_prices(sorted(panel["ticker"].dropna().astype(str).unique().tolist() + ["BTC", "ETH"]))
    trades = run_naive_trades(panel, raw_prices)
    trades.to_csv(OUT_DIR / "naive_1dollar_trades.csv", index=False)
    selection = select_large_loss_trades(trades)
    selection.to_csv(TABLE_DIR / "bad_trade_selection.csv", index=False)
    diag = compute_path_diagnostics(panel, trades, selection)
    diag.to_csv(TABLE_DIR / "bad_trade_path_diagnostics.csv", index=False)
    cases = select_cases(diag)
    cases.to_csv(TABLE_DIR / "representative_bad_trade_cases.csv", index=False)
    case_files = {
        1: "case_1_short_squeeze.png",
        2: "case_2_slow_mean_reversion.png",
        3: "case_3_model_instability.png",
    }
    case_lookup = diag.merge(cases[["case_id", "mechanism", "trade_id", "reason_selected"]], on="trade_id", how="inner")
    for row in case_lookup.itertuples(index=False):
        plot_case(panel, raw_prices, row, case_files[int(row.case_id)])
        plot_case_trade_lifecycle(panel, raw_prices, row, f"case_{int(row.case_id)}_trade_lifecycle_4panel.png")
    overall, by_side, by_hl, by_ticker, by_month = summarize_bad_trades(diag, trades)
    overall.to_csv(TABLE_DIR / "bad_trade_summary_stats.csv", index=False)
    by_side.to_csv(TABLE_DIR / "bad_trade_stats_by_side.csv", index=False)
    by_hl.to_csv(TABLE_DIR / "bad_trade_stats_by_hl_bucket.csv", index=False)
    by_ticker.to_csv(TABLE_DIR / "bad_trade_stats_by_ticker.csv", index=False)
    by_month.to_csv(TABLE_DIR / "bad_trade_stats_by_month.csv", index=False)
    plot_aggregate(diag, trades)
    validation = validate(trades, cases)
    write_report(overall, by_side, by_hl, by_ticker, cases, validation)

    print(f"total naive trades: {len(trades)}")
    print(f"total large-loss diagnostic trades: {diag['trade_id'].nunique()}")
    print("worst long trade:")
    print(trades[trades["side"] == "long"].nsmallest(1, "net_pnl_5bps")[["trade_id", "ticker", "net_pnl_5bps", "holding_hours", "hl_bucket"]].to_string(index=False))
    print("worst short trade:")
    print(trades[trades["side"] == "short"].nsmallest(1, "net_pnl_5bps")[["trade_id", "ticker", "net_pnl_5bps", "holding_hours", "hl_bucket"]].to_string(index=False))
    print("selected representative cases:")
    print(cases[["case_id", "mechanism", "trade_id", "side", "ticker", "net_pnl_5bps", "reason_selected"]].to_string(index=False))
    print("bad trade summary by side:")
    print(by_side.to_string(index=False))
    print("bad trade summary by half-life bucket:")
    print(by_hl.to_string(index=False))
    print("validation:")
    print(validation)
    print(f"tables: {TABLE_DIR}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
