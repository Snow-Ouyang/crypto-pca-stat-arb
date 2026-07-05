from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESIDUAL_FILE = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "advanced_pca_v1" / "advanced_residual_returns_long.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "final_report" / "residual_cleanliness"
FIG_DIR = OUT_DIR / "figures"
WINDOW = 360
ORDINARY_METHOD = "ordinary"
ADVANCED_METHOD = "advanced_lambda_0p5"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def residual_metrics(window: pd.DataFrame) -> dict[str, float]:
    complete = window.notna().sum(axis=0).eq(len(window))
    std = window.std(axis=0, ddof=1).replace([np.inf, -np.inf], np.nan)
    tickers = [c for c in window.columns if bool(complete.get(c, False) and std.get(c, np.nan) > 1e-12)]
    if len(tickers) < 2:
        return {
            "avg_abs_pairwise_residual_corr": np.nan,
            "residual_pc1_evr_corr": np.nan,
            "residual_pc1_evr_cov": np.nan,
            "clean_universe_size": len(tickers),
        }

    x = window[tickers].to_numpy(dtype=float)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    avg_abs_corr = float(np.mean(np.abs(corr[mask])))
    corr_eig = np.linalg.eigvalsh(corr)
    pc1_evr_corr = float(max(corr_eig.max(), 0.0) / corr.shape[0])

    cov = np.cov(x, rowvar=False)
    cov_eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    cov_total = float(cov_eig.sum())
    pc1_evr_cov = float(cov_eig.max() / cov_total) if cov_total > 0 else np.nan
    return {
        "avg_abs_pairwise_residual_corr": avg_abs_corr,
        "residual_pc1_evr_corr": pc1_evr_corr,
        "residual_pc1_evr_cov": pc1_evr_cov,
        "clean_universe_size": len(tickers),
    }


def build_wide_panels() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RESIDUAL_FILE, usecols=["timestamp", "method", "ticker", "residual_return"], parse_dates=["timestamp"])
    panels: dict[str, pd.DataFrame] = {}
    for method in [ORDINARY_METHOD, ADVANCED_METHOD]:
        sub = df[df["method"] == method]
        panel = sub.pivot_table(index="timestamp", columns="ticker", values="residual_return", aggfunc="last")
        panels[method] = panel.sort_index()
    return panels


def compute_timeseries(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_index = panels[ORDINARY_METHOD].index.intersection(panels[ADVANCED_METHOD].index)
    ordinary = panels[ORDINARY_METHOD].reindex(common_index)
    advanced = panels[ADVANCED_METHOD].reindex(common_index)

    rows: list[dict[str, object]] = []
    for i in range(WINDOW - 1, len(common_index)):
        if i == WINDOW - 1 or (i + 1) % 1000 == 0:
            print(f"[residual_cleanliness] processed {i + 1}/{len(common_index)}", flush=True)
        ts = common_index[i]
        o_metrics = residual_metrics(ordinary.iloc[i - WINDOW + 1 : i + 1])
        a_metrics = residual_metrics(advanced.iloc[i - WINDOW + 1 : i + 1])
        o_corr = o_metrics["avg_abs_pairwise_residual_corr"]
        a_corr = a_metrics["avg_abs_pairwise_residual_corr"]
        o_pc1 = o_metrics["residual_pc1_evr_corr"]
        a_pc1 = a_metrics["residual_pc1_evr_corr"]
        rows.append(
            {
                "timestamp": ts,
                "ordinary_avg_abs_pairwise_residual_corr": o_corr,
                "advanced_avg_abs_pairwise_residual_corr": a_corr,
                "ordinary_residual_pc1_evr_corr": o_pc1,
                "advanced_residual_pc1_evr_corr": a_pc1,
                "ordinary_residual_pc1_evr_cov": o_metrics["residual_pc1_evr_cov"],
                "advanced_residual_pc1_evr_cov": a_metrics["residual_pc1_evr_cov"],
                "residual_corr_reduction_pct": (o_corr - a_corr) / o_corr if np.isfinite(o_corr) and o_corr != 0 else np.nan,
                "residual_pc1_evr_reduction_pct": (o_pc1 - a_pc1) / o_pc1 if np.isfinite(o_pc1) and o_pc1 != 0 else np.nan,
                "clean_universe_size": min(o_metrics["clean_universe_size"], a_metrics["clean_universe_size"]),
            }
        )
    return pd.DataFrame(rows)


def build_summary(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_pairs = [
        ("avg_abs_pairwise_residual_corr", "ordinary_avg_abs_pairwise_residual_corr", "advanced_avg_abs_pairwise_residual_corr"),
        ("residual_pc1_evr_corr", "ordinary_residual_pc1_evr_corr", "advanced_residual_pc1_evr_corr"),
        ("residual_pc1_evr_cov", "ordinary_residual_pc1_evr_cov", "advanced_residual_pc1_evr_cov"),
    ]
    for metric, ordinary_col, advanced_col in metric_pairs:
        ordinary_mean = float(ts[ordinary_col].mean())
        advanced_mean = float(ts[advanced_col].mean())
        rows.append(
            {
                "metric": metric,
                "ordinary_mean": ordinary_mean,
                "advanced_mean": advanced_mean,
                "absolute_change": advanced_mean - ordinary_mean,
                "percent_change": (advanced_mean - ordinary_mean) / ordinary_mean if ordinary_mean else np.nan,
            }
        )

    rows.append(
        {
            "metric": "residual_corr_reduction_pct",
            "ordinary_mean": np.nan,
            "advanced_mean": float(ts["residual_corr_reduction_pct"].mean()),
            "absolute_change": np.nan,
            "percent_change": float(ts["residual_corr_reduction_pct"].mean()),
        }
    )
    rows.append(
        {
            "metric": "residual_pc1_evr_reduction_pct",
            "ordinary_mean": np.nan,
            "advanced_mean": float(ts["residual_pc1_evr_reduction_pct"].mean()),
            "absolute_change": np.nan,
            "percent_change": float(ts["residual_pc1_evr_reduction_pct"].mean()),
        }
    )
    return pd.DataFrame(rows)


def plot_bar(summary: pd.DataFrame) -> None:
    keep = summary[summary["metric"].isin(["avg_abs_pairwise_residual_corr", "residual_pc1_evr_corr"])].copy()
    labels = ["Avg abs pairwise\nresidual corr", "Residual PC1\nEVR corr"]
    x = np.arange(len(keep))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, keep["ordinary_mean"], width, label="Ordinary PCA", color="tab:blue")
    ax.bar(x + width / 2, keep["advanced_mean"], width, label="Advanced PCA", color="tab:orange")
    ax.set_xticks(x, labels)
    ax.set_title("Residual Cleanliness: Ordinary vs Advanced PCA")
    ax.set_ylabel("Mean rolling W360 diagnostic value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_cleanliness_bar.png", dpi=160)
    plt.close(fig)


def plot_timeseries(ts: pd.DataFrame) -> None:
    df = ts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["timestamp"], df["ordinary_avg_abs_pairwise_residual_corr"], label="Ordinary PCA", linewidth=1.0)
    ax.plot(df["timestamp"], df["advanced_avg_abs_pairwise_residual_corr"], label="Advanced PCA", linewidth=1.0)
    ax.set_title("Average absolute residual pairwise correlation over time")
    ax.set_ylabel("Rolling W360 avg abs pairwise corr")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_pairwise_corr_over_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["timestamp"], df["ordinary_residual_pc1_evr_corr"], label="Ordinary PCA", linewidth=1.0)
    ax.plot(df["timestamp"], df["advanced_residual_pc1_evr_corr"], label="Advanced PCA", linewidth=1.0)
    ax.set_title("Residual PC1 EVR over time")
    ax.set_ylabel("Rolling W360 residual PC1 EVR corr")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_pc1_evr_over_time.png", dpi=160)
    plt.close(fig)


def write_improvement_table(summary: pd.DataFrame) -> None:
    display = summary[summary["metric"].isin(["avg_abs_pairwise_residual_corr", "residual_pc1_evr_corr"])].copy()
    display["reduction_pct"] = -display["percent_change"]
    display = display[["metric", "ordinary_mean", "advanced_mean", "reduction_pct"]]
    markdown = display.to_markdown(index=False, floatfmt=".4f")
    (OUT_DIR / "residual_cleanliness_improvement_table.md").write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panels = build_wide_panels()
    ts = compute_timeseries(panels)
    summary = build_summary(ts)
    ts.to_csv(OUT_DIR / "residual_cleanliness_timeseries.csv", index=False)
    summary.to_csv(OUT_DIR / "residual_cleanliness_summary.csv", index=False)
    write_improvement_table(summary)
    plot_bar(summary)
    plot_timeseries(ts)
    print(f"timeseries: {OUT_DIR / 'residual_cleanliness_timeseries.csv'}")
    print(f"summary: {OUT_DIR / 'residual_cleanliness_summary.csv'}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
