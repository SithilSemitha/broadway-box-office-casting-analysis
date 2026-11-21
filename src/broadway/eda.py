"""Development-only figures and supporting data from the Spark Parquet handoff."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import AUDIT, FEATURES, FIGURES, TABLES
from .io import write_json


def read_features(path=FEATURES):
    frame = pd.read_parquet(path, engine="pyarrow")
    frame["week_ending"] = pd.to_datetime(frame["week_ending"])
    frame["month"] = frame["month"].astype(str)
    return frame.sort_values(["week_ending", "show", "theatre"]).reset_index(drop=True)


def coverage(frame, column):
    table = frame.groupby(column, observed=True)["eligible"].agg(rows="size", eligible_rows="sum")
    table["ineligible_rows"] = table["rows"] - table["eligible_rows"]
    table["eligibility_rate"] = table["eligible_rows"] / table["rows"]
    return table.reset_index()


def generate_eda(path=FEATURES, figures=FIGURES, tables=TABLES):
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    # Filter before any plot, relationship statistic, sampling or coverage aggregation.
    early = read_features(path).query("year <= 2014").copy()
    if early.empty:
        raise ValueError("No data through 2014 for EDA.")
    eligible = early.loc[early.eligible].copy()
    plt.rcParams.update({"figure.figsize": (10, 5.5), "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "axes.titlepad": 14})
    money = FuncFormatter(lambda value, _: f"${value / 1e6:.1f}m")
    index = []

    def save(fig, number, name, title, scope):
        filename = f"figure_{number:02d}_{name}.png"
        fig.suptitle(f"Figure {number}. {title}", fontsize=13)
        fig.text(0.01, 0.01, f"{scope} | Nominal USD; source: TidyTuesday / Alex Cookson / Playbill", fontsize=8)
        fig.tight_layout(rect=(0, 0.045, 1, 0.95))
        fig.savefig(figures / filename)
        plt.close(fig)
        index.append({"figure": number, "file": filename, "title": title, "scope": scope})

    weekly = early.groupby("week_ending")["weekly_gross"].agg(median_gross="median", reported_shows="size").reset_index()
    weekly.to_csv(tables / "figure_01_weekly_medians.csv", index=False)
    fig, ax = plt.subplots()
    ax.plot(weekly.week_ending, weekly.median_gross, color="#255f85", linewidth=0.9)
    ax.set(xlabel="Reporting week ending", ylabel="Median gross per reported show (nominal USD)")
    ax.yaxis.set_major_formatter(money)
    save(fig, 1, "weekly_median", "Weekly median reported gross per show", "1985–2014; all cleaned observed rows")

    counts, edges = np.histogram(early.weekly_gross, bins=65)
    pd.DataFrame({"lower_usd": edges[:-1], "upper_usd": edges[1:], "rows": counts}).to_csv(tables / "figure_02_histogram_bins.csv", index=False)
    fig, ax = plt.subplots()
    ax.hist(early.weekly_gross, bins=edges, color="#378a86", edgecolor="white", linewidth=0.4)
    ax.set(xlabel="Weekly gross (nominal USD)", ylabel="Show–theatre–week observations")
    ax.xaxis.set_major_formatter(money)
    save(fig, 2, "gross_histogram", "Distribution of weekly gross", "1985–2014; all cleaned observed rows")

    recent = early.query("2010 <= year <= 2014").copy()
    recent["month_number"] = recent.week_ending.dt.month
    recent[["week_ending", "show", "theatre", "month_number", "weekly_gross"]].to_csv(tables / "figure_03_monthly_source.csv", index=False)
    fig, ax = plt.subplots()
    ax.boxplot([recent.loc[recent.month_number == m, "weekly_gross"] for m in range(1, 13)],
               tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
               flierprops={"markersize": 2, "alpha": 0.25}, medianprops={"color": "#c35d26"})
    ax.set(xlabel="Target calendar month", ylabel="Weekly gross (nominal USD)")
    ax.yaxis.set_major_formatter(money)
    save(fig, 3, "month_boxplots", "Calendar-month distributions", "2010–2014; all cleaned observed rows")

    missing = early.groupby("year")[["potential_gross", "top_ticket_price"]].agg(lambda col: col.isna().mean())
    missing.to_csv(tables / "figure_04_missingness_by_year.csv")
    fig, ax = plt.subplots()
    for col, color in [("potential_gross", "#255f85"), ("top_ticket_price", "#c35d26")]:
        ax.plot(missing.index, missing[col], label=col.replace("_", " "), color=color)
    ax.set(xlabel="Year", ylabel="Missing share of reported rows", ylim=(-0.03, 1.03))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    save(fig, 4, "missingness_by_year", "Missingness in optional price-related fields", "1985–2014; all cleaned observed rows")

    rho = spearmanr(eligible.weekly_gross_lag1, eligible.weekly_gross)
    write_json(tables / "lag_relationship.json", {"scope": "eligible target weeks through 2014", "n": len(eligible),
               "spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue),
               "caution": "Repeated observations within runs; p-value is descriptive, not an independence-adjusted test."})
    eligible[["week_ending", "show", "theatre", "weekly_gross_lag1", "weekly_gross"]].to_csv(tables / "figure_05_lag_relationship_source.csv", index=False)
    fig, ax = plt.subplots()
    hb = ax.hexbin(eligible.weekly_gross_lag1, eligible.weekly_gross, gridsize=55, mincnt=1, bins="log", cmap="viridis")
    limit = max(eligible.weekly_gross.max(), eligible.weekly_gross_lag1.max())
    ax.plot([0, limit], [0, limit], "--", color="#bd5c40", linewidth=1, label="Equal gross")
    ax.set(xlabel="Previous reported week's gross (nominal USD)", ylabel="Target week's gross (nominal USD)")
    ax.xaxis.set_major_formatter(money)
    ax.yaxis.set_major_formatter(money)
    fig.colorbar(hb, ax=ax, label="Observation count (log colour scale)")
    ax.text(0.03, 0.95, f"Spearman ρ = {rho.statistic:.4f}; n = {len(eligible):,}", transform=ax.transAxes, va="top")
    save(fig, 5, "lag_hexbin", "Previous-week and target-week gross", "1985–2014; eligible rows only")

    scatter = early[["week_ending", "show", "theatre", "pct_capacity", "weekly_gross"]].dropna()
    scatter.to_csv(tables / "figure_06_occupancy_source.csv", index=False)
    fig, ax = plt.subplots()
    ax.scatter(scatter.pct_capacity * 100, scatter.weekly_gross, s=5, alpha=0.12, color="#255f85", rasterized=True)
    ax.axvline(100, color="#c35d26", linestyle="--", label="100% capacity")
    ax.set(xlabel="Reported occupancy (%)", ylabel="Weekly gross (nominal USD)")
    ax.yaxis.set_major_formatter(money)
    ax.legend()
    save(fig, 6, "occupancy_gross", "Reported occupancy and weekly gross", "1985–2014; all cleaned rows with reported occupancy")

    for column in ["year", "theatre"]:
        coverage(early, column).to_csv(tables / f"coverage_{column}_through_2014.csv", index=False)
    run_lengths = early.groupby(["show", "theatre", "inferred_run"])["week_ending"].transform("size")
    early["observed_run_length_band"] = pd.cut(run_lengths, [0, 4, 12, 52, np.inf], labels=["1–4", "5–12", "13–52", "53+"])
    coverage(early, "observed_run_length_band").to_csv(tables / "coverage_run_length_through_2014.csv", index=False)
    coverage(early, "show").sort_values("rows", ascending=False).to_csv(tables / "coverage_show_through_2014.csv", index=False)
    early.nlargest(20, "weekly_gross")[["week_ending", "show", "theatre", "weekly_gross", "pct_capacity", "eligible"]].to_csv(tables / "outlier_review_through_2014.csv", index=False)
    pd.DataFrame(index).to_csv(tables / "figure_index.csv", index=False)
    write_json(tables / "eda_scope.json", {"max_target_date": str(early.week_ending.max().date()),
        "cleaned_rows": len(early), "eligible_rows": len(eligible), "figures": len(index),
        "coverage_note": "Run lengths are observed within the truncated exploratory period, not official lifetime lengths.",
        "source_parquet": str(path), "final_holdout_used": False})
    return index
