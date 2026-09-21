# 03_src/tft/plot_feature_importance.py
"""Render the TFT v2 Variable Selection Network importance figure.

Reads the `feature_importance.json` written by notebook 13 for an ablation
variant and produces a two-panel figure: the top-N individual features
coloured by feature block, and each block's share of total importance.

The composite `sentiment_score` is always included, annotated with its rank
when it falls outside the top N.  That is the point of the figure for the
thesis: the channel decomposition pushed the composite from 53% of TFT v1's
weight down to rank 21 here.

Reads:   04_outputs/experiment_tracking/TFT/TFTv2/tft_v2_outputs/<variant>/feature_importance.json
Writes:  04_outputs/figures/tft_v2_feature_importance.png

Usage:
    python 03_src/tft/plot_feature_importance.py
    python 03_src/tft/plot_feature_importance.py --variant v2.1 --top 20
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[2]
V2_OUTPUTS = ROOT / "04_outputs/experiment_tracking/TFT/TFTv2/tft_v2_outputs"
FIGURES = ROOT / "04_outputs/figures"

# Canonical variant. See 05_reports/development-decisions/thesis_decisions_log.md:
# v2.2 exp2 is the reported model (v2.3 added EIA features and scored worse).
DEFAULT_VARIANT = "v2.2"

# Feature blocks. Entity flags are matched by the ent_ prefix that
# config.entity_to_column_name() produces, so the 71 flags need no listing.
CHANNEL = {"supply_impact", "demand_impact", "risk_premium"}
OTHER_LLM = {
    "sentiment_score",
    "certainty",
    "magnitude",
    "n_articles",
    # v2.0 only: the int-encoded categoricals that v2.1 replaced with learned
    # embeddings. Kept so the earlier ablation variants still plot.
    "event_type_int",
    "time_horizon_int",
}
MARKET = {"vix", "dxy", "log_volume", "log_return", "amihud", "price_range"}
CALENDAR = {
    "is_wednesday",
    "month",
    "day_of_week",
    "hour",
    "is_us_session",
    "relative_time_idx",
}

COLORS = {
    "channel": "#C0392B",
    "llm": "#E8836F",
    "entity": "#E0A800",
    "market": "#3B7EA1",
    "calendar": "#9AA5AB",
}
LABELS = {
    "channel": "LLM channel (supply / demand / risk)",
    "llm": "Other LLM features",
    "entity": "Canonical entity flags (71)",
    "market": "Market and macro context",
    "calendar": "Calendar and session",
}
LEGEND_ORDER = ("market", "channel", "entity", "llm", "calendar")


def feature_block(name: str) -> str:
    """Map a feature name to its block key.

    Raises on an unrecognised name rather than silently bucketing it, so a
    schema change surfaces here instead of quietly skewing the block shares.
    """
    if name in CHANNEL:
        return "channel"
    if name in OTHER_LLM:
        return "llm"
    if name.startswith("ent_"):
        return "entity"
    if name in MARKET:
        return "market"
    if name in CALENDAR:
        return "calendar"
    raise ValueError(
        f"uncategorized feature {name!r}: add it to a block set in this module"
    )


def plot_feature_importance(
    importance: dict, variant: str, out_path: Path, top_n: int = 15
):
    """Render the two-panel importance figure and save it to out_path."""
    total = sum(importance.values())
    ranked = sorted(importance.items(), key=lambda kv: -kv[1])
    rank_of = {k: i + 1 for i, (k, _) in enumerate(ranked)}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.5, 6.6), gridspec_kw={"width_ratios": [2.15, 1]}
    )

    # ---- left panel: top-N individual features ----
    rows = ranked[:top_n]
    # Always show the composite, even when it ranks below the cut.
    trailing = [(k, v) for k, v in ranked if k == "sentiment_score" and rank_of[k] > top_n]
    rows = rows + trailing

    names = [k for k, _ in rows]
    vals = [v for _, v in rows]
    ypos = list(range(len(rows)))[::-1]

    ax1.barh(
        ypos,
        vals,
        color=[COLORS[feature_block(k)] for k in names],
        edgecolor="white",
        linewidth=0.6,
    )
    for y, v in zip(ypos, vals):
        ax1.text(v + max(vals) * 0.017, y, f"{v:.3f}", va="center",
                 fontsize=9.5, color="#333333")

    ax1.set_yticks(ypos)
    ax1.set_yticklabels(
        [n if rank_of[n] <= top_n else f"{n}  (rank {rank_of[n]})" for n in names],
        fontsize=10.5,
    )
    ax1.set_xlabel("Mean variable-selection weight", fontsize=11)
    ax1.set_title(
        f"TFT {variant} feature importance: "
        f"top {min(top_n, len(importance))} of {len(importance)} features",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax1.set_xlim(0, max(vals) * 1.16)
    ax1.grid(axis="x", alpha=0.3, linewidth=0.7)
    ax1.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax1.spines[side].set_visible(False)

    if trailing:
        # Dotted rule marking the break between the top-N and the appended rows.
        ax1.axhline(len(trailing) - 0.5, color="#BBBBBB", linestyle=":", linewidth=1.2)

    ax1.legend(
        [plt.Rectangle((0, 0), 1, 1, color=COLORS[b]) for b in LEGEND_ORDER],
        [LABELS[b] for b in LEGEND_ORDER],
        loc="lower right",
        fontsize=9.5,
        frameon=True,
        framealpha=0.95,
    )

    # ---- right panel: share of total importance by block ----
    sums = {}
    for name, weight in importance.items():
        block = feature_block(name)
        sums[block] = sums.get(block, 0.0) + weight

    order = sorted(sums, key=lambda b: -sums[b])
    shares = [100 * sums[b] / total for b in order]
    ypos2 = list(range(len(order)))[::-1]

    ax2.barh(
        ypos2,
        shares,
        color=[COLORS[b] for b in order],
        edgecolor="white",
        linewidth=0.6,
    )
    for y, s in zip(ypos2, shares):
        ax2.text(s + 0.9, y, f"{s:.1f}%", va="center", fontsize=10, color="#333333")

    ax2.set_yticks(ypos2)
    ax2.set_yticklabels(
        [
            LABELS[b].replace(" (supply / demand / risk)", "").replace(" (71)", "")
            for b in order
        ],
        fontsize=10.5,
    )
    ax2.set_xlabel("Share of total importance (%)", fontsize=11)
    ax2.set_title("Importance by feature block", fontsize=13,
                  fontweight="bold", loc="left")
    ax2.set_xlim(0, max(shares) * 1.22)
    ax2.grid(axis="x", alpha=0.3, linewidth=0.7)
    ax2.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)

    # Only v2.2 exp2 is the reported model; label the others plainly.
    descriptor = (
        f"Canonical model TFT {variant} exp2"
        if variant == DEFAULT_VARIANT
        else f"Ablation variant TFT {variant}"
    )
    fig.suptitle(
        f"{descriptor}, Variable Selection Network, encoder variables",
        fontsize=10.5,
        color="#666666",
        x=0.5,
        y=0.015,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {b: round(100 * sums[b] / total, 1) for b in order}


def _display(path: Path) -> str:
    """Repo-relative path when the file sits inside the repo, absolute otherwise."""
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=f"ablation variant directory under {V2_OUTPUTS.name} (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="number of individual features to show"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURES / "tft_v2_feature_importance.png",
        help="output PNG path",
    )
    args = parser.parse_args()

    src = V2_OUTPUTS / args.variant / "feature_importance.json"
    if not src.exists():
        raise SystemExit(f"no feature_importance.json for variant {args.variant!r}: {src}")

    importance = json.loads(src.read_text())
    shares = plot_feature_importance(importance, args.variant, args.out, args.top)

    print(f"read  {_display(src)}")
    print(f"wrote {_display(args.out)}")
    print(f"block shares: {shares}")


if __name__ == "__main__":
    main()
