#!/usr/bin/env python3
"""Plot the saved 2021/2022 rows of the original combined Table 1."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
# Here load the proper full-period subset estimates, including the 2021/2022 rows (not supplied).
SOURCE = HERE.parent / 'data/example_subset_effects.csv'
# Columns: outcome (keys in SCENARIOS below), case (2021/2022), term (TV/Home),
# estimate_pct, ci_low_pct, ci_high_pct. Numeric fields are percentages.
SCENARIOS = (
    ("low_risk_contacts", "Non-risky\ncontacts"),
    ("low_risk_exposure", "Non-risky\nexposure"),
    ("risky_contacts", "Risky\ncontacts"),
    ("risky_exposure", "Risky\nexposure"),
    ("all_transmissions", "All\ntransmissions"),
)
YEARS = ("2021", "2022")
TERMS = ("TV", "Home")
COLORS = {"2021": "#4C72B0", "2022": "#C44E52"}
Y_LABELS = {
    "TV": r"TV effect on $R_t$ (%)",
    "Home": r"Home-match effect on $R_t$ (%)",
}


def read_year_rows():
    """Read unrounded percentages; reject incomplete or changed source tables."""
    raw = SOURCE.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    required = {
        "case", "outcome", "term", "estimate_pct", "ci_low_pct", "ci_high_pct",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("The source does not have the expected Table 1 columns.")
    expected_outcomes = {outcome for outcome, _ in SCENARIOS}
    if {r["outcome"] for r in rows} != expected_outcomes:
        raise ValueError("The source must contain all five original scenarios.")
    if not {"2021", "2022"}.issubset({r["case"] for r in rows}):
        raise ValueError("Use the original full-period summary with both year rows.")

    selected = {}
    for row in rows:
        if row["case"] not in YEARS:
            continue
        key = (row["outcome"], row["term"], row["case"])
        if key in selected:
            raise ValueError(f"Duplicate year estimate: {key}")
        values = [float(row[k]) for k in ("estimate_pct", "ci_low_pct", "ci_high_pct")]
        if not all(math.isfinite(v) for v in values) or values[1] > values[2]:
            raise ValueError(f"Invalid saved estimate or interval: {key}")
        selected[key] = row

    expected_keys = {
        (outcome, term, year)
        for outcome, _ in SCENARIOS for term in TERMS for year in YEARS
    }
    if set(selected) != expected_keys:
        raise ValueError("Expected exactly one estimate per scenario, effect and year.")
    return selected, fingerprint


def build_figure(rows, terms=TERMS):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    style = {
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.usetex": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        home_only = terms == ("Home",)
        fig, axes = plt.subplots(1, len(terms),
                                 figsize=(7.5 if home_only else 14, 5.5),
                                 sharey=True, squeeze=False)
        axes = axes.ravel()
        positions = list(range(len(SCENARIOS)))
        # Keep the standalone home panel on the same scale as the combined figure.
        all_values = [float(row["estimate_pct"]) for row in rows.values()]
        low, high = min(0.0, min(all_values)), max(0.0, max(all_values))
        padding = 0.12 * (high - low if high > low else 1.0)

        for ax, term in zip(axes, terms):
            for year, offset in zip(YEARS, (-0.18, 0.18)):
                # estimate_pct is already in percent. Do not rescale or refit.
                heights = [float(rows[outcome, term, year]["estimate_pct"])
                           for outcome, _ in SCENARIOS]
                ax.bar([x + offset for x in positions], heights, width=0.34,
                       color=COLORS[year], edgecolor="none", linewidth=0,
                       label=year)
            ax.set_xticks(positions, [label for _, label in SCENARIOS], fontsize=11)
            ax.set_ylabel(Y_LABELS[term], fontsize=14, labelpad=10)
            ax.set_ylim(low - padding, high + padding)
            ax.set_xlim(-0.6, len(SCENARIOS) - 0.4)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.grid(False, which="both", axis="both")
            ax.axhline(0, color="black", linewidth=0.8, zorder=3)
            ax.minorticks_off()
            ax.tick_params(axis="both", which="major", direction="out",
                           length=3.5, width=0.8, pad=8, bottom=True, left=True,
                           top=False, right=False)
            ax.tick_params(axis="y", labelleft=True)
            for side, spine in ax.spines.items():
                spine.set_visible(side in ("bottom", "left"))
                spine.set_linewidth(0.8)
                spine.set_color("black")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.995), handlelength=1.5, columnspacing=2)
        fig.subplots_adjust(left=0.14 if home_only else 0.075, right=0.99,
                            bottom=0.16, top=0.86, wspace=0.28)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="year_comparison", help="New folder under results/figures.")
    parser.add_argument("--home-only", action="store_true",
                        help="Create a standalone home-match figure instead of both panels.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate the saved table without creating figures or files.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("Run name must contain only letters, digits, underscores or hyphens.")
    try:
        rows, fingerprint = read_year_rows()
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Source validation failed: {exc}\n")
    if args.check_only:
        print("Source validation passed; no figures or output files created.")
        return

    output = HERE.parent / "results/figures" / args.run_name
    if output.exists():
        parser.error("That output folder already exists. Choose a new --run-name.")
    terms = ("Home",) if args.home_only else TERMS
    stem = "table1_home_2021_vs_2022" if args.home_only else "table1_2021_vs_2022"
    fig = build_figure(rows, terms=terms)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=300,
                        bbox_inches="tight", pad_inches=0.12)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Preserve the full original selected rows, including intervals and provenance.
    ordered_rows = [rows[outcome, term, year]
                    for term in terms for outcome, _ in SCENARIOS for year in YEARS]
    with (output / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered_rows[0]))
        writer.writeheader()
        writer.writerows(ordered_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "source_sha256": fingerprint,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": list(YEARS), "panels": list(terms),
        "scenario_order": [outcome for outcome, _ in SCENARIOS],
        "bar_height_column": "estimate_pct", "rescaling": "none",
        "year_colors": COLORS, "confidence_intervals_drawn": False,
        "refitting_performed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
