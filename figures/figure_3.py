"""Risky contacts and exposure: paired stadium/TV bars for every subset."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # Registers the SciencePlots styles with Matplotlib.

WIDTH_CM, HEIGHT_CM = 17.8, 14.5
FONT_PT, TICK_PT, LEGEND_PT = 9, 8, 9

OUTCOMES = ["risky_contacts", "risky_exposure"]
# Saved term, legend label, colour. These colours match Figure 2's bottom panels.
EFFECTS = [("Home", "Home match (stadium)", "#4C72B0"), ("TV", "TV viewing", "#C44E52")]


# Edit these font sizes to adjust the text throughout Figure 3.
@plt.style.context(["science", "no-latex", {
    "font.family": "STIXGeneral",
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": LEGEND_PT,
}])
def draw(data):
    # One panel per outcome, with a stadium/TV pair for each of the 14 subsets.
    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54))
    indexed = data.set_index(["outcome", "case", "term"])
    y = np.arange(len(CASES))
    height = .8 / len(EFFECTS)
    for ax, outcome in zip(axes, OUTCOMES):
        for j, (term, label, color) in enumerate(EFFECTS):
            rows = indexed.loc[[(outcome, case, term) for case in CASES]]
            values, low, high = rows[["estimate_pct", "ci_low_pct", "ci_high_pct"]].to_numpy().T
            positions = y + (j - (len(EFFECTS) - 1) / 2) * height
            ax.barh(positions, values, height=height, color=color, label=label)
            # Plot exact saved endpoints, even if an interval does not contain its estimate.
            ax.errorbar((low + high) / 2, positions, xerr=(high - low) / 2,
                        fmt="none", ecolor="black", capsize=2.5, capthick=.7,
                        elinewidth=.7)
        ax.axvline(0, color="black", linewidth=0.6, zorder=3)
        ax.set_xlabel(f"{LABELS[outcome]}: relative effect (%)")
        ax.set_yticks(y, list(CASES.values()))
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2)
    fig.subplots_adjust(left=.305, right=.985, bottom=.095, top=.925, wspace=.25)
    return fig


import argparse
import json
from matplotlib.text import Text
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ORDER = ["low_risk_contacts", "low_risk_exposure", "risky_contacts",
         "risky_exposure", "single_day_transmissions", "all_transmissions"]


CONTACT_EXPOSURE = ORDER[:4]


LABELS = {
    "low_risk_contacts": "Non-risky contacts",
    "low_risk_exposure": "Non-risky exposure",
    "risky_contacts": "Risky contacts",
    "risky_exposure": "Risky exposure",
    "single_day_transmissions": "Transmissions",
    "all_transmissions": "All transmissions",
    "all_exposures": "All exposures",
}


CASES = {
    "main": "Overall",
    "2021": "2021", "2022": "2022",
    "weekday": "Monday–Friday", "weekend": "Saturday–Sunday",
    "premier": "Premier League", "cups": "FA and EFL Cups",
    "top_teams": "Most-supported clubs", "lesser_teams": "Lesser-supported clubs",
    "no_stadium_lads": "Exclude stadium LTLAs",
    "drop_top_transmissions": "Exclude top 10 outcome LTLAs",
    "drop_top_tv": "Exclude top 10 TV-exposure LTLAs",
    "drop_top_home": "Exclude top 10 home-exposure LTLAs",
    "drop_top_exposure_union": "Exclude top TV/home-exposure LTLAs",
}
def save_figure(fig, output, stem, pdf_flags=False):
    """Keep the native canvas size, embed fonts, and write a PNG preview too."""
    fig.canvas.draw()
    fonts = sorted({item.get_fontsize() for item in fig.findobj(Text)
                    if item.get_visible() and item.get_text().strip()})
    if fonts and (min(fonts) < 6 or max(fonts) > 12):
        raise ValueError("Set visible text to 6-12 pt before exporting at final size.")
    width, height = fig.get_size_inches() * 2.54
    settings = {
        "pdf.fonttype": 42,       # Embedded TrueType fonts; do not outline text.
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "savefig.bbox": None,    # SciencePlots uses tight cropping by default.
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
    if pdf_flags:
        # Preserve the existing flag-PDF fix: avoid image predictor compression.
        settings["pdf.compression"] = 0
    with plt.rc_context(settings):
        # Do not use bbox_inches="tight": that changes the physical page size.
        fig.savefig(output / f"{stem}.pdf", bbox_inches=None, dpi=600)
        fig.savefig(output / f"{stem}.png", bbox_inches=None, dpi=300)
    (output / "export_settings.json").write_text(json.dumps({
        "width_cm": round(float(width), 4),
        "height_cm": round(float(height), 4),
        "font_sizes_pt": fonts,
        "submission_file": f"{stem}.pdf",
        "preview_file": f"{stem}.png",
        "pdf_fonttype": 42,
        "color_mode": "RGB",
        "tight_cropping": False,
        "flag_pdf_workaround": pdf_flags,
        "export_helper": str(Path(__file__).resolve()),
    }, indent=2) + "\n")
    plt.close(fig)

# Here load the proper subset effect estimates and intervals (not supplied).
EFFECTS_FILE = DATA / 'example_subset_effects.csv'
# Columns: outcome (risky_contacts/risky_exposure), case (keys of CASES above),
# term (TV/Home), estimate_pct, ci_low_pct, ci_high_pct. Numeric fields are percentages.

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    data = pd.read_csv(EFFECTS_FILE, dtype={'case': str})
    output = ROOT / 'results/figures/figure_3'
    output.mkdir(parents=True, exist_ok=True)
    save_figure(draw(data), output, 'figure_3')
    print(output)

if __name__ == "__main__":
    main()
