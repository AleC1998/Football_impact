"""Season daily impacts above; home/TV exposure-normalized coefficients below."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scienceplots  # Registers the SciencePlots styles with Matplotlib.
from matplotlib.ticker import PercentFormatter

# Two-column figure. All font sizes are points at the final printed size.
WIDTH_CM, HEIGHT_CM = 17.8, 15.0
FONT_PT, TICK_PT, LEGEND_PT = 9, 8, 8


# Keep the style local and preserve its serif font when exporting later.
@plt.style.context(["science", "no-latex", {
    "font.family": "STIXGeneral", "font.size": FONT_PT,
    "axes.labelsize": FONT_PT, "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT, "legend.fontsize": LEGEND_PT,
}])
def draw(daily, coefficients):
    # Existing explicit colours, sizes, labels and layout override style defaults.
    fig = plt.figure(figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.4, wspace=0.4)
    top = fig.add_subplot(grid[0, :])
    bottom = [fig.add_subplot(grid[1, i]) for i in range(2)]
    dates = pd.date_range(daily.date.min(), daily.date.max())
    x = mdates.date2num(dates.to_pydatetime())
    for j, (term, color, label) in enumerate([
        ("stadium", "blue", "Home match (stadium attendance)"),
        ("tv", "red", "TV viewing (venues/pubs or at home)")]):
        values = daily[daily.term.eq(term)].set_index("date").reindex(dates)
        top.bar(x + (j - .5) * .5, values.relative_impact_pct, width=.6,
                color=color, label=label, edgecolor="none", linewidth=0, zorder=3)
    for boundary in mdates.date2num(dates[dates.dayofweek == 6].to_pydatetime()) + .5:
        top.axvline(boundary, color="#888888", linewidth=.35, alpha=.65, zorder=1)
    top.axhline(0, color="black", linewidth=.8, zorder=4)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    top.xaxis.set_major_locator(locator)
    top.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    top.xaxis.get_offset_text().set_fontsize(TICK_PT)
    top.set_xlim(x[0] - .65, x[-1] + .65)
    top.set_ylabel("Relative daily effect", fontsize=FONT_PT)
    top.tick_params(axis="both", labelsize=TICK_PT)
    top.tick_params(axis="x", labelrotation=20)
    for label in top.get_xticklabels():
        label.set_horizontalalignment("right")
    top.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2,
               frameon=False, fontsize=LEGEND_PT)
    top.spines[["top", "right"]].set_visible(False)

    indexed = coefficients.set_index(["input", "term"])
    x = np.arange(len(CONTACT_EXPOSURE))
    for ax, term, color, label in zip(bottom, ["stadium", "tv"], ["#4C72B0", "#C44E52"],
                                     ["Effect per individual: home match", "Effect per individual: TV viewing"]):
        rows = indexed.loc[[(name, term) for name in CONTACT_EXPOSURE]]
        y, low, high = rows[["coefficient", "ci_low", "ci_high"]].to_numpy().T
        ax.bar(x, y, width=.55, color=color, edgecolor="black", linewidth=.7, zorder=2)
        ax.axhline(0, color="black", linewidth=.8, zorder=3)
        ax.vlines(x, low, high, color="black", linewidth=.8, zorder=4)
        for endpoint in (low, high):
            ax.hlines(endpoint, x - .15, x + .15, color="black", linewidth=.8, zorder=4)
        ax.set_xticks(x, [LABELS[name].replace("Non-risky", "Low-risk") for name in CONTACT_EXPOSURE],
                      rotation=20, ha="right")
        ax.set_ylabel(label, fontsize=FONT_PT)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.tick_params(axis="both", labelsize=TICK_PT)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.105, right=.985, top=.92, bottom=.13)
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

# Here load the proper daily effects and coefficient estimates (not supplied).
DAILY_FILE = DATA / 'example_daily_effects.csv'
# Daily columns: date, term (tv/stadium), relative_impact_pct.
COEFFICIENT_FILE = DATA / 'example_coefficient_estimates.csv'
# Coefficient columns: input (one of CONTACT_EXPOSURE above), term (tv/stadium),
# coefficient, ci_low, ci_high. All three numeric fields must already be in percent.

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    daily = pd.read_csv(DAILY_FILE, parse_dates=['date'])
    coefficients = pd.read_csv(COEFFICIENT_FILE)
    output = ROOT / 'results/figures/figure_2'
    output.mkdir(parents=True, exist_ok=True)
    save_figure(draw(daily, coefficients), output, 'figure_2')
    print(output)

if __name__ == "__main__":
    main()
