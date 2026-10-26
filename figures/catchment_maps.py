"""Export saved LAD catchments as PNG and compact PDF; no refitting."""
from pathlib import Path
import argparse
import gc
import re

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['pdf.compression'] = 9
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from shapely.geometry import Point

FINAL_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]


FINAL_STADIUMS = {
    "Arsenal": (-0.1084, 51.5549),
    "Aston Villa": (-1.8847, 52.5092),
    "Brentford": (-0.2887, 51.4908),
    "Brighton": (-0.0837, 50.8616),
    "Chelsea": (-0.1910, 51.4817),
    "Crystal Palace": (-0.0856, 51.3983),
    "Everton": (-2.9663, 53.4388),  # Goodison Park for the study period
    "Liverpool": (-2.9608, 53.4308),
    "Manchester City": (-2.2004, 53.4831),
    "Manchester United": (-2.2913, 53.4631),
    "Newcastle United": (-1.6224, 54.9752),
    "Tottenham Hotspur": (-0.0665, 51.6043),
    "West Ham United": (-0.0165, 51.5386),  # London Stadium
    "Wolverhampton Wanderers": (-2.1306, 52.5902),
}


ROOT = Path(__file__).resolve().parent
# Fixed intervals across teams and scenarios. Nonlinear bins retain small shares.
EDGES = [0, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 25, 100]
LABELS = ['0', '.01', '.025', '.05', '.1', '.25', '.5', '1', '2', '5', '10', '25', '100']
CMAP = ListedColormap(plt.colormaps['YlGnBu'](np.linspace(.08, .98, len(EDGES)-1)))
NORM = BoundaryNorm(EDGES, CMAP.N)


def draw(ax, frame, stadium, bounds, linewidth=.18, marker_size=28):
    frame.plot(column='share_pct', ax=ax, cmap=CMAP, norm=NORM,
               edgecolor='#6b7c86', linewidth=linewidth, rasterized=False)
    ax.scatter(stadium.x, stadium.y, marker='*', s=marker_size,
               color='#e44736', edgecolors='white', linewidths=.6, zorder=5)
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal'); ax.set_axis_off()


def colorbar(fig, rect, labelsize=9):
    ax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=ax,
                      orientation='horizontal', boundaries=EDGES, ticks=EDGES, spacing='uniform')
    cb.ax.set_xticklabels(LABELS, fontsize=labelsize)
    cb.outline.set_linewidth(.4)
    cb.set_label("Share of this team's modelled attendance allocated to each LAD (%)", fontsize=labelsize+1, labelpad=8)


def save_pdf(fig, path, vector_pdf=False, dpi=600):
    """Rasterize only map polygons; retain vector text, markers and colour bars."""
    layers = [collection for ax in fig.axes for collection in ax.collections
              if isinstance(collection, PatchCollection)]
    previous = [layer.get_rasterized() for layer in layers]
    temporary = path.with_suffix('.pdf.tmp')
    try:
        for layer in layers:
            layer.set_rasterized(not vector_pdf)
        fig.savefig(temporary, format='pdf', dpi=dpi, facecolor='white')
        temporary.replace(path)
    finally:
        for layer, rasterized in zip(layers, previous):
            layer.set_rasterized(rasterized)
        temporary.unlink(missing_ok=True)


def run(scenario, output, results_dir=None, vector_pdf=False):
    results_dir = results_dir or ROOT.parent/'data'
    data = pd.read_csv(results_dir/'map_catchments_317.csv')
    selected = data[data.scenario == scenario].copy()
    if selected.empty:
        raise ValueError(f'Unknown scenario {scenario!r}. Available: {sorted(data.scenario.unique())}')
    boundaries = gpd.read_file(ROOT.parent/'data/map_boundaries_317.geojson')
    boundaries = boundaries[boundaries.lad19cd.str.startswith('E')].to_crs(27700)
    if boundaries.lad19cd.duplicated().any():
        raise ValueError('Duplicate boundary codes')
    if set(selected.lad_code) != set(boundaries.lad19cd):
        raise ValueError('Catchment and boundary LAD codes do not match')
    output.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = boundaries.total_bounds
    bounds = [xmin-10000, ymin-10000, xmax+10000, ymax+10000]
    frames, stadiums, audit = {}, {}, []
    for team in ('Manchester United', 'Newcastle United'):
        values = selected[selected.team == team]
        if len(values) != 317 or values.lad_code.duplicated().any():
            raise ValueError(f'Incomplete or duplicated LAD coverage: {team}')
        if not np.isclose(values.q.sum(), 1) or not values.q.between(0, 1).all():
            raise ValueError(f'Invalid allocation shares: {team}')
        frame = boundaries.merge(values[['lad_code', 'q']], left_on='lad19cd', right_on='lad_code', validate='one_to_one')
        frame['share_pct'] = frame.q*100
        stadium = gpd.GeoSeries([Point(*FINAL_STADIUMS[team])], crs=4326).to_crs(27700).iloc[0]
        frames[team], stadiums[team] = frame, stadium
        fig = plt.figure(figsize=(10, 9), facecolor='white')
        fig.text(.055, .955, team, fontsize=24, weight='bold', color='#163244')
        fig.text(.055, .92, 'Modelled catchment across England’s 317 local authority districts', fontsize=11, color='#546975')
        ax = fig.add_axes([.02, .13, .64, .75])
        draw(ax, frame, stadium, bounds, marker_size=70)
        local = fig.add_axes([.67, .48, .30, .33])
        draw(local, frame, stadium, [stadium.x-45000, stadium.y-45000, stadium.x+45000, stadium.y+45000], .4, 85)
        local.set_title('Around the stadium · 90 km across', fontsize=10, color='#163244')
        # Label only the stadium LAD, avoiding invented town locations.
        containing = frame[frame.geometry.covers(stadium)]
        if len(containing):
            local.annotate(containing.iloc[0].lad19nm, (stadium.x, stadium.y), xytext=(5, -12),
                           textcoords='offset points', fontsize=8, color='#163244',
                           path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        top = frame.loc[frame.q.idxmax()]
        fig.text(.69, .40, '★ Stadium', color='#ca3a2e', fontsize=11)
        fig.text(.69, .34, f'Largest allocated share\n{top.lad19nm}\n{top.share_pct:.1f}%', fontsize=11,
                 color='#163244', linespacing=1.6, va='top')
        fig.text(.69, .25, 'Shares sum to 100% for each team.\nColour intervals are unequal,\nbut identical across all maps.\n\nThese are model estimates,\nnot observed supporter counts.', fontsize=9, color='#546975', linespacing=1.5, va='top')
        colorbar(fig, [.09, .095, .82, .018])
        filename = re.sub(r'\W+', '_', team.lower())+'.png'
        fig.savefig(output/filename, dpi=220, facecolor='white')
        pdf_filename = Path(filename).with_suffix('.pdf').name
        save_pdf(fig, output/pdf_filename, vector_pdf=vector_pdf)
        plt.close(fig)
        gc.collect()
        audit.append(dict(team=team, lad_count=len(frame), share_sum=float(frame.q.sum()),
                          largest_lad=top.lad19nm, largest_share_pct=top.share_pct,
                          file=filename, pdf_file=pdf_filename))
    pd.DataFrame(audit).to_csv(output/'map_checks.csv', index=False)
    (output/'README.md').write_text(
        f'# Catchment maps: {scenario}\n\n'
        'Each map shows q × 100: the percentage of a team’s modelled attendance allocated to each LAD. '
        'It is not W (attendance per resident), and is not the percentage of local residents attending. '
        'Larger-population LADs can receive larger shares. Shares are season-independent in this model; '
        'seasonal attendance scaling changes counts, not these shares.\n\n'
        'All maps use the same unequal colour intervals. Boundaries are the supplied 2019 LAD geometry, '
        'projected to British National Grid. All 317 English LADs are included, including the Isles of Scilly. '
        'Individual figures include a stadium-area inset. No smoothing or interpolation is applied.\n\n'
        'Both individual maps are exported as PNG and PDF. '
        + ('PDF maps are fully vector.\n\n' if vector_pdf else
           'PDF map shapes use 600 dpi for individual maps; '
           'text, stadium markers and colour bars remain vector. No boundary simplification is applied.\n\n') +
        'Run catchment_maps.py --scenario SCENARIO to plot another saved robustness specification.\n')
    print(f'Saved the two requested individual maps as PNG and PDF to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='regional_non_big_six')
    parser.add_argument('--results-dir', type=Path, default=ROOT.parent/'data',
                        help='Directory containing map_catchments_317.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--vector-pdf', action='store_true',
                        help='Keep every boundary vertex as vector paths (larger PDFs)')
    args = parser.parse_args()
    run(args.scenario, args.output_dir or ROOT.parent/'results/figures/catchment_maps'/args.scenario,
        args.results_dir, vector_pdf=args.vector_pdf)
