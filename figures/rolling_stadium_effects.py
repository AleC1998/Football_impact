"""Reproduce the two supplied rolling-window stadium-effect figures."""
import argparse
from datetime import date
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
from datetime import timedelta
def month_start(index):
    year, month = divmod(index, 12)
    return date(year, month + 1, 1)


def make_windows(date_min, date_max):
    """Slide three calendar months by one month, clipping only study endpoints.

    Require three distinct calendar months: do not append two-month or one-month
    trailing windows. Windows may cross December/January and use inclusive dates.
    """
    start, end = date.fromisoformat(date_min), date.fromisoformat(date_max)
    if start > end:
        raise ValueError('Study start must not follow study end')
    first, last = start.year * 12 + start.month - 1, end.year * 12 + end.month - 1
    windows = {}
    for index in range(first, last - 1):
        a, b = month_start(index), month_start(index + 3) - timedelta(days=1)
        label = f'{a:%Y-%m}_to_{b:%Y-%m}'
        windows[label] = (max(a, start).isoformat(), min(b, end).isoformat())
    if not windows:
        raise ValueError('The study must cover at least three calendar months')
    return windows


def window_metadata(window):
    start, end = WINDOWS[window]
    first_month, last_month = window.split('_to_')
    last = date.fromisoformat(last_month + '-01')
    calendar_end = month_start(last.year * 12 + last.month) - timedelta(days=1)
    return dict(window=window, date_min=start, date_max=end,
        calendar_start=first_month + '-01', calendar_end=calendar_end.isoformat(),
        partial_window=start != first_month + '-01' or end != calendar_end.isoformat(),
        calendar_months=3, step_months=1,
        calendar_days=(date.fromisoformat(end) - date.fromisoformat(start)).days + 1)

WINDOWS = make_windows('2021-06-01', '2022-07-09')
OUTCOMES = {'risky_contacts':'Risky contacts', 'risky_exposure':'Risky exposure',
            'all_transmissions':'All transmissions'}
NORMAL_EDGE = 0.7


SIGNIFICANT_EDGE = 3.2


COLORS = {'risky_exposure': '#397DA8', 'risky_contacts': '#359080',
          'all_transmissions': '#B9793D'}


def interval_significant(low, high):
    """Two-sided nominal significance; an interval touching zero is not significant."""
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError('Invalid confidence interval')
    return bool(low > 0 or high < 0)


def window_label(window):
    info = window_metadata(window)
    start, end = date.fromisoformat(info['date_min']), date.fromisoformat(info['date_max'])
    first = start.strftime('%b') if start.year == end.year else start.strftime('%b %Y')
    last = end.strftime('%b %Y')
    if info['partial_window']:
        last = f'{end.day} {last}'
    return f'{first} – {last}'


def axis_limits(values):
    numbers = values.loc[values.available, ['estimate_pct', 'ci_low_pct', 'ci_high_pct']].to_numpy()
    if numbers.size == 0:
        return (-1., 1.)
    lower, upper = min(0., numbers.min()), max(0., numbers.max())
    padding = 0.12 * (upper - lower) if upper > lower else 1.
    return (lower - padding, upper + padding)


def make_figure(values, outcome, case='main', limits=None):
    """Create an in-memory figure. This function does not save or display anything."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    rows = values[values.outcome.eq(outcome)].reset_index(drop=True)
    fig = Figure(figsize=(10.5, max(6.0, 0.43 * len(rows) + 1.4)), facecolor='white')
    ax = fig.subplots()
    fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.14)
    color = COLORS[outcome]
    for y, row in rows.iterrows():
        if row.available:
            # Independent endpoints also support intervals that do not bracket the point estimate.
            ax.barh(y, row.estimate_pct, height=0.62, color=color, edgecolor='#171717',
                    linewidth=row.edge_width, zorder=3)
            ax.hlines(y, row.ci_low_pct, row.ci_high_pct, color='#171717', linewidth=1.15, zorder=4)
            ax.vlines([row.ci_low_pct, row.ci_high_pct], y - 0.10, y + 0.10,
                      color='#171717', linewidth=1.15, zorder=4)
            if row.estimate_pct == 0:
                ax.plot(0, y, marker='|', color='#171717', markersize=9, zorder=5)
        else:
            label = {'failed': 'Fit failed', 'pending': 'Pending', 'running': 'Running',
                     'interrupted': 'Interrupted', 'not_estimable': 'Not estimable',
                     'report_failed': 'Validation failed'}.get(row.row_status, 'Unavailable')
            ax.text(0.98, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center',
                    color='#707070', fontsize=9, fontstyle='italic')
    labels = rows.window_label.tolist()
    ax.set_yticks(np.arange(len(rows)), labels=labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)  # Earliest window at the top.
    ax.set_xlim(limits if limits is not None else axis_limits(rows))
    ax.axvline(0, color='#494949', linewidth=1.1, zorder=2)
    ax.set_xlabel('Stadium effect (% relative impact)', fontsize=11, labelpad=10)
    ax.set_ylabel('Time window', fontsize=11, labelpad=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.xaxis.grid(True, color='#E4E6E8', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#B0B0B0')
    fig.legend(handles=[
        Patch(facecolor=color, edgecolor='#171717', linewidth=SIGNIFICANT_EDGE, label='95% interval excludes zero'),
        Patch(facecolor=color, edgecolor='#171717', linewidth=NORMAL_EDGE, label='95% interval includes zero'),
        Line2D([0], [0], color='#171717', linewidth=1.15, marker='|', label='95% interval')],
        loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=9)
    return fig

# Here load the proper rolling-window estimates and intervals (not supplied).
EFFECTS_FILE = ROOT / 'data/example_window_effects.csv'
# Columns: window (e.g. 2021-06_to_2021-08), outcome (keys of OUTCOMES),
# case (main), term (Home), row_status (completed/failed/pending/not_estimable),
# estimate_pct, ci_low_pct, ci_high_pct. Unavailable rows use blank numeric cells.
# Include all three outcomes to retain the original common horizontal axis.

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    values = pd.read_csv(EFFECTS_FILE)
    values = values[values['case'].eq('main') & values.term.eq('Home')].copy()
    expected = {(w, o) for w in WINDOWS for o in OUTCOMES}
    if values.duplicated(['window', 'outcome']).any() or set(zip(values.window, values.outcome)) != expected:
        raise ValueError('Expected one Home-effect row per outcome and window, including unavailable rows')
    values['window_label'] = values.window.map(window_label)
    values['available'] = values.row_status.eq('completed')
    values['edge_width'] = [SIGNIFICANT_EDGE if r.available and interval_significant(r.ci_low_pct, r.ci_high_pct)
                           else NORMAL_EDGE for r in values.itertuples()]
    values['order'] = values.window.map({w:i for i,w in enumerate(WINDOWS)})
    values = values.sort_values('order', kind='stable')
    limits = axis_limits(values)
    output = ROOT / 'results/figures/rolling_stadium_effects'
    output.mkdir(parents=True, exist_ok=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for outcome in ('risky_contacts', 'risky_exposure'):
        fig = make_figure(values, outcome, 'main', limits)
        FigureCanvasAgg(fig)
        for extension in ('pdf', 'png'):
            fig.savefig(output / f'stadium_effect_{outcome}.{extension}', dpi=300, facecolor='white')
        fig.clear()
    print(output)

if __name__ == '__main__':
    main()
