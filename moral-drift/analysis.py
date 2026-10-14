"""Do the clusters move towards stay or swerve over a 20-year deployment?

    python analysis.py      (after python generate.py)

Reads data/choices.npz. For each scenario and cluster, prints how its average swerve share and
its majority choices change, which way the flips go, and the alignment horizon of a classifier
frozen at deployment, which keeps the year-0 choices. Saves figures to figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIGURES = ROOT / 'figures'
DILEMMAS = ROOT.parent / 'moral-classification' / 'data' / 'dilemmas.jsonl'

REPORT_YEARS = [0, 5, 10, 20]
HORIZON_DISAGREEMENT = 0.02   # the frozen classifier is misaligned once it disagrees on > 2% of dilemmas
TOP_FLIPS = 5

COLORS = {'western': '#2a78d6', 'eastern': '#eb6834', 'southern': '#1baf7a'}
INK, MUTED = '#2b2b2b', '#8a8a85'


def load():
    with np.load(DATA / 'choices.npz') as f:
        return {k: f[k] for k in f.files}


def flips(choice):
    """Shares of dilemmas flipped since year 0: (scenario, run, year, cluster) for each direction."""
    start = choice[:, :, :1]
    to_swerve = ((start == 0) & (choice == 1)).mean(axis=-1)
    to_stay = ((start == 1) & (choice == 0)).mean(axis=-1)
    return to_swerve, to_stay


def horizon(disagreement):
    """First year the disagreement exceeds the threshold, per run; inf if it never does."""
    over = disagreement > HORIZON_DISAGREEMENT
    years = np.argmax(over, axis=-2).astype(float)          # over (..., year, cluster)
    years[~over.any(axis=-2)] = np.inf
    return years


def summary(d):
    p, choice = d['p_swerve'].astype('float32'), d['choice']
    to_swerve, to_stay = flips(choice)
    first = horizon(to_swerve + to_stay)                     # (scenario, run, cluster)
    rows = []
    for s, scenario in enumerate(d['scenarios']):
        for i, cluster in enumerate(d['clusters']):
            for t in REPORT_YEARS:
                rows.append({
                    'scenario': scenario, 'cluster': cluster, 'year': t,
                    'swerve_share': p[s, :, t, i].mean(),
                    'swerve_choices': choice[s, :, t, i].mean(),
                    'stay->swerve': to_swerve[s, :, t, i].mean(),
                    'swerve->stay': to_stay[s, :, t, i].mean(),
                    'flipped_p90': np.percentile(to_swerve[s, :, t, i] + to_stay[s, :, t, i], 90),
                    'horizon_median': np.median(first[s, :, i]) if t == 0 else np.nan,
                })
    return pd.DataFrame(rows)


def most_flipped(d, scenario='social_influence'):
    """Dilemmas whose year-20 choice differs from year 0 in the most runs, per cluster."""
    s = list(d['scenarios']).index(scenario)
    text = {}
    for line in open(DILEMMAS):
        r = json.loads(line)
        text[r['id']] = r['natural']
    lines = []
    for i, cluster in enumerate(d['clusters']):
        changed = (d['choice'][s, :, -1, i] != d['choice'][s, :, 0, i]).mean(axis=0)
        lines.append(f'\n{cluster}: dilemmas most often flipped by year 20 ({scenario})')
        for k in np.argsort(-changed)[:TOP_FLIPS]:
            p0, p20 = d['p_swerve'][s, :, 0, i, k].mean(), d['p_swerve'][s, :, -1, i, k].mean()
            lines.append(f'  {changed[k]:4.0%} of runs, swerve share {p0:.3f} -> {p20:.3f}  '
                         f'{d["dilemma_ids"][k]}: {text[d["dilemma_ids"][k]]}')
    return '\n'.join(lines)


def style(ax):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis='y', color='#e6e6e3', linewidth=0.8)
    ax.set_axisbelow(True)


def plot_over_time(d, values, ylabel, title, path):
    """One panel per scenario, one line per cluster: median over runs and the 10-90% band."""
    years = np.arange(values.shape[2])
    fig, axes = plt.subplots(1, len(d['scenarios']), figsize=(11, 4), sharey=True)
    for s, ax in enumerate(axes):
        for i, cluster in enumerate(d['clusters']):
            v = values[s, :, :, i]
            lo, mid, hi = np.percentile(v, [10, 50, 90], axis=0)
            ax.fill_between(years, lo, hi, color=COLORS[cluster], alpha=0.18, linewidth=0)
            ax.plot(years, mid, color=COLORS[cluster], linewidth=2, label=cluster)
        ax.set_title(str(d['scenarios'][s]).replace('_', ' '), color=INK, fontsize=11)
        ax.set_xlabel('years since deployment', color=INK)
        ax.set_xlim(0, years[-1])
        ax.set_xticks(years[::5])
        style(ax)
    axes[0].set_ylabel(ylabel, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=len(labels))
    fig.suptitle(title, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    d = load()
    table = summary(d)
    pd.set_option('display.width', 160)
    print('Means over runs. swerve_share: average predicted share of swerve votes; swerve_choices: share of')
    print('dilemmas whose majority is swerve; stay->swerve, swerve->stay: shares of dilemmas flipped since')
    print(f'year 0; horizon_median: first year the year-0 choices disagree on > {HORIZON_DISAGREEMENT:.0%} of dilemmas.\n')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(most_flipped(d))

    FIGURES.mkdir(exist_ok=True)
    to_swerve, to_stay = flips(d['choice'])
    plot_over_time(d, d['p_swerve'].astype('float32').mean(axis=-1), 'mean swerve share',
                   'Average share of swerve votes over 2,295 dilemmas', FIGURES / 'swerve_share.png')
    plot_over_time(d, 100 * to_swerve, '% of dilemmas', 'Majority flipped from stay to swerve since deployment',
                   FIGURES / 'flips_to_swerve.png')
    plot_over_time(d, 100 * to_stay, '% of dilemmas', 'Majority flipped from swerve to stay since deployment',
                   FIGURES / 'flips_to_stay.png')
    print(f'\nFigures saved to {FIGURES}/')


if __name__ == '__main__':
    main()
