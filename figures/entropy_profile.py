"""Selected entropy figure; numerical estimates enter as data."""
from pathlib import Path


import json


def build_data(stats):
    """Rebuild the display export from frozen entropy values and intervals."""
    import numpy as np

    palette = json.loads(Path(__file__).with_name('source_palette.json').read_text())
    sources = palette['source_order'][1:]
    labels = ['Human', 'GPT', 'Claude', 'Gemini']
    families = ['arithmetic24', 'maze', 'grid_placement', 'minesweeper_lite', 'mini_sudoku']
    records = []
    for family_index, family in enumerate(families):
        ids = [pid for pid in stats['puzzle_order'] if stats['puzzle_types'][pid] == family]
        for source in sources:
            saved = stats['by_type'][family]['sources'][source]
            values = [stats['values_by_puzzle'][source][pid] for pid in ids]
            assert values == saved['values'] and len(ids) == 20
            records.append(dict(family=family, source=source, puzzle_ids=ids, entropy=values,
                                summary_y=family_index, mean=saved['mean'], ci95=saved['ci95']))
    bin_edges = np.linspace(0, 1, 21)
    histograms = []
    for source, label in zip(sources, labels):
        ids = stats['puzzle_order']
        values = np.asarray([stats['values_by_puzzle'][source][pid] for pid in ids])
        counts, _ = np.histogram(values, bins=bin_edges)
        assert counts.sum() == len(values) == 100
        histograms.append(dict(source=source, label=label, puzzle_ids=ids,
                               entropy=values.tolist(), counts=counts.tolist(),
                               percent=(100 * counts / len(values)).tolist()))
    return dict(
        sources=sources, source_labels=labels, source_colors=palette['colors'][1:],
        families=families, family_labels=['Arithmetic', 'Maze', 'Rooks', 'Minesweeper', 'Sudoku'],
        summary={'records': records},
        histogram=dict(bin_edges=bin_edges.tolist(), bin_width=.05,
            units='percent of puzzles',
            method='Equal-width bins; left-closed and right-open except the final bin, which includes 1',
            sources=histograms),
        display=dict(histogram_fill_alpha=.12,
            summary_interval_style=dict(mean_diameter_pt=5, mean_area_pt2=25,
                interval_width_pt=.65, cap_height_pt=5.8, cap_stroke_width_pt=.55, alpha=.8)))


def render(stats):
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.path import Path as MarkerPath
    from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
    # Font hashes and the source palette are checked by the shared style module.
    from .plot_style import plt

    data = build_data(stats)
    records, histograms = data['summary']['records'], data['histogram']
    style = data['display']['summary_interval_style']
    vertices = np.array([(.25, .5), (.06, .38), (0, .18), (0, 0),
                         (0, -.18), (.06, -.38), (.25, -.5)])
    codes = [MarkerPath.MOVETO] + [MarkerPath.CURVE4] * 6
    caps = [MarkerPath(vertices, codes), MarkerPath(vertices * [-1, 1], codes)]
    with plt.rc_context({'axes.linewidth': .5, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'}):
        width, height = 183, 60
        fig = plt.figure(figsize=(width/25.4, height/25.4), dpi=300, facecolor='white')
        summary = fig.add_axes([26/width, 17/height, 68/width, 32/height])
        histogram = fig.add_axes([114/width, 17/height, 65/width, 32/height])

        # Preserve artist insertion order as well as z-order at tied intervals.
        order_by_family = {}
        for family in data['families']:
            indices = [i for i, r in enumerate(records) if r['family'] == family]
            order_by_family[family] = sorted(indices, key=lambda i: records[i]['ci95'][1]-records[i]['ci95'][0], reverse=True)
        for i, record in enumerate(records):
            color = data['source_colors'][data['sources'].index(record['source'])]
            y = record['summary_y']
            rank = order_by_family[record['family']].index(i)
            summary.plot(record['ci95'], [y, y], color=color,
                         lw=style['interval_width_pt'], alpha=style['alpha'],
                         solid_capstyle='butt', zorder=2+rank*.01)
            summary.scatter(record['mean'], y, s=style['mean_area_pt2'], color=color,
                            edgecolors='none', alpha=style['alpha'], zorder=4+rank*.01)
        for family in data['families']:
            for rank, i in enumerate(order_by_family[family]):
                record = records[i]
                color = data['source_colors'][data['sources'].index(record['source'])]
                for endpoint, path in zip(record['ci95'], caps):
                    summary.plot(endpoint, record['summary_y'], marker=path,
                                 markersize=style['cap_height_pt'], markeredgewidth=style['cap_stroke_width_pt'],
                                 markeredgecolor=color, markerfacecolor='none', linestyle='none',
                                 alpha=style['alpha'], zorder=3+rank*.01)
        summary.set(ylim=(4.55, -.55), xlim=(-.02, 1.02), xticks=[0, .25, .5, .75, 1], yticks=range(5))
        summary.set_yticklabels(data['family_labels'], fontsize=7)

        for record, color in zip(histograms['sources'], data['source_colors']):
            histogram.stairs(record['percent'], histograms['bin_edges'], color=color,
                             fill=True, alpha=data['display']['histogram_fill_alpha'],
                             linewidth=0, zorder=1)
            histogram.stairs(record['percent'], histograms['bin_edges'], color=color,
                             linewidth=.7, zorder=2)
        histogram.set(xlim=(0, 1), xticks=[0, .25, .5, .75, 1],
                      ylim=(0, max(max(r['percent']) for r in histograms['sources']) + 2))
        histogram.yaxis.set_major_locator(MultipleLocator(5))
        histogram.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        histogram.set_ylabel('Percentage of puzzles', labelpad=5)
        for ax, letter, title in [(summary, 'a', 'Mean entropy and 95% bootstrap CI'),
                                  (histogram, 'b', 'Histograms')]:
            ax.set_title(title, loc='left', pad=13, fontsize=7, y=1)
            ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-13, 13),
                        textcoords='offset points', fontweight='bold', fontsize=8, annotation_clip=False)
            ax.set_xlabel('Normalized entropy', labelpad=4)
            ax.grid(False)
            ax.tick_params(length=2, width=.5, pad=3)
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        handles = [Patch(facecolor=color + '22', edgecolor=color, linewidth=.7, label=label)
                   for color, label in zip(data['source_colors'], data['source_labels'])]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.55, 2.2/height),
                   frameon=False, ncol=4, fontsize=7, handlelength=1.5,
                   handletextpad=.5, columnspacing=1.7, borderaxespad=0)
    return fig, data
