"""Relative-difficulty scatter matrix with diagonal puzzle histograms."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from .plot_style import plt, SOURCES, LABEL, COLORS, FAMILIES, FCOLOR, FLABEL, typography, MANUSCRIPT_FIGURE_SCALE

def render(summary):
    order = summary['puzzle_order']
    types = [summary['puzzle_types'][p] for p in order]
    if len(order) != 100 or len(set(order)) != 100:
        raise ValueError('Relative difficulty requires 100 distinct puzzles')
    if set(types) != set(FAMILIES):
        raise ValueError('Relative difficulty requires the five study families')
    families = np.asarray(types)
    common_limits = (-3, 3)
    bin_edges = np.linspace(*common_limits, 21)
    standardization, standardized, histograms = {}, {}, {}
    for source in SOURCES:
        raw = np.asarray([summary['primary'][source]['values_by_puzzle'][p] for p in order])
        if raw.shape != (100,) or not np.isfinite(raw).all():
            raise ValueError(f'{source}: expected 100 finite relative-difficulty values')
        mean, sd = float(raw.mean()), float(raw.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f'{source}: relative difficulty must have positive standard deviation')
        z = (raw - mean) / sd
        if not (z.min() > common_limits[0] and z.max() < common_limits[1]):
            raise ValueError(f'{source}: standardized values exceed the figure limits')
        standardized[source] = z
        standardization[source] = {'n': len(raw), 'mean_before': mean,
            'sample_sd_before': sd, 'ddof': 1, 'mean_after': float(z.mean()),
            'sample_sd_after': float(z.std(ddof=1))}
        counts, _ = np.histogram(z, bins=bin_edges)
        assert counts.sum() == len(z) == 100
        histograms[source] = {'counts': counts.tolist(),
                              'percent': (100 * counts / len(z)).tolist()}

    # Bake the selected manuscript crop into the exported figure itself.
    # Crop values are PDF points: left, bottom, right, top.
    original_width, original_height = 180, 166
    crop_left, crop_bottom, crop_right, crop_top = np.array([32, 4, 52, 30]) * 25.4 / 72
    width = original_width - crop_left - crop_right
    height = original_height - crop_bottom - crop_top
    common_ticks = [-3, -2, -1, 0, 1, 2, 3]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor='white')
    grid = fig.add_gridspec(4, 4, left=(20 - crop_left) / width, right=(160 - crop_left) / width,
                           bottom=(17 - crop_bottom) / height, top=(157 - crop_bottom) / height,
                           wspace=.15, hspace=.15)
    max_percent = max(max(hist['percent']) for hist in histograms.values()) * 1.10


    for row, source_y in enumerate(SOURCES):
        for col, source_x in enumerate(SOURCES[:row + 1]):
            ax = fig.add_subplot(grid[row, col])
            ax.set_xlim(common_limits)
            ax.set_xticks(common_ticks)
            ax.set_box_aspect(1)
            ax.tick_params(length=2, width=.55, pad=2)
            formatter = FuncFormatter(lambda value, pos: '0' if abs(value) < 1e-10 else f'{value:g}')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.spines[['top', 'right']].set_visible(False)
            if row == col:
                marginal = histograms[source_x]
                ax.stairs(marginal['percent'], bin_edges, color=COLORS[source_x],
                          fill=True, alpha=.13, linewidth=0)
                ax.stairs(marginal['percent'], bin_edges, color=COLORS[source_x], lw=.9)
                ax.set_ylim(0, max_percent)
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
            else:
                x, y = standardized[source_x], standardized[source_y]
                for family in FAMILIES:
                    mask = families == family
                    ax.scatter(x[mask], y[mask], color=FCOLOR[family],
                               s=9, edgecolors='none', alpha=.9, zorder=3)
                ax.set_ylim(common_limits)
                ax.set_yticks(common_ticks)
                ax.set_aspect('equal', adjustable='box')
                ax.axhline(0, color='#B3B8BF', lw=.55, zorder=1)
                ax.axvline(0, color='#B3B8BF', lw=.55, zorder=1)
            if row < 3:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(LABEL[source_x], color='black', labelpad=4)
            if col > 0:
                ax.tick_params(axis='y', labelleft=False)
            elif row > 0:
                ax.set_ylabel(LABEL[source_y], color='black', labelpad=5)

    handles = [Line2D([], [], ls='', marker='o', markersize=4.2,
                      markeredgewidth=0, color=FCOLOR[f], alpha=.9, label=FLABEL[f])
               for f in FAMILIES]
    legend = fig.legend(handles=handles, title='Puzzle family', loc='upper left',
                        bbox_to_anchor=((.59 * original_width - crop_left) / width,
                                        (.91 * original_height - crop_bottom) / height),
                        frameon=False, labelspacing=.8,
                        handletextpad=.5, borderaxespad=0)
    legend.get_title().set_fontsize(7)
    fig.text((.50 * original_width - crop_left) / width, (3 - crop_bottom) / height,
             'Standardized relative difficulty (z-score)', ha='center', fontsize=7, color='black')
    typography(fig, scale=1 / MANUSCRIPT_FIGURE_SCALE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
            raise ValueError(f'Figure label is clipped: {label.get_text()}')
    return fig, {'puzzle_ids':order, 'families':types,
        'z_scores':{s:v.tolist() for s,v in standardized.items()},
        'histogram':{'bin_edges':bin_edges.tolist(), 'bin_width':.3,
            'units':'percent of puzzles',
            'method':'Equal-width bins; left-closed and right-open except the final bin, which includes 3',
            'sources':histograms},
        'standardization':standardization}
