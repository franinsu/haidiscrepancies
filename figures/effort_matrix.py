"""Selected effort matrix. Points are per-puzzle summaries, never participant rows."""
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
    common_grid = np.linspace(*common_limits, 801)
    standardization, standardized, curves = {}, {}, {}
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
        # Express the original Scott bandwidth in standardized units.
        bandwidth = (sd * len(raw)**(-1/5)) / sd
        density = np.exp(-.5 * ((common_grid[:, None] - z[None, :]) / bandwidth) ** 2).mean(axis=1)
        density /= bandwidth * np.sqrt(2 * np.pi)
        curves[source] = {'grid': common_grid.tolist(), 'density': density.tolist(),
                          'bandwidth': bandwidth}

    width, height = 180, 166
    common_ticks = [-3, -2, -1, 0, 1, 2, 3]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor='white')
    grid = fig.add_gridspec(4, 4, left=20 / width, right=160 / width,
                           bottom=17 / height, top=157 / height,
                           wspace=.15, hspace=.15)
    max_density = max(max(curve['density']) for curve in curves.values()) * 1.10


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
                marginal = curves[source_x]
                ax.plot(marginal['grid'], marginal['density'], color=COLORS[source_x], lw=.9)
                ax.fill_between(marginal['grid'], 0, marginal['density'], color=COLORS[source_x], alpha=.13)
                ax.vlines(standardized[source_x], 0, max_density * .04,
                          color=COLORS[source_x], lw=.4, alpha=.45)
                ax.set_ylim(0, max_density)
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
                        bbox_to_anchor=(.59, .91), frameon=False, labelspacing=.8,
                        handletextpad=.5, borderaxespad=0)
    legend.get_title().set_fontsize(7)
    fig.text(.50, 3 / height, 'Standardized relative difficulty (z-score)', ha='center', fontsize=7, color='black')
    typography(fig, scale=1 / MANUSCRIPT_FIGURE_SCALE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
            raise ValueError(f'Figure label is clipped: {label.get_text()}')
    return fig, {'puzzle_ids':order, 'families':types, 'z_scores':{s:v.tolist() for s,v in standardized.items()}, 'density_curves':curves, 'standardization':standardization}
