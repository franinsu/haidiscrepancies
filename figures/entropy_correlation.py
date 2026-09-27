"""Lower-triangle entropy comparisons with diagonal puzzle histograms."""

def entropy_scale(v):
    import numpy as np
    pad=max(.02,.05*np.ptp(v));low=round(max(0,.05*np.floor((min(v)-pad)/.05)),10);high=round(min(1,.05*np.ceil((max(v)+pad)/.05)),10)
    span=high-low;step=.5 if span>=.95 else .25 if span>=.8 else .2 if span>=.6 else .1 if span>=.3 else .05
    ticks=np.round(np.arange(step*np.ceil((low-1e-12)/step),step*np.floor((high+1e-12)/step)+.5*step,step),10)
    return (low,high),ticks

def render(data):
    import itertools
    import numpy as np
    from matplotlib.lines import Line2D
    from .plot_style import (plt, SOURCES, LABEL, COLORS, FAMILIES, FLABEL,
                            FCOLOR, clean, dot, finalize)

    order = data['puzzle_order']
    types = np.array([data['puzzle_types'][p] for p in order])
    values = {s: np.array([data['values_by_puzzle'][s][p] for p in order]) for s in SOURCES}
    scales = {s: entropy_scale(values[s]) for s in SOURCES}
    # Match the entropy-profile figure's bins, shared across all sources.
    bin_edges = np.linspace(0, 1, 21)
    histograms = {}
    for source in SOURCES:
        counts, _ = np.histogram(values[source], bins=bin_edges)
        assert counts.sum() == len(values[source]) == 100
        histograms[source] = {'counts': counts.tolist(),
                              'percent': (100 * counts / len(values[source])).tolist()}
    max_percent = max(max(hist['percent']) for hist in histograms.values()) * 1.10

    width, height = 180, 166
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor='white')
    grid = fig.add_gridspec(4, 4, left=20 / width, right=160 / width,
                           bottom=17 / height, top=157 / height,
                           wspace=.15, hspace=.15)
    for row, source_y in enumerate(SOURCES):
        for col, source_x in enumerate(SOURCES[:row + 1]):
            ax = fig.add_subplot(grid[row, col])
            xlim, xticks = scales[source_x]
            ax.set(xlim=xlim, xticks=xticks)
            ax.set_box_aspect(1)
            clean(ax, grid='both')
            ax.tick_params(length=2, width=.55, pad=2)
            if row == col:
                marginal = histograms[source_x]
                ax.stairs(marginal['percent'], bin_edges, color=COLORS[source_x],
                          fill=True, alpha=.13, linewidth=0)
                ax.stairs(marginal['percent'], bin_edges, color=COLORS[source_x], lw=.9)
                ax.set(ylim=(0, max_percent), yticks=[])
                ax.spines['left'].set_visible(False)
                ax.grid(False)
            else:
                x, y = values[source_x], values[source_y]
                for family in FAMILIES:
                    mask = types == family
                    dot(ax, x[mask], y[mask], FCOLOR[family], 9).set_clip_on(False)
                ylim, yticks = scales[source_y]
                ax.set(ylim=ylim, yticks=yticks)
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
    fig.text(.50, 3 / height, 'Normalized entropy', ha='center', fontsize=7)
    # Retain the source-data ordering independently of the new display order.
    points = [{'sources': [sx, sy], 'puzzle_ids': order,
               'x': values[sx].tolist(), 'y': values[sy].tolist(),
               'x_limits': scales[sx][0], 'y_limits': scales[sy][0]}
              for sx, sy in itertools.combinations(SOURCES, 2)]
    payload = {'panels':points, 'puzzle_types':data['puzzle_types'],
        'marginal':'Histogram of normalized entropy across puzzles; descriptive, not a confidence interval.',
        'histogram':{'bin_edges':bin_edges.tolist(), 'bin_width':.05,
            'units':'percent of puzzles',
            'method':'Equal-width bins; left-closed and right-open except the final bin, which includes 1',
            'sources':histograms}}
    finalize(fig)
    return fig, payload
