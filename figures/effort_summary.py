"""Render directly from numerical summaries supplied by the caller."""




def render(data):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter
    from .plot_style import (plt, SOURCES, COLORS, LABEL, FAMILIES, FLABEL, WIDTH,
                            point_ci, finalize, INTERVAL_ALPHA,
                            INTERVAL_LINE_WIDTH, LEGEND_MARKER_SIZE)
    # The wide layout is displayed at 62% of the manuscript text width.
    width = WIDTH * .62
    fig, ax = plt.subplots(figsize=(width, width * 190 / 300))
    fig.subplots_adjust(left=.235, right=.97, top=.94, bottom=.255)
    points = []
    for fi, family in enumerate(FAMILIES):
        for source in SOURCES:
            record = data['primary'][source]['by_type'][family]
            # Like Figure 3a, every source shares its family's row.
            point_ci(ax, record['mean'], fi, record['ci95'],
                     COLORS[source], horizontal=True)
            points.append({'source': source, 'family': family, **record})
    low = min(p['ci95'][0] for p in points)
    high = max(p['ci95'][1] for p in points)
    pad = .05 * (high - low)
    limits = (np.floor((low - pad) * 4) / 4,
              np.ceil((high + pad) * 4) / 4)
    ax.set(xlim=limits, ylim=(4.6, -.6), yticks=range(5),
           yticklabels=[FLABEL[f] for f in FAMILIES], xlabel='Relative difficulty')
    ax.set_xticks(np.arange(np.ceil(limits[0]), np.floor(limits[1]) + 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.tick_params(axis='x', length=2, width=.5)
    ax.tick_params(axis='y', length=2, width=.5, pad=4)
    ax.grid(False)
    ax.axvline(0, color='#9ca3af', lw=.6, zorder=0)
    handles = [Line2D([], [], color=COLORS[s], marker='o', markeredgewidth=0,
                      markersize=LEGEND_MARKER_SIZE, lw=INTERVAL_LINE_WIDTH,
                      alpha=INTERVAL_ALPHA, label=LABEL[s]) for s in SOURCES]
    fig.legend(handles=handles,
               loc='lower center', bbox_to_anchor=(.5, .005),
               frameon=False, ncol=4, handlelength=1.4,
               handletextpad=.5, columnspacing=1)
    finalize(fig)

    assert len(ax.collections) == 20
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            assert (box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width
                    and box.y1 <= fig.bbox.height), label.get_text()
    payload = {'points': points, 'bootstrap': data['bootstrap'],
               'primary_condition': data['primary_condition']}
    return fig, payload
