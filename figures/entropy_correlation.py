"""Lower-triangle entropy comparisons with one density curve per source."""

def entropy_scale(v):
    import numpy as np
    pad=max(.02,.05*np.ptp(v));low=round(max(0,.05*np.floor((min(v)-pad)/.05)),10);high=round(min(1,.05*np.ceil((max(v)+pad)/.05)),10)
    span=high-low;step=.5 if span>=.95 else .25 if span>=.8 else .2 if span>=.6 else .1 if span>=.3 else .05
    ticks=np.round(np.arange(step*np.ceil((low-1e-12)/step),step*np.floor((high+1e-12)/step)+.5*step,step),10)
    return (low,high),ticks

def marginal(values,limits):
    import numpy as np
    # Exact original bounded-entropy display smoother: fixed 0.075 bandwidth,
    # reflections at 0 and 1, and unit-area normalization on the 401-point grid.
    grid=np.linspace(*limits,401);bandwidth=.075
    direct=(grid[:,None]-values[None,:])/bandwidth
    low=(grid[:,None]+values[None,:])/bandwidth
    high=(grid[:,None]-(2-values[None,:]))/bandwidth
    density=np.sum(np.exp(-.5*direct**2)+np.exp(-.5*low**2)+np.exp(-.5*high**2),axis=1)/(np.sqrt(2*np.pi)*bandwidth*len(values))
    return grid,density/np.trapezoid(density,grid)

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
    # Keep the original reflected KDE and common [0,1] evaluation grid.
    curves = {s: marginal(values[s], (0, 1)) for s in SOURCES}
    max_density = max(density.max() for _, density in curves.values()) * 1.10

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
                density_grid, density = curves[source_x]
                ax.plot(density_grid, density, color=COLORS[source_x], lw=.9)
                ax.fill_between(density_grid, 0, density, color=COLORS[source_x], alpha=.13)
                ax.vlines(values[source_x], 0, max_density * .04,
                          color=COLORS[source_x], lw=.4, alpha=.45)
                ax.set(ylim=(0, max_density), yticks=[])
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
    fig.text(.50, 3 / height, 'Normalized entropy on both axes', ha='center', fontsize=7)
    # Retain the source-data ordering independently of the new display order.
    points = [{'sources': [sx, sy], 'puzzle_ids': order,
               'x': values[sx].tolist(), 'y': values[sy].tolist(),
               'x_limits': scales[sx][0], 'y_limits': scales[sy][0]}
              for sx, sy in itertools.combinations(SOURCES, 2)]
    payload = {'panels':points,'puzzle_types':data['puzzle_types'],'marginal':'Gaussian KDE with fixed bandwidth 0.075 and boundary reflections at 0 and 1; normalized over [0,1]; descriptive, not a confidence interval.'}
    finalize(fig)
    return fig, payload
