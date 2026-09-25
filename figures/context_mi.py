"""Plot ten computed MI contrasts."""
from matplotlib.text import Text
from .plot_style import plt, WIDTH, SOURCES, LABEL, COLORS, point_ci, legend, clean, finalize, FONT_FAMILY

def render(result):
    contrasts = result['contrasts']
    names = [LABEL[s] for s in SOURCES]
    colour = {LABEL[s]: COLORS[s] for s in SOURCES}
    contexts = ['Related', 'Unrelated']

    fig = plt.figure(figsize=(WIDTH, 72 / 25.4))
    left = fig.add_axes([.15, .27, .30, .55])
    right = fig.add_axes([.64, .27, .34, .55])
    fig.text(.035, .93, 'a', fontsize=8, fontweight='bold')
    fig.text(.15, .93, 'Related − unrelated', fontsize=7)
    fig.text(.53, .93, 'b', fontsize=8, fontweight='bold')
    fig.text(.64, .93, 'Human − model', fontsize=7)
    plotted = []
    for source in names:
        item, = [c for c in contrasts if c['comparison'] == 'Related minus unrelated'
                 and c['source'] == source]
        position = names.index(source)
        point_ci(left, item['estimate_bits'], position,
                 [item['ci95_low'], item['ci95_high']], colour[source], True)
        plotted.append({'panel': 'a', 'y': position, **item})
    for context in contexts:
        for source in names[1:]:
            item, = [c for c in contrasts if c['comparison'] == 'Human minus model'
                     and c['source'] == source and c['context'] == context]
            position = contexts.index(context)
            point_ci(right, item['estimate_bits'], position,
                     [item['ci95_low'], item['ci95_high']], colour[source], True)
            plotted.append({'panel': 'b', 'y': position, **item})
    left.set(yticks=range(4), yticklabels=names, ylim=(3.55, -.55),
             xlim=(-.10, .27), xticks=[-.1, 0, .1, .2])
    right.set(yticks=range(2), yticklabels=contexts, ylim=(1.55, -.55),
              xlim=(-.03, .65), xticks=[0, .2, .4, .6])
    for ax in [left, right]:
        ax.set_xlabel('MI difference (bits)')
        clean(ax, 'x')
        ax.grid(False)
        ax.axvline(0, color='#9ca3af', lw=.6, zorder=0)
    fig.legend(handles=legend(), ncol=4, frameon=False,
               loc='center', bbox_to_anchor=(.55, .055), handletextpad=.4, columnspacing=1.4)
    finalize(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), label.get_text()
            assert label.get_fontfamily() == [FONT_FAMILY]
    assert len(plotted) == 10
    return fig, [{k:r[k] for k in ('panel','comparison','source','context','estimate_bits','ci95_low','ci95_high') if k in r} for r in plotted]
