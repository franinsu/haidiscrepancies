"""Shared colors, typography and interval marks."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.path import Path as MarkerPath
from matplotlib.ticker import FuncFormatter, MaxNLocator

SOURCES = ['Human', 'GPT 5.6 Sol', 'Claude Opus 4.8', 'Gemini 3.5 Flash']
LABEL = dict(zip(SOURCES, ['Human', 'GPT', 'Claude', 'Gemini']))
LABEL['Uniform'] = 'Uniform'
SOURCE_PALETTE = json.loads(Path(__file__).with_name('source_palette.json').read_text())
assert SOURCE_PALETTE['source_order'] == ['Uniform'] + SOURCES
COLORS = dict(zip(SOURCE_PALETTE['source_order'], SOURCE_PALETTE['colors']))
FAMILIES = ['arithmetic24', 'maze', 'grid_placement', 'minesweeper_lite', 'mini_sudoku']
FLABEL = dict(zip(FAMILIES, ['Arithmetic', 'Maze', 'Rooks', 'Minesweeper', 'Sudoku']))
FCOLOR = dict(zip(FAMILIES, ['#E377C2', '#F0C808', '#1B5E20', '#56B4E9', '#F58518']))
WIDTH = 180 / 25.4
# The paper's arxiv.sty uses a 6.6-inch text block. Shared 180-mm artwork is
# reduced to that width (and 117-mm artwork to .65 of it) on inclusion.
MANUSCRIPT_FIGURE_SCALE = (6.6 * 25.4) / 180
# DejaVu Sans is distributed with the pinned Matplotlib package. An optional
# locally licensed font may be selected explicitly; no proprietary font is bundled.
FONT_FAMILY = os.environ.get('STUDY_FONT_FAMILY', 'DejaVu Sans')
FONT_DIRECTORY = os.environ.get('STUDY_FONT_DIR')
if FONT_DIRECTORY:
    for font in sorted(Path(FONT_DIRECTORY).iterdir()):
        if font.suffix.lower() in {'.ttf', '.otf'}:
            font_manager.fontManager.addfont(str(font))
font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
# Match the selected Figure 3 means and Figure 2 geometry markers.
INTERVAL_MARKER_AREA = 25
INTERVAL_LINE_WIDTH = .65
INTERVAL_ALPHA = .8
INTERVAL_CAP_HEIGHT = 5.8
INTERVAL_CAP_WIDTH = .55
GEOMETRY_MARKER_AREA = 55
GEOMETRY_LABEL_SIZE = 7
LEGEND_MARKER_SIZE = 5
plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': 7,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT_FAMILY,
    'mathtext.it': FONT_FAMILY + ':italic', 'mathtext.bf': FONT_FAMILY + ':bold',
    'mathtext.sf': FONT_FAMILY, 'mathtext.fallback': 'stixsans',
    'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 7, 'axes.linewidth': .6,
    'lines.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})

def clean(ax, grid='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color='#e5e7eb', linewidth=.4)
    ax.tick_params(length=2, width=.5)
    formatter=lambda value,pos: '0' if abs(value)<1e-10 else f'{value:g}'
    if grid in ('x','both'):ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    if grid in ('y','both'):ax.yaxis.set_major_formatter(FuncFormatter(formatter))

def dot(ax, x, y, color, size=15, *, alpha=.9):
    return ax.scatter(x, y, color=color, s=size, edgecolors='none', alpha=alpha, zorder=3)

def geometry_dot(ax, x, y, color):
    return dot(ax, x, y, color, GEOMETRY_MARKER_AREA, alpha=1)

def confidence_interval(ax, ci, position, color, *, horizontal=False):
    """Draw the interval with inward-facing parentheses at its exact endpoints."""
    if horizontal:
        line, = ax.plot(ci, [position, position], color=color,
                       lw=INTERVAL_LINE_WIDTH, alpha=INTERVAL_ALPHA,
                       solid_capstyle='butt', zorder=2)
    else:
        line, = ax.plot([position, position], ci, color=color,
                       lw=INTERVAL_LINE_WIDTH, alpha=INTERVAL_ALPHA,
                       solid_capstyle='butt', zorder=2)
    vertices = [(.25, .5), (.06, .38), (0, .18), (0, 0),
                (0, -.18), (.06, -.38), (.25, -.5)]
    codes = [MarkerPath.MOVETO] + [MarkerPath.CURVE4] * 6
    caps = []
    for endpoint, sign in zip(ci, [1, -1]):
        shape = [(sign*x, y) for x, y in vertices]
        if not horizontal:
            shape = [(-y, x) for x, y in shape]
        x, y = (endpoint, position) if horizontal else (position, endpoint)
        cap, = ax.plot(x, y, marker=MarkerPath(shape, codes),
                       markersize=INTERVAL_CAP_HEIGHT, markeredgewidth=INTERVAL_CAP_WIDTH,
                       markeredgecolor=color, markerfacecolor='none', linestyle='none',
                       alpha=INTERVAL_ALPHA, zorder=2)
        caps.append(cap)
    return line, caps

def point_ci(ax, x, y, ci, color, horizontal=False):
    confidence_interval(ax, ci, y if horizontal else x, color, horizontal=horizontal)
    dot(ax, x, y, color, INTERVAL_MARKER_AREA, alpha=INTERVAL_ALPHA)

def legend(sources=SOURCES):
    return [Line2D([], [], ls='', marker='o', markeredgewidth=0,
                   color=COLORS[s], alpha=INTERVAL_ALPHA, markersize=LEGEND_MARKER_SIZE, label=LABEL[s]) for s in sources]

def panel(ax, letter, title=''):
    if title:
        ax.set_title(title, loc='left', pad=8)
    ax.annotate(letter, (-.10, 1), xycoords='axes fraction', xytext=(0, 8),
                textcoords='offset points', fontweight='bold', fontsize=8,
                annotation_clip=False)


def typography(fig, scale=1):
    """Physical type sizes: labels/legends 7 pt, numbers 6 pt, panel letters 8 pt.

    Scale compensates for manuscript inclusion, keeping existing plot sizes
    while making the type hierarchy consistent on the printed page.
    """
    from matplotlib.legend import Legend
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(text.get_fontsize() * scale)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(7 * scale)
            axis.offsetText.set_fontsize(6 * scale)
            for label in axis.get_ticklabels():
                try:
                    float(label.get_text().replace('\N{MINUS SIGN}', '-'))
                    size = 6
                except ValueError:
                    size = 7
                label.set_fontsize(size * scale)
        for title in (ax.title, ax._left_title, ax._right_title):
            is_panel = len(title.get_text()) == 1 and title.get_fontweight() == 'bold'
            title.set_fontsize((8 if is_panel else 7) * scale)
    for item in fig.findobj(Legend):
        for label in item.get_texts():
            label.set_fontsize(7 * scale)


def finalize(fig):
    """Use left-aligned regular panel titles in every quantitative renderer."""
    for ax in fig.axes:
        title=ax.get_title(loc='center')
        if title:
            color=ax.title.get_color()
            ax.set_title('',loc='center')
            ax.set_title(title,loc='left',pad=8,fontweight='normal',color=color)
        if ax.get_aspect()==1 and ax.get_adjustable()=='box':
            bounds=ax.get_position(original=True); width,height=fig.get_size_inches()
            ratio=bounds.width*width/(bounds.height*height)
            xl,yl=ax.get_xlim(),ax.get_ylim()
            dx,dy=xl[1]-xl[0],yl[1]-yl[0]
            dx,dy=max(dx,dy*ratio),max(dy,dx/ratio)
            cx,cy=sum(xl)/2,sum(yl)/2
            ax.set_xlim(cx-dx/2,cx+dx/2);ax.set_ylim(cy-dy/2,cy+dy/2)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
    typography(fig, scale=1 / MANUSCRIPT_FIGURE_SCALE)
