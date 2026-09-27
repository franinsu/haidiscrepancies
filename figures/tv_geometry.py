"""Figure 2: selected 23 September layout driven by recomputed statistics."""
import itertools
import json
import re
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from .plot_style import plt, FONT_FAMILY, INTERVAL_ALPHA
FAMILY_KEYS = ['arithmetic24', 'maze', 'grid_placement', 'minesweeper_lite', 'mini_sudoku']
FAMILIES = ['Arithmetic', 'Maze', 'Rooks', 'Minesweeper', 'Sudoku']
def axes_at(fig, box, page):
    x0, y0, x1, y1 = box
    w, h = page
    return fig.add_axes([x0/w, (h-y1)/h, (x1-x0)/w, (y1-y0)/h])

def cap_patch(fig, template, center, color, page):
    """Translate a frozen glyph outline to the recomputed scientific endpoint."""
    vertices, codes = [], []
    def mapped(point):
        return ((center[0]+point[0])/page[0], 1-(center[1]+point[1])/page[1])
    for segment in template['items']:
        kind, *points = segment
        if not vertices:
            vertices.append(mapped(points[0])); codes.append(MplPath.MOVETO)
        if kind == 'c':
            vertices.extend(mapped(p) for p in points[1:]); codes.extend([MplPath.CURVE4]*3)
        elif kind == 'l':
            vertices.append(mapped(points[1])); codes.append(MplPath.LINETO)
        else:
            raise ValueError(kind)
    fig.add_artist(PathPatch(MplPath(vertices, codes), transform=fig.transFigure,
                             facecolor=color, edgecolor=color, linewidth=.2, alpha=INTERVAL_ALPHA,
                             joinstyle='round', capstyle='butt', zorder=3))

def draw(bootstrap, geometry, layout, ab, output, *, previews=False):
    plt.rcParams.update({'axes.unicode_minus': False})
    page = layout['page_size_pt']
    fig = plt.figure(figsize=(page[0]/72, page[1]/72), dpi=300, facecolor='white')
    tv = np.asarray(bootstrap['global']['matrix'])
    ci = bootstrap['global']['ci95_puzzle_matrix']
    a = ab['panel_a']; box = a['plot_bbox_pt']
    ax = axes_at(fig, box, page)
    ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5)); ax.set_axis_off()
    cmap = plt.get_cmap(a['colormap'])
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                                  facecolor=cmap(tv[i, j]), edgecolor='none'))
    for v in [.5, 1.5, 2.5, 3.5]:
        ax.axhline(v, color='white', lw=a['white_grid_width_pt'], solid_capstyle='projecting')
        ax.axvline(v, color='white', lw=a['white_grid_width_pt'], solid_capstyle='projecting')
    bar = a['colorbar']
    cax = axes_at(fig, bar['axes_bbox_pt'], page)
    # Vector color scale: no reference image is reused in the reproduction.
    for k in range(256):
        color = cmap((k+.5)/256)
        cax.add_patch(Rectangle((0, k/256), 1, 1/256,
                               facecolor=color, edgecolor=color, linewidth=.3))
    cax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=bar['ticks'])
    cax.yaxis.tick_right()
    cax.tick_params(length=2, width=.6, labelright=False)
    for spine in cax.spines.values():
        spine.set_visible(False)

    b = ab['panel_b']; boxb = b['plot_bbox_pt']
    axb = axes_at(fig, boxb, page)
    axb.set(xlim=b['xlim'], ylim=b['ylim'], xticks=b['x_tick_values'], yticks=[])
    axb.tick_params(length=3, width=.6, labelbottom=False)
    for side in ['left', 'top', 'right']:
        axb.spines[side].set_visible(False)
    axb.spines['bottom'].set_linewidth(b['bottom_spine']['stroke_width_pt'])
    records = []
    for ri, (i, j) in enumerate(itertools.combinations(range(5), 2)):
        y = b['row_coordinates'][ri]
        for fi, key in enumerate(FAMILY_KEYS):
            stats = bootstrap['families'][key]
            color = b['family_colors'][fi]
            interval = stats['ci95_matrix'][i][j]
            mean = stats['matrix'][i][j]
            axb.plot(interval, [y, y], color=color, lw=.7, alpha=INTERVAL_ALPHA,
                     solid_capstyle='butt', zorder=2)
            records.append({'row': ri, 'family': FAMILIES[fi], 'mean': mean,
                            'ci95': interval, 'y': y, 'color': color})
    for row in records:
        y_pdf = boxb[1]+(row['y']-b['ylim'][1])/(b['ylim'][0]-b['ylim'][1])*(boxb[3]-boxb[1])
        for endpoint, side in zip(row['ci95'], ['left', 'right']):
            x_pdf = boxb[0]+endpoint*(boxb[2]-boxb[0])
            cap_patch(fig, b['cap_templates'][side], (x_pdf, y_pdf), row['color'], page)
    # Figure-level points sit above figure-level cap outlines.
    for row in records:
        x = (boxb[0]+row['mean']*(boxb[2]-boxb[0]))/page[0]
        y_pdf = boxb[1]+(row['y']-b['ylim'][1])/(b['ylim'][0]-b['ylim'][1])*(boxb[3]-boxb[1])
        fig.add_artist(Line2D([x], [1-y_pdf/page[1]], transform=fig.transFigure,
                             marker='o', markersize=4.2, markeredgewidth=1,
                             color=row['color'], alpha=INTERVAL_ALPHA, linestyle='none', zorder=4))
    for entry in ab['family_legend']:
        x0, y, x1, _ = entry['line']['bbox_pt']; color = entry['line']['stroke_color_hex']
        fig.add_artist(Line2D([x0/page[0], x1/page[0]], [1-y/page[1]]*2,
                             transform=fig.transFigure, color=color, lw=.7, alpha=INTERVAL_ALPHA,
                             solid_capstyle='projecting'))
        fig.add_artist(Line2D([(x0+x1)/2/page[0]], [1-y/page[1]],
                             transform=fig.transFigure, marker='o', markersize=4.2,
                             markeredgewidth=1, color=color, alpha=INTERVAL_ALPHA, linestyle='none'))

    for key in ['c', 'd']:
        style = layout['panel_'+key]
        box = style['plot_bbox_pt']
        data = geometry['source' if key == 'c' else 'family']
        graph = axes_at(fig, box, page)
        xlim = style['xlim'] if key == 'c' else style['axes_limits']['x']
        ylim = style['ylim'] if key == 'c' else style['axes_limits']['y']
        xticks = style['xticks'] if key == 'c' else style['tick_values']['x']
        yticks = style['yticks'] if key == 'c' else style['tick_values']['y']
        graph.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
        graph.tick_params(length=3, width=.6, labelbottom=False, labelleft=False)
        for side in ['top', 'right']:
            graph.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            graph.spines[side].set_linewidth(.8)
        graph.axhline(0, color='#9AA0B0', lw=.7, zorder=0)
        graph.axvline(0, color='#9AA0B0', lw=.7, zorder=0)
        for (x, y), marker in zip(data['coordinates'], style['markers']):
            color = marker['fill_rgb'] if key == 'c' else marker['fill_color_hex']
            graph.plot(x, y, linestyle='none', marker='o', markersize=7.5,
                       markeredgewidth=1, color=color, zorder=3)

    reference_font = FONT_FAMILY.casefold() == 'arial'
    for span in layout['text_spans']:
        text = span['text']; x, y = span['origin']
        # Regenerate all45 numerical heatmap labels from recalculated statistics.
        heat = a['plot_bbox_pt']
        if heat[0] <= x <= heat[2] and heat[1] <= y <= heat[3] and re.fullmatch(r'[\d.,\[\]]+', text):
            center_x = (span['bbox'][0]+span['bbox'][2])/2
            j = int((center_x-heat[0])/(heat[2]-heat[0])*5)
            i = int((y-heat[1])/(heat[3]-heat[1])*5)
            text = f'[{ci[i][j][0]:.2f},{ci[i][j][1]:.2f}]' if text.startswith('[') else f'{tv[i,j]:.2f}'
        for prefix, data_key in [('MDS', 'source'), ('CKA', 'family')]:
            if text.startswith(prefix+' '):
                component = int(text[4])-1
                text = f'{prefix} {component+1} ({geometry[data_key]["shares"][component]*100:.1f}%)'
        rgb = tuple(((span['color'] >> shift) & 255)/255 for shift in [16, 8, 0])
        # Center labels on their design boxes; portable font metrics differ from Arial.
        horizontal = span['direction'] != [0, -1]
        if horizontal and not reference_font:
            x = (span['bbox'][0] + span['bbox'][2]) / 2
        fig.text(x/page[0], 1-y/page[1], text, fontsize=span['size'] * (1 if reference_font else .9), color=rgb,
                 ha='center' if horizontal and not reference_font else 'left', va='baseline', fontweight='bold' if 'Bold' in span['font'] else 'normal',
                 rotation=90 if span['direction'] == [0, -1] else 0, rotation_mode='anchor', zorder=5)
    fig.savefig(output, dpi=300, facecolor='white', metadata={
        'Title': 'Figure 2: Solution-distribution distances across sources',
        'CreationDate': None, 'ModDate': None})
    if previews:
        fig.savefig(output.with_suffix('.png'), dpi=300, facecolor='white')
    plt.close(fig)
    return output, records

def render(statistics, geometry, output, *, previews=False):
    layout_dir = Path(__file__).with_name('layouts')
    layout = json.loads((layout_dir/'figure2.json').read_text())
    style = json.loads((layout_dir/'figure2_intervals.json').read_text())
    bootstrap = {'global': statistics['global_tv_primary'], 'families': statistics['pairwise_tv_by_type_primary']}
    coords = {key: {'coordinates': geometry[name]['coordinates'], 'shares': geometry[name]['shares']}
              for key, name in [('source', 'mean_tv_mds'), ('family', 'tv_family_geometry_cka')]}
    result, records = draw(bootstrap, coords, layout, style, output, previews=previews)
    return {'matrix': bootstrap['global']['matrix'], 'ci95': bootstrap['global']['ci95_puzzle_matrix'],
            'family_intervals': records, 'geometry': coords}
