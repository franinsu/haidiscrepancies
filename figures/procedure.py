"""Study procedure schematic in the September 23 paper layout.

Geometry and bar heights are illustrative. The local schematic palette belongs
with this drawing and is separate from the quantitative figures' source palette.
"""
from pathlib import Path
import json
from . import plot_style
from .plot_style import plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.path import Path as DrawPath
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch
import matplotlib.patheffects as pe

INK = '#16181B'
MUTED = '#5B6168'
BORDER = '#C9CDD3'
SOLUTION_COLORS = ['#E8C9A4', '#D0A881', '#B9885E', '#A1673B', '#8A4718']
SCHEMATIC_COLORS = {'Human': '#8E1C1C', 'GPT 5.6 Sol': '#6A3FA0',
                    'Claude Opus 4.8': '#2C5E8E', 'Gemini 3.5 Flash': '#1C8B78'}


def pipeline(path, *, previews=False):
    geometry = json.loads(Path(__file__).with_name('layouts').joinpath('procedure_geometry.json').read_text())
    design = json.loads(Path(__file__).with_name('layouts').joinpath('study_design.json').read_text())
    protocol, assignment = design['protocol'], design['assignment']
    maximum_submissions = protocol['maximum_submissions_per_trial']
    font = plot_style.FONT_FAMILY
    # Page-point coordinates start at the supplied figure's top left. Preserve
    # its physical composition even when the selected portable font is wider.
    walls = {tuple(v) for v in geometry['walls']}
    width, height = 487.5835266113, 176.1593780518
    ink, muted, border = INK, MUTED, BORDER
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')

    def text(x, y, label, size, color=ink, bold=False, ha='left', **kwargs):
        return ax.text(x, y, label, fontsize=size, fontfamily=font, color=color,
                       fontweight='bold' if bold else 'normal', ha=ha, va='baseline', **kwargs)

    def rounded(x, y, w, h, radius, fill, edge='none', linewidth=0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f'round,pad=0,rounding_size={radius}',
                     facecolor=fill, edgecolor=edge, linewidth=linewidth))

    def fitted_size(labels, size, available_width):
        properties = font_manager.FontProperties(family=font, size=size)
        renderer = fig.canvas.get_renderer()
        longest = max(renderer.get_text_width_height_descent(label, properties, False)[0]
                      for label in labels)
        return size * min(1, available_width * fig.dpi / 72 / longest)

    def arrow(shaft, head):
        ax.add_patch(PathPatch(DrawPath(shaft, [DrawPath.MOVETO] + [DrawPath.CURVE4] * 3),
                     fill=False, edgecolor=muted, linewidth=1, capstyle='round', joinstyle='round'))
        ax.add_patch(PathPatch(DrawPath(head + [head[0]], [DrawPath.MOVETO,
                     DrawPath.LINETO, DrawPath.LINETO, DrawPath.CLOSEPOLY]),
                     facecolor=muted, edgecolor=muted, linewidth=1, joinstyle='round'))

    def maze(left, top, extent, cell_linewidth, outer_linewidth, labels, routes=False):
        cell = extent / 8
        for row in range(8):
            for col in range(8):
                ax.add_patch(Rectangle((left + col * cell, top + row * cell), cell, cell,
                             facecolor='#3A3E44' if (row, col) in walls else 'white',
                             edgecolor=border, linewidth=cell_linewidth))
        if routes:
            for points, color in zip(geometry['paths'], SOLUTION_COLORS):
                ax.plot([left + (p[0] - 33.669670) * extent / 102.540660 for p in points],
                        [top + (155.126620 - p[1]) * extent / 102.540659 for p in points],
                        color=color, linewidth=2.1, solid_capstyle='round', solid_joinstyle='round')
        ax.add_patch(Rectangle((left, top), extent, extent, fill=False,
                     edgecolor=ink, linewidth=outer_linewidth))
        for x, y, label, size in labels:
            t = text(x, y, label, size, bold=True, zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])

    maze(8.489670, 23.105515, 102.540660, .6, 1.3,
         [(12.273462, 32.144197, 'S', 8.6), (101.043413, 96.232109, 'G', 8.6)], routes=True)
    separator = '·' if font.casefold() == 'arial' else '⋅'
    text(59.76, 148.226593, f'270 puzzles {separator} five families', 7.9, muted, ha='center')
    text(59.76, 161.134491, '100 main + 170 perturbation', 7.9, muted, ha='center')
    rounded(146.720317, 23.792786, 57.773370, 115.202852, 3.438891, 'white', border, 1.1)
    for y, w in [(33.937515, 32.325577), (45.285858, 41.266694), (56.634201, 26.823351)]:
        rounded(154.973655, y, w, 4.986392, 2.235280, '#E7E9EC')
    maze(150.571875, 78.952606, 50.070254, .5, .9,
         [(152.021578, 83.762090, 'S', 5.5), (195.223676, 115.056000, 'G', 5.5)])
    text(175.607002, 153.602356, 'Matched puzzle images', 7.8, muted, ha='center')

    for cx in [242.321489 + i * 14.443343 for i in range(4)]:
        ax.add_patch(plt.Circle((cx, 21.729456), 4.470559,
                     facecolor=SCHEMATIC_COLORS['Human'], edgecolor='none'))
        rounded(cx - 6.018059, 26.887787, 12.036119, 12.036119, 5.158336, SCHEMATIC_COLORS['Human'])
    for y, label, color in [(11.646469, '521 participants', ink),
                           (24.026474, 'Across both studies', muted),
                           (36.406479, f'Up to {maximum_submissions} submissions', muted),
                           (48.786484, 'per trial', muted)]:
        text(300.094859, y, label, 8.1, color)

    for i, source in enumerate(plot_style.SOURCES[1:]):
        color = SCHEMATIC_COLORS[source]
        fill = [.9 + .1 * c for c in to_rgb(color)]
        rounded(242.321489, 94.633949 + i * 19.945568, 44.705584, 15.818899, 4.470559, fill)
        text(264.674281, 104.762215 + i * 19.945568, plot_style.LABEL[source],
             8.2, color, bold=True, ha='center')
    model_detail_size = fitted_size(['100 samples per puzzle', 'For each reasoning effort',
                                    'and prompt setting'], 8.1, 100)
    for y, label, color in [(111.030418, '100 samples per puzzle', ink),
                           (124.098206, 'For each reasoning effort', muted),
                           (137.165985, 'and prompt setting', muted)]:
        text(295.280426, y, label, model_detail_size, color)
    arrow([(121.708801,84.572556),(125.164803,84.572556),(128.248123,84.572556),(130.958771,84.572556)],
          [(127.358765,82.772552),(130.958771,84.572556),(127.358765,86.372551)])
    arrow([(208.107803,66.506050),(209.648956,51.750320),(216.085144,40.184555),(227.416397,31.808716)],
          [(223.451477,32.501144),(227.416397,31.808716),(225.591385,35.396103)])
    arrow([(208.174255,107.623444),(210.100952,121.451477),(216.496033,132.128967),(227.359497,139.655930)],
          [(225.425522,136.126099),(227.359497,139.655930),(223.375244,139.085205)])

    # Preserve original physical heights. Their separately normalized proportions
    # remain Human [.24,.18,.22,.16,.20] and Model [.55,.10,.17,.06,.12].
    for baseline, key in [(54.055031, 'human_bars'), (152.751205, 'model_bars')]:
        for (_, bar_height), i, color in zip(geometry[key], range(5), SOLUTION_COLORS):
            ax.add_patch(Rectangle((399.478812 + i * 16.506677, baseline - bar_height),
                         14.443342, bar_height, facecolor=color, edgecolor='none'))
        ax.plot([395.696032, 483.731644], [baseline, baseline], color=muted,
                linewidth=.9, solid_capstyle='projecting')
    text(439.713838, 13.561218, 'Human', 9, ha='center')
    text(439.713838, 72.710152, 'Model', 9, ha='center')
    footer_size = fitted_size(['Distribution over solutions', 'Illustrative valid answers'], 7.8,
                             2 * (width - 439.713838) - .5)
    text(439.713838, 161.855698, 'Distribution over solutions', footer_size, muted, ha='center')
    text(439.713838, 172.516251, 'Illustrative valid answers', footer_size, muted, ha='center')
    fig.canvas.draw()
    bounds = [t.get_window_extent(fig.canvas.get_renderer()) for t in ax.texts]
    clipped = [t.get_text() for t, b in zip(ax.texts, bounds) if
               b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height]
    if clipped:
        raise ValueError(f'Procedure labels extend beyond the page: {clipped}')
    fig.savefig(path, facecolor='white', metadata={'CreationDate': None, 'ModDate': None})
    if previews:
        fig.savefig(path.with_suffix('.png'), facecolor='white')
    plt.close(fig)
    examples = {}
    for label, key in [('Human', 'human_bars'), ('Model', 'model_bars')]:
        heights = [float(value) for _, value in geometry[key]]
        examples[label] = [value / sum(heights) for value in heights]
    payload = {
        'kind': 'procedure schematic; path colors and bar heights are illustrative, not measurements',
        'adapted_from': 'study procedure schematic',
        'scope': 'general procedure across main and perturbation studies',
        'total_puzzles': 270, 'core_puzzles': 100, 'perturbation_puzzles': 170,
        'families': 5, 'valid_classes_range': [3, 8], 'single_solution_primers': 10,
        'retained_humans_total': 521, 'retained_module_humans': assignment['retained_participants'],
        'retained_core_humans': 104, 'maximum_human_attempts_per_trial': maximum_submissions,
        'human_trial_protocol': protocol,
        'model_requests_per_puzzle_per_condition': 100,
        'model_conditions': 'provider x low/medium effort x plain/persona prompt',
        'core_and_module_studies_separate': True,
        'module_participant_assignment': assignment,
        'distribution_denominator': 'retained valid responses only',
        'display': {'layout': 'original schematic: maze, rendered puzzle, people/model branches, solution distributions',
                    'stages': ['design', 'collect'],
                    'illustrative_probabilities': examples, 'shared_probability_axis': [0, .6],
                    'class_colors': SOLUTION_COLORS,
                    'bar_provenance': 'Original schematic bar proportions, normalized separately for Human and Model; not pooled model data'}
    }
    return payload
