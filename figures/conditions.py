"""Selected perturbation and condition layouts; inputs are computed summaries."""
import numpy as np
from matplotlib.lines import Line2D
from .plot_style import plt, SOURCES, LABEL, COLORS, WIDTH, INTERVAL_MARKER_AREA, INTERVAL_ALPHA, LEGEND_MARKER_SIZE, clean, point_ci, legend, panel, confidence_interval, geometry_dot, GEOMETRY_LABEL_SIZE
from .plot_style import finalize
def condition(source):
    return source if source == 'Human' else source + '|low|plain'

def presentation(P, V, T):
    figure_height=115/25.4
    fig=plt.figure(figsize=(WIDTH,figure_height))
    panel_bounds=[[.17,.20,.31,.66], [.66,.57,.31,.29], [.66,.20,.31,.25]]
    axs=[fig.add_axes(bounds) for bounds in panel_bounds]
    response_rows=['cue','pair_related','transfer']
    invariance_rows=['spatial','formulation']
    row_labels=['Highlighting','Related context','Strategy primer']
    tv_rows=response_rows+invariance_rows
    tv_y=[0,1,2,3.45,4.45]
    for key,center in zip(tv_rows,tv_y):
        for si,s in enumerate(SOURCES):
            value=P['summaries'][key][condition(s)]['movement']
            point_ci(axs[0],value['estimate'],center,value['ci95'],COLORS[s],True)
    for ri,key in enumerate(response_rows):
        for si,s in enumerate(SOURCES):
            cond=condition(s);y=ri
            value=V['summaries']['b'][key][cond]
            point_ci(axs[1],value['estimate'],y,value['ci95'],COLORS[s],True)
    axs[0].set_yticks(tv_y,row_labels+['Spatial reflection','Number ordering'])
    axs[0].set_ylim(5.05,-.6)
    axs[1].set_yticks(range(3),row_labels);axs[1].set_ylim(2.6,-.6)
    for ax in axs[:2]:clean(ax,'x')
    axs[0].set(xlim=(0,1),xticks=[0,.5,1],xlabel='TV from ref.',title='Distribution change')
    axs[1].set(xlim=(-.85,1.05),xticks=[-.5,0,.5,1],xlabel='Normalized prob. change',title='Shift toward target')
    axs[1].axvline(0,color='#9ca3af',lw=.7,zorder=1)
    context_points=[]
    context_rows=['pair_related_mi','pair_control_mi']
    for si,s in enumerate(SOURCES):
        for y,key in enumerate(context_rows):
            value=T['summaries'][key][LABEL[s]]
            point_ci(axs[2],value['estimate'],y,value['ci95'],COLORS[s],True)
            context_points.append({'panel':'c','metric':key,'source':s,'condition':condition(s),**value})
    axs[2].set(yticks=range(2),yticklabels=['Related','Unrelated'],ylim=(1.6,-.6))
    clean(axs[2],'x')
    axs[2].set(xlim=(0,.7),xticks=[0,.3,.6],xlabel='MI',
               title='Dependence between answers')
    for ax,letter in zip(axs,'abc'):
        ax.grid(False, which='both', axis='x')
        panel(ax,letter)
    fig.legend(handles=legend(),frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(.55,.047))
    finalize(fig)
    return fig, {'tv': [{ 'module':key, 'source':source, 'estimate':P['summaries'][key][condition(source)]['movement']['estimate'], 'ci95':P['summaries'][key][condition(source)]['movement']['ci95']} for key in tv_rows for source in SOURCES], 'gain': [{'module':key,'source':source, 'estimate':V['summaries']['b'][key][condition(source)]['estimate'],'ci95':V['summaries']['b'][key][condition(source)]['ci95']} for key in response_rows for source in SOURCES], 'mi': [{k:r[k] for k in ['metric','source','estimate','ci95']} for r in context_points]}

def conditions(S, projection):
    CONDITIONS = [
        {'key':'low|plain', 'label':'Low plain', 'marker':'o', 'filled':True},
        {'key':'low|as-human', 'label':'Low persona', 'marker':'o', 'filled':False},
        {'key':'medium|plain', 'label':'Medium plain', 'marker':'s', 'filled':True},
        {'key':'medium|as-human', 'label':'Medium persona', 'marker':'s', 'filled':False},
    ]
    uniform_reference = projection['uniform_reference']
    coordinates = np.asarray(projection['embedding']['coordinates'])
    condition_order = projection['geometry_input']['source_order']
    shares = projection['embedding']['shares']
    reference_limits = projection['embedding']['reference_limits']
    def condition_point(ax, x, y, spec, color, size=INTERVAL_MARKER_AREA):
        return ax.scatter(x, y, s=size, marker=spec['marker'],
            facecolors=color if spec['filled'] else 'white',
            edgecolors='none' if spec['filled'] else color,
            linewidths=.65, alpha=INTERVAL_ALPHA, zorder=4 if spec['filled'] else 3)

    fig = plt.figure(figsize=(WIDTH, 5.8))
    fig.text(.47, .96, 'b', fontweight='bold', fontsize=8)
    condition_levels = []
    ax = fig.add_axes([.625, .59, .345, .34])
    ax.axvspan(*uniform_reference['ci95'],color=COLORS['Uniform'],alpha=.10,zorder=0)
    ax.axvline(uniform_reference['estimate'],color=COLORS['Uniform'],lw=.85,ls='--',zorder=2)
    for si, provider in enumerate(SOURCES[1:]):
        g = S['tv_conditions'][provider]
        h = g['order'].index('Human')
        for ci, spec in enumerate(CONDITIONS):
            key = provider+'|'+spec['key']; j = g['order'].index(key)
            mean, interval = g['matrix'][h][j], g['ci95_matrix'][h][j]
            confidence_interval(ax, interval, ci, COLORS[provider], horizontal=True)
            condition_point(ax, mean, ci, spec, COLORS[provider])
            condition_levels.append({'provider':provider, 'source':key,
                'condition':spec['label'], 'estimate':mean, 'ci95':interval})
    ax.set_yticks(range(4), [c['label'] for c in CONDITIONS])
    ax.set_ylim(3.6, -.6); ax.set_xlim(.25, .60)
    ax.set_xticks([.3, .4, .5, .6])
    ax.set_xlabel('TV from Human')
    clean(ax, 'x')
    ax.grid(False, which='both', axis='x')
    fig.legend(handles=legend(SOURCES[1:]), loc='center',
               bbox_to_anchor=(.52, .025), ncol=3, frameon=False,
               columnspacing=1.5, handletextpad=.5)
    fig.legend(handles=[Line2D([], [], ls='', marker=c['marker'], color='#333333',
        markerfacecolor='#333333' if c['filled'] else 'white', markeredgewidth=.65,
        markersize=LEGEND_MARKER_SIZE, alpha=INTERVAL_ALPHA, label=c['label']) for c in CONDITIONS] +
        [Line2D([],[],color=COLORS['Uniform'],ls='--',lw=.85,label='Uniform')],
        loc='center', bbox_to_anchor=(.52, .070), ncol=5, frameon=False,
        columnspacing=1.2, handletextpad=.5)

    fig.text(.025, .715, 'a', fontweight='bold', fontsize=8)
    ax = fig.add_axes([.075, .42, .345, .255])
    for i, label, offset, ha, va in [(0,'Uniform',(-5,7),'right','bottom'),(1,'Human',(0,-8),'center','top')]:
        geometry_dot(ax,*coordinates[i],COLORS[label])
        ax.annotate(label,coordinates[i],xytext=offset,textcoords='offset points',
                    ha=ha,va=va,fontsize=GEOMETRY_LABEL_SIZE,color='#222222',
                    bbox={'facecolor':'white','edgecolor':'none','pad':.4})
    for si, provider in enumerate(SOURCES[1:]):
        xy = coordinates[2+4*si:6+4*si]
        for point, spec in zip(xy, CONDITIONS):
            condition_point(ax, *point, spec, COLORS[provider], size=19)
        centroid = xy.mean(axis=0)
        offset = [(0, 12), (0, 14), (0, -16)][si]
        ax.annotate(LABEL[provider], centroid, xytext=offset, textcoords='offset points',
                    ha='center', color=COLORS[provider], fontsize=7,
                    bbox={'facecolor':'white','edgecolor':'none','pad':.4})
    ax.set_xlim(reference_limits[0]); ax.set_ylim(reference_limits[1])
    ax.set_xticks([-.4,-.2,0,.2,.4]); ax.set_yticks([-.2,-.1,0,.1,.2])
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='#9ca3af', lw=.6, zorder=0)
    ax.axvline(0, color='#9ca3af', lw=.6, zorder=0)
    ax.set_xlabel(f'MDS 1 ({100*shares[0]:.1f}%)')
    ax.set_ylabel(f'MDS 2 ({100*shares[1]:.1f}%)')
    clean(ax, 'both')
    ax.grid(False, which='both', axis='both')

    fig.text(.47, .52, 'c', fontweight='bold', fontsize=8)
    condition_pairs = []
    pair_indices = [(0,j) for j in range(1,4)]
    pair_labels = [CONDITIONS[j]['label'] for i,j in pair_indices]
    pair_ax = fig.add_axes([.625,.18,.345,.30])
    for si, provider in enumerate(SOURCES[1:]):
        cell = S['tv_conditions'][provider]
        for row,(i,j) in enumerate(pair_indices):
            source_i,source_j = [provider+'|'+CONDITIONS[k]['key'] for k in (i,j)]
            ii,jj = [cell['order'].index(k) for k in (source_i,source_j)]
            mean,interval = cell['matrix'][ii][jj],cell['ci95_matrix'][ii][jj]
            y=row
            confidence_interval(pair_ax, interval, y, COLORS[provider], horizontal=True)
            condition_point(pair_ax,mean,y,CONDITIONS[j],COLORS[provider])
            condition_pairs.append({'provider':provider,'conditions':[CONDITIONS[k]['label'] for k in (i,j)],
                'sources':[source_i,source_j],'estimate':mean,'ci95':interval,'n_puzzles':cell['n_puzzles']})
    pair_ax.set_yticks(range(3),pair_labels)
    pair_ax.set(xlim=(0,.21),ylim=(2.6,-.6),xticks=[0,.05,.1,.15,.2],xlabel='TV from Low plain')
    clean(pair_ax,'x')
    pair_ax.grid(False, which='both', axis='x')
    finalize(fig)
    return fig, {'levels':condition_levels, 'condition_pairs':condition_pairs, 'uniform_reference':uniform_reference, 'coordinates':coordinates.tolist(), 'source_order':condition_order, 'shares':list(shares)}
