"""Render directly from numerical summaries supplied by the caller."""




def scatter_geometry(ax,xy,order,labels,colors,shares,prefix,offsets=None):
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    from .plot_style import geometry_dot, GEOMETRY_LABEL_SIZE, clean
    xy=np.asarray(xy)
    for i,(item,(x,y)) in enumerate(zip(order,xy)):
        geometry_dot(ax,x,y,colors[item])
        if offsets:
            dx,dy,ha,va=offsets[i]
            ax.annotate(labels[item],(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va=va,fontsize=GEOMETRY_LABEL_SIZE,color='#222222')
    pads=np.maximum(.06,.30*np.ptp(xy,axis=0))
    ax.set_xlim(xy[:,0].min()-pads[0],xy[:,0].max()+pads[0])
    ax.set_ylim(xy[:,1].min()-pads[1],xy[:,1].max()+pads[1])
    ax.axhline(0,color='#9ca3af',lw=.6,zorder=0)
    ax.axvline(0,color='#9ca3af',lw=.6,zorder=0)
    clean(ax,grid='both')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(f'{prefix} 1 ({100*shares[0]:.1f}%)')
    ax.set_ylabel(f'{prefix} 2 ({100*shares[1]:.1f}%)')

def render(data):
    from .plot_style import (plt, SOURCES, LABEL, COLORS, FAMILIES, FLABEL,
                            FCOLOR, WIDTH, panel, finalize)

    v=data['source_svd_projection'];fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.85))
    fig.subplots_adjust(left=.09,right=.975,bottom=.18,top=.88,wspace=.42)
    for ax,letter,title,key,order,labels,colors,offsets in [
     (axes[0],'a','Family coefficients','left_singular_vectors_top_two',FAMILIES,FLABEL,FCOLOR,[(-5,-5,'right','top'),(5,-5,'left','top'),(5,5,'left','bottom'),(-5,5,'right','bottom'),(5,5,'left','bottom')]),
     (axes[1],'b','Source scores','source_scores_top_two',SOURCES,LABEL,COLORS,[(-5,5,'right','bottom'),(5,-5,'left','top'),(5,5,'left','bottom'),(5,5,'left','bottom')])]:
        scatter_geometry(ax,[v[key][s] for s in order],order,labels,colors,v['frobenius_energy_share'],'Component',offsets)
        ax.grid(False, which='both', axis='both')
        panel(ax,letter,title)
    finalize(fig)
    return fig, v
