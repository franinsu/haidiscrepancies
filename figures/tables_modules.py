"""Current module tables; every estimate and interval is supplied by analysis."""
from __future__ import annotations
from .tables_tv import result_cell

SOURCES = ['Human','ChatGPT','Claude','Gemini']
CONDITIONS = ['Human','GPT 5.6 Sol|low|plain','Claude Opus 4.8|low|plain','Gemini 3.5 Flash|low|plain']
GROUPS = [('minesweeper_lite','Minesweeper'),('mini_sudoku','Sudoku'),(None,'Mean')]


def grouped_table(caption,labels,metrics,select):
    lines=[r'% Generated from numerical summaries; no inference occurs in this renderer.',
        r'\begin{table}[H]',r'  \caption{'+caption+'}',
        *[r'  \label{'+label+'}' for label in labels],r'  \centering',r'  \setlength{\tabcolsep}{2pt}',
        r'  \renewcommand{\arraystretch}{1.2}',r'  \begin{adjustbox}{max width=\linewidth}',
        r'  \begin{tabular}{@{}lcccc@{}}',r'    \toprule',r'    Quantity & Human & ChatGPT & Claude & Gemini \\',r'    \midrule']
    for index,(family,title) in enumerate(GROUPS):
        if index:lines.append(r'    \midrule')
        lines.append(r'    \multicolumn{5}{l}{\textit{'+title+r'}} \\')
        for metric,label in metrics:
            cells=[]
            for source,condition in zip(SOURCES,CONDITIONS):
                item=select(metric,source,condition,family)
                cells.append(result_cell(item['estimate'],item['ci95']))
            lines.append('    '+label+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'    \bottomrule',r'  \end{tabular}',r'  \end{adjustbox}',r'\end{table}'])
    return '\n'.join(lines)+'\n'


def family_value(summary,family):
    return summary if family is None else summary['by_family'][family]


def render_cue_table(normalized):
    metrics=[('cue_all','Prob. change'),('cue_normalized','Normalized prob. change'),('cue_movement','TV from ref.')]
    return grouped_table(
        r'Highlighting. TV is $d_{\mathrm{TV}}$ from the unhighlighted reference; probability changes concern the highlighted solution set (Section~\ref{app:stimulus-measures}). Entries average arms within puzzles. Mean weights families equally; brackets show 95\% puzzle-bootstrap CIs.',
        ['tab:cue-detail'],metrics,
        lambda key,source,condition,family:family_value(normalized['summaries'][key][source],family))


def render_spatial_table(summary):
    return grouped_table(
        r'Spatial reflection. TV from ref. is $d_{\mathrm{TV}}$ from the reference presentation, after mapping answers back to their original cells. Mean rows average the three reflections; the Mean group weights families equally. Brackets show 95\% puzzle-bootstrap CIs.',
        ['tab:spatial-detail'],[('spatial_lr','Left--right'),('spatial_tb','Top--bottom'),('spatial_lrtb','Both'),('spatial','Mean')],
        lambda key,source,condition,family:family_value(summary['summaries'][key][source],family))


def render_transfer_table(summary,normalized):
    def select(key,source,condition,family):
        data=normalized if key=='transfer_normalized' else summary
        return family_value(data['summaries'][key][source],family)
    return grouped_table(
        r'Strategy-primer distribution and probability changes. TV is $d_{\mathrm{TV}}$ from the no-primer reference. Probability changes concern the designated solution (Section~\ref{app:stimulus-measures}). Mean weights families equally; brackets show 95\% puzzle-bootstrap CIs.',
        ['tab:transfer-attraction','tab:transfer-movement'],
        [('transfer_tv','TV from ref.'),('transfer_lift','Prob. change'),('transfer_normalized','Normalized prob. change')],select)


def render_pair_table(context,summary):
    def select(key,source,condition,family):
        if key.startswith('pair_'):return family_value(summary['summaries'][key][source],family)
        return context['summaries'][condition][family or 'Mean'][key]
    return grouped_table(
        r'Preceding-puzzle context by family. Measures are defined in Methods, Section~\ref{sec:target-methods}; $\Delta$ denotes related minus unrelated context. Estimates use pairs with two valid answers from five puzzles per family. Mean weights families equally. Brackets show 95\% whole-puzzle bootstrap intervals.',
        ['tab:pair-movement'],
        [('related','Related TV from ref.'),('unrelated','Unrelated TV from ref.'),('difference',r'$\Delta\mathrm{TV}$'),
         ('related_normalized_change',r'\shortstack[l]{Normalized prob. change\\(related)}'),
         ('pair_related_mi','Related MI'),('pair_control_mi','Unrelated MI'),('pair_related_minus_control_mi',r'$\Delta I$')],select)
