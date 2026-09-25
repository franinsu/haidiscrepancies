"""Generate the manuscript tables and export only their displayed cells."""
from __future__ import annotations
import math
from . import tables_tv, tables_entropy, tables_effort, tables_divergence
from . import tables_regression, tables_maze, tables_modules
from .regression_format import interval_cell

SOURCES = ['Human', 'GPT-5.6 Sol|low|plain', 'Claude Opus 4.8|low|plain', 'Gemini 3.5 Flash|low|plain']
LABELS = ['Human', 'ChatGPT', 'Claude', 'Gemini']
FAMILIES = [('arithmetic','Arithmetic'),('maze','Maze'),('grid','Rooks'),('minesweeper','Minesweeper'),('sudoku','Sudoku')]


def selected_fits(stem, data):
    if stem == 'maze':
        return {task:{source: data['fits'][task][label] for source,label in zip(SOURCES,LABELS) if label in data['fits'][task]} for task in ['choice','source']}
    return {'choice':data['selected_solution_prediction']['conditional']['fits'], 'source':data['source_prediction']['binary_vs_human']['conditional']['fits']}


def accuracy_table(read):
    lines = [r'\begin{table}[!htbp]',r'\centering',
        r'\caption{\textbf{Normalized held-out prediction accuracy.} Accuracy is normalized within each held-out puzzle before averaging over 20 puzzles per family (Section~\ref{sec:methods-prediction}). Brackets give 95\% intervals from 2,000 whole-puzzle bootstrap refits. Model responses use low effort and the plain prompt.}',
        r'\label{tab:feature-accuracy-overview}',r'\small',r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.15}',r'\begin{tabular}{@{}lcccc@{}}',r'\toprule']
    values=[]
    for task,title in [('choice','Selected-solution prediction'),('source','Model-versus-Human prediction')]:
        lines.extend([r'\multicolumn{5}{@{}l}{\textbf{'+('a' if task=='choice' else 'b')+r'}\quad '+title+r'} \\',r'Family & Human & ChatGPT & Claude & Gemini \\',r'\midrule'])
        for stem,label in FAMILIES:
            data=read('maze_three_feature_models' if stem=='maze' else stem+'_feature_models')
            fits=selected_fits(stem,data)[task]; cells=[]
            for source in SOURCES:
                if source=='Human' and task=='source':cells.append('---');continue
                fit=fits[source]; point=fit['cv_normalized_accuracy'];ci=fit['cv_normalized_accuracy_ci_95']
                cells.append(interval_cell(point,ci,bold_if_excludes_zero=True))
                values.append({'family':label,'task':task,'source':source,'estimate':point,'ci95':ci})
            lines.append(label+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\midrule' if task=='choice' else r'\bottomrule')
    lines.extend([r'\end{tabular}',r'\end{table}'])
    return '\n'.join(lines)+'\n',values


def perturbation_table(data):
    rows=data['paper_tests']
    specs=[('cue','Highlighting'),('pair_related','Related context'),('transfer','Strategy primer'),('spatial','Spatial reflection'),('formulation','Number ordering')]
    sources=['Human','GPT 5.6 Sol|low|plain','Claude Opus 4.8|low|plain','Gemini 3.5 Flash|low|plain']
    lines=[r'\begin{table}[H]',r'\centering',
        r'\caption{\textbf{TV permutation tests.} Cells report mean unadjusted $p$, followed by the number of comparisons with Holm-adjusted $p\le0.05$ out of the total.}',
        r'\label{tab:perturbation_tests}',r'\small',r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{@{}lcccc@{}}',r'\toprule',r'Module & Human & ChatGPT & Claude & Gemini \\',r'\midrule']
    for key,label in specs:
        cells=[]
        for source in sources:
            item,=[r for r in rows if r['row']==key and r['source']==source]
            cells.append(f"{item['mean_unadjusted_p']:.4f} ({item['holm_below_005']}/{item['n_comparisons']})")
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    return '\n'.join(lines)+'\n'


def mi_table(data):
    means=[]
    for source in LABELS:
        for context in ['Related','Unrelated']:
            rows=[r for r in data['individual_independence_tests'] if r['source']==source and r['context']==context]
            if len(rows)!=10:raise ValueError(f'Expected ten MI tests for {source}/{context}')
            means.append({'source':source,'context':context,'mean_p':math.fsum(r['p_unadjusted'] for r in rows)/10,'below_005':sum(r['p_unadjusted']<.05 for r in rows),'n':10})
    lines=[r'\begin{table}[H]',r'\centering',r'\caption{\textbf{Individual-puzzle MI permutation tests.} Mean $p$ averages ten unadjusted $p$-values per source and context (five puzzles per family); counts give the number below 0.05. Mean $p$ is descriptive, not a combined $p$-value or a test of mean MI. Models use low effort and plain prompts; tests are described in Section~\ref{supp:perturbation-bootstrap}.}',r'\label{tab:context-mi-mean-p}',r'\setlength{\tabcolsep}{10pt}',r'\begin{tabular}{@{}lcccc@{}}',r'\toprule',r'& \multicolumn{2}{c}{Related} & \multicolumn{2}{c}{Unrelated} \\',r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',r'Source & Mean $p$ & $p<0.05$ & Mean $p$ & $p<0.05$ \\',r'\midrule']
    for source in LABELS:
        rows=[r for r in means if r['source']==source]
        cells=[cell for r in rows for cell in [f"{r['mean_p']:.4f}",f"{r['below_005']}/10"]]
        lines.append(source+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    return '\n'.join(lines)+'\n',means


def generate(read, names):
    """Return table text and public values; no input object is copied wholesale."""
    result={};public={}
    dispatch={
      'tv_global_table':('figure_statistics',tables_tv.render_global),
      'tv_condition_contrasts':('figure_statistics',tables_tv.render_condition_contrasts),
      'alternative_divergence_tables':('figure_statistics',tables_divergence.render),
      'entropy_deficit_table':('entropy_deficit_stats',tables_entropy.render_table_tex),
      'entropy_svd_table':('entropy_deficit_stats',tables_entropy.render_svd_table_tex),
      'effort_correlation_table':('effort_difficulty_stats',tables_effort.render_correlation_table),
      'perturbation_tests':('perturbation_statistics',perturbation_table),
    }
    for name in names:
        if name in dispatch:
            file,render=dispatch[name];result[name]=render(read(file))
        elif name=='context_mi_mean_p':result[name],public[name]=mi_table(read('context_mi_summary'))
        elif name=='feature_accuracy_overview':result[name],public[name]=accuracy_table(read)
        elif name.endswith('_primary_tables'):
            stem=name.removesuffix('_primary_tables')
            result[name]=(tables_maze.render(read('maze_three_feature_models')) if stem=='maze' else tables_regression.render_primary(read(stem+'_feature_models')))
        elif name in {'cue_table','spatial_table','pair_table','transfer_table'}:
            d=read('stimulus_sensitivity_stats') if name!='cue_table' else None
            if name=='cue_table':result[name]=tables_modules.render_cue_table(read('normalized_attraction_stats'))
            elif name=='spatial_table':result[name]=tables_modules.render_spatial_table(d)
            elif name=='pair_table':result[name]=tables_modules.render_pair_table(read('context_dependence_table'),d)
            else:result[name]=tables_modules.render_transfer_table(d,read('normalized_attraction_stats'))
        else:raise ValueError(f'Unknown table {name}')
        # These strings are exactly the displayed numerical cells and headings.
        if name not in public:
            public[name]={'rows':[line.strip().removesuffix('\\\\').split(' & ') for line in result[name].splitlines() if ' & ' in line and not line.lstrip().startswith('%')]}
    return result,public
