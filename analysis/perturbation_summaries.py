"""Only the module summaries displayed in the current paper."""
from collections import defaultdict
import numpy as np
from . import stimulus_modules as sm
from .normalized_attraction import equal_family_summary, normalized_attraction

LABELS = {'Human':'Human','GPT 5.6 Sol|low|plain':'GPT',
          'Claude Opus 4.8|low|plain':'Claude','Gemini 3.5 Flash|low|plain':'Gemini'}


def block_summary(values, strata, repeats):
    """Preserve the established shared whole-block draws and family intervals."""
    draws = sm.bootstrap_draws(tuple(len(ids) for _,ids in strata),repeats=repeats,seed=0)
    result = sm.summarize_blocks(values,strata,draws)
    result['by_family'] = equal_family_summary(values,strata,repeats=repeats,seed=0)['by_family']
    return result


def compute(design, effects, mi_tests, repeats=5000):
    blocks = defaultdict(lambda:defaultdict(dict))
    normalized_blocks = defaultdict(lambda:defaultdict(dict))
    module_for = {'cue_all':'cue','cue_movement':'cue','transfer_lift':'transfer',
                  'transfer_tv':'transfer','spatial':'spatial','spatial_lr':'spatial',
                  'spatial_tb':'spatial','spatial_lrtb':'spatial',
                  'pair_related_mi':'pair','pair_control_mi':'pair',
                  'pair_related_minus_control_mi':'pair'}
    for source in effects['sources']:
        for module in ('cue','spatial','transfer'):
            selected = [r for r in effects['records'] if r['source']==source and r['module']==module]
            for block in design.family_order(module):
                arms = [r for r in selected if r['block']==block]
                if module == 'spatial':
                    for arm,key in zip(sm.SPATIAL_REFLECTIONS,('spatial_lr','spatial_tb','spatial_lrtb')):
                        row, = [r for r in arms if r['arm']==arm]
                        blocks[source][key][block] = row['movement']
                    blocks[source]['spatial'][block] = float(np.mean([r['movement'] for r in arms]))
                else:
                    movement_key = 'cue_movement' if module=='cue' else 'transfer_tv'
                    lift_key = 'cue_all' if module=='cue' else 'transfer_lift'
                    blocks[source][movement_key][block] = float(np.mean([r['movement'] for r in arms]))
                    blocks[source][lift_key][block] = float(np.mean([r['mass_gain'] for r in arms]))
                    ratios = [normalized_attraction(r['baseline_mass'],r['mass_gain']) for r in arms]
                    finite = [v for v in ratios if v is not None]
                    if finite:
                        normalized_blocks[source][module+'_normalized'][block] = float(np.mean(finite))
        for block in design.family_order('pair'):
            values = {r['arm']:r['mi_bits'] for r in mi_tests if r['source']==source and r['block']==block}
            blocks[source]['pair_related_mi'][block] = values['related']
            blocks[source]['pair_control_mi'][block] = values['unrelated_control']
            blocks[source]['pair_related_minus_control_mi'][block] = values['related']-values['unrelated_control']
    summaries = defaultdict(dict)
    normalized = defaultdict(dict)
    for source,label in LABELS.items():
        for key,values in blocks[source].items():
            result = block_summary(values,design.strata(module_for[key]),repeats)
            if key in ('cue_all','cue_movement'):
                normalized[key][label] = result
            else:
                summaries[key][label] = result
        for key,module in (('cue_normalized','cue'),('transfer_normalized','transfer')):
            normalized[key][label] = equal_family_summary(normalized_blocks[source][key],design.strata(module),repeats=repeats,seed=0)
    common = dict(primary_condition=dict(effort='low',prompt='plain',aggregation='none'),
                  sources=LABELS,bootstrap=dict(repeats=repeats,seed=0,unit='Whole blocks within family'))
    return ({**common,'summaries':dict(summaries)},
            {**common,'summaries':dict(normalized),
             'method':{'definition':'A=q(S)-p(S); normalized_A=A/(1-p(S)) for p(S)<1',
                       'zero_denominator_policy':'Exclude from normalized summaries only',
                       'averaging':'Eligible arms within block, blocks within family, then equal families',
                       'bootstrap_repeats':repeats,'bootstrap_seed':0}})
