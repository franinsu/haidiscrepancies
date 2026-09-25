"""Marginal and complete-pair counts for the current perturbations."""
from collections import Counter
from . import stimulus_modules as sm
from . import main_statistics as core

def extract_counts(data_dir, processed_dir):
    design,_=sm.load_design(data_dir)
    sources={'Human':sm.load_human_answers(processed_dir,design),
             **sm.load_model_answers(processed_dir,design,core.PROVIDERS,core.PROMPTS,core.condition_key)}
    primary=['Human']+[p+'|low|plain' for p in core.PROVIDERS]
    trials={t.id:{'ids':list(t.abstract_ids),'coordinates':t.coord_of,
                  'counts':{s:dict(sources[s].counts(t.id)) for s in primary}}
            for t in design.trials.values()}
    comparisons=[]
    for module,families in design.families.items():
        for family in families:
            if module=='cue':
                pairs=[(family.by_role(sm.ROLE_BASELINE)[0],family.by_condition(c),c)
                       for c in sm.CUE_ARMS]
            elif module=='spatial':
                pairs=[(family.by_condition(sm.SPATIAL_ORIGINAL),family.by_condition(c),c)
                       for c in sm.SPATIAL_REFLECTIONS]
            elif module=='formulation':
                pairs=[(family.by_condition(sm.FORMULATION_ORIGINAL),family.by_condition(sm.FORMULATION_PERTURBED),'reversed')]
            else:
                sequences=family.sequences()
                pairs=[(sequences[sm.PAIR_ALONE if module=='pair' else sm.TRANSFER_ALONE]['B'],
                        sequences[c]['B'],c) for c in
                       ([sm.PAIR_RELATED,sm.PAIR_CONTROL] if module=='pair' else [sm.TRANSFER_PRIME])]
            for original,perturbed,arm in pairs:
                row={'module':module,'block':family.family_id,'family':family.puzzle_type,
                     'arm':arm,'original':original.id,'perturbed':perturbed.id,
                     'target_ids':sorted(perturbed.target_ids) if perturbed.target_ids else None}
                human_original={a.pair_key[1] for a in sources['Human'].all_rows(original.id)}
                human_perturbed={a.pair_key[1] for a in sources['Human'].all_rows(perturbed.id)}
                assert not human_original & human_perturbed
                if module=='pair':
                    a=family.sequences()[arm]['A']
                    row.update({'preceding':a.id,'answer_map':perturbed.related_map,
                        'joint_counts':{s:[{'a':aa,'b':bb,'n':n} for (aa,bb),n in
                            sorted(Counter(sm.paired_classes(sources[s],a,perturbed)).items())] for s in primary}})
                comparisons.append(row)
    payload={'primary_sources':primary,'trials':trials,'comparisons':comparisons,
             'sampling':'Marginals condition on validity; joint tables require both answers valid. Within each block, Human reference and perturbed cohorts are disjoint.'}
    return payload
