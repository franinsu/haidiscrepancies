"""Current TV effects, whole-block intervals and Holm-320 tests."""
import hashlib

import json

import random


import numpy as np

NBOOT=5000

NPERM=10000

ROW_ORDER=['cue','spatial','formulation','pair_related','pair_control','transfer']


def tv(p,q):return float(np.abs(p-q).sum()/2)

def distribution(trial,source,ids):
    counts=np.array([trial['counts'][source].get(k,0) for k in ids],dtype=int)
    assert counts.sum()>0
    return counts,counts/counts.sum()

def rng_for(*parts):
    seed=int.from_bytes(hashlib.sha256('|'.join(parts).encode()).digest()[:8],'little')
    return np.random.Generator(np.random.PCG64(seed))

def bootstrap(groups):
    sizes=[len(v) for v in groups];rng=random.Random(0);offsets=np.cumsum([0]+sizes[:-1])
    draws=np.array([[o+rng.randrange(n) for o,n in zip(offsets,sizes) for _ in range(n)] for _ in range(NBOOT)])
    values=np.concatenate(groups);cuts=np.cumsum([0]+sizes)
    sampled=np.mean([values[draws[:,cuts[i]:cuts[i+1]]].mean(axis=1) for i in range(len(groups))],axis=0)
    return [float(x) for x in np.quantile(sampled,[.025,.975])]

def simple_test(n0,n1,observed,rng):
    total=n0+n1;nq=int(n1.sum());np0=int(n0.sum())
    draws=rng.multivariate_hypergeometric(total,nq,size=NPERM)
    null=np.abs(draws/nq-(total-draws)/np0).sum(axis=1)/2
    return float((1+np.count_nonzero(null>=observed-1e-12))/(NPERM+1))

def joint_test(joint,observed,rng):
    na=joint.sum(axis=1);nb=joint.sum(axis=0);n=int(joint.sum())
    a=np.repeat(np.arange(len(na)),na);pool=np.repeat(np.arange(len(nb)),nb)
    null_reference=(na/n)[:,None]*(nb/n)[None,:]
    exceed=0
    for start in range(0,NPERM,200):
        count=min(200,NPERM-start)
        shuffled=rng.permuted(np.tile(pool,(count,1)),axis=1)
        null_joint=np.zeros((count,len(na),len(nb)),dtype=int)
        np.add.at(null_joint,(np.arange(count)[:,None],a[None,:],shuffled),1)
        values=np.abs(null_joint/n-null_reference).sum(axis=(1,2))/2
        exceed+=int(np.count_nonzero(values>=observed-1e-12))
    return (1+exceed)/(NPERM+1)

def apply_holm(records):
    """Adjust only the prespecified family; unrelated context remains unadjusted."""
    order = sorted((i for i,r in enumerate(records) if r['row'] != 'pair_control'),
                   key=lambda i:records[i]['p_value'])
    for record in records:
        value = record['p_value']
        if not 0 <= value <= 1: raise ValueError('Invalid permutation p-value')
        record['p_holm'] = None
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*records[index]['p_value']))
        records[index]['p_holm'] = previous
    return order


def compute(data):
    records = []
    for comparison in data['comparisons']:
        module, arm = comparison['module'], comparison['arm']
        row_key = ('pair_related' if arm == 'related' else 'pair_control') if module == 'pair' else module
        original, perturbed = [data['trials'][comparison[k]] for k in ('original','perturbed')]
        ids = original['ids']
        if set(ids) != set(perturbed['ids']):
            raise ValueError('Compared variants have different answer classes')
        for source in data['primary_sources']:
            n0,p = distribution(original,source,ids)
            n1,q = distribution(perturbed,source,ids)
            record = {k:comparison[k] for k in ('module','block','family','arm','original','perturbed')}
            record.update(row=row_key,source=source,n_original=int(n0.sum()),n_perturbed=int(n1.sum()))
            rng = rng_for(source,comparison['block'],arm,'effect-null')
            if module == 'pair':
                aids = data['trials'][comparison['preceding']]['ids']
                joint = np.zeros((len(aids),len(ids)),dtype=int)
                for cell in comparison['joint_counts'][source]:
                    joint[aids.index(cell['a']),ids.index(cell['b'])] = cell['n']
                Q = joint/joint.sum()
                qa,qb = Q.sum(axis=1),Q.sum(axis=0)
                reference = qa[:,None]*qb[None,:]
                record.update(n_complete_pairs=int(joint.sum()),movement=tv(Q,reference))
                if comparison['answer_map']:
                    mapped = [ids.index(comparison['answer_map'][a]) for a in aids]
                    record.update(repeat=float(sum(Q[i,j] for i,j in enumerate(mapped))),
                                  reference_repeat=float(sum(reference[i,j] for i,j in enumerate(mapped))))
                record['p_value'] = joint_test(joint,record['movement'],rng)
            else:
                record['movement'] = tv(p,q)
                if module in ('cue','transfer'):
                    inside = np.array([k in comparison['target_ids'] for k in ids])
                    # Exact integer fractions make the p=1 boundary unambiguous.
                    pmass = int(n0[inside].sum())/int(n0.sum())
                    qmass = int(n1[inside].sum())/int(n1.sum())
                    record.update(baseline_mass=pmass,perturbed_mass=qmass,mass_gain=qmass-pmass)
                record['p_value'] = simple_test(n0,n1,record['movement'],rng)
            records.append(record)
    order = apply_holm(records)
    summaries = {}
    for row_key in ROW_ORDER:
        summaries[row_key] = {}
        for source in data['primary_sources']:
            selected = [r for r in records if r['row']==row_key and r['source']==source]
            groups, block_values = [], {}
            for family in sorted({r['family'] for r in selected}):
                values = []
                for block in sorted({r['block'] for r in selected if r['family']==family}):
                    block_values[block] = float(np.mean([r['movement'] for r in selected if r['block']==block]))
                    values.append(block_values[block])
                groups.append(np.asarray(values))
            summaries[row_key][source] = {'movement':{
                'estimate':float(np.mean([g.mean() for g in groups])),
                'ci95':bootstrap(groups),'block_values':block_values,
                'retained_comparisons':len(selected),'total_comparisons':len(selected),
                'n_blocks':len(block_values)}}
    return dict(schema_version=3,row_order=ROW_ORDER,sources=data['primary_sources'],
        bootstrap=dict(unit='Whole block within family, all arms retained together',repeats=NBOOT,seed=0,family_weight='equal'),
        permutation=dict(repeats=NPERM,seed='SHA256 of source, block, arm and effect-null',
            p_formula=f'(1 + null statistics at least observed) / ({NPERM} + 1)',
            null_simple='Exchange variant labels conditional on the valid counts and group sizes.',
            null_pair='Q(A,B) = Q_A(A) Q_B(B); permute B labels among complete valid pairs.',
            correction='Holm across 320 primary-source comparisons, excluding unrelated context',family_size=len(order)),
        counts_sha256=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        records=records,summaries=summaries)
