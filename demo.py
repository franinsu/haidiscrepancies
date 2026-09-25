"""A small, completely invented response example using the actual scorer and metrics."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


def run(output: Path) -> dict:
    from processing.response_parser import PARSER_VERSION, score_raw_answer
    from analysis.metrics import distribution, entropy, tv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (root / 'data/stimuli/all_puzzles.jsonl').open() as stream:
        puzzle = next(row for row in map(json.loads, stream) if row['puzzle_type'] == 'maze')
    answers = [s['moves'] for s in puzzle['solutions']]
    # Frequencies are invented deliberately, not sampled from study records.
    invented = {
        'Synthetic human': [answers[0]] * 4 + [answers[1]] * 3 + [answers[2], 'INVALID'],
        'Synthetic model': [answers[0]] * 2 + [answers[1]] * 5 + [answers[2], 'INVALID'],
    }
    scored = []
    counts = {}
    for source, texts in invented.items():
        outcomes = [score_raw_answer(puzzle, text) for text in texts]
        counts[source] = Counter(row['solution_id'] for row in outcomes if row['is_valid'])
        for text, result in zip(texts, outcomes):
            scored.append({'source': source, 'synthetic': True, 'answer': text,
                           'is_valid': result['is_valid'], 'solution_id': result['solution_id']})
    classes = [s['solution_id'] for s in puzzle['solutions']]
    probability = {source: {cls: distribution(c).get(cls, 0.0) for cls in classes}
                   for source, c in counts.items()}
    summary = {
        'synthetic': True,
        'description': 'Invented responses to a public study puzzle. These are not study results.',
        'puzzle_id': puzzle['id'], 'parser_version': PARSER_VERSION,
        'probabilities': probability,
        'valid_responses': {s: sum(c.values()) for s, c in counts.items()},
        'normalized_entropy': {s: entropy(c) / math.log2(len(classes)) for s, c in counts.items()},
        'total_variation': tv(*counts.values()),
    }
    (output / 'invented_responses.json').write_text(json.dumps(scored, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    fig, axis = plt.subplots(figsize=(7, 3.6))
    for i, (source, probs) in enumerate(probability.items()):
        axis.bar([j + (i - 0.5) * 0.36 for j in range(len(classes))],
                 list(probs.values()), width=0.36, label=source)
    axis.set_xticks(range(len(classes)), [str(i + 1) for i in range(len(classes))])
    axis.set(xlabel='Valid solution class', ylabel='Fraction of correct answers',
             title='Synthetic example — invented responses', ylim=(0, 1))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'synthetic_distributions.png', dpi=160)
    plt.close(fig)
    return summary
