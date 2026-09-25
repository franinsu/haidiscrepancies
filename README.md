# Investigating Human–AI Discrepancies via Multiple-Solution Problems

Code and study materials for comparing how humans and AI models choose among
valid solutions to the same puzzle. The study covers arithmetic, mazes, grid
placement, Minesweeper and Sudoku, including changes to presentation and context.

The repository contains response collection, processing, statistical analysis,
and figure and table generation. The reference manuscript is the 23 September
2026 revision. Reproduction runs locally from recorded responses and requires
no new participants or paid model requests. Differences from that manuscript's
numerical results and artwork are described under [Reproducibility](#reproducibility).

## Start here

Use Python **3.13**. Run these commands from this folder:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python reproduce.py demo --output-dir /tmp/haidiscrepancies-demo
python -m unittest discover -s tests -v
```

The demo scores invented answers to a public puzzle and writes a small plot and
numerical summary to `/tmp/haidiscrepancies-demo`. Choose a destination that does
not already exist. The demo and tests work with the files included in this
repository; collected responses are needed only for the full study analysis.

## Reproduce the study

Full reproduction requires the recorded human and model response archives.
These archives are **not included in this repository**, and a data-access link
is not yet provided here. Human records require restricted access. Once the
archives are available, arrange the inputs as follows, keeping the accompanying
archive and run manifests:

```text
data/
├── stimuli/                         included public study materials
├── private/
│   ├── main/{participants,responses}.jsonl
│   ├── module/{participants,responses}.jsonl
│   └── anonymization_manifest.json
└── ai/
    ├── openai/{main,module}/responses.jsonl
    ├── anthropic/{main,module}/responses.jsonl
    └── gemini/formal/responses.jsonl
```

Run the complete workflow:

```sh
python reproduce.py full --workers 1
```

Both `intermediate/` and `results/` must be absent before a full run. To retain an
existing run, choose fresh paths with `--intermediate-dir` and `--output-dir`.
Use `--data-dir` for another input root containing `stimuli/`, `private/` and
`ai/`, and `--workers` for parallel regression fits. Full settings require
substantial CPU time. Smaller resampling counts can test the pipeline, but do
not reproduce the paper's uncertainty estimates.

## Data and privacy

| Location | Contents | Availability and handling |
|---|---|---|
| `data/stimuli/` | Frozen puzzles, complete solution catalogs, module design and selection provenance | Included in the repository |
| `data/private/` | Pseudonymized human response archives | Restricted input; excluded from Git |
| `data/ai/` | Recorded model requests and run/batch metadata | Separate input archive; excluded from Git |
| `intermediate/` | Processed trials, full statistics, regression checkpoints and run records | Generated locally; excluded from Git; contains private or restricted derivatives |
| `results/` | Figures, LaTeX tables, displayed-data exports and rendering manifest | Generated locally; excluded from Git by default |

Public human-data outputs are limited to the summary statistics used in figures
and tables. Individual response histories remain private after participant IDs
are replaced. Keep raw human records and detailed derivatives in restricted
storage; intermediate files can be regenerated from the response archives.

Reproduction needs `data/private/{main,module}/`, not an identity map. These
archives preserve final answers, exclusions, timings and within-study pairing
while removing identifying and administrative fields. The model archive has
288,000 requests, expanding to 324,000 scored trials because sequence requests
contain two puzzles. Retention yields 104 main and 417 module participants.

The following diagram traces data from recorded responses to final outputs.
Full statistical files remain local; selected plotted values and table cells
are exported under `results/`.

```text
data/
├── stimuli/ [PUBLIC: puzzles, solutions and conditions]
│   ├── study illustrations → results/figures/
│   └── catalogs, design and features → statistics below
├── private/ [PRIVATE]
│   ├── {main,module}/participants.jsonl + responses.jsonl
│   │   └── intermediate/processed/human/ [PRIVATE]
│   │       ├── main_retained.jsonl
│   │       └── module_retained.jsonl
│   └── anonymization_manifest.json
└── ai/ [separate model archive]
    └── responses + run/batch manifests
        └── intermediate/processed/ai/*_{main,module}.jsonl

Retained/scored observations + public study design
                         ↓
intermediate/statistics/ [LOCAL, REGENERABLE]
├── Main study
│   ├── figure_statistics.json
│   │   ├── entropy_deficit_stats.json → entropy_correlation_stats.json
│   │   └── source_geometry.json + condition distances → condition_geometry.json
│   ├── main_solution_probabilities.json
│   └── effort_difficulty_stats.json
├── Modules
│   └── perturbation_counts.json
│       ├── perturbation_statistics.json
│       │   ├── gain_summaries.json
│       │   └── context_dependence_table.json
│       └── context_mi_tests.json → context_mi_summary.json
│       Effects + MI tests + design also produce:
│       stimulus_sensitivity_stats.json and normalized_attraction_stats.json
└── Features + main answers
    ├── five family *_feature_models.json files
    └── regression_checkpoints/
                         ↓ select displayed quantities
results/ [FIGURES, TABLES AND DISPLAYED SUMMARIES]
├── figures/
├── tables/
├── source_data/ [displayed summaries and plotted aggregates]
└── render_manifest.json
```

[figures/manifest.json](figures/manifest.json) gives exact inputs for each
figure and table. Arrows between statistics show calculation dependencies;
values are passed in memory and saved as snapshots. `source_data/` contains
selected displayed summaries; the renderer requires the full statistics in
`intermediate/statistics/`, not these exports alone.
`intermediate/run.json` and stage manifests record settings, input/code hashes
and output hashes.

## Code and individual stages

| Folder | Responsibility | Start reading |
|---|---|---|
| `collection/` | Human interface, model requests, stimulus generation | `human/server.py`, `models/run.py` |
| `processing/` | Retention, model parsing and scoring | `run.py`, `retention.py`, `response_parser.py` |
| `analysis/` | Estimates, uncertainty, geometry and feature models | `pipeline.py`, `metrics.py`, `regressions.py` |
| `figures/` | Present results and export displayed values | `manifest.json`, `run.py`, `tables.py` |
| `puzzles/` | Shared exact solvers and stimulus generation helpers | Family-specific modules |
| `tests/` | Automated tests | `test_analysis_methods.py`, `test_regression_penalty.py` |

Run stages separately with fresh output destinations:

```sh
python reproduce.py process --stage all --output-dir intermediate/processed
python reproduce.py analyze --processed-dir intermediate/processed --output-dir intermediate/statistics
python reproduce.py render --previews
```

Processing preserves raw inputs and writes retained human rows, a retention
summary and six scored model files. Analysis requires an explicit processed
input directory; its modes are `summaries`, `regressions`, `mi` and `all`.
Compatible regression checkpoints can be resumed. Changed inputs, settings,
code or runtime require a fresh checkpoint directory.

Rendering produces 16 figures and 18 table files containing 23 numbered tables.
The destinations `results/{figures,tables,source_data}/` must be empty and no
`results/render_manifest.json` may exist. `--only` selects asset names from the
figure manifest; `--previews` adds PNG copies of PDF figures. Rendering does not
refit statistics. Rendering archived numerical inputs requires explicit
`--reference`; it is recorded as reference rendering, not fresh analysis.

The portable font is DejaVu Sans. To match the paper's font, select a locally
available Arial installation with `STUDY_FONT_FAMILY=Arial`; `STUDY_FONT_DIR`
can point to a font directory. Arial is not bundled. Font-file hashes are
recorded in the rendering manifest.

## Analysis conventions

- The primary model condition is low effort with the plain prompt. Model
  answers use parser **v4**, including its historical sequence fallback.
- Correctness uses all relevant trials. Solution distributions condition on
  correct/valid answers and retain every catalogued class, including zeros.
- Human retention requires complete response coverage and excludes testers.
  Main participants have at most 30 bad records overall and 14 per family;
  module participants need at least 40 correct answers and two per module.
  Duration and completion-code issuance are not analysis exclusions. These
  are implemented post-collection rules, not a claim of preregistration.
- TV is half the summed absolute probability difference. Normalized entropy
  is Shannon entropy in bits divided by `log2(k)`. Average puzzles within
  families, then weight families equally.
- Effort is log median correct-response time or reasoning-token count,
  centered within source across the 100 main puzzles; model medians have a
  one-token floor.
- Context MI uses complete pairs with both answers correct, in bits, without
  bias correction. Its marginals come from the same joint table. Individual
  independence tests, mean p-values and Human–model contrasts answer different
  questions; mean p is descriptive.
- Normalized directional change is `(q(S)-p(S))/(1-p(S))`, computed before
  averaging. Undefined cases with `p(S)=1` are excluded only from this
  normalized summary; negative changes are retained.
- Regressions use 13 puzzle features and seven models per family: four choice
  models and three model-versus-human models. Validation holds out whole puzzle
  identities, including every duplicate in a bootstrap draw. Constant-feature
  replacements use nonconstant training puzzles only, with uniform candidate
  weights and fallback `0.5`. The exploratory feature set is fixed.

| Uncertainty calculation | Full settings |
|---|---|
| Main intervals | 5,000 whole-puzzle resamples within family; shared draws across comparisons |
| Trial/joint TV intervals | Resample human profiles and model requests, then reapply correctness filters; joint intervals also resample puzzles |
| Perturbation intervals | 5,000 whole-block resamples within family, preserving arms |
| TV permutation tests | 10,000 permutations; Holm adjustment over 320 comparisons, excluding the 40 unrelated-context controls before adjustment |
| MI independence tests | 2,000 permutations with canonical pair ordering and specified SHA-256/PCG64 streams |
| MI contrasts | Paired whole-puzzle bootstrap within family |
| Regression intervals | 2,000 puzzle bootstrap refits, including held-out validation within each draw |

Most puzzle/block intervals condition on the observed response distributions;
they do not include response-sampling uncertainty. Intervals are pointwise.
The root command pins numerical-library threads and `PYTHONHASHSEED`; seeds,
draw counts and runtime are recorded in outputs. Paper table renderers require
full draw counts.

## Reproducibility

The frozen stimuli and catalogs define the study inputs. Regenerating candidate
puzzles from a seed does not recover the final Arithmetic puzzle replacement;
its provenance is recorded under `data/stimuli/selection_provenance/`.
New collection produces a new sample and cannot recreate historical model
availability or responses.

Compared with the reference manuscript, the default L2 penalty changes some
regression estimates and interval endpoints. The fully specified MI permutation
stream changes some p-values while preserving observed MI and all individual
0.05 significance decisions in the recorded study. Figure 2 recomputes its
intervals from the stated procedure; the exact random realization used in the
reference artwork is not documented. Font metrics and figure layouts can also
differ. Numerical settings, random seeds and file hashes are recorded with
each run.

## Export a new pseudonymized human archive

This step is for data custodians preparing an archive from original human
records. It is not needed when reproducing the study from an already
pseudonymized archive. The input must contain `main/` and `module/`:

```sh
python reproduce.py anonymize --input-dir /path/to/original_archive \
  --output-dir /path/to/fresh_archive --id-map data/private/id_map.csv
```

The exporter requires a fresh destination, preserves all participants before
retention, removes absolute dates and administrative fields, and preserves
bootstrap ordering. Valid existing maps are reused. A restricted deposit should
contain `main/`, `module/` and `anonymization_manifest.json`; keep `id_map.csv`
separately restricted. This is reversible pseudonymization.

## Collecting new responses

For an offline mock run:

```sh
python collection/models/run.py prepare --out_root /tmp/haidiscrepancies-model-demo --run_id mock --samples 1 --dataset_scope main --api_model_conditions OAI_GPT56_SOL_LOW --prompt_conditions direct_solve
python collection/models/run.py mock --run_dir /tmp/haidiscrepancies-model-demo/mock --limit 2
```

Live collection uses `prepare --run_id new_study --samples 100`, followed by
`live --run_dir data/ai/new_runs/new_study`. Batch collection has separate
`batch prepare`, `batch submit` and `batch collect` commands. Live requests and
batch submission call paid providers. Credentials come from `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

Frozen prompts, 240 images, settings and their manifests live in
`collection/models/`. Runs verify their hashes and the v4 parser hash.
The two prompt conditions are `direct_solve` and `human_participant`; A/B
sequences use one combined image. Collection is one-shot, with no solver
feedback or invalid-answer retries. Images give rules and answer formats,
without revealing solution counts or candidate answers.

For the human interface:

```sh
python collection/human/server.py --host 127.0.0.1 --port 8015 --db data/private/new_runs/local_study.sqlite3
```

Open `http://127.0.0.1:8015/study_web/app/?formal=1&study=main` or change the study
to `module`. Participant, authenticated admin and researcher-test pages live
under `/study_web/`; database files are never static assets. Admin exports and
backups remain private. Formal deployments require environment-configured admin
credentials, completion codes and participant-verification settings; local
development credentials are refused on nonlocal hosts.

Stimulus generation and validation commands live in `collection/stimuli/`.
New generation requires a fresh destination. Image rendering uses Pillow's
BASIC layout engine; matching the frozen pixels also requires the original
font. Use each entrypoint's `--help` for generation and deployment options.

## Cleanup

```sh
python clean_temp.py                   # Preview caches and build leftovers
python clean_temp.py --apply           # Remove those temporary files
python clean_temp.py --results         # Also preview intermediate/ and results/
python clean_temp.py --results --apply # Delete generated outputs for a fresh run
```

Stop running jobs first. Ordinary cleanup removes caches and build leftovers.
Adding `--results` also removes `intermediate/` and `results/`, so use it only
when those outputs are no longer needed. Raw archives, public stimuli, source
code and the Python environment are preserved; symbolic links are not followed.
Outputs written to custom locations can be removed separately.
