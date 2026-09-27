# Reproduction and collection

Run commands from the repository root. Reproduction uses recorded responses without new participants
or paid requests. Full analysis requires authorized access to restricted human responses.

## Set up Python and run the demo

Use Python **3.13** and the pinned dependencies:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python reproduce.py demo --output-dir /tmp/haidiscrepancies-demo
python -m unittest discover -s tests -v
```

Use a fresh demo destination. The demo writes a plot and summary from invented answers. Figures use
the bundled, portable DejaVu Sans font.

## Inputs and data access

Public puzzle materials and five compressed model response files are included. **Access to the human
response archive is pending Stanford approval; no human data download link is currently provided.**
Once authorized inputs are available, arrange them as follows:

```text
data/
├── stimuli/                         public puzzles, solutions, and study design
├── human/                           restricted; excluded from Git
│   ├── main/{participants,responses}.jsonl
│   ├── module/{participants,responses}.jsonl
│   └── anonymization_manifest.json
└── ai/                              public recorded model answers
    ├── manifest.json
    ├── openai/{main,module}/responses.jsonl.gz
    ├── anthropic/{main,module}/responses.jsonl.gz
    └── gemini/formal/responses.jsonl.gz
```

The model manifest records file hashes and request counts. The loader also accepts uncompressed
`responses.jsonl`; keep only one representation per input folder. Use the frozen stimuli for
reproduction, rather than regenerating puzzles.

Human archives are pseudonymized, not anonymous. Participant histories and detailed derivatives
remain restricted after IDs are replaced. Only the summary statistics underlying figures and tables
are intended for public human-data outputs. Keep raw records, processed trials, and full statistics
in restricted storage. An identity map is **not needed for reproduction** and must remain outside
this repository in separately restricted storage.

## Run the complete workflow

Run `python reproduce.py full --workers 1` when `intermediate/` and `results/` are absent.
To preserve an existing run, choose fresh destinations separate from each other and the input data:

```sh
python reproduce.py full --data-dir /path/to/authorized_data \
  --intermediate-dir /restricted/path/new_intermediate \
  --output-dir /path/to/new_results --workers 1
```

The input root must contain `stimuli/`, `human/`, and `ai/`. Full settings require substantial CPU
time; `--workers` controls parallel regression fits. Run records preserve settings, runtime, seeds,
and file hashes. The complete workflow's status is recorded in `intermediate/run.json`.

## Run individual stages

Use fresh output destinations rather than overwriting a completed run:

```sh
python reproduce.py process --stage all --output-dir intermediate/processed
python reproduce.py analyze --processed-dir intermediate/processed --output-dir intermediate/statistics
python reproduce.py render --input-dir intermediate/statistics --output-dir results --previews
```

Processing preserves inputs. `--stage models` processes public model responses; `--stage human`
applies human retention. Analysis requires a processed directory and supports `summaries`,
`regressions`, `mi`, or `all`. Compatible regression checkpoints can resume; changed inputs,
settings, code, or runtime require fresh ones.

Rendering reads saved statistics without refitting and checks that the analysis completed and its
recorded output hashes match the statistics being used. Its `figures/`, `tables/`, and `source_data/`
destinations must be empty, with no existing `render_manifest.json`. `--only` selects names from the
[figure and table manifest](../figures/manifest.json); `--previews` adds PNG copies of PDF figures.

Refresh README images with `python reproduce.py render --only fig_procedure entropy_profile --previews --output-dir /tmp/haidiscrepancies-readme`
using a fresh destination, then copy its two PNGs to `docs/figures/`.

## Outputs and source code

| Location | Purpose |
|---|---|
| `intermediate/processed/` | Retained/scored observations; restricted derivatives |
| `intermediate/statistics/` | Full statistics, regression checkpoints, and analysis manifest |
| `results/figures/` | 16 rendered figures, with optional PNG previews |
| `results/tables/` | 18 LaTeX files containing 23 numbered tables |
| `results/source_data/` | Displayed summaries, plotted aggregates, and table cells |
| `results/render_manifest.json` | Rendering settings, input/output hashes, and font hashes |

Source folders follow the workflow: `collection/` acquires responses; `processing/` retains/parses
and scores them; `analysis/` computes statistics; `figures/` renders figures, tables, and their source data.
`puzzles/` contains exact solvers and stimulus helpers; `tests/` contains automated tests.

Generated outputs are excluded from Git by default. Displayed-data exports omit individual records
and bootstrap draws; rendering requires the full statistics.

## Defaults and interpretation

The primary model condition is low effort with the plain `direct_solve` prompt. Parser **v5** leaves
missing sequence answers invalid without reusing another answer. Correctness uses all relevant
trials; solution distributions condition on correct/valid responses. Retention is applied after
collection; the rules are not a claim of preregistration.

| Setting | Default |
|---|---|
| `--bootstrap` | 5,000 puzzle/block resamples |
| `--regression-draws` | 2,000 puzzle bootstrap refits |
| `--permutations` | 10,000 TV permutations |
| `--mi-permutations` | 2,000 MI independence-test permutations |
| `--l2-penalty` | `0.0001`; use `0` for unpenalized fits |
| `--workers` | `1` |

Most puzzle/block intervals condition on observed response distributions and omit response-sampling
uncertainty. They are pointwise intervals. The separately reported trial/joint TV intervals resample
human profiles and model requests; joint intervals also resample puzzles. Context MI is empirical
and uncorrected for small-sample bias; mean p-values are descriptive. Reduced draw counts are for
workflow validation; paper table renderers require full settings.

## Prepare archives for reproduction

Data custodians can export original human records containing `main/` and `module/`. This is
unnecessary for an already pseudonymized input archive:

```sh
python reproduce.py anonymize --input-dir /path/to/original_archive \
  --output-dir data/human --id-map /restricted/location/id_map.csv
python processing/export_models.py --input-dir /path/to/original_ai \
  --output-dir /path/to/fresh_model_export
```

Both exporters require fresh destinations and leave original archives unchanged. The human exporter
preserves participants before retention and response pairing, removing absolute dates and
administrative fields. The required ID map must be outside the repository and both archives;
matching maps may be reused. Deposit `main/`, `module/`, and `anonymization_manifest.json` together;
keep the map separate.

Model exports retain answer text, study settings, token counts, pairing IDs, and image/prompt hashes.
Provider IDs, opaque payloads, local image/prompt paths, and run/batch manifests are omitted.
Unreviewed fields and populated API errors are rejected. Review new answer text before publication;
field filtering cannot remove arbitrary sensitive text.

## Collect new responses

Start with an offline mock run in a fresh destination:

```sh
python collection/models/run.py prepare --out_root /tmp/haidiscrepancies-model-demo --run_id mock --samples 1 \
  --dataset_scope main --api_model_conditions OAI_GPT56_SOL_LOW \
  --prompt_conditions direct_solve
python collection/models/run.py mock --run_dir /tmp/haidiscrepancies-model-demo/mock --limit 2
```

Live collection uses `live --run_dir <run_directory>` after preparation; batch collection uses
`batch prepare`, `batch submit`, and `batch collect`. Live/submission commands call paid providers.
Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the
environment, never committed files. Use `--help` for options; stimulus tools are in `collection/stimuli/`.

Start the local human interface with a separate private database:

```sh
python collection/human/server.py --host 127.0.0.1 --port 8015 \
  --db data/human/new_runs/local_study.sqlite3
```

Open [the main study](http://127.0.0.1:8015/study_web/app/?formal=1&study=main), or change
`study=main` to `study=module`. Admin exports, backups, and databases remain private. Formal
deployments require environment-configured admin credentials, completion codes, and
participant-verification settings; local development credentials are refused on nonlocal hosts.

## Remove generated files

After a run has stopped, you can delete `intermediate/` and `results/` when their outputs are no
longer needed. Both can be regenerated from the original inputs in `data/`; keep those inputs.
Caches such as `.cache/` and `__pycache__/` can also be deleted and are excluded from Git.
