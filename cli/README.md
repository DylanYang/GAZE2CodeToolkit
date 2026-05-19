# GAZE2CodeToolkit CLI scripts

Headless replacements for the five Jupyter notebooks at the repository
root. Each script wraps the same `g2c` Python package the notebooks
import, but exposes a `python -m cli.<name>` interface for batch /
scripted / pipeline use.

The original notebooks are still present and unchanged — they remain the
research-history record. The CLIs are the recommended entry points for
new runs.

## Setup

Run from the `GAZE2CodeToolkit/` directory so that `g2c` and the
`datasets/`, `output/` relative paths resolve correctly:

```bash
cd GAZE2CodeToolkit
python -m cli.<name> --help
```

The CLIs depend only on the existing `g2c` package — no new Python
deps. `evaluate_ocr.py` optionally uses `scikit-learn` for the ROC
curve; pass `--skip-roc` to bypass.

## Parser unification

Two datasets are now routed through a single config-driven Tobii
parser (`g2c/parsers/tobii.py` + `g2c/parsers/datasets_config.py`):

| Dataset | Legacy default sample_size | New API |
|---|---|---|
| UNL_UM  | 44 | `parsers.load("UNL_UM")` |
| YMU_UM  | 84 | `parsers.load("YMU_UM")` |

The legacy `parsers.UNL_UM()` and `parsers.YMU_UM()` functions still
work and are forwarded to `load(...)` internally — existing notebooks
do not need to change.

Adding a third Tobii dataset (same hardware export format) is a config
entry in `datasets_config.DATASETS`, no code edit required.

## The five CLIs

### 1. `extract_fixations.py`

Replaces `g2c_fixation_extractor.ipynb`. Loads a dataset and writes
fixation CSVs.

```bash
# All-in-one CSV
python -m cli.extract_fixations --dataset UNL_UM --mode all \
    --out-dir output/unl_um/all/fixations

# Per-trial CSVs
python -m cli.extract_fixations --dataset YMU_UM --mode by-task \
    --out-dir output/ymu_um/group/fixations

# Per (participant, trial) CSVs
python -m cli.extract_fixations --dataset UNL_UM --mode per-participant \
    --out-dir output/unl_um/individual/fixations --sample-size 50
```

### 2. `extract_aoi.py`

Replaces `g2c_aoi_extractor.ipynb`. OCR → token AOIs → fixation hit-test.

```bash
# Single participant, single trial
python -m cli.extract_aoi --dataset YMU_UM \
    --trial-id introduction-Q1 \
    --image-dir datasets/YMU_UM/stimuli \
    --image-prefix "Quiz - " --image-suffix " (localhost).png" \
    --experiment-id Participant2 \
    --out-dir output/ymu_um

# Task-level (all participants on one trial)
python -m cli.extract_aoi --dataset YMU_UM --by-task \
    --trial-id introduction-Q5 \
    --image-dir datasets/YMU_UM/stimuli \
    --image-prefix "Quiz - " --image-suffix " (localhost).png" \
    --out-dir output/ymu_um
```

### 3. `visualize.py`

Replaces `g2c_visionizer.ipynb`. Renders trial overlay, heatmap,
fixation duration, fixation timeline as PNG (matplotlib `Agg`
backend).

```bash
# Individual heatmap + trial overlay
python -m cli.visualize --dataset YMU_UM \
    --trial-id introduction-Q5 \
    --experiment-id Participant52 \
    --out-dir output/ymu_um/viz \
    --kinds trial heatmap

# Task-level heatmap from an existing AOI fixation CSV
python -m cli.visualize \
    --aoi-csv output/ymu_um/group/aoi/aoi_fixations_introduction-Q5.csv \
    --out-dir output/ymu_um/viz \
    --kinds heatmap --sigma 35 --vmax 1200
```

### 4. `score_expertise.py`

Replaces `g2c_expertise.ipynb`. Scores MCQ answers and emits per-
participant totals.

```bash
python -m cli.score_expertise \
    --input data/ymu_um/python_mcq_answers.csv \
    --output data/ymu_um/python_mcq_scores.csv
```

The default answer key targets the YMU-UM Python MCQ task. Override
via `--answer-key-json` and `--points-json` if needed.

### 5. `evaluate_ocr.py`

Replaces `evaluate_ocr.ipynb`. Compares OCR output against a
ground-truth CSV. Writes metrics CSV, per-match CSV, and ROC PNG.

```bash
python -m cli.evaluate_ocr \
    --ground-truth output/ocr_groundtruth/Q5_ground_truth04.csv \
    --detected output/orc_detection/Q5_detected_tokens.csv \
    --out-dir output/ocr_eval/Q5
```

Both `gt` and `detected` CSVs must contain columns:
`line_num, x, y, width, height, text`. `detected` may additionally
contain `confidence`.

## Migration checklist (per script)

If you previously ran the notebook by editing cells in place, the
equivalent CLI invocations are:

| Notebook                       | CLI                          |
|--------------------------------|------------------------------|
| `g2c_fixation_extractor.ipynb` | `python -m cli.extract_fixations` |
| `g2c_aoi_extractor.ipynb`      | `python -m cli.extract_aoi`       |
| `g2c_visionizer.ipynb`         | `python -m cli.visualize`         |
| `g2c_expertise.ipynb`          | `python -m cli.score_expertise`   |
| `evaluate_ocr.ipynb`           | `python -m cli.evaluate_ocr`      |
