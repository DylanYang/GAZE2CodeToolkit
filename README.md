# GAZE2CodeToolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-005F73.svg)](https://github.com/tesseract-ocr/tesseract)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](https://peps.python.org/pep-0008/)
[![Last commit](https://img.shields.io/github/last-commit/DylanYang/GAZE2CodeToolkit)](https://github.com/DylanYang/GAZE2CodeToolkit/commits/main)
[![Repo stars](https://img.shields.io/github/stars/DylanYang/GAZE2CodeToolkit?style=social)](https://github.com/DylanYang/GAZE2CodeToolkit/stargazers)

End-to-end pipeline for turning raw eye-tracking recordings of programmers
reading source code into **fixation × code-token AOI** tables, and from
there into **participant-level expertise predictions** via an
interpretable prototype-based classifier (PCGC). The transformer line
(EC-GazeFormer, Stage III) remains separate.

```
                    ┌──────────────────────────────────────────────┐
                    │ Raw Tobii TSV   →   Canonical fixation rows  │
                    │                       (parsers / I-DT)       │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────────┐
                    │ Stimulus image  →   OCR token AOIs           │
                    │                       (aoi.aoi_detector)     │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────────┐
                    │ Fixation hit-test → fixation × token table   │
                    │                       (aoi.hit_test)         │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────────┐
                    │ Token features  →   PCGC + hybrid classifier │
                    │                       (g2c.classification)   │
                    └────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                          Expertise prediction (AUC / macro-F1)
                                     │
                                     ▼
                                EC-GazeFormer (separate repo)
```

## Quick start

### Option A — Conda (recommended, includes the Tesseract binary)

```bash
git clone <repo-url>
cd GAZE2CodeToolkit

conda env create -f environment.yml
conda activate gaze2code

# Sanity check
python -c "from g2c.parsers import available_datasets; print(available_datasets())"
# → ['UNL_UM', 'YMU_UM']
```

### Option B — pip / virtualenv

You will need to install the Tesseract OCR binary separately. See the
"System dependencies" section below.

```bash
git clone <repo-url>
cd GAZE2CodeToolkit

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

python -c "from g2c.parsers import available_datasets; print(available_datasets())"
```

## System dependencies

Only one item is not pip-installable: the **Tesseract OCR binary**, which
backs `pytesseract`. If you use **Option A (conda)** above, this is
already handled — `environment.yml` pulls Tesseract from conda-forge and
you can skip the rest of this section.

If you use **Option B (pip/venv)**, install Tesseract through your OS
package manager:

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt-get install tesseract-ocr` |
| macOS (Homebrew) | `brew install tesseract` |
| Windows 11 | UB-Mannheim installer — <https://github.com/UB-Mannheim/tesseract/wiki> |

Verify the install:

```bash
tesseract --version
```

### Windows 11 notes

1. Download `tesseract-ocr-w64-setup-5.x.x.exe` from the
   [UB-Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)
   and run it. Keep the default install path
   (`C:\Program Files\Tesseract-OCR\`) and make sure the **English**
   language pack is selected.
2. Add `C:\Program Files\Tesseract-OCR` to your system `PATH`
   (`Win+R` → `sysdm.cpl` → Advanced → Environment Variables → Path →
   New), then open a new terminal and confirm with `tesseract --version`.
3. If you would rather not edit `PATH`, point `pytesseract` at the
   binary explicitly inside your code:

   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = (
       r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   )
   ```

### macOS notes

Homebrew puts `tesseract` on `PATH` automatically — `pytesseract` will
find it with no further configuration. If `brew` itself is not
installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon the binary lives at `/opt/homebrew/bin/tesseract`; on
Intel Macs at `/usr/local/bin/tesseract`. Either way `which tesseract`
should return a path after `brew install tesseract`.

### Linux notes

On Debian / Ubuntu the `tesseract-ocr` package installs both the binary
and the default English language data:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
# Optional: extra language packs, e.g. simplified Chinese
sudo apt-get install tesseract-ocr-chi-sim
```

Other distros:

| Distro | Command |
|---|---|
| Fedora / RHEL | `sudo dnf install tesseract` |
| Arch / Manjaro | `sudo pacman -S tesseract tesseract-data-eng` |
| openSUSE | `sudo zypper install tesseract-ocr` |

The package manager places the binary at `/usr/bin/tesseract`, already on
`PATH`. WSL2 users follow the same Debian/Ubuntu instructions — the
binary is shared with your Linux side, not your Windows side, so
Tesseract has to be installed in both environments if you want to use
both.

## Repository layout

```
GAZE2CodeToolkit/
├── environment.yml          # Conda env (Python 3.10 + Tesseract)
├── requirements.txt         # pip-installable deps
├── README.md                # ← you are here
├── g2c/                     # Python package
│   ├── parsers/             # Dataset parsers (UNL_UM, YMU_UM, ...)
│   │   ├── tobii.py             # Unified Tobii TSV parser
│   │   ├── datasets_config.py   # Per-dataset config (paths, columns, ...)
│   │   ├── load.py              # parsers.load(name, sample_size=...)
│   │   ├── UNL_UM.py / YMU_UM.py  # Thin back-compat wrappers
│   │   └── ...
│   ├── fixation_classification/   # I-DT classifier
│   ├── aoi/                       # OCR AOI detection, hit-test
│   ├── classification/            # PCGC prototype classifier + hybrid CV
│   ├── util/                      # Helpers (lines, stimuli, exports)
│   └── visualization/             # Trial overlay, heatmap, timeline
├── cli/                     # Command-line entry points (see cli/README.md)
│   ├── extract_fixations.py
│   ├── extract_aoi.py
│   ├── visualize.py
│   ├── score_expertise.py
│   ├── evaluate_ocr.py
│   └── classify_expertise.py
├── app.py                   # Streamlit web UI entry point
├── webapp/                  # Streamlit tab modules and shared helpers
│   ├── components.py
│   ├── state.py
│   └── tabs/{extract,aoi,visualize,evaluate}.py
├── datasets/                # Dataset directories (see config; not in git)
├── output/                  # Default location for generated CSVs / PNGs
└── *.ipynb                  # Legacy Jupyter notebooks (kept as research record)
```

## The four things this toolkit does

### 1. Parse raw Tobii TSV → canonical fixation rows

```python
from g2c.parsers import load

eye_events, samples = load("YMU_UM", sample_size=10)
eye_events.head()
# eye_tracker, experiment_id, participant_id, filename, trial_id,
# stimuli_module, stimuli_name, timestamp, duration, x0, y0, x1, y1,
# token, pupil_l, pupil_r, amplitude, peak_velocity, eye_event_type
```

A single unified Tobii parser handles both `UNL_UM` and `YMU_UM`. Adding
a third Tobii-export dataset is a configuration entry in
`g2c/parsers/datasets_config.py` — no new code required.

The legacy `UNL_UM()` / `YMU_UM()` functions still work for backward
compatibility:

```python
from g2c.parsers import UNL_UM, YMU_UM
ee, sm = YMU_UM(sample_size=10)
```

### 2. OCR the stimulus → token-level AOIs → hit-test fixations

```python
from g2c import aoi

aoi_df = aoi.aoi_detector(
    "datasets/YMU_UM/stimuli/Quiz - introduction-Q1 (localhost).png",
    scale_factor=2.0, min_confidence=60, psm="6", oem="3",
)
fixation_aoi = aoi.aoi_tokens_matcher(
    "output/ymu_um/aoi_tokens_structure/aoi_introduction-Q1_tokens_structure.csv",
    eye_events, trial_id="introduction-Q1", radius=25,
)
```

### 3. Visualize

```python
from g2c import visualization

visualization.draw_trial(eye_events, samples, draw_fixation=True, r3=3, r5=1)
visualization.draw_heatmap(eye_events, contours=False, sigma_value=17,
                           vmin=0, vmax=100)
```

### 4. Classify expertise (PCGC / hybrid)

The fixation × token table is then rolled up into 8 vocabulary-agnostic
**token features** per participant (Shannon entropy, Gini, top-3 %,
revisit rate …) and fed into a Mahalanobis **PCGC** prototype scorer.
A downstream classifier (`lr | linsvm | xgb`) plus participant-level
StratifiedKFold CV produces honest AUC / macro-F1 estimates:

```python
from g2c.classification import (
    extract_token_features_from_csv, load_participant_feature_csv,
    run_multi_seed_experiment, summarize_results,
)

# 1) Per-question token features  (output of cli.extract_aoi + expertise label)
feat_df = extract_token_features_from_csv("output/unl_um/aoi_labelled/Q1.csv")

# 2) Participant-level CV
df, feat_cols = load_participant_feature_csv("participant_features_Q1_token.csv")
results = run_multi_seed_experiment(
    df=df, feature_cols=feat_cols,
    representation="A", model_name="xgb", agg="mean",
    seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50], n_splits=5,
)
print(summarize_results(results))
```

Six hybrid representations (`A`, `A_v2`, `A_v3`, `B`, `C`, `D`) and two
multi-task aggregation modes (training-AUC weighted and OOF-AUC
weighted) are exposed via `cli/classify_expertise.py`. The locked
XGBoost baseline is the reference comparison point; do not change the
`StandardScaler.fit_transform` ordering inside
`g2c.classification.evaluation` without re-running the locked baseline,
or the no-leakage guarantee will silently break.

## Web UI (Streamlit)

A browser UI wraps the CLIs into four interactive tabs (Extract,
AOI Detection, Visualize, Evaluate OCR):

```bash
cd GAZE2CodeToolkit
streamlit run app.py
```

The app opens at <http://localhost:8501>. Both pre-configured datasets
and uploaded files are supported; OCR runs are blocking with a spinner;
generated CSVs are written to `output/` exactly like the CLIs, and also
downloadable from the UI.

Deploy to [Streamlit Cloud](https://streamlit.io/cloud) by pointing at
this repo and `GAZE2CodeToolkit/app.py` if you want a public link
(e.g. for a thesis Appendix A reference).

## Command-line interface

Five headless CLI scripts replace the legacy `.ipynb` notebooks. They are
the recommended entry points for new runs.

```bash
# From inside GAZE2CodeToolkit/

# 1) Parse a dataset and dump fixations as CSVs
python -m cli.extract_fixations --dataset UNL_UM --mode by-task \
    --out-dir output/unl_um/group/fixations

# 2) OCR a stimulus image and hit-test fixations against the AOIs
python -m cli.extract_aoi --dataset YMU_UM --by-task \
    --trial-id introduction-Q5 \
    --image-dir datasets/YMU_UM/stimuli \
    --image-prefix "Quiz - " --image-suffix " (localhost).png" \
    --out-dir output/ymu_um

# 3) Save visualizations as PNG
python -m cli.visualize --dataset YMU_UM \
    --trial-id introduction-Q5 --experiment-id Participant52 \
    --kinds trial heatmap \
    --out-dir output/ymu_um/viz

# 4) Score MCQ answers into per-participant expertise totals
python -m cli.score_expertise \
    --input data/ymu_um/python_mcq_answers.csv \
    --output data/ymu_um/python_mcq_scores.csv

# 5) Evaluate OCR output against a ground-truth CSV
python -m cli.evaluate_ocr \
    --ground-truth output/ocr_groundtruth/Q5_ground_truth04.csv \
    --detected output/orc_detection/Q5_detected_tokens.csv \
    --out-dir output/ocr_eval/Q5

# 6a) Build participant-level token features from labelled fixation×token CSVs
python -m cli.classify_expertise build-features \
    --raw-dir  output/unl_um/aoi/labelled \
    --out-dir  output/unl_um/classification

# 6b) Train + evaluate (multi-task, weighted by training-fold AUC)
python -m cli.classify_expertise train \
    --multitask-dir output/unl_um/classification \
    --multitask-suffix token \
    --feature-type repr_A --weighting w1 --model xgb \
    --output-dir output/unl_um/classification/results
```

See `cli/README.md` for the full list of CLI flags and additional
examples.

## Datasets

Two Tobii-based datasets are wired up out of the box:

| Dataset | Hardware | Default sample size | Configured in |
|---|---|---|---|
| `UNL_UM`  | Tobii Pro Nano (Tobii I-VT export) | 44 participants | `datasets_config.DATASETS["UNL_UM"]` |
| `YMU_UM`  | Tobii Pro Nano (Tobii I-VT export) | 84 participants | `datasets_config.DATASETS["YMU_UM"]` |

Raw TSVs are not shipped with the toolkit. Configure paths to your
local copy in `g2c/parsers/datasets_config.py`:

```python
"YMU_UM": {
    "raw_dir": "datasets/YMU_UM/rawdata",
    "stimuli_dir": "datasets/YMU_UM/stimuli",
    ...
}
```

### Adding a new Tobii dataset

For a new dataset that uses the same Tobii Pro Nano export format,
edit `g2c/parsers/datasets_config.py`:

```python
DATASETS["MY_NEW_DATASET"] = {
    "eye_tracker": "Tobii I-VT (Fixation)",
    "raw_dir": "datasets/MyDataset/rawdata",
    "stimuli_dir": "datasets/MyDataset/stimuli",
    "stimuli_names": ("Task1 (localhost)", "Task2 (localhost)", ...),
    "n_stimuli": 7,
    "columns": TOBII_PRO_COLUMNS,    # already defined at the top of the file
    "participant_col": "Participant name",
    "trial_split": {"strategy": "paired_markers", "per_trial": 3},
    "stimuli_name_template": "{event_value}.png",
    "trial_id_strategy": "first_word",
    "fixation_label": "Fixation",
    "default_sample_size": 30,
    ...
}
```

Then call `parsers.load("MY_NEW_DATASET")` — no new code.

## Notebooks (legacy)

The original five Jupyter notebooks at the repository root are kept as
research-history artefacts and still run unchanged:

| Notebook | CLI replacement |
|---|---|
| `g2c_fixation_extractor.ipynb` | `python -m cli.extract_fixations` |
| `g2c_aoi_extractor.ipynb`      | `python -m cli.extract_aoi`       |
| `g2c_visionizer.ipynb`         | `python -m cli.visualize`         |
| `g2c_expertise.ipynb`          | `python -m cli.score_expertise`   |
| `evaluate_ocr.ipynb`           | `python -m cli.evaluate_ocr`      |

To launch the notebooks, install jupyter into the same environment and
run `jupyter notebook`.

## Citation

If you use this toolkit in academic work, please cite the parent thesis
(or the associated GAZE2Code paper for the OCR-AOI methodology) — see
the top-level repository `README` for the full reference.

## License

See `COPYING.txt` and `LICENSE` at the repository root.

## 📬 Contact

For questions, collaboration, or reuse inquiries:

* **Wudao Yang**
  Universiti Malaya / Yunnan Minzu University
  📧 [s2137045@siswa.um.edu.my](mailto:s2137045@siswa.um.edu.my)
