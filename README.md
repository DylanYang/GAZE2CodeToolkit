# GAZE2CodeToolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-005F73.svg)](https://github.com/tesseract-ocr/tesseract)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](https://peps.python.org/pep-0008/)
[![Last commit](https://img.shields.io/github/last-commit/DylanYang/GAZE2CodeToolkit)](https://github.com/DylanYang/GAZE2CodeToolkit/commits/main)
[![Repo stars](https://img.shields.io/github/stars/DylanYang/GAZE2CodeToolkit?style=social)](https://github.com/DylanYang/GAZE2CodeToolkit/stargazers)

End-to-end pipeline for turning raw eye-tracking recordings of programmers
reading source code into **fixation × code-token AOI** tables. Outputs feed
the ECPG (Stage II) and EC-GazeFormer (Stage III) modeling lines of the
parent PhD thesis.

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
                                     ▼
                             ECPG  /  EC-GazeFormer
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
backs `pytesseract`. The `environment.yml` installs it via conda-forge.
If you use pip/venv, install it through your OS package manager:

| OS | Command |
|---|---|
| macOS (Homebrew) | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt-get install tesseract-ocr` |
| Windows | <https://github.com/UB-Mannheim/tesseract/wiki> |

Confirm with `tesseract --version`.

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
│   ├── util/                      # Helpers (lines, stimuli, exports)
│   └── visualization/             # Trial overlay, heatmap, timeline
├── cli/                     # Command-line entry points (see cli/README.md)
│   ├── extract_fixations.py
│   ├── extract_aoi.py
│   ├── visualize.py
│   ├── score_expertise.py
│   └── evaluate_ocr.py
├── app.py                   # Streamlit web UI entry point
├── webapp/                  # Streamlit tab modules and shared helpers
│   ├── components.py
│   ├── state.py
│   └── tabs/{extract,aoi,visualize,evaluate}.py
├── datasets/                # Dataset directories (see config; not in git)
├── output/                  # Default location for generated CSVs / PNGs
└── *.ipynb                  # Legacy Jupyter notebooks (kept as research record)
```

## The three things this toolkit does

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
