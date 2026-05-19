"""Token-level gaze feature extraction (per participant, per question).

For each question and participant, extract 8 vocabulary-agnostic
statistics that describe **HOW** attention is distributed across code
tokens. The features transfer across datasets regardless of programming
language (Python YMU vs Java UNL).

Features per participant per question
--------------------------------------
=====================  ====================================================
``tok_entropy``        Shannon entropy of fixation-count distribution
                       over tokens.
``tok_n_unique_ratio`` ``unique tokens fixated / total fixations`` —
                       breadth measure.
``tok_max_pct``        Fraction of total fixations on the single
                       most-viewed token.
``tok_top3_pct``       Fraction of total fixations on the top-3
                       most-viewed tokens.
``tok_gini``           Gini coefficient of fixation counts —
                       concentration measure.
``tok_revisit_rate``   Mean fixations per unique token — re-reading
                       tendency.
``tok_dur_entropy``    Shannon entropy of total-duration distribution
                       over tokens.
``tok_mean_token_dur`` Mean total fixation duration (ms) per unique
                       token.
=====================  ====================================================

Ported verbatim from ``ECPG/extract_token_features.py``. The CLI entry
point now lives in :mod:`cli.classify_expertise` (subcommand
``build-features``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


QUESTIONS = ["Q1", "Q2A", "Q2B", "Q3", "Q4A", "Q4B", "Q5"]
LONG_FIX_THRESHOLD_MS = 200.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (range [0, 1])."""
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    x = np.sort(counts.astype(float))
    n = len(x)
    total = x.sum()
    index = np.arange(1, n + 1, dtype=float)   # 1-indexed ranks
    g = (2.0 * (index * x).sum() / (n * total)) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_token_features_from_csv(raw_csv: str) -> pd.DataFrame:
    """
    Extract per-participant token-level features from one question's raw CSV.

    The input CSV is expected to contain at least the columns
    ``p_id``, ``expertise``, ``duration`` and ``aoi_token`` — i.e. the
    fixation × token hit-test table produced by :mod:`cli.extract_aoi`
    after expertise labels have been joined onto it.

    Returns a DataFrame with columns:
        p_id, expertise,
        tok_entropy, tok_n_unique_ratio, tok_max_pct, tok_top3_pct,
        tok_gini, tok_revisit_rate, tok_dur_entropy, tok_mean_token_dur
    """
    df = pd.read_csv(raw_csv)

    required = ["p_id", "expertise", "duration", "aoi_token"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {raw_csv}: {missing}")

    df = df[required].dropna(subset=required).copy()
    df["p_id"] = df["p_id"].astype(str).str.lower()
    df["expertise"] = df["expertise"].astype(int)
    df["duration"] = df["duration"].astype(float)
    df["aoi_token"] = df["aoi_token"].astype(str).str.strip()

    rows = []
    for pid, g in df.groupby("p_id"):
        expertise = int(g["expertise"].iloc[0])

        # Per-token stats
        tok_counts = g["aoi_token"].value_counts()           # Series: token -> count
        tok_dur = g.groupby("aoi_token")["duration"].sum()   # token -> total dur

        counts = tok_counts.values.astype(float)
        durs   = tok_dur.reindex(tok_counts.index).fillna(0).values.astype(float)

        n_fix    = float(counts.sum())
        n_unique = float(len(counts))

        # 1. tok_entropy
        tok_ent = _entropy(counts)

        # 2. tok_n_unique_ratio — breadth relative to total fixations
        tok_n_uniq_ratio = n_unique / max(n_fix, 1.0)

        # 3. tok_max_pct
        tok_max_pct = float(counts.max() / max(n_fix, 1.0))

        # 4. tok_top3_pct
        top3 = float(np.sort(counts)[::-1][:3].sum() / max(n_fix, 1.0))

        # 5. tok_gini
        tok_gin = _gini(counts)

        # 6. tok_revisit_rate
        tok_revisit = float(n_fix / max(n_unique, 1.0))

        # 7. tok_dur_entropy
        tok_dur_ent = _entropy(durs)

        # 8. tok_mean_token_dur (ms)
        tok_mean_dur = float(durs.mean()) if len(durs) > 0 else 0.0

        rows.append({
            "p_id": pid,
            "expertise": expertise,
            "tok_entropy": tok_ent,
            "tok_n_unique_ratio": tok_n_uniq_ratio,
            "tok_max_pct": tok_max_pct,
            "tok_top3_pct": top3,
            "tok_gini": tok_gin,
            "tok_revisit_rate": tok_revisit,
            "tok_dur_entropy": tok_dur_ent,
            "tok_mean_token_dur": tok_mean_dur,
        })

    return pd.DataFrame(rows)


TOKEN_FEAT_COLS = [
    "tok_entropy",
    "tok_n_unique_ratio",
    "tok_max_pct",
    "tok_top3_pct",
    "tok_gini",
    "tok_revisit_rate",
    "tok_dur_entropy",
    "tok_mean_token_dur",
]
