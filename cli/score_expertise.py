"""Score MCQ answers and emit per-participant expertise totals.

Replaces g2c_expertise.ipynb. Inputs a CSV of participant MCQ answers
and writes a CSV of (experiment_id, Score) ready for ECPG expertise
labelling.

Hard-coded answer key matches the YMU-UM Python MCQ task; override via
--answer-key-json if a different question set is used.

Example
-------
    python -m cli.score_expertise \\
        --input data/ymu_um/python_mcq_answers.csv \\
        --output data/ymu_um/python_mcq_scores.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ANSWER_KEY = {
    "Q1": "B",
    "Q2A": "A",
    "Q2B": "B",
    "Q3": "B",
    "Q4A": {"B", "C"},   # multi-select
    "Q4B": "Step1: 2, Step2: 1, Step3: 3",
    "Q5": "D",
}

DEFAULT_POINTS = {
    "Q1": 10, "Q2A": 10, "Q2B": 10, "Q3": 10,
    "Q4A": 20, "Q4B": 20, "Q5": 20,
}


def calculate_score(row, key: dict, points: dict) -> int:
    score = 0
    for q, correct in key.items():
        if q not in row or pd.isnull(row[q]):
            continue
        answer = str(row[q]).strip()
        if isinstance(correct, (set, frozenset)):
            selected = set(map(str.strip, answer.split(",")))
            if selected == set(correct):
                score += points[q]
        else:
            if answer == correct:
                score += points[q]
    return score


def _load_overrides(path: Path | None, default: dict) -> dict:
    if path is None:
        return default
    data = json.loads(path.read_text())
    # JSON-encoded sets become lists; promote them back.
    fixed = {}
    for k, v in data.items():
        fixed[k] = set(v) if isinstance(v, list) else v
    return fixed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="Path to MCQ answer CSV.")
    p.add_argument("--output", required=True, help="Path to write the scores CSV.")
    p.add_argument("--id-column", default="experiment_id",
                   help="Participant identifier column in the answer CSV.")
    p.add_argument("--sep", default=",", help="CSV separator.")
    p.add_argument("--answer-key-json", type=Path, default=None,
                   help="Optional JSON file overriding the default answer key.")
    p.add_argument("--points-json", type=Path, default=None,
                   help="Optional JSON file overriding the default question point values.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    df = pd.read_csv(args.input, sep=args.sep)
    df.columns = df.columns.str.strip()
    if args.id_column not in df.columns:
        print(f"[score_expertise] id column {args.id_column!r} not in CSV. "
              f"Available: {df.columns.tolist()}", file=sys.stderr)
        return 2

    key = _load_overrides(args.answer_key_json, DEFAULT_ANSWER_KEY)
    points = _load_overrides(args.points_json, DEFAULT_POINTS)

    df["Score"] = df.apply(lambda r: calculate_score(r, key, points), axis=1)
    out_df = df[[args.id_column, "Score"]].rename(
        columns={args.id_column: "experiment_id"}
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[score_expertise] {len(out_df)} participants → {args.output}",
          file=sys.stderr)
    print(out_df.describe().T[["mean", "min", "max"]].to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
