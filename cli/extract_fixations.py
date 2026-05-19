"""Extract fixations from a Tobii eye-tracking dataset and save as CSVs.

Replaces g2c_fixation_extractor.ipynb. Loads a dataset via the unified
`parsers.load()`, then exports fixations either as one combined CSV, one
CSV per task, or one CSV per (trial, participant) pair.

Examples
--------
    # All-in-one CSV
    python -m cli.extract_fixations --dataset UNL_UM --mode all \\
        --out-dir output/unl_um/all/fixations

    # Per-trial CSVs (group by task)
    python -m cli.extract_fixations --dataset YMU_UM --mode by-task \\
        --out-dir output/ymu_um/group/fixations

    # One CSV per participant x trial
    python -m cli.extract_fixations --dataset UNL_UM --mode per-participant \\
        --out-dir output/unl_um/individual/fixations --sample-size 50
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from g2c import parsers, util


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   choices=parsers.available_datasets(),
                   help="Dataset name from the unified loader registry.")
    p.add_argument("--mode", required=True,
                   choices=["all", "by-task", "per-participant"],
                   help="Export grouping.")
    p.add_argument("--out-dir", required=True,
                   help="Directory under which fixation CSVs are written.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Cap on participant TSV files to parse.")
    p.add_argument("--trial-ids", nargs="*", default=None,
                   help="Optional subset of trial IDs to export.")
    p.add_argument("--experiment-ids", nargs="*", default=None,
                   help="Optional subset of experiment IDs to export.")
    return p


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = build_arg_parser().parse_args(argv)

    print(f"[extract_fixations] loading {args.dataset} "
          f"(sample_size={args.sample_size}) ...", file=sys.stderr)
    eye_events, samples = parsers.load(args.dataset, sample_size=args.sample_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_range = pd.Series(
        args.trial_ids if args.trial_ids is not None
        else samples["trial_id"].unique()
    )
    experiment_range = pd.Series(
        args.experiment_ids if args.experiment_ids is not None
        else eye_events["experiment_id"].unique()
    )

    print(f"[extract_fixations] {len(experiment_range)} participants × "
          f"{len(trial_range)} trials → {out_dir} (mode={args.mode})",
          file=sys.stderr)

    if args.mode == "all":
        util.export_fixations(eye_events, samples, experiment_range,
                              trial_range, str(out_dir), byall=True)
    elif args.mode == "by-task":
        util.export_fixations(eye_events, samples, experiment_range,
                              trial_range, str(out_dir), bytask=True)
    else:  # per-participant
        util.export_fixations(eye_events, samples, experiment_range,
                              trial_range, str(out_dir))

    n_files = sum(1 for _ in out_dir.rglob("*.csv"))
    print(f"[extract_fixations] wrote {n_files} CSV file(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
