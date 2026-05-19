"""Extract token-level AOIs from a stimulus image and match fixations to them.

Replaces g2c_aoi_extractor.ipynb. Pipeline:

    parsers.load(dataset)
      → aoi.aoi_detector(image)                # Tesseract OCR token boxes
      → aoi.aoi_save_tokens_structure(...)     # serialize per-trial AOI CSV
      → aoi.aoi_tokens_matcher(...)            # hit-test fixations to AOIs
      → write aoi_fixations CSV(s)

Examples
--------
    # Single trial, single participant
    python -m cli.extract_aoi --dataset YMU_UM \\
        --image-dir datasets/YMU_UM/stimuli \\
        --image-suffix " (localhost).png" \\
        --trial-id introduction-Q1 \\
        --experiment-id Participant2 \\
        --out-dir output/ymu_um

    # All participants on one trial (task-level aggregation)
    python -m cli.extract_aoi --dataset YMU_UM \\
        --trial-id introduction-Q5 \\
        --image-dir datasets/YMU_UM/stimuli \\
        --image-suffix " (localhost).png" \\
        --out-dir output/ymu_um --by-task
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from g2c import aoi, parsers


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   choices=parsers.available_datasets())
    p.add_argument("--trial-id", required=True,
                   help="Trial identifier to process (e.g. 'Q1', 'introduction-Q5').")
    p.add_argument("--image-dir", required=True,
                   help="Directory containing stimulus images.")
    p.add_argument("--image-suffix", default=".png",
                   help="Filename suffix appended to trial-id, e.g. ' (localhost).png'.")
    p.add_argument("--image-prefix", default="",
                   help="Optional filename prefix, e.g. 'Quiz - '.")
    p.add_argument("--out-dir", required=True,
                   help="Root output directory. Will create "
                        "{out-dir}/aoi_tokens_structure and {out-dir}/{individual,group}/aoi.")
    p.add_argument("--experiment-id", default=None,
                   help="If set, hit-test only this participant. Otherwise all.")
    p.add_argument("--by-task", action="store_true",
                   help="Task-level (all participants) aggregation. Overrides --experiment-id.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Cap on participant TSV files to parse.")

    # OCR knobs
    p.add_argument("--ocr-scale", type=float, default=2.0)
    p.add_argument("--ocr-min-confidence", type=int, default=60)
    p.add_argument("--ocr-psm", default="6")
    p.add_argument("--ocr-oem", default="3")
    p.add_argument("--ocr-debug", action="store_true")
    p.add_argument("--ocr-preprocess", action="store_true",
                   help="Apply CLAHE + adaptive threshold preprocessing before OCR.")

    # Hit-test
    p.add_argument("--hit-radius", type=int, default=35,
                   help="Pixel radius for fixation→AOI matching. "
                        "Default 35 matches the effective historical "
                        "behaviour used to generate the locked ECPG tables.")
    return p


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = build_arg_parser().parse_args(argv)

    out_root = Path(args.out_dir)
    aoi_struct_dir = out_root / "aoi_tokens_structure"
    aoi_struct_dir.mkdir(parents=True, exist_ok=True)

    image_filename = f"{args.image_prefix}{args.trial_id}{args.image_suffix}"
    image_path = str(Path(args.image_dir) / image_filename)

    print(f"[extract_aoi] OCR on {image_path}", file=sys.stderr)
    aoi_df = aoi.aoi_detector(
        image_path,
        scale_factor=args.ocr_scale,
        debug=args.ocr_debug,
        use_preprocessing=args.ocr_preprocess,
        min_confidence=args.ocr_min_confidence,
        psm=args.ocr_psm,
        oem=args.ocr_oem,
    )
    print(f"[extract_aoi] detected {len(aoi_df)} AOI tokens", file=sys.stderr)

    # Persist the AOI token structure so it can be reused later.
    aoi.aoi_save_tokens_structure(aoi_df, str(aoi_struct_dir) + "/")
    aoi_struct_csv = aoi_struct_dir / f"aoi_{args.trial_id}_tokens_structure.csv"
    print(f"[extract_aoi] AOI tokens → {aoi_struct_csv}", file=sys.stderr)

    print(f"[extract_aoi] loading {args.dataset} for fixation hit-test ...",
          file=sys.stderr)
    eye_events, _ = parsers.load(args.dataset, sample_size=args.sample_size)

    if args.by_task:
        trial_data = eye_events[eye_events["trial_id"] == args.trial_id]
        out_dir = out_root / "group" / "aoi"
        out_csv = out_dir / f"aoi_fixations_{args.trial_id}.csv"
    else:
        if args.experiment_id is None:
            print("[extract_aoi] --experiment-id required unless --by-task is set.",
                  file=sys.stderr)
            return 2
        trial_data = eye_events[
            (eye_events["experiment_id"] == args.experiment_id)
            & (eye_events["trial_id"] == args.trial_id)
        ]
        out_dir = out_root / "individual" / "aoi"
        out_csv = (out_dir
                   / f"aoi_fixations_{args.trial_id}_{args.experiment_id}.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    aoi_fixations = aoi.aoi_tokens_matcher(
        str(aoi_struct_csv), trial_data, args.trial_id, radius=args.hit_radius,
    )
    aoi_fixations.to_csv(out_csv, index=False)
    print(f"[extract_aoi] wrote {len(aoi_fixations)} fixation-AOI rows → {out_csv}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
