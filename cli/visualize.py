"""Render gaze visualizations as PNG files (no interactive plt.show).

Replaces g2c_visionizer.ipynb. Saves any combination of: trial overlay,
heatmap, line-level fixation duration, and fixation timeline.

Examples
--------
    # Heatmap + trial overlay for one participant on one trial
    python -m cli.visualize --dataset YMU_UM \\
        --trial-id introduction-Q5 \\
        --experiment-id Participant52 \\
        --out-dir output/ymu_um/viz \\
        --kinds trial heatmap

    # Task-level heatmap from a previously hit-tested AOI fixation CSV
    python -m cli.visualize --aoi-csv output/ymu_um/group/aoi/aoi_fixations_introduction-Q5.csv \\
        --out-dir output/ymu_um/viz --kinds heatmap --sigma 35 --vmax 1200
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend — required for CLI rendering.
import matplotlib.pyplot as plt
import pandas as pd

from g2c import parsers, visualization

KINDS = ("trial", "heatmap", "duration", "timeline")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Data source — either parser+trial OR pre-built AOI CSV.
    src = p.add_argument_group("data source (mutually exclusive groups)")
    src.add_argument("--dataset", choices=parsers.available_datasets(),
                     help="Dataset name; pairs with --trial-id, optional --experiment-id.")
    src.add_argument("--aoi-csv", help="Path to a pre-built fixation-AOI CSV.")
    src.add_argument("--trial-id", help="Required with --dataset.")
    src.add_argument("--experiment-id", default=None,
                     help="With --dataset, restrict to one participant (else all).")
    src.add_argument("--sample-size", type=int, default=None)

    p.add_argument("--out-dir", required=True,
                   help="Directory under which PNGs are saved.")
    p.add_argument("--kinds", nargs="+", default=["trial", "heatmap"],
                   choices=KINDS,
                   help="Which visualizations to render.")

    # trial overlay
    p.add_argument("--r3", type=float, default=3.0,
                   help="Trial overlay: base fixation-circle radius.")
    p.add_argument("--r5", type=float, default=0.8,
                   help="Trial overlay: per-100ms radius growth.")
    p.add_argument("--draw-saccade", action="store_true")
    p.add_argument("--draw-aoi", action="store_true")
    p.add_argument("--draw-raw-data", action="store_true")
    p.add_argument("--sample-x-col", default="Gaze point X [DACS px]")
    p.add_argument("--sample-y-col", default="Gaze point Y [DACS px]")

    # heatmap
    p.add_argument("--sigma", type=float, default=17.0,
                   help="Heatmap Gaussian sigma.")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=100.0)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--contours", action="store_true")
    p.add_argument("--figsize", type=float, nargs=2, default=(18.0, 10.0))
    return p


def _load_trial_data(args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (trial_data, samples_data); samples_data is None when loading from AOI CSV."""
    if args.aoi_csv:
        return pd.read_csv(args.aoi_csv), None
    if not (args.dataset and args.trial_id):
        raise SystemExit("Provide either --aoi-csv OR (--dataset AND --trial-id).")
    eye_events, samples = parsers.load(args.dataset, sample_size=args.sample_size)
    if args.experiment_id:
        mask = ((eye_events["experiment_id"] == args.experiment_id)
                & (eye_events["trial_id"] == args.trial_id))
        smask = ((samples["experiment_id"] == args.experiment_id)
                 & (samples["trial_id"] == args.trial_id))
    else:
        mask = eye_events["trial_id"] == args.trial_id
        smask = samples["trial_id"] == args.trial_id
    return eye_events.loc[mask].copy(), samples.loc[smask].copy()


def _save_current_fig(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = build_arg_parser().parse_args(argv)

    trial_data, samples_data = _load_trial_data(args)
    out_dir = Path(args.out_dir)
    tag = (args.experiment_id or "all") + "_" + (args.trial_id or Path(args.aoi_csv).stem)

    if "trial" in args.kinds:
        print("[visualize] trial overlay ...", file=sys.stderr)
        img = visualization.draw_trial(
            trial_data,
            samples_data if samples_data is not None else trial_data,
            draw_raw_data=args.draw_raw_data,
            draw_fixation=True,
            draw_saccade=args.draw_saccade,
            draw_aoi=args.draw_aoi,
            sample_x_col=args.sample_x_col,
            sample_y_col=args.sample_y_col,
            r3=args.r3,
            r5=args.r5,
        )
        if img is not None and hasattr(img, "save"):
            out = out_dir / f"trial_{tag}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            print(f"  wrote {out}", file=sys.stderr)
        else:
            _save_current_fig(out_dir / f"trial_{tag}.png")

    if "heatmap" in args.kinds:
        print("[visualize] heatmap ...", file=sys.stderr)
        visualization.draw_heatmap(
            trial_data,
            contours=args.contours,
            figsize=tuple(args.figsize),
            alpha=args.alpha,
            sigma_value=args.sigma,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        _save_current_fig(out_dir / f"heatmap_{tag}.png")

    if "duration" in args.kinds:
        print("[visualize] fixation duration ...", file=sys.stderr)
        img = visualization.fixation_duration(trial_data)
        if img is not None and hasattr(img, "save"):
            out = out_dir / f"duration_{tag}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            print(f"  wrote {out}", file=sys.stderr)
        else:
            _save_current_fig(out_dir / f"duration_{tag}.png")

    if "timeline" in args.kinds:
        print("[visualize] fixation timeline ...", file=sys.stderr)
        visualization.fixation_timeline(trial_data, figsize=tuple(args.figsize))
        _save_current_fig(out_dir / f"timeline_{tag}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
