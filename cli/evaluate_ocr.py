"""Evaluate OCR token-detection quality against a ground-truth CSV.

Replaces evaluate_ocr.ipynb. Computes:
- token-level precision / recall / F1 with IoU bounding-box matching
- character-level accuracy and CER (Levenshtein-based)
- word-level accuracy and WER
- average IoU and average detection confidence
- full-text CER / WER over concatenated tokens
- optional ROC AUC for matched-vs-unmatched detections

Examples
--------
    python -m cli.evaluate_ocr \\
        --ground-truth output/ocr_groundtruth/Q5_ground_truth04.csv \\
        --detected output/orc_detection/Q5_detected_tokens.csv \\
        --out-dir output/ocr_eval/Q5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["line_num", "x", "y", "width", "height", "text"]


# ---------------------------------------------------------------------------
# Bounding-box + text metrics
# ---------------------------------------------------------------------------

def compute_iou(box1: dict, box2: dict) -> float:
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1["width"] * box1["height"] + box2["width"] * box2["height"] - inter
    return inter / union if union > 0 else 0.0


def _levenshtein_dist(a, b):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)


def calculate_cer(gt: str, ocr: str) -> float:
    if not gt:
        return 1.0 if ocr else 0.0
    if not ocr:
        return 1.0
    dist = _levenshtein_dist(gt, ocr)
    max_len = max(len(gt), len(ocr))
    return dist / max_len if max_len > 0 else 0.0


def calculate_wer(gt: str, ocr: str) -> float:
    gt_w, ocr_w = gt.split(), ocr.split()
    if not gt_w:
        return 1.0 if ocr_w else 0.0
    if not ocr_w:
        return 1.0
    dist = _levenshtein_dist(gt_w, ocr_w)
    max_n = max(len(gt_w), len(ocr_w))
    return dist / max_n if max_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class OCREvaluator:
    def __init__(self, iou_threshold: float = 0.5,
                 text_sim_threshold: float = 0.8):
        self.iou_threshold = iou_threshold
        self.text_sim_threshold = text_sim_threshold

    def evaluate(self, gt_df: pd.DataFrame,
                 detected_df: pd.DataFrame,
                 progress_callback=None) -> Tuple[Dict, List[Dict]]:
        """Pair GT and detected tokens, return aggregate metrics.

        Uses Hungarian (`linear_sum_assignment`) for **globally
        optimal 1-to-1 matching** instead of the older two-pass
        greedy approach. Greedy struggles whenever the GT or
        detected sides contain repeated tokens (e.g. `int`/`for`
        in code, or "world" appearing in every multiple-choice
        answer in Q1) — it locks early pairings to the first
        candidate that meets the threshold and leaves later
        duplicates unmatched, even when a different assignment
        would pair every duplicate cleanly. Hungarian fixes this
        by optimising total similarity across the whole matrix.

        After assignment, each pair is classified:
          - strict TP: IoU ≥ iou_threshold AND text_sim ≥ text_sim_threshold
          - fuzzy  TP: not strict, but IoU > 0 AND text_sim ≥ 0.5×text_sim_threshold
          - else: discarded (becomes FP on the det side, FN on the GT side)
        """
        from rapidfuzz import fuzz
        from scipy.optimize import linear_sum_assignment
        self._validate(gt_df, detected_df)

        def _report(frac, msg):
            if progress_callback is not None:
                try:
                    progress_callback(frac, msg)
                except Exception:
                    pass

        metrics = {
            "character_level": {"cer": [], "correct_chars": 0, "total_chars": 0},
            "word_level": {"wer": [], "correct_words": 0, "total_words": 0},
            "detection": {"tp": 0, "iou": []},
            "confidence": [] if "confidence" in detected_df.columns else None,
        }
        matched_pairs: list[dict] = []
        gt_matched: set[int] = set()
        det_matched: set[int] = set()

        n_gt, n_det = len(gt_df), len(detected_df)
        if n_gt == 0 or n_det == 0:
            return self._finalize(metrics, n_gt, n_det), matched_pairs

        # Build the score and admissibility matrices.
        #
        # `sim[i, j]` is the goodness of pairing GT_i with DET_j
        # (weighted 50/50 text similarity + IoU).
        #
        # `iou_mat[i, j]` is the raw IoU, needed for threshold checks
        # after assignment.
        #
        # `admissible[i, j]` marks pairs that COULD pass the strict OR
        # fuzzy threshold. Non-admissible pairs get a huge cost so
        # Hungarian will not pick them unless forced (which only
        # happens for "filler" assignments on rectangular matrices,
        # and those get filtered out post-hoc).
        _report(0.00, f"Building similarity matrix {n_gt:,}×{n_det:,} …")
        sim = np.zeros((n_gt, n_det), dtype=np.float64)
        iou_mat = np.zeros((n_gt, n_det), dtype=np.float64)
        admissible = np.zeros((n_gt, n_det), dtype=bool)
        strict_thr_text = self.text_sim_threshold
        fuzzy_thr_text = self.text_sim_threshold * 0.5

        gt_rows = list(gt_df.iterrows())
        det_rows = list(detected_df.iterrows())
        step = max(1, n_gt // 100)
        for i, (gt_idx, gt_row) in enumerate(gt_rows):
            if i % step == 0 or i == n_gt - 1:
                _report(0.45 * (i + 1) / n_gt,
                        f"Similarity matrix: {i + 1:,}/{n_gt:,}")
            gt_text = str(gt_row["text"])
            for j, (det_idx, det_row) in enumerate(det_rows):
                ts = fuzz.ratio(gt_text, str(det_row["text"])) / 100.0
                io = compute_iou(gt_row, det_row)
                sim[i, j] = 0.5 * ts + 0.5 * io
                iou_mat[i, j] = io
                is_strict = (io >= self.iou_threshold and ts >= strict_thr_text)
                is_fuzzy = (not is_strict and io > 0 and ts >= fuzzy_thr_text)
                admissible[i, j] = is_strict or is_fuzzy

        # Hungarian wants a COST matrix; negate similarity, and add a
        # large penalty to inadmissible cells so the optimiser only
        # picks them when no real candidate exists for that row/col.
        BIG_COST = 10.0  # similarity is in [0, 1] so 10 is "infinity"
        cost = np.where(admissible, -sim, BIG_COST)

        _report(0.50, "Running Hungarian assignment …")
        row_ind, col_ind = linear_sum_assignment(cost)

        # Classify each assignment. Inadmissible cells (cost == BIG_COST)
        # are filler — skip them.
        _report(0.85, "Classifying matches …")
        for i, j in zip(row_ind, col_ind):
            if not admissible[i, j]:
                continue
            gt_idx = gt_rows[i][0]
            det_idx = det_rows[j][0]
            gt_row = gt_df.loc[gt_idx]
            det_row = detected_df.loc[det_idx]
            ts = fuzz.ratio(str(gt_row["text"]), str(det_row["text"])) / 100.0
            io = iou_mat[i, j]
            is_strict = (io >= self.iou_threshold and ts >= strict_thr_text)
            self._record(gt_idx, det_idx, gt_row, det_row,
                         metrics, matched_pairs, gt_matched, det_matched,
                         is_fuzzy=not is_strict, fuzz=fuzz)

        _report(0.97, "Computing final metrics …")
        result = self._finalize(metrics, n_gt, n_det), matched_pairs
        _report(1.0, f"Done — {len(matched_pairs)} matched pair(s).")
        return result

    # internals
    def _validate(self, gt_df, det_df):
        for df, name in [(gt_df, "ground truth"), (det_df, "detected")]:
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {name}: {missing}")

    def _best_match(self, gt_row, det_df, det_matched, *, strict, fuzz):
        """Find the best candidate detection for a GT row.

        Strict pass: require IoU >= iou_threshold AND text_sim >= text
        threshold. The previous implementation also required
        `gt_row["line_num"] == det_row["line_num"]`, but `line_num` is
        a free-form label (different casing / numbering schemes used
        by GT builders vs OCR pipelines) — making it a strict equality
        gate disqualifies every candidate when the two sources use
        different conventions, collapsing matching to the relaxed
        text-only pass and tanking IoU-based metrics like AUC.

        Relaxed pass: text similarity only (used as fallback).
        """
        best, best_iou, best_sim = None, 0.0, 0.0
        threshold = self.text_sim_threshold if strict else self.text_sim_threshold * 0.5
        for det_idx, det_row in det_df.iterrows():
            if det_idx in det_matched:
                continue
            text_sim = fuzz.ratio(str(gt_row["text"]), str(det_row["text"])) / 100.0
            iou = compute_iou(gt_row, det_row)
            if not strict:
                # Relaxed pass requires at least *some* positional
                # overlap (IoU > 0). Without this, repeating tokens
                # in code (e.g. `int`, `for`, `System.out.print`)
                # get paired arbitrarily across the page, dropping
                # mean IoU to ~0 and tanking ROC/AUC.
                if (text_sim >= threshold and iou > 0
                        and (text_sim > best_sim
                             or (text_sim == best_sim and iou > best_iou))):
                    best, best_iou, best_sim = det_idx, iou, text_sim
                continue
            if (iou >= self.iou_threshold and text_sim >= threshold
                    and (iou > best_iou or text_sim > best_sim)):
                best, best_iou, best_sim = det_idx, iou, text_sim
        return best

    def _record(self, gt_idx, det_idx, gt_row, det_row, metrics, pairs,
                gt_matched, det_matched, *, is_fuzzy, fuzz):
        cer = calculate_cer(str(gt_row["text"]), str(det_row["text"]))
        wer = calculate_wer(str(gt_row["text"]), str(det_row["text"]))
        iou = compute_iou(gt_row, det_row)
        text_sim = fuzz.ratio(str(gt_row["text"]), str(det_row["text"])) / 100.0

        gt_len = len(str(gt_row["text"]))
        det_len = len(str(det_row["text"]))
        metrics["character_level"]["cer"].append(cer)
        metrics["character_level"]["correct_chars"] += (
            gt_len - int(cer * max(gt_len, det_len))
        )
        metrics["character_level"]["total_chars"] += gt_len

        gt_words = str(gt_row["text"]).split()
        det_words = str(det_row["text"]).split()
        metrics["word_level"]["wer"].append(wer)
        metrics["word_level"]["correct_words"] += (
            len(gt_words) - int(wer * max(len(gt_words), len(det_words)))
        )
        metrics["word_level"]["total_words"] += len(gt_words)

        metrics["detection"]["tp"] += 1
        metrics["detection"]["iou"].append(iou)
        if metrics["confidence"] is not None:
            metrics["confidence"].append(det_row.get("confidence", 0))

        pairs.append({
            "gt_index": gt_idx, "det_index": det_idx,
            "gt_text": gt_row["text"], "det_text": det_row["text"],
            "cer": cer, "wer": wer, "iou": iou,
            "text_similarity": text_sim, "is_fuzzy": is_fuzzy,
        })
        gt_matched.add(gt_idx)
        det_matched.add(det_idx)

    def _finalize(self, metrics, total_gt, total_det) -> Dict:
        tp = metrics["detection"]["tp"]
        fp = total_det - tp
        fn = total_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        char_acc = (metrics["character_level"]["correct_chars"]
                    / metrics["character_level"]["total_chars"]
                    if metrics["character_level"]["total_chars"] > 0 else 0.0)
        word_acc = (metrics["word_level"]["correct_words"]
                    / metrics["word_level"]["total_words"]
                    if metrics["word_level"]["total_words"] > 0 else 0.0)
        avg_cer = float(np.mean(metrics["character_level"]["cer"])
                        if metrics["character_level"]["cer"] else 1.0)
        avg_wer = float(np.mean(metrics["word_level"]["wer"])
                        if metrics["word_level"]["wer"] else 1.0)
        avg_iou = float(np.mean(metrics["detection"]["iou"])
                        if metrics["detection"]["iou"] else 0.0)
        avg_conf = (float(np.mean(metrics["confidence"]))
                    if metrics["confidence"] else None)

        return {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1_score": f1,
            "character_error_rate": avg_cer, "character_accuracy": char_acc,
            "word_error_rate": avg_wer, "word_accuracy": word_acc,
            "average_iou": avg_iou, "average_confidence": avg_conf,
            "match_rate": tp / total_gt if total_gt > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Auxiliary outputs
# ---------------------------------------------------------------------------

def full_text_cer_wer(gt_df: pd.DataFrame, det_df: pd.DataFrame) -> dict:
    import Levenshtein  # type: ignore
    gt_full = " ".join(gt_df["text"].astype(str).tolist()).strip()
    det_full = " ".join(det_df["text"].astype(str).tolist()).strip()
    ops = Levenshtein.editops(gt_full, det_full)
    s = sum(1 for o in ops if o[0] == "replace")
    i = sum(1 for o in ops if o[0] == "insert")
    d = sum(1 for o in ops if o[0] == "delete")
    n = len(gt_full)
    gt_words = gt_full.split()
    det_words = det_full.split()
    wer_dist = Levenshtein.distance(" ".join(gt_words), " ".join(det_words))
    return {
        "ground_truth_length": n,
        "cer": (s + i + d) / n if n > 0 else 0.0,
        "wer": wer_dist / max(len(gt_words), 1),
        "substitutions": s, "insertions": i, "deletions": d,
    }


def plot_roc(matched_pairs, det_df, gt_df, out_path: Path) -> Optional[float]:
    """Plot ROC and return AUC for the detection task.

    Score semantics:
        (a) OCR's per-token `confidence` (when the detected CSV has
            that column) — the natural "how sure am I this is text?"
            score that ROC is designed for.
        (b) Otherwise, "max IoU against any GT box" — independent of
            whether this detection was actually matched. Detections
            that spatially overlap *some* GT (correct text or not)
            score high; detections floating in empty space score 0.
            This is a genuine signal even without matching, unlike
            the older "matched IoU else 0" approach which made AUC
            degenerate to 1.0 by construction once the matcher
            enforced IoU > 0.
    """
    from sklearn.metrics import roc_curve, auc
    matched_det_ids = {p["det_index"] for p in matched_pairs}

    use_confidence = "confidence" in det_df.columns
    y_true: list[int] = []
    y_scores: list[float] = []
    for i, det_row in det_df.iterrows():
        y_true.append(1 if i in matched_det_ids else 0)
        if use_confidence:
            try:
                y_scores.append(float(det_row["confidence"]))
            except (TypeError, ValueError):
                y_scores.append(0.0)
        else:
            # Best IoU vs ANY GT — gives unmatched-yet-overlapping
            # detections a non-trivial score and avoids the trivial
            # AUC=1.0 collapse that "matched IoU else 0" produces.
            best_iou = 0.0
            for _, gt_row in gt_df.iterrows():
                iou_val = compute_iou(gt_row, det_row)
                if iou_val > best_iou:
                    best_iou = iou_val
            y_scores.append(best_iou)
    if not any(y_true) or all(y_true):
        return None
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for AOI Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return roc_auc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ground-truth", required=True, help="Ground-truth CSV.")
    p.add_argument("--detected", required=True, help="OCR detected-tokens CSV.")
    p.add_argument("--out-dir", required=True,
                   help="Directory for metrics CSV, matches CSV, and ROC PNG.")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--text-sim-threshold", type=float, default=0.8)
    p.add_argument("--skip-roc", action="store_true",
                   help="Skip ROC curve generation (avoids sklearn dependency).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.ground_truth):
        print(f"Ground truth not found: {args.ground_truth}", file=sys.stderr)
        return 2
    if not os.path.exists(args.detected):
        print(f"Detection file not found: {args.detected}", file=sys.stderr)
        return 2

    gt_df = pd.read_csv(args.ground_truth)
    det_df = pd.read_csv(args.detected)

    for df, name in [(gt_df, "ground_truth"), (det_df, "detected")]:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"Missing columns in {name}: {missing}", file=sys.stderr)
            return 2

    print(f"[evaluate_ocr] GT rows={len(gt_df)}  detected rows={len(det_df)}",
          file=sys.stderr)

    evaluator = OCREvaluator(iou_threshold=args.iou_threshold,
                             text_sim_threshold=args.text_sim_threshold)
    metrics, matches = evaluator.evaluate(gt_df, det_df)
    metrics["full_text"] = full_text_cer_wer(gt_df, det_df)

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(matches).to_csv(out_dir / "matches.csv", index=False)

    if not args.skip_roc:
        roc_auc = plot_roc(matches, det_df, gt_df, out_dir / "roc.png")
        if roc_auc is not None:
            metrics["roc_auc"] = roc_auc

    print("\nOCR Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}  {v:>12.4f}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                kvv = f"{vv:.4f}" if isinstance(vv, float) else vv
                print(f"  {k}.{kk:<22}  {kvv:>12}")
        else:
            print(f"  {k:<28}  {v}")
    print(f"\nMetrics → {out_dir / 'metrics.csv'}")
    print(f"Matches → {out_dir / 'matches.csv'}")
    if not args.skip_roc:
        print(f"ROC     → {out_dir / 'roc.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
