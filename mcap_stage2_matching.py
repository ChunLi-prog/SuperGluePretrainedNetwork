"""Stage 2: SuperPoint + SuperGlue matching on pre-extracted images.

Reads original-resolution images from Stage 1 output, feeds to
SuperPoint+SuperGlue, and produces .npz match results.

Production features:
  - Checkpoint/resume: skips pairs whose .npz already exists
  - Per-pair error isolation: one pair failure does not crash the batch
  - --parking-ids filter: process only selected parking lots
  - Rich .npz metadata: timestamps, filenames, parking_id, case_name

Usage:
    python mcap_stage2_matching.py --config config/mcap_pipeline.yaml
    python mcap_stage2_matching.py --config config/mcap_pipeline.yaml \\
        --input-dir /path --workers 2 --parking-ids P1 P2
"""

import argparse
import csv
import datetime
import json
import logging
import os
import re
import time
import traceback
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.matching import Matching
from models.utils import frame2tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("Stage2")

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_file_logging(log_dir: str) -> str:
    """Add file handler to root logger for persistent log storage."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"stage2_{ts}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    logger.info("Log file: %s", log_path)
    return log_path


def load_scannet_pairs(pairs_txt: str) -> List[Tuple[str, str]]:
    """Load image pairs from scannet_pairs.txt."""
    pairs = []
    with open(pairs_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def extract_timestamp_from_filename(filename: str) -> float:
    """Extract millisecond timestamp from filenames like 'case-1716197226500.jpg'.

    Returns 0.0 on failure (non-fatal).
    """
    m = re.search(r'-(\d{13})', filename)
    if m:
        return float(m.group(1))
    m = re.search(r'-(\d{10,})', filename)
    return float(m.group(1)) if m else 0.0


def ransac_filter_matches(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    mconf: np.ndarray,
    threshold: float = 3.0,
    method: str = "fundamental",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """RANSAC geometric verification as optional post-filter."""
    if len(mkpts0) < 8:
        return mkpts0, mkpts1, mconf, None
    if method == "homography":
        _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, threshold)
    else:
        _, mask = cv2.findFundamentalMat(
            mkpts0, mkpts1, cv2.FM_RANSAC, threshold, 0.999
        )
    if mask is None:
        return mkpts0, mkpts1, mconf, None
    mask = mask.ravel().astype(bool)
    return mkpts0[mask], mkpts1[mask], mconf[mask], mask


# ---------------------------------------------------------------------------
# Core worker
# ---------------------------------------------------------------------------

def process_one_case_camera(args_tuple):
    """Process all pairs for one (case_dir, camera) with checkpoint/resume.

    Returns list[dict] of per-pair metrics.
    """
    case_camera_dir_str, config = args_tuple
    case_camera_dir = Path(case_camera_dir_str)
    case_name = case_camera_dir.parent.name
    camera_label = case_camera_dir.name

    pairs_txt = case_camera_dir / "scannet_pairs.txt"
    if not pairs_txt.exists():
        logger.warning("No scannet_pairs.txt in %s", case_camera_dir)
        return []

    pairs = load_scannet_pairs(str(pairs_txt))
    if not pairs:
        logger.warning("Empty pairs in %s", pairs_txt)
        return []

    day_dir = case_camera_dir / "day"
    night_dir = case_camera_dir / "night"

    # Device
    device = "cuda" if torch.cuda.is_available() and not config.get("force_cpu", False) else "cpu"
    logger.info("[%s/%s] device=%s, pairs=%d", case_name, camera_label, device, len(pairs))

    sp_resolution = tuple(config.get("sp_input_resolution", [1920, 1536]))

    # Build Matching model
    matching_config = {
        "superpoint": config.get("superpoint", {
            "nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 2048,
        }),
        "superglue": config.get("superglue", {
            "weights": "outdoor", "sinkhorn_iterations": 20, "match_threshold": 0.2,
        }),
    }
    matching = Matching(matching_config).eval().to(device)
    logger.info(
        "[%s/%s] Loaded SP+SG (weights=%s)",
        case_name, camera_label,
        matching_config["superglue"].get("weights", "outdoor"),
    )

    do_ransac = config.get("ransac", True)
    ransac_thresh = config.get("ransac_threshold", 3.0)
    ransac_method = config.get("ransac_method", "fundamental")

    match_dir = case_camera_dir / "matches"
    match_dir.mkdir(exist_ok=True)

    save_vis = config.get("save_visualization", False)
    vis_dir = case_camera_dir / "vis"
    if save_vis:
        vis_dir.mkdir(exist_ok=True)
    vis_interval = max(1, config.get("vis_sample_interval", 10))

    # Derive parking_id from case_name (e.g. "P1_ent1_route1_case1-P1_ent1_route1_case3")
    parking_id = ""
    pm = re.match(r'(P\d+)', case_name)
    if pm:
        parking_id = pm.group(1)

    # Error/alert thresholds
    alert_fail_ratio = config.get("alert_fail_ratio", 0.3)

    metrics = []
    skipped = 0
    errors = 0
    t_start = time.monotonic()

    for pair_idx, (day_name, night_name) in enumerate(
        tqdm(pairs, desc=f"{case_name}/{camera_label}", unit="pair")
    ):
        stem0, stem1 = Path(day_name).stem, Path(night_name).stem
        npz_path = match_dir / f"{stem0}_{stem1}.npz"

        # --- Checkpoint: skip if already processed ---
        if npz_path.exists():
            skipped += 1
            continue

        try:
            day_path = str(day_dir / day_name)
            night_path = str(night_dir / night_name)

            img0_gray = cv2.imread(day_path, cv2.IMREAD_GRAYSCALE)
            img1_gray = cv2.imread(night_path, cv2.IMREAD_GRAYSCALE)

            if img0_gray is None or img1_gray is None:
                logger.warning("[%s/%s] pair %d: read failed (day=%s, night=%s)",
                               case_name, camera_label, pair_idx,
                               img0_gray is not None, img1_gray is not None)
                errors += 1
                continue

            original_shape0 = img0_gray.shape
            img0_resized = cv2.resize(img0_gray, sp_resolution)
            img1_resized = cv2.resize(img1_gray, sp_resolution)

            inp0 = frame2tensor(img0_resized, device)
            inp1 = frame2tensor(img1_resized, device)

            pred = matching({"image0": inp0, "image1": inp1})

            kpts0 = pred["keypoints0"][0].cpu().numpy()
            kpts1 = pred["keypoints1"][0].cpu().numpy()
            scores0 = pred["scores0"][0].cpu().numpy()
            scores1 = pred["scores1"][0].cpu().numpy()
            desc0 = pred["descriptors0"][0].cpu().numpy()
            desc1 = pred["descriptors1"][0].cpu().numpy()
            matches0 = pred["matches0"][0].cpu().numpy()
            match_conf = pred["matching_scores0"][0].cpu().numpy()

            valid = matches0 > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches0[valid]]
            mconf = match_conf[valid]

            if do_ransac and len(mkpts0) >= 8:
                mkpts0_r, mkpts1_r, mconf_r, _ = ransac_filter_matches(
                    mkpts0, mkpts1, mconf, ransac_thresh, ransac_method
                )
            else:
                mkpts0_r, mkpts1_r, mconf_r = mkpts0, mkpts1, mconf

            # Timestamps from filenames
            day_ts = extract_timestamp_from_filename(day_name)
            night_ts = extract_timestamp_from_filename(night_name)

            result = {
                "keypoints0": kpts0,
                "keypoints1": kpts1,
                "scores0": scores0,
                "scores1": scores1,
                "descriptors0": desc0.T,
                "descriptors1": desc1.T,
                "matches": matches0,
                "match_confidence": match_conf,
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
                "mconf": mconf,
                "mkpts0_ransac": mkpts0_r,
                "mkpts1_ransac": mkpts1_r,
                "mconf_ransac": mconf_r,
                # Metadata
                "camera": camera_label,
                "image_shape": np.array(img0_resized.shape),
                "original_shape": np.array(original_shape0),
                "matcher": "superglue",
                "day_timestamp": np.float64(day_ts),
                "night_timestamp": np.float64(night_ts),
                "day_image": day_name,
                "night_image": night_name,
                "parking_id": parking_id,
                "case_name": case_name,
            }
            np.savez(str(npz_path), **result)

            n_matches, n_inliers = len(mkpts0), len(mkpts0_r)
            metrics.append({
                "camera": camera_label,
                "pair_idx": pair_idx,
                "day_img": day_name,
                "night_img": night_name,
                "n_kpts_day": len(kpts0),
                "n_kpts_night": len(kpts1),
                "n_matches": n_matches,
                "n_inliers": n_inliers,
                "inlier_ratio": n_inliers / max(n_matches, 1),
            })

            # Visualization
            if save_vis and pair_idx % vis_interval == 0:
                _save_vis(day_path, night_path, sp_resolution,
                          mkpts0, mkpts1, mkpts0_r, mkpts1_r,
                          n_matches, n_inliers, camera_label,
                          vis_dir, stem0, stem1)

        except Exception:
            errors += 1
            logger.error("[%s/%s] pair %d FAILED:\n%s",
                         case_name, camera_label, pair_idx, traceback.format_exc())
            continue

    # Summary
    total_processed = len(metrics)
    fail_pairs = sum(1 for m in metrics if m["n_inliers"] == 0)
    avg_r = float(np.mean([m["inlier_ratio"] for m in metrics])) if metrics else 0
    avg_inliers = float(np.mean([m["n_inliers"] for m in metrics])) if metrics else 0
    elapsed = time.monotonic() - t_start

    logger.info(
        "[%s/%s] done: processed=%d, skipped=%d, errors=%d, "
        "fail=%d, avg_inliers=%.1f, avg_ratio=%.3f, %.1fs",
        case_name, camera_label,
        total_processed, skipped, errors,
        fail_pairs, avg_inliers, avg_r, elapsed,
    )

    # Error alert
    if total_processed > 0:
        error_ratio = errors / (total_processed + errors)
        if error_ratio > alert_fail_ratio:
            logger.warning(
                "⚠️ ALERT [%s/%s]: error_ratio=%.1f%% exceeds threshold %.0f%%",
                case_name, camera_label, error_ratio * 100, alert_fail_ratio * 100,
            )

    # Save metrics CSV (append-safe: write new run, don't merge with old)
    if metrics:
        csv_path = case_camera_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    return metrics


def _save_vis(day_path, night_path, sp_resolution,
              mkpts0, mkpts1, mkpts0_r, mkpts1_r,
              n_matches, n_inliers, camera_label,
              vis_dir, stem0, stem1):
    """Render and save match visualization image."""
    try:
        day_img = cv2.imread(day_path)
        night_img = cv2.imread(night_path)
        if day_img is None or night_img is None:
            return
        dv = cv2.resize(day_img, sp_resolution)
        nv = cv2.resize(night_img, sp_resolution)
        h, w = dv.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w], vis[:, w:] = dv, nv

        inlier_set = set()
        for i in range(len(mkpts0_r)):
            inlier_set.add((float(mkpts0_r[i, 0]), float(mkpts0_r[i, 1])))

        for i in range(min(len(mkpts0), 300)):
            p0 = (int(mkpts0[i, 0]), int(mkpts0[i, 1]))
            p1 = (int(mkpts1[i, 0]) + w, int(mkpts1[i, 1]))
            if (float(mkpts0[i, 0]), float(mkpts0[i, 1])) not in inlier_set:
                cv2.line(vis, p0, p1, (0, 0, 200), 1, cv2.LINE_AA)

        for i in range(min(len(mkpts0_r), 200)):
            p0 = (int(mkpts0_r[i, 0]), int(mkpts0_r[i, 1]))
            p1 = (int(mkpts1_r[i, 0]) + w, int(mkpts1_r[i, 1]))
            cv2.line(vis, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)

        ratio_pct = 100.0 * n_inliers / max(n_matches, 1)
        label = f"SuperGlue {camera_label} | inliers:{n_inliers}/{n_matches} ({ratio_pct:.0f}%)"
        cv2.putText(vis, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(vis_dir / f"{stem0}_{stem1}.jpg"), vis)
    except Exception:
        logger.debug("Vis save failed for %s_%s", stem0, stem1)


# ---------------------------------------------------------------------------
# Discovery & filtering
# ---------------------------------------------------------------------------

def discover_work_units(input_dir: str, parking_ids: Optional[List[str]] = None) -> List[str]:
    """Find all case/camera dirs with scannet_pairs.txt.

    Args:
        input_dir: Root output directory from Stage 1.
        parking_ids: If given, only include cases whose dir name starts with one of these.
    """
    units = []
    for case_dir in sorted(Path(input_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        # Filter by parking ID
        if parking_ids:
            matched = any(case_dir.name.startswith(pid) for pid in parking_ids)
            if not matched:
                continue
        for cam_dir in sorted(case_dir.iterdir()):
            if cam_dir.is_dir() and (cam_dir / "scannet_pairs.txt").exists():
                units.append(str(cam_dir))
    return units


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: SuperPoint + SuperGlue matching (production)"
    )
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--input-dir", default=None, help="Override input_dir")
    parser.add_argument("--workers", type=int, default=None, help="Override num_gpu_workers")
    parser.add_argument("--parking-ids", nargs="+", default=None,
                        help="Only process these parking lot IDs (e.g. P1 P2)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_dir = args.input_dir or config["input_dir"]
    num_workers = args.workers or config.get("num_gpu_workers", 1)

    log_dir = config.get("log_dir", os.path.join(input_dir, "log"))
    setup_file_logging(log_dir)

    logger.info("=== Stage 2: SuperPoint + SuperGlue Matching (Production) ===")
    logger.info("Config: %s", json.dumps(config, indent=2, default=str))
    if args.parking_ids:
        logger.info("Parking ID filter: %s", args.parking_ids)

    work_units = discover_work_units(input_dir, args.parking_ids)
    logger.info("Found %d work units in %s", len(work_units), input_dir)

    if not work_units:
        logger.error("No work units found")
        return

    work_args = [(u, config) for u in work_units]

    all_metrics = []
    failed_units = []

    if num_workers <= 1:
        for wa in work_args:
            try:
                all_metrics.extend(process_one_case_camera(wa))
            except Exception:
                failed_units.append(wa[0])
                logger.error("Work unit FAILED: %s\n%s", wa[0], traceback.format_exc())
    else:
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass
        with Pool(processes=num_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(process_one_case_camera, work_args)
            ):
                try:
                    all_metrics.extend(result)
                except Exception:
                    failed_units.append(work_args[i][0])
                    logger.error("Collecting result failed for unit %d", i)

    # Global summary
    logger.info("Stage 2 complete: %d total pairs, %d failed units",
                len(all_metrics), len(failed_units))

    summary = {
        "matcher": "superglue",
        "total_pairs": len(all_metrics),
        "avg_inlier_ratio": float(np.mean([m["inlier_ratio"] for m in all_metrics])) if all_metrics else 0,
        "avg_inliers": float(np.mean([m["n_inliers"] for m in all_metrics])) if all_metrics else 0,
        "fail_pairs": sum(1 for m in all_metrics if m["n_inliers"] == 0),
        "failed_work_units": failed_units,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    summary_path = os.path.join(input_dir, "stage2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary))

    if failed_units:
        logger.warning("⚠️ %d work units had fatal errors: %s", len(failed_units), failed_units)


if __name__ == "__main__":
    main()
