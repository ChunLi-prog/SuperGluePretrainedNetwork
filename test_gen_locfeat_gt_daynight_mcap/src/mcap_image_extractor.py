"""Stage 1: Extract original-resolution images from MCAP files.

Decodes H.265 camera frames from MCAP ADAS files and saves them as JPEG
at original resolution. Also generates scannet_pairs.txt for Stage 2.

Production features:
  - Checkpoint/resume: skips pairs whose day+night images already exist
  - Per-pair error isolation: one decode failure does not crash the batch
  - scannet_pairs.txt incremental: preserves already-written lines
  - Error alerting: warns when failure ratio exceeds threshold

Usage:
    python mcap_image_extractor.py \\
        --config config/gen_locfeat.yaml \\
        --case-pairs /path/to/case_pairs.json \\
        --keyframe-pairs-dir /path/to/keyframe_pairs/ \\
        --workers 4
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Add SuperGlue repo root to path for mcap_utils
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from mcap_utils.h265_decoder import H265Decoder
from mcap_utils.mcap_image_reader import MCAPImageReader
from mcap_utils.mcap_gnss_reader import MCAPGnssReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("Stage1")


def _load_existing_pairs(pairs_txt_path: Path) -> set:
    """Load already-written pair lines from scannet_pairs.txt as a set."""
    existing = set()
    if pairs_txt_path.exists():
        with open(pairs_txt_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    existing.add(stripped)
    return existing


def extract_one_case(args_tuple):
    """Extract images for one (day_case, night_case) pair with checkpoint/resume.

    Returns dict with status, counts, elapsed time.
    """
    case_pair, kf_dir, out_path, config = args_tuple

    day_case = case_pair["day_case"]
    night_case = case_pair["night_case"]
    case_name = f"{day_case}-{night_case}"

    try:
        return _extract_one_case_inner(case_pair, kf_dir, out_path, config)
    except Exception:
        logger.error("[%s] FATAL:\n%s", case_name, traceback.format_exc())
        return {
            "case": case_name, "status": "fatal_error", "pairs": 0,
            "decode_fail": 0, "skipped": 0, "errors": 1,
            "attempted": 0, "elapsed_s": 0,
        }


def _extract_one_case_inner(case_pair, kf_dir, out_path, config):
    """Inner logic for extract_one_case (separated for clean try/except)."""
    day_case = case_pair["day_case"]
    night_case = case_pair["night_case"]
    case_name = f"{day_case}-{night_case}"

    # Find keyframe pairs JSON
    kf_json_path = os.path.join(kf_dir, f"{case_name}_pairs.json")
    if not os.path.exists(kf_json_path):
        candidates = list(Path(kf_dir).glob(f"*{day_case}*{night_case}*_pairs.json"))
        if not candidates:
            logger.warning("[%s] No keyframe pairs found", case_name)
            return {"case": case_name, "status": "no_kf_json", "pairs": 0,
                    "decode_fail": 0, "skipped": 0, "errors": 0, "attempted": 0, "elapsed_s": 0}
        kf_json_path = str(candidates[0])

    with open(kf_json_path, "r") as f:
        kf_data = json.load(f)

    ts_pairs = []
    for key, val in kf_data.items():
        if not key.startswith("pair_"):
            continue
        ts_pairs.append({
            "day_ts": val["day_timestamp"],
            "night_ts": val["night_timestamp"],
            "day_position": val.get("day_position", None),
        })

    if not ts_pairs:
        logger.warning("[%s] No pairs in %s", case_name, kf_json_path)
        return {"case": case_name, "status": "empty_pairs", "pairs": 0,
                "decode_fail": 0, "skipped": 0, "errors": 0, "attempted": 0, "elapsed_s": 0}

    ts_pairs.sort(key=lambda x: x["day_ts"])

    cameras = config.get("cameras", [
        {"channel": "7", "label": "front"},
        {"channel": "6", "label": "rear"},
    ])
    ts_tol = config.get("timestamp_tolerance_ms", 500)
    mcap_paths = case_pair.get("mcap_paths", {})
    alert_fail_ratio = config.get("alert_fail_ratio", 0.3)

    # GNSS readers
    day_gnss_mcaps = mcap_paths.get("day", {}).get("gnssfusion", [])
    night_gnss_mcaps = mcap_paths.get("night", {}).get("gnssfusion", [])
    day_gnss_local = [p for p in day_gnss_mcaps if os.path.exists(p)]
    night_gnss_local = [p for p in night_gnss_mcaps if os.path.exists(p)]
    day_gnss = MCAPGnssReader(day_gnss_local) if day_gnss_local else None
    night_gnss = MCAPGnssReader(night_gnss_local) if night_gnss_local else None

    if not day_gnss_local or not night_gnss_local:
        logger.warning("[%s] GNSS mcap missing (day=%d, night=%d)",
                       case_name, len(day_gnss_local), len(night_gnss_local))

    total_saved = 0
    total_skipped = 0
    total_decode_fail = 0
    total_errors = 0
    t_case_start = time.monotonic()

    for cam_cfg in cameras:
        channel = cam_cfg["channel"]
        cam_label = f"adas_{channel}"

        day_mcaps = mcap_paths.get("day", {}).get(cam_label, [])
        night_mcaps = mcap_paths.get("night", {}).get(cam_label, [])

        if not day_mcaps or not night_mcaps:
            logger.warning("[%s/%s] no MCAP paths", case_name, cam_label)
            continue

        day_mcaps_local = [p for p in day_mcaps if os.path.exists(p)]
        night_mcaps_local = [p for p in night_mcaps if os.path.exists(p)]

        if not day_mcaps_local or not night_mcaps_local:
            logger.warning("[%s/%s] MCAP not on disk (day=%d/%d, night=%d/%d)",
                           case_name, cam_label,
                           len(day_mcaps_local), len(day_mcaps),
                           len(night_mcaps_local), len(night_mcaps))
            continue

        cam_out = Path(out_path) / case_name / cam_label
        day_out = cam_out / "day"
        night_out = cam_out / "night"
        day_out.mkdir(parents=True, exist_ok=True)
        night_out.mkdir(parents=True, exist_ok=True)

        day_reader = MCAPImageReader(day_mcaps_local, channel, target_size=None)
        night_reader = MCAPImageReader(night_mcaps_local, channel, target_size=None)

        pairs_txt_path = cam_out / "scannet_pairs.txt"
        labels_path = cam_out / "pair_labels.txt"

        # Load already-written pairs for incremental append
        existing_pairs = _load_existing_pairs(pairs_txt_path)
        existing_labels = _load_existing_pairs(labels_path)

        new_pair_lines = []
        new_label_lines = []
        cam_saved = 0
        cam_skipped = 0
        cam_decode_fail = 0
        cam_errors = 0
        t_cam_start = time.monotonic()

        for pair in tqdm(ts_pairs, desc=f"{case_name}/{cam_label}",
                         unit="pair", disable=len(ts_pairs) < 50):
            try:
                day_ts_ms = int(pair["day_ts"] * 1000)
                night_ts_ms = int(pair["night_ts"] * 1000)

                # Predict filenames
                day_filename = f"{day_case}-{day_ts_ms}.jpg"
                night_filename = f"{night_case}-{night_ts_ms}.jpg"

                # --- Checkpoint: skip if both images already exist ---
                day_img_path = day_out / day_filename
                night_img_path = night_out / night_filename
                pair_line = f"{day_filename} {night_filename}"

                if day_img_path.exists() and night_img_path.exists():
                    cam_skipped += 1
                    # Ensure pair is in scannet_pairs.txt
                    if pair_line not in existing_pairs:
                        new_pair_lines.append(pair_line)
                        existing_pairs.add(pair_line)
                    continue

                # Decode
                day_result = day_reader.get_frame_by_ts(day_ts_ms, ts_tol)
                night_result = night_reader.get_frame_by_ts(night_ts_ms, ts_tol)

                if day_result is None or night_result is None:
                    cam_decode_fail += 1
                    if day_result is None:
                        logger.debug("[%s/%s] day decode fail ts=%d", case_name, cam_label, day_ts_ms)
                    if night_result is None:
                        logger.debug("[%s/%s] night decode fail ts=%d", case_name, cam_label, night_ts_ms)
                    continue

                day_img, actual_day_ts = day_result
                night_img, actual_night_ts = night_result

                # Actual filenames (use real decoded timestamp)
                day_filename = f"{day_case}-{actual_day_ts}.jpg"
                night_filename = f"{night_case}-{actual_night_ts}.jpg"
                pair_line = f"{day_filename} {night_filename}"

                # Indoor/outdoor classification
                day_label = day_gnss.get_label(actual_day_ts, ts_tol) if day_gnss else "unknown"
                night_label = night_gnss.get_label(actual_night_ts, ts_tol) if night_gnss else "unknown"

                # Re-check with actual timestamp filename
                day_img_path = day_out / day_filename
                night_img_path = night_out / night_filename
                if day_img_path.exists() and night_img_path.exists():
                    cam_skipped += 1
                    if pair_line not in existing_pairs:
                        new_pair_lines.append(pair_line)
                        existing_pairs.add(pair_line)
                    continue

                # Save original-resolution images
                cv2.imwrite(str(day_img_path), day_img)
                cv2.imwrite(str(night_img_path), night_img)

                if pair_line not in existing_pairs:
                    new_pair_lines.append(pair_line)
                    existing_pairs.add(pair_line)

                label_line = f"{day_filename} {night_filename} {day_label} {night_label}"
                if label_line not in existing_labels:
                    new_label_lines.append(label_line)
                    existing_labels.add(label_line)

                cam_saved += 1

            except Exception:
                cam_errors += 1
                logger.error("[%s/%s] pair error:\n%s",
                             case_name, cam_label, traceback.format_exc())
                continue

        # Append new lines to files (incremental, not overwrite)
        if new_pair_lines:
            with open(pairs_txt_path, "a") as f:
                f.write("\n".join(new_pair_lines) + "\n")
        if new_label_lines:
            with open(labels_path, "a") as f:
                f.write("\n".join(new_label_lines) + "\n")

        day_reader.close()
        night_reader.close()

        total_saved += cam_saved
        total_skipped += cam_skipped
        total_decode_fail += cam_decode_fail
        total_errors += cam_errors
        cam_elapsed = time.monotonic() - t_cam_start

        logger.info(
            "[%s/%s] saved=%d, skipped=%d, decode_fail=%d, errors=%d, %.1fs",
            case_name, cam_label, cam_saved, cam_skipped, cam_decode_fail, cam_errors, cam_elapsed,
        )

        # Error alert
        total_attempted = cam_saved + cam_decode_fail + cam_errors
        if total_attempted > 0 and (cam_decode_fail + cam_errors) / total_attempted > alert_fail_ratio:
            logger.warning(
                "⚠️ ALERT [%s/%s]: fail_ratio=%.0f%% exceeds threshold",
                case_name, cam_label,
                100.0 * (cam_decode_fail + cam_errors) / total_attempted,
            )

    if day_gnss:
        day_gnss.close()
    if night_gnss:
        night_gnss.close()

    case_elapsed = time.monotonic() - t_case_start
    logger.info(
        "[%s] total saved=%d, skipped=%d, decode_fail=%d, errors=%d, %.1fs",
        case_name, total_saved, total_skipped, total_decode_fail, total_errors, case_elapsed,
    )
    return {
        "case": case_name, "status": "ok", "pairs": total_saved,
        "decode_fail": total_decode_fail, "skipped": total_skipped,
        "errors": total_errors, "attempted": len(ts_pairs),
        "elapsed_s": round(case_elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Extract original-resolution images from MCAP (production)"
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--case-pairs", required=True, help="case_pairs.json from Phase 0")
    parser.add_argument("--keyframe-pairs-dir", required=True, help="Dir with keyframe pair JSONs")
    parser.add_argument("--output", default=None, help="Override out_path from config")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--parking-ids", default=None,
                        help="Comma-separated parking IDs to process (default=all)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_path = args.output or config.get("out_path", "output")
    os.makedirs(out_path, exist_ok=True)

    # Persistent file logging
    log_dir = os.path.join(out_path, "log")
    os.makedirs(log_dir, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"stage1_{ts_str}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    logger.info("Log file: %s", log_path)

    with open(args.case_pairs, "r") as f:
        case_pairs = json.load(f)

    if args.parking_ids:
        ids = set(args.parking_ids.split(","))
        case_pairs = [c for c in case_pairs if c["parking_id"] in ids]

    logger.info("=== Stage 1: Image Extraction (Production) ===")
    logger.info("Processing %d case pairs with %d workers", len(case_pairs), args.workers)

    work_args = [
        (cp, args.keyframe_pairs_dir, out_path, config) for cp in case_pairs
    ]

    results = []
    if args.workers <= 1:
        for wa in work_args:
            results.append(extract_one_case(wa))
    else:
        with Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(extract_one_case, work_args):
                results.append(r)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    total_pairs = sum(r["pairs"] for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)
    total_fails = sum(r.get("decode_fail", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    fatal = [r for r in results if r["status"] == "fatal_error"]

    logger.info(
        "Stage 1 complete: %d/%d cases OK, %d fatal, "
        "saved=%d, skipped=%d, decode_fail=%d, errors=%d",
        len(ok), len(results), len(fatal),
        total_pairs, total_skipped, total_fails, total_errors,
    )

    if fatal:
        logger.warning("⚠️ %d cases had fatal errors: %s",
                       len(fatal), [r["case"] for r in fatal])

    summary_path = os.path.join(out_path, "stage1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
