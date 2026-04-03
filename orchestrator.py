#!/usr/bin/env python3
"""Pipeline orchestrator: Phase 0 → Phase 1 → Stage 1 → Stage 2.

Runs the full MCAP day/night GT generation pipeline end-to-end with:
  - Checkpoint/resume via checkpoint.json
  - Per-stage error isolation
  - Configurable stage selection (--stages)
  - Parking lot filtering (--parking-ids)
  - Aggregated summary + error alerting

Usage:
    # Full pipeline
    python orchestrator.py --config config/mcap_pipeline.yaml

    # Resume from where it left off
    python orchestrator.py --config config/mcap_pipeline.yaml

    # Only Stage 2
    python orchestrator.py --config config/mcap_pipeline.yaml --stages stage2

    # Specific parking lots
    python orchestrator.py --config config/mcap_pipeline.yaml --parking-ids P1 P2
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("Orchestrator")

STAGE_ORDER = ["phase0", "phase1", "stage1", "stage2"]


def setup_file_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"orchestrator_{ts}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    return log_path


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(out_path: str) -> dict:
    cp_path = os.path.join(out_path, "checkpoint.json")
    if os.path.exists(cp_path):
        with open(cp_path, "r") as f:
            return json.load(f)
    return {"completed_stages": [], "stage_results": {}}


def save_checkpoint(out_path: str, checkpoint: dict):
    cp_path = os.path.join(out_path, "checkpoint.json")
    checkpoint["last_updated"] = datetime.datetime.now().isoformat()
    with open(cp_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info("Checkpoint saved: %s", cp_path)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_phase0(config: dict, args) -> dict:
    """Phase 0: Generate case_pairs.json from CSV + JSON index."""
    script = os.path.join(
        config["repo_root"],
        "test_gen_locfeat_gt_daynight_mcap/src/parse_parking_csv.py",
    )
    case_pairs_path = os.path.join(config["out_path"], "case_pairs.json")

    if os.path.exists(case_pairs_path):
        logger.info("[Phase0] case_pairs.json already exists, skipping")
        return {"status": "skipped", "output": case_pairs_path}

    cmd = [
        sys.executable, script,
        "--csv", config["csv_file"],
        "--json", config["raw_mcap_dataset"],
        "--output", case_pairs_path,
    ]
    logger.info("[Phase0] Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("[Phase0] FAILED:\n%s\n%s", result.stdout, result.stderr)
        return {"status": "error", "returncode": result.returncode, "stderr": result.stderr[:2000]}

    logger.info("[Phase0] Done: %s", case_pairs_path)
    return {"status": "ok", "output": case_pairs_path}


def run_phase1(config: dict, args) -> dict:
    """Phase 1: Trajectory matching → keyframe_pairs/*.json."""
    script = os.path.join(
        config["repo_root"],
        "test_gen_locfeat_gt_daynight_mcap/src/gen_locfeat_gt_daynight.py",
    )
    case_pairs_path = os.path.join(config["out_path"], "case_pairs.json")
    kf_dir = os.path.join(config["out_path"], "keyframe_pairs")

    # Check if already has keyframe pairs
    if os.path.isdir(kf_dir) and len(list(Path(kf_dir).glob("*_pairs.json"))) > 0:
        existing = len(list(Path(kf_dir).glob("*_pairs.json")))
        logger.info("[Phase1] %d keyframe pair files exist, skipping", existing)
        return {"status": "skipped", "existing_files": existing}

    cmd = [
        sys.executable, script,
        "--config", config.get("stage1_config", ""),
        "--case-pairs", case_pairs_path,
    ]
    if args.parking_ids:
        cmd += ["--parking-ids", ",".join(args.parking_ids)]

    logger.info("[Phase1] Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("[Phase1] FAILED:\n%s\n%s", result.stdout, result.stderr)
        return {"status": "error", "returncode": result.returncode, "stderr": result.stderr[:2000]}

    n_files = len(list(Path(kf_dir).glob("*_pairs.json"))) if os.path.isdir(kf_dir) else 0
    logger.info("[Phase1] Done: %d keyframe pair files", n_files)
    return {"status": "ok", "keyframe_files": n_files}


def run_stage1(config: dict, args) -> dict:
    """Stage 1: MCAP → original-resolution images + scannet_pairs.txt."""
    script = os.path.join(
        config["repo_root"],
        "test_gen_locfeat_gt_daynight_mcap/src/mcap_image_extractor.py",
    )
    case_pairs_path = os.path.join(config["out_path"], "case_pairs.json")
    kf_dir = os.path.join(config["out_path"], "keyframe_pairs")

    cmd = [
        sys.executable, script,
        "--config", config.get("stage1_config", ""),
        "--case-pairs", case_pairs_path,
        "--keyframe-pairs-dir", kf_dir,
        "--output", config["out_path"],
        "--workers", str(config.get("stage1_workers", 4)),
    ]
    if args.parking_ids:
        cmd += ["--parking-ids", ",".join(args.parking_ids)]

    logger.info("[Stage1] Running: %s", " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        logger.error("[Stage1] FAILED (%.1fs):\n%s\n%s", elapsed, result.stdout[-2000:], result.stderr[-2000:])
        return {"status": "error", "returncode": result.returncode, "elapsed_s": round(elapsed, 1)}

    # Read summary
    summary_path = os.path.join(config["out_path"], "stage1_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    logger.info("[Stage1] Done in %.1fs", elapsed)
    return {"status": "ok", "elapsed_s": round(elapsed, 1), "summary": summary}


def run_stage2(config: dict, args) -> dict:
    """Stage 2: SuperPoint + SuperGlue matching → .npz."""
    script = os.path.join(config["repo_root"], "mcap_stage2_matching.py")

    cmd = [
        sys.executable, script,
        "--config", config.get("stage2_config", "config/mcap_pipeline.yaml"),
        "--input-dir", config["out_path"],
        "--workers", str(config.get("stage2_workers", 1)),
    ]
    if args.parking_ids:
        cmd += ["--parking-ids"] + args.parking_ids

    logger.info("[Stage2] Running: %s", " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        logger.error("[Stage2] FAILED (%.1fs):\n%s\n%s", elapsed, result.stdout[-2000:], result.stderr[-2000:])
        return {"status": "error", "returncode": result.returncode, "elapsed_s": round(elapsed, 1)}

    # Read summary
    summary_path = os.path.join(config["out_path"], "stage2_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    logger.info("[Stage2] Done in %.1fs", elapsed)
    return {"status": "ok", "elapsed_s": round(elapsed, 1), "summary": summary}


STAGE_RUNNERS = {
    "phase0": run_phase0,
    "phase1": run_phase1,
    "stage1": run_stage1,
    "stage2": run_stage2,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCAP GT Pipeline Orchestrator")
    parser.add_argument("--config", required=True, help="Orchestrator YAML config")
    parser.add_argument("--stages", nargs="+", default=None,
                        help="Stages to run (default=all). Choices: phase0 phase1 stage1 stage2")
    parser.add_argument("--parking-ids", nargs="+", default=None,
                        help="Only process these parking lot IDs")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoint, re-run all requested stages")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve repo root (directory containing this script)
    config.setdefault("repo_root", str(Path(__file__).resolve().parent))

    out_path = config["out_path"]
    os.makedirs(out_path, exist_ok=True)

    log_dir = config.get("log_dir", os.path.join(out_path, "log"))
    log_path = setup_file_logging(log_dir)

    logger.info("=" * 60)
    logger.info("MCAP GT Pipeline Orchestrator")
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", out_path)
    logger.info("Log: %s", log_path)
    if args.parking_ids:
        logger.info("Parking filter: %s", args.parking_ids)
    logger.info("=" * 60)

    # Determine which stages to run
    stages_to_run = args.stages if args.stages else STAGE_ORDER

    # Load checkpoint
    checkpoint = load_checkpoint(out_path) if not args.force else {"completed_stages": [], "stage_results": {}}

    pipeline_ok = True
    pipeline_start = time.monotonic()

    for stage in stages_to_run:
        if stage not in STAGE_RUNNERS:
            logger.error("Unknown stage: %s (valid: %s)", stage, STAGE_ORDER)
            continue

        # Skip if already completed (checkpoint)
        if stage in checkpoint["completed_stages"] and not args.force:
            logger.info("[%s] Already completed (checkpoint), skipping", stage)
            continue

        logger.info("--- Starting %s ---", stage)
        t0 = time.monotonic()

        try:
            result = STAGE_RUNNERS[stage](config, args)
        except Exception as e:
            result = {"status": "exception", "error": str(e)}
            logger.error("[%s] Exception: %s", stage, e, exc_info=True)

        elapsed = time.monotonic() - t0
        result["wall_time_s"] = round(elapsed, 1)
        checkpoint["stage_results"][stage] = result

        if result.get("status") in ("ok", "skipped"):
            checkpoint["completed_stages"].append(stage)
            logger.info("[%s] ✓ completed in %.1fs", stage, elapsed)
        else:
            logger.error("[%s] ✗ FAILED (%.1fs): %s", stage, elapsed, result.get("status"))
            pipeline_ok = False
            save_checkpoint(out_path, checkpoint)
            # Stop pipeline on stage failure
            logger.error("Pipeline stopped at %s. Fix and re-run to resume.", stage)
            break

        save_checkpoint(out_path, checkpoint)

    # Final summary
    total_elapsed = time.monotonic() - pipeline_start
    logger.info("=" * 60)
    if pipeline_ok:
        logger.info("Pipeline COMPLETED in %.1fs", total_elapsed)
    else:
        logger.warning("Pipeline INCOMPLETE (%.1fs) — check errors above", total_elapsed)

    for stage in STAGE_ORDER:
        sr = checkpoint["stage_results"].get(stage, {})
        status = sr.get("status", "not_run")
        wt = sr.get("wall_time_s", 0)
        logger.info("  %-10s: %-10s (%.1fs)", stage, status, wt)

    logger.info("=" * 60)

    # Alert on any errors
    failed_stages = [s for s in STAGE_ORDER
                     if checkpoint["stage_results"].get(s, {}).get("status")
                     not in ("ok", "skipped", None)]
    if failed_stages:
        logger.warning("⚠️ ALERT: Failed stages: %s", failed_stages)
        sys.exit(1)


if __name__ == "__main__":
    main()
