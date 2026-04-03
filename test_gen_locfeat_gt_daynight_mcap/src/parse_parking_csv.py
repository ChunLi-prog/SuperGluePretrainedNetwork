"""Phase 0: Parse parking lot CSV and generate day/night case combos.

Reads the complex parking lot CSV, groups routes by parking lot,
separates day/night, and produces standardized case pairs.

Usage:
    python parse_parking_csv.py --csv path/to/csv --json path/to/odo_dump.json \
        --gt-path path/to/gt_samples --output case_pairs.json
"""

import argparse
import json
import logging
import os
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def parse_parking_csv(csv_path: str) -> Dict[str, Dict[str, List]]:
    """Parse parking lot CSV into structured day/night route groups.

    Returns:
        {parking_num: {"day": [route_nums], "night": [route_nums]}}
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"CSV too short: {len(lines)} lines")

    # Line 0: category headers (ignored)
    # Line 1: column headers
    # Line 2+: data rows
    parking_groups = {}

    for line in lines[2:]:
        cols = line.strip().split(",")
        if len(cols) < 15:
            continue

        parking_num = cols[2].strip()
        route_num = cols[5].strip()
        day_night = cols[10].strip()

        if not parking_num or not route_num:
            continue

        if parking_num not in parking_groups:
            parking_groups[parking_num] = {"day": [], "night": []}

        if "白天" in day_night:
            parking_groups[parking_num]["day"].append(route_num)
        elif "黑夜" in day_night:
            parking_groups[parking_num]["night"].append(route_num)

    logger.info(
        "Parsed %d parking lots from CSV", len(parking_groups)
    )
    return parking_groups


def resolve_mcap_paths(
    odo_data: Dict,
    parking_num: str,
    route_num: str,
    camera_channels: List[str] = ("6", "7"),
) -> Dict[str, List[str]]:
    """Resolve MCAP file paths for a given parking lot + route.

    Args:
        odo_data: Pre-loaded odo_dump_process dict (load once, pass many times).
        parking_num: Parking lot number (string).
        route_num: Route number (string).
        camera_channels: Camera channels to filter for.

    Returns:
        {"adas_6": [...], "adas_7": [...], "gnssfusion": [...]}
    """
    routes = odo_data.get(parking_num, {})
    sessions = routes.get(route_num, {})

    result = {f"adas_{ch}": [] for ch in camera_channels}
    result["gnssfusion"] = []

    for session_id, mcap_urls in sessions.items():
        for url in mcap_urls:
            basename = os.path.basename(url)

            # ADAS camera mcaps
            if basename.startswith("ADAS_"):
                m = re.match(r"ADAS_.+_(\d+)\.mcap$", basename)
                if m:
                    ch = m.group(1)
                    key = f"adas_{ch}"
                    if key in result:
                        result[key].append(url)

            # GNSSFUSION mcaps (for indoor/outdoor classification)
            elif basename.startswith("GNSSFUSION"):
                result["gnssfusion"].append(url)

    return result


def find_gt_trajectory(
    gt_path: str, parking_num: str, case_num: str
) -> str:
    """Find GT trajectory file for a given parking + case.

    Looks for: gt_path/outdoor-P{num}_ent1_route1_case{case}/lio_offline_10HZ.txt
    """
    if not os.path.exists(gt_path):
        return ""

    prefix = f"outdoor-P{parking_num}"
    suffix = f"case{case_num}"

    for dirname in os.listdir(gt_path):
        if dirname.startswith(prefix) and suffix in dirname:
            traj_file = os.path.join(gt_path, dirname, "lio_offline_10HZ.txt")
            if os.path.exists(traj_file):
                return traj_file
    return ""


def generate_case_pairs(
    parking_groups: Dict,
    odo_json_path: str,
    gt_path: str,
    camera_channels: List[str] = ("6", "7"),
    mcap_local_root: str = "",
) -> List[Dict]:
    """Generate case pairs with resolved MCAP paths and GT trajectories.

    For each parking lot, creates all (day_route, night_route) combos.
    """
    # Load JSON once (can be large, ~100MB+)
    logger.info("Loading MCAP index JSON: %s", odo_json_path)
    with open(odo_json_path, "r") as f:
        odo_data = json.load(f)
    logger.info("Loaded MCAP index (%d parking lots)", len(odo_data))

    case_pairs = []

    for parking_num, routes in sorted(parking_groups.items()):
        day_routes = routes["day"]
        night_routes = routes["night"]

        if not day_routes or not night_routes:
            logger.warning(
                "Parking %s: missing day(%d) or night(%d) routes",
                parking_num, len(day_routes), len(night_routes),
            )
            continue

        for day_route in day_routes:
            for night_route in night_routes:
                # Resolve MCAP paths (uses pre-loaded dict, fast)
                day_mcaps = resolve_mcap_paths(
                    odo_data, parking_num, day_route, camera_channels
                )
                night_mcaps = resolve_mcap_paths(
                    odo_data, parking_num, night_route, camera_channels
                )

                # Convert dmpv2:// paths to local if root provided
                if mcap_local_root:
                    day_mcaps = _localize_paths(day_mcaps, mcap_local_root)
                    night_mcaps = _localize_paths(night_mcaps, mcap_local_root)

                # Build case naming
                day_case = f"P{parking_num}_ent1_route1_case{day_route}"
                night_case = f"P{parking_num}_ent1_route1_case{night_route}"

                # Find GT trajectories
                day_gt = find_gt_trajectory(gt_path, parking_num, day_route)
                night_gt = find_gt_trajectory(gt_path, parking_num, night_route)

                pair = {
                    "parking_id": parking_num,
                    "day_route": day_route,
                    "night_route": night_route,
                    "day_case": day_case,
                    "night_case": night_case,
                    "day_gt_trajectory": day_gt,
                    "night_gt_trajectory": night_gt,
                    "mcap_paths": {
                        "day": day_mcaps,
                        "night": night_mcaps,
                    },
                }
                case_pairs.append(pair)

    logger.info("Generated %d case pairs", len(case_pairs))
    return case_pairs


def _localize_paths(
    mcap_dict: Dict[str, List[str]], local_root: str
) -> Dict[str, List[str]]:
    """Convert dmpv2:// URLs to local file paths."""
    result = {}
    for key, urls in mcap_dict.items():
        local_paths = []
        for url in urls:
            # dmpv2://bucket_name/path/to/file.mcap -> local_root/path/to/file.mcap
            if url.startswith("dmpv2://"):
                parts = url.replace("dmpv2://", "").split("/", 1)
                if len(parts) == 2:
                    local_path = os.path.join(local_root, parts[1])
                else:
                    local_path = url
            else:
                local_path = url
            local_paths.append(local_path)
        result[key] = local_paths
    return result


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Phase 0: Parse CSV + resolve MCAP paths -> case_pairs.json"
    )
    parser.add_argument("--csv", required=True, help="Parking lot CSV file")
    parser.add_argument("--json", required=True, help="odo_dump_process.json")
    parser.add_argument("--gt-path", required=True, help="GT trajectory directory")
    parser.add_argument("--output", required=True, help="Output case_pairs.json")
    parser.add_argument(
        "--mcap-local-root", default="",
        help="Local root to convert dmpv2:// paths (empty = keep URLs)",
    )
    parser.add_argument(
        "--cameras", default="6,7",
        help="Camera channels to include (comma-separated)",
    )
    args = parser.parse_args()

    channels = [c.strip() for c in args.cameras.split(",")]
    parking_groups = parse_parking_csv(args.csv)
    case_pairs = generate_case_pairs(
        parking_groups, args.json, args.gt_path,
        camera_channels=channels,
        mcap_local_root=args.mcap_local_root,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(case_pairs, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d case pairs to %s", len(case_pairs), args.output)


if __name__ == "__main__":
    main()
