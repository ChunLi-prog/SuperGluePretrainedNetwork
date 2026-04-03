"""Patch case_pairs.json to use local MCAP paths for P1 sample.

Replaces dmpv2:// URLs with local file paths for cases where
MCAP files have been downloaded to local disk.

Usage:
    python patch_p1_local_paths.py \
        --case-pairs /path/to/case_pairs.json \
        --mcap-dir /path/to/p1_sample/ \
        --output /path/to/case_pairs_local.json
"""

import argparse
import json
import os


# dmpv2 bucket → JuiceFS mount (for reference)
BUCKET_MAP = {
    "dmpv2://carizon_collect_jfs4/": "/horizon-bucket/carizon_collect_jfs4/",
    "dmpv2://carizon_maploc_jfs/": "/horizon-bucket/carizon_maploc_jfs/",
    "dmpv2://carizon_fillback_jfs/": "/horizon-bucket/carizon_fillback_jfs/",
}


def dmpv2_to_local(url: str, mcap_dir: str) -> str:
    """Convert dmpv2:// URL to local path by matching filename."""
    basename = os.path.basename(url)
    local_path = os.path.join(mcap_dir, basename)
    if os.path.exists(local_path):
        return local_path
    return url  # keep original if not found locally


def main():
    parser = argparse.ArgumentParser(description="Patch dmpv2 URLs to local paths")
    parser.add_argument("--case-pairs", required=True)
    parser.add_argument("--mcap-dir", required=True, help="Dir with downloaded .mcap files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parking-ids", default=None, help="Only patch these parking IDs (comma-sep)")
    args = parser.parse_args()

    with open(args.case_pairs) as f:
        cases = json.load(f)

    target_ids = set(args.parking_ids.split(",")) if args.parking_ids else None
    patched = 0
    total_urls = 0

    for case in cases:
        if target_ids and case["parking_id"] not in target_ids:
            continue

        mcap_paths = case.get("mcap_paths", {})
        for time_of_day in ["day", "night"]:
            channels = mcap_paths.get(time_of_day, {})
            for channel, urls in channels.items():
                new_urls = []
                for url in urls:
                    total_urls += 1
                    new_url = dmpv2_to_local(url, args.mcap_dir)
                    if new_url != url:
                        patched += 1
                    new_urls.append(new_url)
                channels[channel] = new_urls

    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Patched {patched}/{total_urls} URLs")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
