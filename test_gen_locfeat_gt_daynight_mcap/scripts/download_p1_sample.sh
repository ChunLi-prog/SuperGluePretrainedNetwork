#!/bin/bash
# ============================================================
# P1 Sample MCAP Download Script
# Run this on the dev server (10.29.20.24) where dmpv2 tools are available.
#
# Usage:
#   1. SSH to server: ssh chun.li@10.29.20.24
#   2. Copy this script to server
#   3. chmod +x download_p1_sample.sh && ./download_p1_sample.sh
#   4. SCP results back to local:
#      scp -r chun.li@10.29.20.24:/tmp/p1_mcap_sample/ \
#          /home/user/data/maploc_data/gen_kpts_gt_datasets/orig_mcap_datasets/
# ============================================================

set -e
DEST="/tmp/p1_mcap_sample"
mkdir -p "$DEST"

echo "=== Downloading P1 sample MCAP files ==="
echo "Destination: $DEST"
echo ""

# ---- Day Route 1 Session 20240520-173206_582 ----
echo "[1/12] Day session 173206 - ADAS adas_6 (rear)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-173206_582/ADAS_20240520-173206_582_6.mcap" "$DEST/"

echo "[2/12] Day session 173206 - ADAS adas_7 (front)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-173206_582/ADAS_20240520-173206_582_7.mcap" "$DEST/"

echo "[3/12] Day session 173206 - GNSSFUSION"
dmpv2 cp "dmpv2://carizon_maploc_jfs/chenchen.li/parking_daily_data/daily_fix_data/LITE/odometry/raw/1/1/20240520-173206_582/GNSSFUSION#_20240520-173206_582_204.mcap" "$DEST/"

# ---- Day Route 1 Session 20240520-172706_582 ----
echo "[4/12] Day session 172706 - ADAS adas_6 (rear)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-172706_582/ADAS_20240520-172706_582_6.mcap" "$DEST/"

echo "[5/12] Day session 172706 - ADAS adas_7 (front)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-172706_582/ADAS_20240520-172706_582_7.mcap" "$DEST/"

echo "[6/12] Day session 172706 - GNSSFUSION"
dmpv2 cp "dmpv2://carizon_maploc_jfs/chenchen.li/parking_daily_data/daily_fix_data/LITE/odometry/raw/1/1/20240520-172706_582/GNSSFUSION#_20240520-172706_582_204.mcap" "$DEST/"

# ---- Night Route 3 Session 20240520-192453_467 ----
echo "[7/12] Night session 192453 - ADAS adas_6 (rear)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-192453_467/ADAS_20240520-192453_467_6.mcap" "$DEST/"

echo "[8/12] Night session 192453 - ADAS adas_7 (front)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-192453_467/ADAS_20240520-192453_467_7.mcap" "$DEST/"

echo "[9/12] Night session 192453 - GNSSFUSION"
dmpv2 cp "dmpv2://carizon_maploc_jfs/chenchen.li/parking_daily_data/daily_fix_data/LITE/odometry/raw/1/3/20240520-192453_467/GNSSFUSION#_20240520-192453_467_204.mcap" "$DEST/"

# ---- Night Route 3 Session 20240520-191953_467 ----
echo "[10/12] Night session 191953 - ADAS adas_6 (rear)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-191953_467/ADAS_20240520-191953_467_6.mcap" "$DEST/"

echo "[11/12] Night session 191953 - ADAS adas_7 (front)"
dmpv2 cp "dmpv2://carizon_collect_jfs4/collect/mcap/BT5537/BT5537_20240520_D/20240520-191953_467/ADAS_20240520-191953_467_7.mcap" "$DEST/"

echo "[12/12] Night session 191953 - GNSSFUSION"
dmpv2 cp "dmpv2://carizon_maploc_jfs/chenchen.li/parking_daily_data/daily_fix_data/LITE/odometry/raw/1/3/20240520-191953_467/GNSSFUSION#_20240520-191953_467_204.mcap" "$DEST/"

echo ""
echo "=== Download complete ==="
ls -lh "$DEST/"
echo ""
echo "Total size:"
du -sh "$DEST/"
echo ""
echo "Now SCP back to your local machine:"
echo "  scp -r chun.li@10.29.20.24:$DEST/ /home/user/data/maploc_data/gen_kpts_gt_datasets/orig_mcap_datasets/"
