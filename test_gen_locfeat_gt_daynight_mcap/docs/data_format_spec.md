# Data Format Specifications

## 1. Input Data Formats

### 1.1 Parking Lot CSV (`50_50_UPDATE_c_a_parking.csv`)

Complex multi-column CSV with 2 header rows + data rows.

**Key columns**:

| Col Index | Header | Usage | Example |
|-----------|--------|-------|---------|
| 2 | 编号 | Parking lot ID, maps to JSON key | `1`, `2`, `3` |
| 5 | 路线 | Route number, maps to JSON sub-key | `1`, `2`, `3`, `4` |
| 10 | 白天黑夜 | Day/night classification | `白天`, `黑夜` |
| 27 | extrace_pack_name | Session ID(s), comma-separated | `20240520-173206_582,...` |

**Grouping**: By parking_num → split by day/night → generate all (day, night) combos.

### 1.2 MCAP Index JSON (`odo_dump_process.json`)

```json
{
  "<parking_num>": {
    "<route_num>": {
      "<session_id>": [
        "dmpv2://bucket/path/ADAS_{session}_6.mcap",
        "dmpv2://bucket/path/ADAS_{session}_7.mcap",
        "..."
      ]
    }
  }
}
```

**MCAP file naming**: `ADAS_{session}_{channel}.mcap`
- Channel 6 → rear fisheye (adas_6)
- Channel 7 → front fisheye (adas_7)

### 1.3 GT Trajectory Files

```text
Location: gt_dataset_samples/outdoor-P{num}_ent1_route1_case{case}/lio_offline_10HZ.txt
Format: timestamp px py pz qx qy qz qw (space-separated, 8 cols, 10Hz)
```

## 2. MCAP Message Formats

### 2.1 ADAS Camera Image

```text
File:  ADAS_{session}_{channel}.mcap
Topic: "image-/dc/adas_{channel}"
Proto: CameraFrame

Fields:
  proto.time_stamp → int64 (milliseconds)
  data[0]          → bytes (H.265 encoded frame)
```

**H.265 decoding**: IDR frame (NAL type 19/20) caching + P-frame prepend strategy.

## 3. Stage 1 Output: Extracted Images

### 3.1 Directory Structure

```text
{out_path}/
  {day_case}-{night_case}/
    adas_7/                              # front fisheye camera
      day/
        {day_case}-{ts_ms}.jpg           # ORIGINAL resolution (e.g. 1920×1536)
        ...
      night/
        {night_case}-{ts_ms}.jpg
        ...
      scannet_pairs.txt                  # pair listing for Stage 2
    adas_6/                              # rear fisheye camera
      day/
        {day_case}-{ts_ms}.jpg
        ...
      night/
        {night_case}-{ts_ms}.jpg
        ...
      scannet_pairs.txt
```

### 3.2 Image Format

- **Resolution**: Original camera resolution (NO resize), e.g. 1920×1536
- **Format**: JPEG (BGR color, standard quality)
- **Naming**: `{case_name}-{timestamp_ms}.jpg`
  - Example: `P1_ent1_route1_case1-1716197226500.jpg`

### 3.3 Scannet Pairs File (`scannet_pairs.txt`)

```text
P1_ent1_route1_case1-1716197226500.jpg P1_ent1_route1_case3-1716208415300.jpg
P1_ent1_route1_case1-1716197228500.jpg P1_ent1_route1_case3-1716208417100.jpg
...
```

One pair per line, space-separated. Day filename first, night filename second.

## 4. Stage 2 Output: Match Results

### 4.1 Directory Structure

```text
{out_path}/
  {day_case}-{night_case}/
    adas_7/
      matches/
        {day_case}-{ts}_{night_case}-{ts}.npz
        ...
      metrics.csv
      vis/                               # visualization (distance-sampled)
        day{ts}_night{ts}_matches.jpg
    adas_6/
      matches/
      metrics.csv
      vis/
```

### 4.2 Match Result `.npz`

```python
{
    # SuperPoint keypoints (in 960×720 coordinate space)
    "keypoints0": ndarray (N0, 2),       # day SP keypoints (x, y)
    "keypoints1": ndarray (N1, 2),       # night SP keypoints
    "scores0": ndarray (N0,),            # day SP confidence
    "scores1": ndarray (N1,),            # night SP confidence

    # SP descriptors (256-dim, L2-normalized, saved as N×256)
    "descriptors0": ndarray (N0, 256),
    "descriptors1": ndarray (N1, 256),

    # SuperGlue match results (learned matching via attention-based GNN)
    "matches": ndarray (N0,),            # index into keypoints1 (-1 = unmatched)
    "match_confidence": ndarray (N0,),

    # Matched pairs (before RANSAC)
    "mkpts0": ndarray (M, 2),
    "mkpts1": ndarray (M, 2),
    "mconf": ndarray (M,),

    # RANSAC inliers (final GT — SuperGlue matches filtered by RANSAC)
    "mkpts0_ransac": ndarray (R, 2),
    "mkpts1_ransac": ndarray (R, 2),
    "mconf_ransac": ndarray (R,),

    # Metadata
    "day_timestamp": float,              # day frame timestamp (ms)
    "night_timestamp": float,            # night frame timestamp (ms)
    "camera": str,                       # "adas_7" or "adas_6"
    "image_shape": ndarray (2,),         # (720, 960) - SP input resolution
    "original_shape": ndarray (2,),      # (1536, 1920) - original image resolution
    "matcher": str,                      # "superglue" - matching method identifier
    "day_image": str,                    # day image filename
    "night_image": str,                  # night image filename
    "parking_id": str,                   # parking lot ID (e.g. "P1")
    "case_name": str,                    # case pair name (e.g. "P1_ent1_route1_case1-P1_ent1_route1_case3")
}
```

All metadata fields enable full traceability from .npz back to source images and MCAP files.

### 4.3 Metrics CSV

```csv
case,camera,pair_idx,day_ts,night_ts,n_kpts_day,n_kpts_night,n_matches,n_inliers,inlier_ratio
P1_case1-P1_case3,adas_7,0,1716197226500,1716208415300,512,487,234,156,0.667
```

### 4.4 Visualization Images (distance-sampled)

Save a match visualization every ~1m along the trajectory:
- Side-by-side: day image (960×720) | night image (960×720)
- RANSAC inlier match lines drawn between them
- Label: camera, inlier count/total matches

## 5. Configuration Files

### 5.1 Stage 1 Config (navinet repo, `gen_locfeat.yaml`)

```yaml
# Data indexing inputs
csv_file: "path/to/50_50_UPDATE_c_a_parking.csv"
raw_mcap_dataset: "path/to/odo_dump_process.json"
raw_mcap_path: "path/to/local/mcap/cache"

# GT trajectories
gt_path: "path/to/gt_dataset_samples/"

# Stage 1 output
out_path: "path/to/output/"

# Trajectory matching parameters
keyframe_dist: 2.0
max_match_dist_threshold: 4.0
max_match_angle_threshold: 10.0
neighbor_time_threshold: 10.0
neighbor_distance_threshold: 4.0
neighbor_keyframe_dist: 0.5

# Cameras
cameras:
  - channel: "7"
    label: "front"
  - channel: "6"
    label: "rear"

# Stage 1 parallelism
num_extract_workers: 4
timestamp_tolerance_ms: 500

# Logging
log_level: 10
```

### 5.2 Stage 2 Config (SuperGlue repo, `mcap_pipeline.yaml`)

```yaml
# Input: Stage 1 output directory
input_dir: "path/to/stage1/output/"

# Output
output_dir: "path/to/stage2/output/"

# Image processing (resize in memory for SP)
sp_input_resolution: [960, 720]

# SuperPoint
superpoint:
  nms_radius: 4
  keypoint_threshold: 0.005
  max_keypoints: 1024

# Matching (SuperGlue — learned graph neural network matcher)
matcher: "superglue"
superglue:
  weights: "outdoor"                 # "outdoor" or "indoor"
  sinkhorn_iterations: 20
  match_threshold: 0.2

# RANSAC
ransac: true
ransac_threshold: 3.0
ransac_method: "fundamental"

# Visualization
save_visualization: true
vis_distance_threshold: 1.0       # (m) save vis every ~1m

# Parallelism
num_gpu_workers: 1
force_cpu: false
```

## 6. Server Access & Data Download

### 6.1 Dev Server

```text
Host:     10.29.20.24 (jvgpu-dev002.hogpu.cc)
User:     chun.li
Password: aA123!@#2025
```

### 6.2 JuiceFS Bucket Mounts

`dmpv2://` URLs map directly to filesystem paths on the dev server:

| dmpv2 bucket | JuiceFS mount path |
|---|---|
| `dmpv2://carizon_collect_jfs4/...` | `/horizon-bucket/carizon_collect_jfs4/...` |
| `dmpv2://carizon_maploc_jfs/...` | `/horizon-bucket/carizon_maploc_jfs/...` |
| `dmpv2://carizon_fillback_jfs/...` | `/horizon-bucket/carizon_fillback_jfs/...` |

**No need for `dmpv2` CLI tool** — files are directly accessible via the mount.

### 6.3 Download Workflow

```bash
# 1. SSH to server
sshpass -p 'aA123!@#2025' ssh chun.li@10.29.20.24

# 2. Files are at /horizon-bucket/{bucket_name}/... (same path as dmpv2:// URL)

# 3. SCP back to local
sshpass -p 'aA123!@#2025' scp -r chun.li@10.29.20.24:/horizon-bucket/... /local/dest/
```

### 6.4 Local MCAP Storage

```text
/home/user/data/maploc_data/gen_kpts_gt_datasets/orig_mcap_datasets/p1_sample/
  ADAS_20240520-173206_582_6.mcap    # Day session 173206, rear (adas_6)
  ADAS_20240520-173206_582_7.mcap    # Day session 173206, front (adas_7)
  ADAS_20240520-172706_582_6.mcap    # Day session 172706, rear
  ADAS_20240520-172706_582_7.mcap    # Day session 172706, front
  ADAS_20240520-192453_467_6.mcap    # Night session 192453, rear
  ADAS_20240520-192453_467_7.mcap    # Night session 192453, front
  ADAS_20240520-191953_467_6.mcap    # Night session 191953, rear
  ADAS_20240520-191953_467_7.mcap    # Night session 191953, front
  GNSSFUSION#_20240520-173206_582_204.mcap  # Day GNSS
  GNSSFUSION#_20240520-172706_582_204.mcap
  GNSSFUSION#_20240520-192453_467_204.mcap  # Night GNSS
  GNSSFUSION#_20240520-191953_467_204.mcap
```

## 7. Disk Space Estimation

```text
Per case pair (50k timestamp pairs, 2 cameras):
  Stage 1 images: 50k × 4 (cameras×daynight) × ~500KB = ~100GB
  Stage 2 .npz:   50k × 2 cameras × ~50KB = ~5GB
  Vis images:      ~2k (sampled) × ~200KB = ~400MB
  Total per case pair: ~105GB

For all 2896 case pairs (full dataset):
  Note: each case pair has varying pair counts
  Rough estimate: 2896 × avg_pairs × 4 × 500KB = very large
  → Process in batches, clean up intermediate data as needed
```
