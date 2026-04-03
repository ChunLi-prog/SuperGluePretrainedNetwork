# Old vs New Pipeline Detailed Comparison

## 1. End-to-End Pipeline Flow

### Old Pipeline (pack-based)

```text
[Phase 1: Trajectory Matching] (Python)
  Simple CSV → (day, night) combos → GT trajectory matching → keyframe_pairs.json

[Phase 2: Image + Feature Extraction] (C++, serial)
  .pack → pack-sdk → NV12 decode → resize 960x720 → save JPG + kpts.txt + desc.txt
  Per pair: 6 files written (2 images + 4 feature files)
  Camera: front-view only

[Phase 3: SP+SG Matching] (Python, serial)
  Read images + features from disk → SuperGlue match → RANSAC → .npz
```

### New Pipeline (mcap-based, two-stage parallel)

```text
[Phase 0: Data Indexing] (Python, one-time)
  Complex CSV + JSON → case_pairs.json (2896 pairs for 50 parking lots)

[Phase 1: Trajectory Matching] (Python, one-time)
  GT trajectories → keyframe_pairs.json (same algorithm as old)

[Stage 1: Image Extraction] (Python, multi-process parallel, IO-bound)
  MCAP → H.265 decode → save ORIGINAL resolution JPG (no resize!)
  Camera: front (adas_7) + rear (adas_6) fisheye
  Per pair: 4 images (2 cameras × day + night) + scannet_pairs.txt
  Parallelism: one process per case, no shared state

[Stage 2: SP+SG Matching] (Python, multi-GPU parallel, GPU-bound)
  Read images → resize 960x720 in memory → SuperPoint+SuperGlue → RANSAC → .npz
  Parallelism: one process per (case, camera) tuple
```

## 2. Key Differences

| Dimension | Old (.pack) | New (.mcap) |
|-----------|-------------|-------------|
| Data format | `.pack` (proprietary binary) | `.mcap` (open container) |
| Camera | Front-view only | Front (adas_7) + Rear (adas_6) fisheye |
| Image encoding | NV12 bitstream | H.265 encoded |
| Saved image resolution | Resized 960×720 | **Original resolution** (e.g. 1920×1536) |
| SP input resolution | From disk (already 960×720) | Resize in memory to 960×720 |
| Feature extraction | C++ (J6-128 model from pack proto) | Python (SuperPoint on decoded images) |
| Data source | Local disk | NAS Bucket (`dmpv2://`) via JSON index |
| Case list | Simple 5-col CSV | Complex parking CSV + JSON index |
| Image extraction | C++ serial process | Python multi-process parallel |
| SP+SG matching | Python serial process | Python multi-GPU parallel |
| C++ build dependency | Required (pack-sdk) | **None** (pure Python) |

## 3. Image Handling Comparison

### Old: Resize then save

```text
pack-sdk → NV12 → BGR (3840×2160 or 1920×1536)
  → resize to 960×720
  → cv2.imwrite("day/{case}-{ts}.jpg")   # 960×720 on disk
SP reads: cv2.imread → already 960×720 → frame2tensor → inference
```

### New: Save original, resize only for SP

```text
MCAP → H.265 → BGR (original resolution, e.g. 1920×1536)
  → cv2.imwrite("day/{case}-{ts}.jpg")   # ORIGINAL resolution on disk
SP reads: cv2.imread → resize to 960×720 IN MEMORY → frame2tensor → inference
```

**Why save original resolution?**
- Original images are the ground truth data deliverable
- Allows re-running SP with different target resolutions without re-extracting
- Fisheye images at original resolution preserve more detail for future use

## 4. Feature Extraction Comparison

### Old: J6-128 model features from pack proto (C++)

```text
Source: feature_map_v2::FeatureFrame in LocalFeatures_*.pack
Output: .kpts.txt (x, y, confidence) + .desc.txt (256-dim floats)
These are embedded model features, not SuperPoint features.
SP+SG matching in Phase 3 uses these pre-extracted features via SuperGlue.
```

### New: SuperPoint features directly (Python)

```text
Source: SuperPoint model inference on decoded + resized image
Input: grayscale 960×720 → frame2tensor → (1,1,720,960) float [0,1]
Output: keypoints (N,2), scores (N,), descriptors (256,N) → in memory
Then: SuperGlue match → RANSAC → save .npz only
No intermediate .kpts.txt / .desc.txt files needed.
```

## 5. Parallelism Comparison

### Old: Serial everything

```text
Phase 2 (C++): Process cases one by one, sequential pack reading
Phase 3 (Python): Process pairs one by one, sequential SP inference
Total: O(N_cases × N_pairs) with no parallelism
```

### New: Two-level parallelism

```text
Stage 1 (IO-bound): multiprocessing.Pool, one case per worker
  - Each worker: own MCAP reader, own H265Decoder, own output dir
  - No shared state, no locking
  - num_workers ≈ min(num_cases, cpu_cores / 2)
  - Bottleneck: H.265 decode (CPU) + disk write

Stage 2 (GPU-bound): multiprocessing.Pool, one (case, camera) per worker
  - Each worker: own SuperPoint model on assigned GPU
  - CUDA_VISIBLE_DEVICES for multi-GPU
  - num_workers ≈ num_GPUs × 1-2
  - Bottleneck: SuperPoint forward pass (GPU)
```

## 6. Disk Usage Comparison

```text
                        Old Pipeline              New Pipeline
Images per pair:        2 (day+night, 960×720)    4 (2 cameras × day+night, ORIGINAL)
Image size:             ~100KB each               ~500KB each (original res)
Feature files/pair:     4 (.kpts + .desc × 2)     0 (in-memory only)
Match output/pair:      1 (.npz ~50KB)            1 (.npz ~50KB)
Vis image/pair:         1 (~200KB)                1 per ~1m (~200KB)

For 50k pairs:
  Images:               100k files, ~10GB         200k files, ~100GB
  Features:             200k files, ~5GB          0
  Matches:              50k files, ~2.5GB         50k×2cam files, ~5GB
  Total:                ~17.5GB                   ~105GB
```

**Note**: Disk usage increases significantly because we save original resolution + dual cameras. This is by design — the original images are deliverables.

## 7. What Can Be Reused vs What Needs New Code

### Reusable (no changes)

- SuperPoint model + weights (`models/superpoint.py`, `superpoint_v1.pth`)
- SuperGlue model + weights (`models/superglue.py`, `superglue_outdoor.pth`, `superglue_indoor.pth`)
- Matching wrapper (`models/matching.py`) — combines SP+SG in single forward pass
- RANSAC filter (`filters/ransac_filter.py`) — optional post-filter after SuperGlue
- `frame2tensor()` utility
- Trajectory math: `calculate_distance`, `find_closest_frame`, `iterate_and_find_keyframe_pairs`
- Config parameters: keyframe_dist, match thresholds

### Needs Adaptation

- `read_csv()` → new CSV format
- `load_trajectory_files()` → new GT naming
- `pipeline_baseline.py` → adapt for Stage 2 (read original images, resize in memory)

### New Code

| Module | Location | Stage | Purpose |
|--------|----------|-------|---------|
| `parse_parking_csv.py` | navinet | Phase 0 | CSV + JSON → case_pairs.json (✅ done) |
| `mcap_image_extractor.py` | navinet | Stage 1 | MCAP → original images + scannet_pairs.txt |
| `mcap_stage2_matching.py` | SuperGlue | Stage 2 | Images → SP+SuperGlue → RANSAC → .npz |
| `mcap_utils/` | SuperGlue | Shared | H.265 decoder + MCAP reader (✅ done) |

### Eliminated

- All C++ code: `test_gen_locfeat_baseline.cc`, `test_dump_imgs.cc`
- All pack-sdk managers: `adas_data_manager`, `feature_data_manager`, `gnss_data_manager`, `pack_proc`
- Intermediate feature files: `.kpts.txt`, `.desc.txt` (SP features stay in memory)
