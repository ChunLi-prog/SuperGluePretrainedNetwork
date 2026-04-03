# MCAP-based Day/Night Local Feature GT Generation - Development Plan

## 1. Project Overview

Upgrade the SuperPoint+SuperGlue local feature ground truth generation pipeline from `.pack` to `.mcap` format, with support for front+rear fisheye cameras, and NAS-based data sourcing.

## 2. Architecture: Two-Stage Parallel Design

### 2.1 Design Rationale

All original-resolution fisheye images (front+rear, day+night) must be saved to disk as deliverables. Since disk IO is an unavoidable requirement, we split the pipeline into two stages with different parallelism strategies:

- **Stage 1 (IO-bound)**: MCAP decode + image extraction → multi-process parallel by case/session
- **Stage 2 (GPU-bound)**: SuperPoint + SuperGlue match → batch parallel, multi-GPU friendly

This separation also enables:
- Re-running Stage 2 with different SP/SG configs without re-extracting images
- Quality inspection of raw images before running expensive GPU inference
- Independent scaling: add disk bandwidth for Stage 1, add GPUs for Stage 2

### 2.2 Pipeline Flow

```text
Phase 0: Data Indexing (one-time, fast)
  CSV + JSON → case_pairs.json

Phase 1: Trajectory Matching (one-time, fast)
  GT trajectories → keyframe_pairs/*.json (day_ts, night_ts per pair)

Stage 1: Image Extraction (IO-bound, multi-process parallel)
  MCAP files + keyframe_pairs.json → original-resolution images on disk
  ┌─ Worker 1: P1_case1-case3 ─→ MCAP decode → save front+rear day+night JPGs
  ├─ Worker 2: P1_case1-case4 ─→ ...
  ├─ Worker 3: P1_case2-case3 ─→ ...
  ├─ Worker 4: P2_case1-case3 ─→ ...
  └─ ...
  Output per case:
    {case}/adas_7/day/{day_case}-{ts}.jpg      (original resolution)
    {case}/adas_7/night/{night_case}-{ts}.jpg
    {case}/adas_6/day/{day_case}-{ts}.jpg
    {case}/adas_6/night/{night_case}-{ts}.jpg
    {case}/adas_7/scannet_pairs.txt
    {case}/adas_6/scannet_pairs.txt

Stage 2: SP+SG Matching (GPU-bound, batch parallel)
  Images + scannet_pairs.txt → .npz match results
  ┌─ GPU Worker 1: P1/adas_7 ─→ read images → SP+SuperGlue → RANSAC → .npz
  ├─ GPU Worker 2: P1/adas_6 ─→ ...
  ├─ GPU Worker 3: P2/adas_7 ─→ ...
  └─ ...
  Input: reads images from Stage 1 output, resizes to 960x720 for SP
  Output: .npz per pair + metrics CSV + visualization images
```

### 2.3 Key Design Decisions

1. **Save ALL original images**: Front (adas_7) + Rear (adas_6) × Day + Night × ALL keyframe timestamps. Original resolution, no resize during extraction.

2. **Resize only for SP inference**: Stage 2 reads original images, resizes to 960×720 in memory before feeding to SuperPoint. Original images on disk stay untouched.

3. **GT = SuperPoint + SuperGlue matches**: SP→SuperGlue→RANSAC inlier matches are the final ground truth. SuperGlue uses attention-based graph neural network for learned matching, producing higher quality matches than KNN especially for day/night conditions.

4. **No fisheye undistortion**: SP runs directly on fisheye images.

5. **Visualization by distance**: Save vis images every ~1m along trajectory.

## 3. Stage 1: Image Extraction (Detail)

### 3.1 Per-case processing

For each (day_case, night_case) pair, for each camera (adas_6, adas_7):
1. Open MCAP files for day session + night session
2. Build timestamp index: scan all frames, record (ts_ms → raw_h265_data)
3. For each (day_ts, night_ts) in keyframe_pairs.json:
   a. Find closest frame within tolerance (500ms)
   b. H.265 decode → BGR numpy array (original resolution)
   c. Save as JPEG: `{case}/{camera}/day/{day_case}-{ts}.jpg`
   d. Record pair: `{day_filename} {night_filename}` in scannet_pairs.txt

### 3.2 Parallelism

```python
# Multi-process pool, one case per worker
from multiprocessing import Pool

with Pool(processes=num_workers) as pool:
    pool.map(extract_one_case, case_list)
```

- Each worker is independent: own MCAP readers, own H265Decoder, own output directory
- No shared state, no locking needed
- `num_workers` = min(num_cases, CPU_cores / 2) (H.265 decode is CPU-heavy)
- Memory: each worker holds ~1 MCAP index in memory (~few hundred MB)

### 3.3 Output structure

```text
{out_path}/
  {day_case}-{night_case}/
    adas_7/
      day/
        {day_case}-{ts_ms}.jpg          # original resolution (e.g. 1920x1536)
      night/
        {night_case}-{ts_ms}.jpg
      scannet_pairs.txt                  # "day_filename.jpg night_filename.jpg"
    adas_6/
      day/
        {day_case}-{ts_ms}.jpg
      night/
        {night_case}-{ts_ms}.jpg
      scannet_pairs.txt
```

## 4. Stage 2: SP+SG Matching (Detail)

### 4.1 Per-case-camera processing

For each case directory with `scannet_pairs.txt`:
1. Load SuperPoint model (once per GPU worker)
2. For each pair in scannet_pairs.txt:
   a. Read day/night images from disk
   b. Resize to 960×720 (in memory only, for SP)
   c. Convert to grayscale → frame2tensor → SuperPoint + SuperGlue (combined forward pass)
   d. Optional RANSAC post-filter (threshold=3.0)
   e. Save .npz result

### 4.2 Parallelism

```python
# Multi-process, each with own GPU (or time-share single GPU)
# Each worker processes a (case, camera) tuple independently
work_units = [(case_dir, "adas_7"), (case_dir, "adas_6"), ...]

with Pool(processes=num_gpu_workers) as pool:
    pool.map(match_one_case_camera, work_units)
```

- GPU memory: SuperPoint ~200MB + SuperGlue ~400MB per worker
- Single GPU: 1-2 workers (serial or interleaved)
- Multi GPU: one worker per GPU via CUDA_VISIBLE_DEVICES

### 4.3 Output structure

```text
{out_path}/
  {day_case}-{night_case}/
    adas_7/
      matches/                           # .npz per pair
        {day_case}-{ts}_{night_case}-{ts}.npz
      metrics.csv                        # summary statistics
      vis/                               # visualization images (sampled)
        day{ts}_night{ts}_matches.jpg
    adas_6/
      matches/
      metrics.csv
      vis/
```

## 5. Development Plan

### Phase 1: Local Sample Pipeline (Current Goal)

#### Step 1: Data Preparation
- [ ] Download P1 sample MCAP files from NAS (adas_6, adas_7 for day+night)
- [ ] Verify GT trajectories exist (✅ already confirmed for P1)
- [ ] Phase 0 case_pairs.json (✅ already generated, 2896 pairs)

#### Step 2: Phase 1 Trajectory Matching
- [ ] Adapt `read_csv()` in gen_locfeat_gt_daynight.py for new case pair format
- [ ] Run on P1 → keyframe_pairs JSON

#### Step 3: Stage 1 - Image Extraction
- [ ] Create `mcap_image_extractor.py` (in navinet or standalone repo)
  - MCAPImageReader + H265Decoder (✅ already written)
  - Per-case extraction logic with original-resolution save
  - Multi-process wrapper
  - scannet_pairs.txt generation
- [ ] Test on P1 sample MCAP → verify output images

#### Step 4: Stage 2 - SP+SG Matching
- [ ] Adapt `pipeline_baseline.py` or create new Stage 2 script
  - Input: Stage 1 output directories
  - Resize 960×720 in memory for SP
  - Output: .npz matches + metrics
  - Multi-process wrapper
- [ ] Test on P1 → verify .npz output

#### Step 5: Integration Test
- [ ] End-to-end: Phase 0 → Phase 1 → Stage 1 → Stage 2 on P1
- [ ] Visual inspection of matches

### Phase 2: Server Deployment (Future)
- [ ] NAS download/streaming for MCAP files
- [ ] Scale to all parking lots with job scheduler
- [ ] Multi-GPU batch processing for Stage 2
- [ ] Progress tracking and resume

## 6. File Structure (All in SuperGlue repo, branch: feature-mcap_streaming_pipeline)

```text
SuperGluePretrainedNetwork/
  # --- Shared MCAP utilities ---
  mcap_utils/
    __init__.py
    h265_decoder.py                # H.265 decode with IDR caching
    mcap_image_reader.py           # MCAP ADAS image reader by timestamp
    mcap_gnss_reader.py            # MCAP GNSSFUSION reader (indoor/outdoor)

  # --- Stage 2: SP+SG matching ---
  mcap_stage2_matching.py          # Images -> SP+SuperGlue -> RANSAC -> .npz
  config/
    mcap_pipeline.yaml             # Stage 2 config

  # --- Phase 0 + Phase 1 + Stage 1 ---
  test_gen_locfeat_gt_daynight_mcap/
    config/
      gen_locfeat.yaml             # Local dev config
      gen_locfeat_in_server.yaml   # Server batch config
    docs/
      development_plan.md          # This file
      pipeline_comparison.md       # Old vs new comparison
      data_format_spec.md          # Data format specifications
    scripts/
      download_p1_sample.sh        # MCAP download script for P1 sample
    src/
      parse_parking_csv.py         # Phase 0: CSV + JSON -> case_pairs.json
      gen_locfeat_gt_daynight.py   # Phase 1: Trajectory matching
      gen_locfeat_baseline.py      # Phase 1 variant (legacy reference)
      mcap_image_extractor.py      # Stage 1: MCAP -> original images + labels
    log/

  # --- Existing (unchanged) ---
  models/                          # SuperPoint, SuperGlue models + weights
  matchers/                        # KNN matcher
  filters/                         # RANSAC filter
  pipeline_baseline.py             # Original SP+SG pipeline (reference)
```

## 7. Design Decisions (Confirmed)

1. **Save ALL original-resolution images**: Both adas_6 and adas_7, day and night, all keyframe timestamps. No resize during extraction.

2. **Resize only for SP inference**: Stage 2 resizes to 960×720 in memory when feeding to SuperPoint. Disk images remain original resolution.

3. **GT = SuperPoint + SuperGlue matches**: SP→SuperGlue→RANSAC inlier matches are the final ground truth labels. No J6-128/DESCRIPTOR mcap needed.

4. **No fisheye undistortion**: SP runs directly on fisheye images.

5. **Two-stage parallel design**: Stage 1 (IO-bound, multi-process) and Stage 2 (GPU-bound, multi-GPU) with different parallelism strategies.

6. **Visualization by distance**: Save vis images every ~1m along trajectory.

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| H.265 decode failure (missing IDR) | Medium | High | IDR caching; skip + log on failure |
| Large disk usage (original res) | Expected | Medium | ~500KB/img × 4cam × 50k pairs ≈ 100GB; ensure sufficient disk |
| GPU OOM in Stage 2 | Low | High | Process one pair at a time; clear GPU memory |
| NAS download bottleneck | High | Medium | Local caching, parallel download |
| MCAP session spanning multiple files | Medium | Medium | MCAPImageReader handles multi-file per session |
