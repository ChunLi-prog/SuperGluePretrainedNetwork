# MCAP GT Pipeline → L20 集群部署计划

## Context
Stage 1（MCAP→图片）和 Stage 2（SP+SuperGlue 匹配）已在本地 P1 样本上验证通过（764对，0失败）。现需：
1. 生产加固（断点续传、错误处理、编排脚本）
2. Docker化 + L20集群部署
3. GT格式完善 + 代码提交

## Phase A: 生产加固（核心代码改造）

### A1. Stage 2 断点续传 + 错误处理
**文件**: `mcap_stage2_matching.py`
**改动**:
- `process_one_case_camera()` 中：检查 `.npz` 是否已存在，跳过已处理的 pair
- 用 `try/except` 包裹每个 pair 的处理逻辑，单 pair 失败不影响整体
- 用 `Pool.imap_unordered` 替代 `Pool.map`，加 `error_callback`
- 添加 `--parking-ids` 过滤参数（与 Stage 1 一致）
- .npz 新增字段：`day_timestamp`, `night_timestamp`, `day_image`, `night_image`, `parking_id`, `case_name`
**验证**: 跑一半中断，重启后只处理剩余 pair

### A2. Stage 1 断点续传 + 错误处理
**文件**: `test_gen_locfeat_gt_daynight_mcap/src/mcap_image_extractor.py`
**改动**:
- 检查 day/night 图片是否已存在，跳过
- `extract_one_case()` 用 try/except 包裹，失败记录但不 crash
- `pair_labels.txt` 追加模式（已有的不重复写）
**验证**: 同 A1

### A3. 顶层编排脚本 `orchestrator.py`
**文件**: 新建 `orchestrator.py`（SuperGlue 仓库根目录）
**功能**:
- 接收 `--config`, `--parking-ids`, `--stages` 参数
- 顺序执行: Phase 0 → Phase 1 → Stage 1 → Stage 2
- 每个阶段完成写 checkpoint JSON（`{out_path}/checkpoint.json`）
- 断点续传：读取 checkpoint，跳过已完成阶段
- 汇总日志 + 失败报告
- 错误预警：失败率超阈值时输出 WARNING
**验证**: 端到端跑 P1，验证 checkpoint.json 和续传

### A4. GT 格式完善
**文件**: `mcap_stage2_matching.py`, `docs/data_format_spec.md`
**新增 .npz 字段**:
```python
"day_timestamp": float,        # day 帧时间戳 (ms)
"night_timestamp": float,      # night 帧时间戳 (ms)
"day_image": str,              # day 图片文件名
"night_image": str,            # night 图片文件名
"parking_id": str,             # 停车场 ID
"case_name": str,              # case pair 名称
```
**验证**: 加载 .npz 检查所有字段完整

## Phase B: L20 集群部署

### B1. L20 配置文件
**文件**: 新建 `submit/k8s_config.py`
```python
job_name = "mcap_gt_gen"
num_machines = 1
num_gpus_per_machine = 2
docker_image = "cr-aidi-harbor-..../hat:fsd_multitask-cu11-20230616-v3.9"
input_bucket = "carizon_collect_jfs4, carizon_maploc_jfs, carizon_fillback_jfs"
output_bucket = "carizon_perception_2"
max_jobtime = 20160  # 14 days
upload_folder_name = "SuperGluePretrainedNetwork"
folder_list = [
    "../SuperGluePretrainedNetwork",
    "./run.sh", "./run_sleep.sh"
]
```

### B2. L20 运行脚本
**文件**: 新建 `submit/run.sh`
```bash
cd /running_package/
virtualenv -p python3 env_3
source env_3/bin/activate
cd SuperGluePretrainedNetwork

# 安装依赖（集群只能用 pypi.hobot.cc）
pip3 install torch==2.4.1+cu121 torchvision --index-url https://pypi.hobot.cc/simple
pip3 install opencv-python-headless numpy pyyaml tqdm scipy -i https://pypi.hobot.cc/simple
pip3 install ./wheels/cmdk-*.whl  # 手动提供的 whl

# 运行 pipeline
export CUDA_VISIBLE_DEVICES=$1
python3 orchestrator.py \
    --config config/mcap_pipeline_l20.yaml \
    --stages all \
    --workers 2
```

### B3. L20 专用配置
**文件**: 新建 `config/mcap_pipeline_l20.yaml`
**改动**: 所有路径改为 `/bucket/input/` 前缀
```yaml
# MCAP 数据通过 JuiceFS bucket 直接访问
mcap_base_path: "/bucket/input/carizon_collect_jfs4"
gt_path: "/bucket/input/carizon_maploc_jfs/gt_dataset_samples"
out_path: "/job_data/gt_results"
log_dir: "/job_log/mcap_gt_gen"
```
**dmpv2:// URL 映射规则**:
- `dmpv2://carizon_collect_jfs4/...` → `/bucket/input/carizon_collect_jfs4/...`
- 在 `parse_parking_csv.py` 或 orchestrator 中自动转换

### B4. submit.py 适配
**文件**: 新建 `submit/submit.py`（基于 L20ServerManual/scripts/submit.py 模板）
**改动**: 
- `mmengine.Config` → 直接用 dict（减少依赖）
- 适配 mcap_gt_gen 的 folder_list 和 run_cmd
- 添加 `--parking-ids` 透传参数

### B5. cmdk whl 打包
**文件**: 新建 `wheels/` 目录
**动作**: 从当前环境导出 cmdk wheel：
```bash
pip download cmdk --no-deps -d wheels/
```
打包进 upload_folder

## Phase C: 文档 & Git

### C1. 代码提交
```bash
git add -A
git commit -m "feat(navi): [PRO-1173] Add production pipeline with L20 deployment"
git push origin feature-mcap_streaming_pipeline
```

### C2. 文档更新
**文件**: `docs/development_plan.md`, `docs/data_format_spec.md`
- 新增 L20 部署章节
- 更新 .npz 格式（新增字段）
- 添加 orchestrator.py 使用说明

## Phase D: 测试验证

### D1. 本地 Docker 测试（模拟集群环境）
```bash
docker run --gpus all -v /home/user/data:/data \
    -v $(pwd):/running_package/SuperGluePretrainedNetwork \
    <docker_image> bash submit/run.sh 0,1
```
**验证**: Stage 1 + Stage 2 在 Docker 内跑通

### D2. L20 Sleep 测试
```bash
cd submit && python submit.py --cluster carizon-4dgt-gpu --gpu 2 --sleep
```
进入 pod 手动执行 `bash run.sh 0,1`，验证：
- JuiceFS bucket 路径可访问
- cmdk 安装成功
- torch.cuda.is_available() == True
- 跑通 P1 sample

### D3. L20 正式提交
```bash
cd submit && python submit.py --cluster carizon-4dgt-gpu --gpu 2
```
**监控**: 检查 /job_log/ 中的日志，确认各阶段正常推进

### D4. 全量生产验证
- 先跑 10 个 parking lot 验证
- 检查 .npz 质量（inlier ratio > 0.3）
- 确认断点续传正常工作
- 全量 2896 case pairs 生产

## 执行顺序 & 依赖

```
A1 ──┐
A2 ──┤── A3 (需要 A1+A2 完成) ── A4 ── C1 (git commit)
     │                                      │
     └──────── B1+B2+B3+B4 (可与 A 并行) ──┘── B5 ── D1 ── D2 ── D3 ── D4
```

**关键路径**: A1 → A3 → C1 → B5 → D1 → D2 → D3

## 关键风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| cmdk whl 不兼容集群 Python/CUDA | Stage 1 无法运行 | D2 sleep 测试提前验证 |
| JuiceFS 读取速度慢 | Stage 1 超时 | 增加 timeout，添加重试 |
| PyTorch 在集群 CUDA 11 上不兼容 | Stage 2 无法运行 | 降级到 torch 1.13+cu117 |
| 2896 case pairs 磁盘空间不足 | 生产中断 | 分批处理，及时清理 |

## 验证清单

- [ ] Stage 2 断点续传：中断后重启只处理未完成 pair
- [ ] Stage 1 断点续传：中断后重启只处理未完成图片
- [ ] orchestrator.py 端到端：P1 sample 全流程通过
- [ ] .npz 包含所有必要字段（含时间戳、文件名、parking_id）
- [ ] Docker 内跑通完整 pipeline
- [ ] L20 sleep pod 内手动验证 JuiceFS + CUDA + cmdk
- [ ] L20 正式提交 10 个 parking lot 验证
- [ ] 全量生产启动
