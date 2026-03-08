# VLM Robot Eval

`vlm_robot_eval` 是一个面向机器人控制场景的视觉语言模型（VLM）评测项目。
当前支持对以下模型进行统一评估：

- `SmolVLM-256M-Instruct`
- `Qwen2.5-VL-3B-Instruct`

项目基于 COCO 2017（`val2017 + annotations`）构建任务数据，评估模型在动作生成、目标识别与场景判断上的表现。

---

## 1. 项目结构

- `vlm_robot_eval/main.py`：主入口，负责数据准备、评估执行、结果校验
- `vlm_robot_eval/core/dataset_builder_v3.py`：v3 数据集构建
- `vlm_robot_eval/core/evaluator.py`：核心评估与绘图逻辑
- `vlm_robot_eval/models/smol_vlm.py`：SmolVLM 适配
- `vlm_robot_eval/models/qwen_vl.py`：Qwen2.5-VL 适配
- `vlm_robot_eval/results/`：评测结果输出目录

---

## 2. 环境依赖

Python 3.10+，安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 主要包含：`torch`、`transformers`、`accelerate`、`pycocotools`、`sentence-transformers`、`matplotlib` 等。

---

## 3. 数据集准备

默认按以下优先级寻找 COCO 根目录（需包含 `val2017/` 与 `annotations/`）：

1. `COCO_ROOT`
2. `COCO2017_ROOT`
3. `<repo>/coco2017`
4. `<repo>/vlm_robot_eval/coco2017`
5. `~/coco2017`

建议显式设置：

```bash
export COCO_ROOT=/home/jiang/hand-main/coco2017
```

> 说明：`coco2017` 数据集体积较大，默认不应提交到 Git 仓库。

---

## 4. 快速开始

### 4.1 只评估关注指标（推荐）

该模式仅输出以下指标与图：

- 空输出率（`empty_output_rate`）
- 非法动作率（`illegal_action_rate`）
- 目标识别精确率/召回率（`precision/recall`）
- 场景分类准确率（`scene_accuracy`）
- 场景有效性指标（`valid_scene_rate`、`valid_scene_accuracy`）

示例：

```bash
cd /home/jiang/hand-main

export COCO_ROOT=/home/jiang/hand-main/coco2017
export SBERT_MODEL_PATH=/home/jiang/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf
export SBERT_DEVICE=cpu
export HF_HUB_OFFLINE=1

# 显存/卸载策略（按机器调整）
export VLM_GPU_MEMORY_RATIO=0.90
export VLM_GPU_RESERVE_GB=1.0
export VLM_CPU_MAX_MEMORY=64GiB
export VLM_OFFLOAD_DIR=/tmp/vlm_offload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 模型推理参数
export SMOL_DEVICE_MAP=cpu
export SMOL_MAX_NEW_TOKENS=64
export QWEN_MAX_NEW_TOKENS=96
export STRICT_NO_FALLBACK=0

python3 -m vlm_robot_eval.main \
  --samples 60 \
  --repeat 3 \
  --run_name v3_eval_focus_metrics_only \
  --focus_metrics_only
```

结果默认输出到：

- `vlm_robot_eval/results/<run_name>/all_results.csv`
- `vlm_robot_eval/results/<run_name>/sample_metrics.csv`
- `vlm_robot_eval/results/<run_name>/plots/*.png`

### 4.2 全量评估

去掉 `--focus_metrics_only` 即可执行全量指标评估。

---

## 5. 常用参数

`python3 -m vlm_robot_eval.main [args]`

- `--coco_root`：COCO 根目录
- `--samples`：评估样本数（默认 `60`）
- `--repeat`：每样本重复推理次数（默认 `5`）
- `--run_name`：结果目录名
- `--enable_stress_test`：开启显存压力测试
- `--disable_qwen_deterministic`：关闭 Qwen 的确定性解码
- `--focus_metrics_only`：仅输出关注指标

---

## 6. 输出与校验

每次运行结束会自动执行结果校验（`[validate] PASS/FAIL`）。
当使用 `--focus_metrics_only` 时，校验会检查以下图表文件：

- `empty_output_rate.png`
- `illegal_action_rate.png`
- `precision_recall.png`
- `scene_accuracy.png`
- `scene_diagnostics.png`

---

## 7. 备注

- 若使用离线权重，请确保模型路径可访问（如 `SMOL_MODEL_PATH`、`QWEN_MODEL_PATH` 对应目录存在）。
- 若显存不足，建议：
  - `SMOL_DEVICE_MAP=cpu`
  - 降低 `SMOL_MAX_NEW_TOKENS` / `QWEN_MAX_NEW_TOKENS`
  - 增大 `VLM_GPU_RESERVE_GB`
