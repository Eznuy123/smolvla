# aloha_sim_vla

用 uv 管理 Python 环境，在 MuJoCo 的 ALOHA 双臂环境里微调和运行 SmolVLA。
默认任务是把方块从右臂转移到左臂；训练使用官方 LeRobot 演示数据集。

## 模型与训练方式

本项目使用 **SmolVLA**，基座是 [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)，约 4.5 亿参数。
视觉语言主干为 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，动作专家通过流匹配生成连续动作序列。

| 项目 | 当前配置 |
|---|---|
| 任务 | 右臂抓起方块并转交左臂 |
| 图像输入 | 一路顶视 RGB 图像 |
| 状态输入 | 14 维：每臂 6 个关节位置和 1 个夹爪位置 |
| 语言输入 | `Pick up the cube with the right arm and transfer it to the left arm.` |
| 动作输出 | 14 维绝对关节与夹爪目标 |
| 预测长度 | `chunk_size=50` |
| 连续执行长度 | `n_action_steps=10`，随后根据新观测重新预测 |
| 仿真控制频率 | 50 Hz；每 0.2 秒仿真时间重新预测一次 |
| 演示数据 | 50 条 episode，共 20,000 帧 |

当前训练属于**后训练中的监督微调（SFT）**；从机器人学习角度看，是使用专家演示进行的**模仿学习／行为克隆（BC）**。
流程为：预训练 SmolVLA → ALOHA 专家演示微调 → 保存 checkpoint → MuJoCo 推理与评测。

- 训练样本包含图像、关节状态、语言指令及对应的专家动作序列。
- 模型通过**流匹配损失**学习动作生成；不是简单的动作均方误差回归。
- 默认冻结视觉编码器和 VLM 主干，主要更新动作专家与状态投影。
- 当前没有进行强化学习。仿真 `reward` 用于观察和评测，不参与模型参数更新。
- “在线推理”指根据正在运行的仿真观测选择动作，运行时不会继续训练或更新模型权重。

已有 checkpoint 时，可以直接使用第 6 节的 MuJoCo 窗口在线推理命令。

## 1. 安装与检查

当前固定 Python 3.12、LeRobot 0.6.1、PyTorch 2.11.0 和 torchvision 0.26.0。
PyTorch 从 CUDA 12.8 官方索引安装。系统 NVIDIA 驱动需要支持该 CUDA 运行时；无需单独安装 CUDA Toolkit。

```bash
cd ~/aloha_sim_vla
uv sync --locked
uv run --locked python scripts/check_env.py --require-cuda
```

检查必须输出 GPU 型号，以及 `CUDA tensor / matmul: OK`。本机 RTX 5060 Laptop GPU 有约 8 GiB 显存，训练默认 batch size 为 1。
如果宿主机 `nvidia-smi` 正常、脚本却检测不到 CUDA，请检查当前终端、容器或沙箱是否能访问 GPU。

离屏渲染需要系统 EGL 库，缺少时安装：

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

离屏评测与冒烟测试默认使用 EGL；窗口在线推理 `infer.py` 默认使用 GLFW。
可在命令前设置 `MUJOCO_GL=glfw`（桌面显示）或 `MUJOCO_GL=osmesa`（离屏软件渲染，需系统 OSMesa 库）。

日常统一用 uv，无需手动激活 `.venv`：

```bash
uv add <包名>                  # 添加运行依赖，同时更新 pyproject.toml 和 uv.lock
uv add --dev <包名>            # 添加开发依赖
uv remove <包名>
uv sync --locked               # 按锁文件恢复环境
uv run --locked python <脚本>
```

不要用 `uv pip install` 代替 `uv add` 管理项目依赖，它不会更新项目的依赖声明。
脚本默认把模型缓存放在 `data/hf_cache/`，数据集缓存放在 `data/lerobot/`；可用 `HF_HOME`、`HF_LEROBOT_HOME` 覆盖。

## 2. 验证仿真

```bash
uv run --locked python scripts/test_sim.py --steps 200
uv run --locked python scripts/eval.py --policy random --episodes 1 --max-steps 100
```

冒烟测试输出 `state: (14,)`、`action_dim: 14` 和一张 `out/smoke_test/cam_high.png`。
随机评测输出 `out/eval/episode_000.mp4` 与 `results.json`。

## 3. 先跑两步训练

使用数据集 [lerobot/aloha_sim_transfer_cube_scripted](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted)
和基座 [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)。
首次运行需要联网下载模型权重、VLM 配置/分词器和数据集。基座权重为 GB 量级，请预留下载时间和磁盘空间。
这些是公开资源，无需上传自己的模型或数据；脚本默认关闭 W&B 和 Hub 上传。

```bash
uv run --locked python scripts/train.py \
  --steps 2 \
  --episodes 0 \
  --log-freq 1 \
  --save-freq 2 \
  --out out/train/smolvla_smoke
```

两步训练用于检查数据加载、前向、反向传播和 checkpoint 保存，不能据此判断任务成功率。
输出模型位于：

```text
out/train/smolvla_smoke/checkpoints/last/pretrained_model/
```

只查看实际传给 LeRobot 的参数，不下载、不训练：

```bash
uv run --locked python scripts/train.py --dry-run
```

## 4. 正式微调

以下训练由你在自己的终端启动。若输出目录已经存在，请通过 `--out` 换一个新目录，或使用下文的断点续训命令。

可选：先跑 2,000 步观察 loss 与仿真效果：

```bash
uv run --locked python scripts/train.py \
  --steps 2000 \
  --out out/train/smolvla_aloha_2k
```

训练日志保存到 `out/train/smolvla_aloha_2k.log`，可以用 `tail -f` 查看。
导出 loss、梯度范数、学习率和显存指标：

```bash
uv run --locked python scripts/summarize_train.py out/train/smolvla_aloha_2k.log
```

会生成同名 `.metrics.csv` 和 `.summary.json`，比较开头与结尾各十条日志的平均 loss。
然后按评测命令将 checkpoint 路径替换为 `out/train/smolvla_aloha_2k/checkpoints/last/pretrained_model`。

正式配置：使用执行长度 10、batch size 1 训练 20,000 步：

```bash
uv run --locked python scripts/train.py \
  --steps 20000 \
  --batch-size 1 \
  --n-action-steps 10 \
  --save-freq 5000 \
  --out out/train/smolvla_aloha_20k_a10
```

在另一个终端查看日志：

```bash
tail -f out/train/smolvla_aloha_20k_a10.log
```

训练完成后导出完整指标：

```bash
uv run --locked python scripts/summarize_train.py out/train/smolvla_aloha_20k_a10.log
```

每 5,000 步保存一次 checkpoint；最终模型和处理器保存在：

```text
out/train/smolvla_aloha_20k_a10/checkpoints/last/pretrained_model/
```

默认使用全部 50 条演示，冻结视觉编码器，训练动作专家和状态投影，启用 CUDA BF16 AMP。
数据加载默认 `num-workers=0`，减少这台 16 GB 内存机器的额外开销。显存允许时可增加 `--batch-size`；内存允许时再增加 `--num-workers`。
batch size 1 时，20,000 步约对应 20,000 个观测样本，即当前 20,000 帧数据集的一个 epoch；每个样本还包含未来 50 步的动作监督。
`--steps` 表示优化器更新次数，不是 episode 数，也不是新采集的仿真帧数。
loss 下降表示训练目标在改善，不能代替闭环抓取成功率；后者需要单独评测。

`smolvla_base` 原配置使用 6 维状态/动作及 camera1/2/3；训练脚本会从当前数据集重建特征定义，
使用 14 维状态/动作及 `observation.images.top`，并使用该数据集的统计量重新配置处理器。
保存的 checkpoint 包含推理所需的配置、权重与预处理/后处理参数。

其他数据集或已经下载的本地 LeRobot 数据：

```bash
uv run --locked python scripts/train.py \
  --dataset-repo-id <所有者/数据集名> \
  --dataset-root /绝对路径/数据集目录 \
  --out out/train/custom
```

默认仍按 ALOHA 任务设计；改变机器人、相机和动作语义后，需要相应修改推理适配。
已有输出目录不会被覆盖。断点续训使用同一 BF16 训练后端，训练参数会从 checkpoint 恢复：

```bash
uv run --locked python scripts/_train_backend.py \
  --config_path=out/train/smolvla_aloha_20k_a10/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

若增加总训练步数，可附加 `--steps=30000`。训练后端会继续使用相同的项目缓存目录。

### 本机训练记录（2026-10-15）

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，标称 8 GB 显存。
- 已保存的训练状态：20,000 步，batch size 1，执行长度 10。
- 日志记录的训练循环耗时约 40 分 44 秒，不含模型初始化时间。
- 最后一条训练日志的 loss 为 0.015；这是训练指标，不代表任务成功率。
- 已通过该 checkpoint 的 50 步 MuJoCo 窗口在线推理测试，完整任务表现请按第 5 节评测。

## 5. 加载 checkpoint 推理与评测

先用两步训练生成的模型检查端到端流程：

```bash
uv run --locked python scripts/eval.py \
  --policy smolvla \
  --checkpoint out/train/smolvla_smoke/checkpoints/last/pretrained_model \
  --device cuda \
  --prompt "Pick up the cube with the right arm and transfer it to the left arm." \
  --episodes 1 \
  --max-steps 10 \
  --out out/eval/smolvla_smoke
```

正式模型评测：

```bash
uv run --locked python scripts/eval.py \
  --policy smolvla \
  --checkpoint out/train/smolvla_aloha_20k_a10/checkpoints/last/pretrained_model \
  --prompt "Pick up the cube with the right arm and transfer it to the left arm." \
  --n-action-steps 10 \
  --episodes 10 \
  --max-steps 600 \
  --out out/eval/smolvla_aloha_20k_a10
```

语言指令应与训练数据一致，可从数据集的 `meta/tasks.parquet` 查看。
每条 episode 使用不同种子，输出 50 FPS MP4，并把成功率、最大奖励、步数和成功/超时标记保存到 `results.json`。
`--max-steps` 同时配置 Gym 时间上限，避免底层默认 300 步过早截断。

推理会加载 checkpoint 自带的归一化与反归一化处理器，调用 LeRobot 的 `select_action()` 管理动作队列。
默认预测长度 `chunk_size=50`，每次只执行前 `n_action_steps=10` 个动作后重新根据观测预测，即每 0.2 秒仿真时间重新规划。
训练和评测均可用 `--n-action-steps` 修改执行长度；评测时也可覆盖旧 checkpoint 的 50 步执行配置，模型权重保持原样。
评测的 `results.json` 会记录实际使用的执行长度和预测长度。
必须使用适配过 ALOHA 的 checkpoint；原始 6 维基座不能直接控制这套双臂。

## 6. MuJoCo 窗口在线推理

在本机桌面终端运行，默认加载本次训练完成的 20,000 步 checkpoint：

```bash
cd ~/aloha_sim_vla
MUJOCO_GL=glfw uv run --locked python scripts/infer.py \
  --checkpoint out/train/smolvla_aloha_20k_a10/checkpoints/last/pretrained_model \
  --n-action-steps 10
```

程序打开 MuJoCo 原生交互窗口，在同一套仿真物理状态上执行“相机观测 → 策略推理 → 关节控制”。
默认连续运行，每次成功或达到 600 步后自动换一个种子重置。
不需要重新训练，也不需要启动单独的策略服务器。

- **空格**：暂停或继续。
- **R**：重置当前种子的场景，同时清空策略动作队列。
- **N**：切换到下一个种子的初始场景。
- **Esc、关闭窗口或 Ctrl+C**：退出。
- 默认是可旋转、缩放的三维视角；`--camera top` 可切换到训练使用的顶视相机。
- `--speed 0.5` 放慢播放；`--episodes 3` 在完成 3 次后退出，默认一直运行。

例如，以顶视角慢速运行 3 次：

```bash
MUJOCO_GL=glfw uv run --locked python scripts/infer.py \
  --camera top \
  --speed 0.5 \
  --episodes 3 \
  --n-action-steps 10
```

默认控制步长仍为 0.02 秒、每 10 个动作重新预测一次。程序尝试按正常速度推进；
如果推理或渲染来不及，仿真会放慢，不跳过动作。终端的 `sim/wall` 表示仿真时间与实际时间的比值。

先检查窗口与仿真，而不加载模型：

```bash
MUJOCO_GL=glfw uv run --locked python scripts/infer.py --policy random --episodes 1 --max-steps 100
```

在线窗口要求可用的桌面显示会话。无显示器服务器上的推理评测继续使用 `scripts/eval.py` 保存视频。
录制评测视频和统计结果也使用 `eval.py`；`infer.py` 用于交互观察。

参考：[MuJoCo passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。

## 观测与动作

| 字段 | 形状 | 含义 |
|---|---|---|
| `images/cam_high` | `(480, 640, 3)` uint8 | 顶视 RGB 图像，推理映射为 `observation.images.top` |
| `state` | `(14,)` float32 | 每臂 6 个关节位置 + 1 个归一化夹爪位置 |
| `action` | `(14,)` float32 | 每臂 6 个绝对关节目标 + 1 个归一化夹爪目标 |

关节单位是弧度，夹爪 0 表示闭合、1 表示张开。动作不是关节增量。
本地 gym-aloha 0.1.5 的 Gym 观测只返回顶视相机；虽然 XML 定义了腕部相机，封装不会自动提供腕部图像。
随机策略只用于检查仿真与视频流程，不用于生成训练演示。

`collect_data.py` 目前只能用随机策略验证 NPZ 录制，脚本专家与 NPZ → LeRobot 转换尚未实现。
当前训练直接使用已有的 LeRobot 数据集。

任务奖励按阶段定义：

| reward | 含义 |
|---|---|
| 0 | 未达到后续接触条件 |
| 1 | 右夹爪接触方块 |
| 2 | 右夹爪接触方块，且方块未接触桌面 |
| 3 | 左夹爪接触方块 |
| 4 | 左夹爪接触方块，且方块未接触桌面；环境判定成功 |

这些条件反映瞬时接触状态。最大奖励为 2 不能证明已经稳定抓住方块，短暂带起或弹起也可能达到该条件。

## 脚本入口

| 脚本 | 用途 |
|---|---|
| `scripts/check_env.py` | 检查依赖、GPU 和 CUDA 运算 |
| `scripts/test_sim.py` | 仿真渲染与短程物理稳定性检查 |
| `scripts/train.py` | 启动 SmolVLA 监督微调，保存训练日志 |
| `scripts/_train_backend.py` | 配置 BF16 后进入 LeRobot 训练器；也用于断点续训 |
| `scripts/summarize_train.py` | 将日志中的训练指标导出为 CSV 和 JSON |
| `scripts/eval.py` | 离屏闭环评测，保存视频与成功率 |
| `scripts/infer.py` | MuJoCo 交互窗口在线推理 |
| `scripts/collect_data.py` | NPZ 录制流程验证，专家策略尚未实现 |

## 可选：远端 OpenPI

远端 OpenPI 客户端保留在 `policies/openpi_client.py`，需要另行用 `uv add` 添加兼容的 `openpi-client` 依赖，
并核对服务端图像键名与动作定义。它不参与上述本地 SmolVLA 训练和推理流程。

参考：[LeRobot SmolVLA 官方说明](https://huggingface.co/docs/lerobot/smolvla)。
