# ZAP-AI4S: four-task visual laboratory reproduction

**New experiments:** use [LAB_GUIDE.md](LAB_GUIDE.md) for `pick`, `transport`,
`insert`, `unscrew`, and held-out `rack_transfer`. The new `tube_agent.lab` flow
uses two RGB-D views, native SAM 3.1, Qwen3.5-4B, fixed-seed sampling, model-prefix
search, multimodal self-success SFT and online GRPO with staged rewards.
SAM weights download anonymously from ModelScope with checksum verification.
See [LAB_VALIDATION.md](LAB_VALIDATION.md) for tested scope and pending H100 checks.
This is a scaled method reproduction, not a claim to the paper's benchmark scores.

**Latest server result:** the v3 fixed-scene pick probe completed with 4/9
successes and no training. One supplied success was independently replayed with
all states matching. See [LAB_GUIDE.md](LAB_GUIDE.md) for the new Git checkout,
shared-runtime links, and `--success-source` reuse with mandatory replay checks.

**Four-H100 automatic experiment:** after applying the offline auto-pipeline
patch, run `bash scripts/start_lab_all.sh runs/lab_success_first/fixed_seed0_v3`
on the GPU instance. SAM runs on GPU 1; GPUs 0, 2, 3 sample and later train one
shared policy. **No optimizer updates before success admission:** each task
needs 10 unique, physically replay-verified visual successes and at least 3/10
fresh independent successes. Zero-success acquisition stops for diagnosis.
The v1 zero-success warm-up/RL fallback has been removed. Use a new output path.
See [LAB_GUIDE.md](LAB_GUIDE.md) for exact criteria, artifacts and commands.

## Legacy two-task oracle-state baseline

The following instructions describe the earlier `tube_agent.run` baseline.
They do not describe the new visual pipeline above.

用 **Qwen3.5-4B → 简单 JSON 动作 → MuJoCo Panda 机械臂**，完成两个实验室基础操作。
核心只有仿真、模型接口、运行记录和自采成功轨迹微调；不依赖 Open-RUA 的服务调度、MCP、SAM、动作模型或多机器人框架。

本地 Python 3.13.5 已通过 48 项测试；规则策略的常规拿起、搬运检查均为 20/20。
这些不是 Qwen 的成绩，完整验证范围和示例视频见 [VALIDATION.md](VALIDATION.md)。

## 当前实现范围

| 项目 | 实现 |
|---|---|
| `pick` | 夹住红色试管，使整个管身离开原支撑面至少 6 cm，并持续至少 0.35 秒；最终仍须保持 |
| `transport` | 先完成拿起，再将整根试管放入青色托盘，张开夹爪、撤离，稳定至少 0.4 秒 |
| 动作 | `move(xyz, yaw)`、`gripper(opening)`、`finish` |
| 物理 | 真实 Panda 关节、逆运动学、关节伺服、重力、自由刚体试管与摩擦接触；没有吸附或焊接试管 |
| 模型输入 | 首轮默认使用仿真器提供的物体位置、轴向和机器人状态，明确标为 **oracle state** |
| 评测 | 固定场景种子、独立成功判定、动作记录、成功率及 Wilson 区间，可选视频 |
| 训练 | 模型自行采样成功轨迹 → 去重和过滤 → 文本状态 LoRA SFT → 相同测试集复测 |

**边界：** 第一版试管是带颜色和盖子外观的实心胶囊刚体，初始横放，夹爪保持朝下，只开放 XYZ 和 yaw。托盘是平面目标区域。暂未实现插孔、拧盖、液体、视觉定位训练、真实机器人迁移、GRPO 预热、前缀树搜索和正式 GRPO。
加入 `--vision` 会附加 RGB 图片，但仍保留 oracle state，不能当成 RGB-only 实验。当前 SFT 导出器只接收纯文本状态轨迹。

## 1. 在联网 CPU 实例准备

### 下载不稳定时：上传离线包（推荐本服务器使用）

离线包 `ZAP-AI4S-offline-linux-py311-20261106-v2.tar.gz` 包含完整代码、机器人资产和 **42 个 Linux/Python 3.11 安装包**。Torch/CUDA 从已确认存在的 DiG 环境本地复制，模型权重使用原共享目录。服务器不需要 `git pull` 或访问任何包下载站。

将压缩包上传到 `wangningchong` 目录后，在 CPU 实例执行：

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong &&
tar -xzf ZAP-AI4S-offline-linux-py311-20261106-v2.tar.gz &&
cd ZAP-AI4S-offline-20261106 &&
bash scripts/prepare_offline.sh
```

入口先核对所有代码和安装包的 SHA256，再创建独立环境、复制 Torch/CUDA，使用 `--no-index --require-hashes` 从包内文件安装，最后检查依赖和运行测试。环境要求为 Linux x86_64、Python 3.11、glibc ≥ 2.28；脚本会在开始复制前检查。Git 下载失败后，逐行粘贴的下一条命令仍可能运行旧代码；离线包包含同一提交的全部代码，不需要更新原仓库。

GPU 实例也进入新目录 `ZAP-AI4S-offline-20261106`，然后沿用后文的环境激活和实验命令。后文示例中的 `ZAP-AI4S` 路径需替换为这个新目录。

已有首版离线包、且安装末尾只有 `test_dependency_copy_is_independent_and_excludes_unrelated_packages` 失败时，可上传小补丁 `ZAP-AI4S-offline-cache-fix.tar.gz` 到 `wangningchong`，在该目录解压后重跑准备脚本。补丁含修复文件和更新的校验清单，保留已安装环境与 42 个安装包。修复会在本地复制完成后刷新 Python 的包发现缓存，避免目录时间戳未变化时读到旧的空列表。

### 网络正常时：从 GitHub 准备

在共享目录克隆，使 CPU 和 GPU 实例看到**同一个绝对路径**：

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong
git clone https://github.com/Vannico233/ZAP-AI4S.git
cd ZAP-AI4S
export TUBE_PYTHON_BIN=/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/.runtime/py311/bin/python
export TUBE_NO_PROXY=1
export TUBE_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/prepare_cpu.sh
```

脚本使用共享目录中已有的 **Python 3.11** 创建项目专用环境 `.venv-linux`，不再下载 uv 或 Python，也不修改 DiG 环境的依赖。未设置 `TUBE_PYTHON_BIN` 时使用当前终端的 `python`；已有可用的 `.venv-linux` 则直接复用。接受 Python 3.10–3.13，GPU 实例系统自带的 Python 3.13.5 不限制项目环境的版本。

**默认从本地复用 PyTorch。** 用户已确认 DiG 中有 `torch 2.10.0+cu128`、`torchvision 0.25.0+cu128`，与本项目需要的版本完全一致。脚本检查版本、依赖和 CPU 算子后，将这两个包及其依赖复制到项目环境，不联网下载 PyTorch、CUDA 包。复制使用独立文件，不使用硬链接，也不引入 DiG 的其他训练框架或可选加速扩展；会先检查剩余磁盘空间。重复运行时，已通过检查的完整 Torch 环境会直接复用。

随后在项目环境补装 Transformers 5.3.0 等所需依赖，并用 `.venv-linux/torch-constraints.txt` 固定已经复用的 Torch/CUDA 版本。DiG 的 Transformers 5.22.1 不会被修改。没有 vLLM / FlashAttention / FLA 等编译依赖；先用 PyTorch 自带实现，吞吐量并非最优。

虚拟环境仍依赖基础 Python 的文件，因此**基础解释器和项目都必须在 CPU/GPU 共享挂载中**，且两端绝对路径相同。用户确认的基础解释器是 `DiG/.runtime/py311`；GPU 实验完成前请保留该目录。脚本会检查基础解释器和项目是否都位于共享根目录下，默认共享根目录为项目的父目录（这里是 `wangningchong`）；其他布局可设置 `TUBE_SHARED_ROOT`。这个路径检查不能替代 GPU 上的实际启动检查。

上面的 `TUBE_NO_PROXY=1` 针对已观察到的代理断连：只在安装脚本及其子进程内清除代理变量、绕过 pip 配置文件，不修改终端或系统的代理设置。示例将剩余依赖从清华镜像直连下载；如果服务器只能通过代理联网，请不设置 `TUBE_NO_PROXY`，并使用管理员提供的可用代理。镜像是否可达仍需在服务器上验证。

未指定 `TUBE_PYPI_INDEX_URL` 时使用 `https://pypi.org/simple`。脚本直接使用现有 pip，不强制升级；依赖下载出现哈希不符、超时或连接中断时最多尝试三轮，后两轮绕过 pip 缓存重新下载，保留 HTTPS 和哈希校验。版本不存在等非下载错误会立即停止。剩余依赖的完整输出保存在 `.bootstrap-linux/pip-packages.log`。

其他机器若没有现成的对应 Torch 环境，可以显式设置 `TUBE_TORCH_MODE=download` 从 CUDA 12.8 官方源下载；这不是本服务器的推荐路径。该模式的 PyTorch 源可用 `TUBE_TORCH_INDEX_URL` 指定。重新下载不能保证修复持续的网络问题；pip 24 的这种重试不是大文件断点续传。

如果旧版在 uv、pip 或 Torch/CUDA 下载阶段失败，现有项目环境可以保留，更新并执行：

```bash
git pull --ff-only && TUBE_NO_PROXY=1 TUBE_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/prepare_cpu.sh
```

GPU 端仍使用 `source .venv-linux/bin/activate`；随后 `python --version` 应显示项目环境的版本（新建时为 3.11.18），无需安装到 GPU 的系统 Python 中。PyTorch 和模型依赖版本保持不变。

模型默认位置：

```text
/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model
```

该目录可直接包含 `config.json` 与权重，也可只包含一个 Qwen3.5 模型子目录；有多个候选时会报错而不随意选择。修改路径使用：

```bash
export TUBE_MODEL_PATH=/实际的模型目录
bash scripts/prepare_cpu.sh
```

机器人资产已经随仓库提供，GPU 实例不需要再下载。CPU 实例没有 CUDA 设备也能安装 CUDA 版本 PyTorch。需要 CPU/GPU 实例使用兼容的 Linux 系统、相同共享挂载路径；不要把 Windows 虚拟环境复制到服务器。

## 2. 先检查环境，不调用大模型

```bash
source .venv-linux/bin/activate
python -m tube_agent.doctor --sim-only
python -m tube_agent.run --policy scripted --task pick --split test --episodes 20
python -m tube_agent.run --policy scripted --task transport --split test --episodes 20
```

`scripted` 是手写控制策略，只证明环境能完成任务，不是 Qwen 的成绩，也不会被导出成自采训练数据。MuJoCo 物理在 CPU 上运行；默认不创建图形窗口、不要求 GPU 渲染。

### 固定种子逐步调试人工搬运命令

无需调用模型，使用 `train/seed=0` 和可编辑的九条原始动作：

```bash
python -m tube_agent.manual --commands examples/manual_transport_seed0.json --output runs/manual_transport_seed0_v1
```

命令依次为靠近、下降、闭爪、抬升、横移、下降、松爪、撤离、提交。
每步输出执行反馈和独立物理判定；目录保存前后观测、原始动作、`result.json`、
`messages.json` 和用于录制的 `replay.jsonl`。调参数后使用新的输出目录，保留之前的尝试。
不自动补全动作或追加 `finish`；中断/未完成与物理判定分开记录，拒绝覆盖旧目录。

该方案以已有规则控制的几何参数为初值，固定场景首个完整方案即成功。
`examples/manual_transport_controller_v1.json` 冻结了跨场景生成命令的规则：
位置及方向取自各场景初始观测，高度和夹爪参数固定。一次性验证三个预定种子区间：

```bash
python scripts/validate_manual_transport.py --output runs/manual_transport_benchmark_v1
```

默认 validation 0–19、test 20–39、ood 20–39 各 20 个场景，全部结果均保留。
本地结果为 20/20、20/20、15/20，详见 `examples/manual_transport_validation.json`。
这是 `policy=manual` 的环境/控制可行性结果，不是 Qwen 成功率；这些轨迹不会进入现有模型自采 SFT 导出。
控制参数、物理、奖励和成功判定没有因评测结果而修改，RL 尚未开始。

有可用 OpenGL 的电脑可直接重放录制完整搬运：

```bash
python scripts/render_replays.py --data runs/manual_transport_seed0_v1/replay.jsonl --output runs/manual_transport_video_v1
```

视频会明确标注 `MANUAL CONTROL`，近景跟随试管，显示是否曾抬起及稳定放置时间。

## 3. GPU 实例：启动 Qwen

终端 A：

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S
source .venv-linux/bin/activate
python -m tube_agent.doctor --gpu
CUDA_VISIBLE_DEVICES=0 bash scripts/start_model.sh
```

模型只从本地加载，服务只监听 `127.0.0.1:8000`。终端显示 `Ready` 后，在同一 GPU 实例的终端 B：

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S
source .venv-linux/bin/activate
python -m tube_agent.run --policy model --task pick --split test --episodes 20 --output runs/base
python -m tube_agent.run --policy model --task transport --split test --episodes 20 --output runs/base
```

这是明早最先需要的原模型基线。默认温度 0、每任务最多 20 个动作、每次生成最多 192 token，关闭长思考。模型没有一键抓取工具，必须自己决定位置、朝向、夹爪开合和顺序。执行反馈不暴露成功判定或奖励。

### 人工单例示范辅助的固定场景试跑

在人工搬运已经跑通后，让模型参考该示例并自己逐步产生动作。终端 A 的原模型服务保持运行，
在终端 B 执行：

```bash
bash scripts/probe_transport_demo.sh
```

脚本自动进入项目并激活 `.venv-linux`，绕过代理检查服务必须 `ready=true`、`adapter=null`。
随后使用 `manual_demo_v1`、`train/seed=0`、温度 0.8，采样 8 次，每次仍最多 20 个动作、每步最多 192 token。
只汇总本次唯一目录 `runs/transport_demo_probe/<时间戳_PID>/` 的结果与首个成功/失败动作；不启动训练或自动导出数据。

示例文件 `examples/manual_transport_demo_v1.json` 来自已验证成功的人工 `train/seed=0` 轨迹，
包含初始可见观测和九条原始动作。它只进入系统提示，既不自动执行，也不作为 assistant 训练标签。
当前动作仍取自模型输出，原 `basic_v1` 反馈、物理、判定和动作预算不变。
config、result 和导出数据会记录人工示范来源、内容 SHA256 和完整提示 SHA256。

此条件应称为“人工示例辅助的模型评测”，不属于零示范自学习或 RL 收益。
同一场景反复尝试只验证示例跟随与执行链；跑通后再从 `validation seed=20` 起检查新场景。
后续如训练这些模型轨迹，需保留示范条件，不能与无示范条件混作同一个实验。

### 几何提示与反馈修订实验

首轮服务器基线中，拿起任务目标恰好增加 6 cm，实际跟踪误差使物体抬升不足；
搬运任务出现直接把托盘表面高度当作 TCP 高度、反复请求越界位置的情况。
`--prompt-variant geometry_v1` 是可选的人工提示增强：说明 TCP/物体/支撑面的区别、
执行误差裕度和先抬升再搬运，并让越界反馈指出具体字段和边界。
动作仍由模型产生，越界仍拒绝，不自动修正坐标；物理、动作预算、奖励和成功标准不变。
默认 `base` 保留原提示与原通用拒绝反馈。此实验尚未经完整 Qwen 权重验证，不能称为训练收益。

模型服务已启动时，先在验证集各做 5 次，输出与原基线分开保存：

```bash
python -m tube_agent.run --policy model --prompt-variant geometry_v1 --task pick --split validation --episodes 5 --output runs/geometry_v1
python -m tube_agent.run --policy model --prompt-variant geometry_v1 --task transport --split validation --episodes 5 --output runs/geometry_v1
```

此选项同时改变提示与拒绝反馈，应将结果称为“几何提示与反馈修订后的基线”。
`config.json`、每轮 `result.json` 和导出训练数据会记录提示版本、提示 SHA256 和反馈版本。
旧结果保留；后续正式比较用尚未查看的测试种子，例如两种方法都使用 `--seed-start 20 --episodes 20`。
若随后采集训练数据或评测微调模型，需明确使用同一个提示版本，避免把提示变化误算为训练收益。

## 4. 自采成功轨迹，再微调

在模型服务仍运行时采样训练场景；每个场景尝试 8 次：

```bash
python -m tube_agent.run --policy model --task pick --split train --episodes 40 --attempts 8 --temperature 0.8 --output runs/collect
python -m tube_agent.run --policy model --task transport --split train --episodes 40 --attempts 8 --temperature 0.8 --output runs/collect
python -m tube_agent.export_sft runs/collect --output data/self_success.jsonl
```

只保留 `train`、`model`、真正成功且没有无效动作的轨迹。不会把测试集、规则策略或失败记录混入数据；输出文件已存在时拒绝覆盖。**如果没有成功轨迹，导出会明确失败**，此时先检查失败录像/动作，而不是用手写轨迹冒充自采成功。

回到终端 A 用 Ctrl+C 结束模型服务，释放显存，再训练：

```bash
bash scripts/train_sft.sh --data data/self_success.jsonl --output checkpoints/tube-lora
```

默认使用 4 张卡；每卡一份模型、轻量 LoRA 更新，只训练语言模型中的线性层，不训练视觉编码器。损失只覆盖当前模型动作，观测和之前的动作都被屏蔽。过长样本会计数并跳过。单卡运行可设置 `CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1`。

训练结束后启动微调模型，重复**相同**测试种子：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/start_model.sh --adapter checkpoints/tube-lora
# 在另一个终端中：
python -m tube_agent.run --policy model --task pick --split test --episodes 20 --output runs/sft
python -m tube_agent.run --policy model --task transport --split test --episodes 20 --output runs/sft
```

## 5. 数据划分与结果

- `train`、`validation`、`test` 使用不相交的种子区间，来自同一摆放分布。
- `ood` 扩大位置和 yaw 范围，改变试管尺寸，并加入蓝色干扰试管。这是参数和场景分布变化，**不是组合泛化**。
- `--attempts` 的每次尝试都计入分母；汇总是逐次尝试成功率，不是 best-of-N。
- 所有方法使用相同动作预算、观察信息和判定器。

每次运行创建新的目录，包含：

```text
config.json                    # 设置与关键依赖版本
summary.json                   # 成功率、动作数、无效动作、错误数
seed_00000_try_00/
  trajectory.jsonl             # 逐步观测、动作与执行反馈
  messages.json                # 原始模型对话；用于自采 SFT
  result.json                  # 独立环境判定与最终状态
  episode.mp4                  # 仅 --video 时生成
```

接口超时、模型服务错误等会计为失败并记录，不会偷偷换成规则策略。任何运行错误会令命令以非零状态退出。`finish` 只表示模型提交，不代表成功。

## 可选视频与视觉输入

H100 上先做不渲染的状态实验。视频需要可用的 OpenGL 后端，不应假定 H100 能提供桌面图形渲染。安装了系统 OSMesa 库的 Linux 实例可使用软件渲染：

```bash
MUJOCO_GL=osmesa python -m tube_agent.run --policy scripted --task transport --episodes 1 --video
```

如环境已具备有效 EGL 渲染，也可改成 `MUJOCO_GL=egl`。`doctor` 会显示是否能找到 OSMesa。没有图形库时仍可完成全部默认评测和文本状态训练。

### 把已保存的成功轨迹录成视频

在已有可用 OpenGL 渲染的电脑上运行；不需要加载模型或重新采样：

```bash
python scripts/render_replays.py --data data/base_pick_train_360_success.jsonl --output runs/success_videos
```

脚本按原场景种子逐条执行保存的模型动作，生成全景/抓取近景双视角 MP4、九宫格合集、
重放结果 JSON 和 ZIP。原物理与判定不变；报告核对每次动作前观测是否与原记录一致。
画面使用英文标签，标明重放性质、动作、实际抬升和接触状态；播放时间是仿真时间，
不包含原推理等待。首尾停留只重复画面，不额外推进物理。
输出文件已存在时拒绝覆盖。选中成功样本的重放通过率不代表模型重新评测的成功率。

## 代码位置

| 文件 | 用途 |
|---|---|
| `tube_agent/env.py` | 场景、Panda 控制、动作检查、独立判定和奖励 |
| `tube_agent/policy.py` | 三个动作的提示词、模型 HTTP 调用、规则检查策略 |
| `tube_agent/run.py` | 同一套任务循环、评测和记录 |
| `tube_agent/serve.py` | 直接用 Transformers 加载本地 Qwen，可加载 LoRA |
| `tube_agent/export_sft.py` / `train_sft.py` | 自采成功数据筛选与轻量微调 |

来源、资产许可和相对原论文的差异见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库提供初步任务复现；不宣称复现 ZAP 论文的成功率。
