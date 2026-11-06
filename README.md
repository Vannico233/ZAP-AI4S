# ZAP-AI4S：试管拿取与搬运

用 **Qwen3.5-4B → 简单 JSON 动作 → MuJoCo Panda 机械臂**，完成两个实验室基础操作。
核心只有仿真、模型接口、运行记录和自采成功轨迹微调；不依赖 Open-RUA 的服务调度、MCP、SAM、动作模型或多机器人框架。

本地 Python 3.13.5 已通过 34 项测试；规则策略的常规拿起、搬运检查均为 20/20。
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

在共享目录克隆，使 CPU 和 GPU 实例看到**同一个绝对路径**：

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong
git clone https://github.com/Vannico233/ZAP-AI4S.git
cd ZAP-AI4S
bash scripts/prepare_cpu.sh
```

脚本把独立的 **Python 3.13.5** 下载到共享目录 `.python-linux`，再创建 `.venv-linux`，保留系统 Python。这样不会因为 CPU/GPU 镜像的系统 Python 路径不同而丢失解释器。安装 PyTorch 2.10.0 / torchvision 0.25.0 的 **CUDA 12.8** 版本，以及 Transformers 5.3.0。没有 vLLM / FlashAttention / FLA 等编译依赖；先用 PyTorch 自带实现，吞吐量并非最优。

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

## 代码位置

| 文件 | 用途 |
|---|---|
| `tube_agent/env.py` | 场景、Panda 控制、动作检查、独立判定和奖励 |
| `tube_agent/policy.py` | 三个动作的提示词、模型 HTTP 调用、规则检查策略 |
| `tube_agent/run.py` | 同一套任务循环、评测和记录 |
| `tube_agent/serve.py` | 直接用 Transformers 加载本地 Qwen，可加载 LoRA |
| `tube_agent/export_sft.py` / `train_sft.py` | 自采成功数据筛选与轻量微调 |

来源、资产许可和相对原论文的差异见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库提供初步任务复现；不宣称复现 ZAP 论文的成功率。
