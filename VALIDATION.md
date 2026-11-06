# 本地验证记录

日期：2026-11-06。运行平台：Windows，Python 3.13.5。
关键版本：MuJoCo 3.3.7、NumPy 2.2.6、PyTorch 2.10.0+cpu、
torchvision 0.25.0+cpu、Transformers 5.3.0、PEFT 0.18.1。

## 已实际运行

- `python -m pytest -q`：**34 passed**。涵盖物理抓取与搬运、掉落、空提交、越界、
  无效动作预算、真实本机 HTTP 闭环、自采数据过滤、训练标签屏蔽等。
- 小型随机初始化 **Qwen3.5 架构**：LoRA 前向、反向、保存、重载和生成通过。
  这用于接口验证，不是 Qwen3.5-4B 权重的训练结果。
- 使用官方 Qwen3.5-4B 的真实分词器和处理配置，离线验证推理与训练的提示 token 一致：
  528 个提示 token，示例动作及结束符共 6 个 token，提示部分全部屏蔽训练损失。
- 三个 Linux shell 脚本通过 Bash 语法检查；Python 环境通过依赖一致性检查。
- 视频生成和静态图片渲染通过。

## 规则策略的环境检查

每组 20 个场景，种子 0–19，最多 20 个动作。数据由
`python scripts/validate_physics.py` 生成。

| 任务 | 测试分布 | 成功次数 | 成功率 |
|---|---|---:|---:|
| 拿起 | 常规 | 20/20 | 100% |
| 搬运 | 常规 | 20/20 | 100% |
| 拿起 | 扩大摆放/尺寸范围并加入干扰物 | 19/20 | 95% |
| 搬运 | 扩大摆放/尺寸范围并加入干扰物 | 15/20 | 75% |

以上全部是 **scripted 规则策略** 的环境检查，不是 Qwen 成绩。
逐次结果、置信区间和依赖版本保存在
[examples/physics_validation.json](examples/physics_validation.json)。
较难场景中的失败被保留，未剔除失败种子。

## 尚未在本机验证

- 用户的完整 Qwen3.5-4B 权重加载、基线评测和自采成功轨迹。
- H100 / Linux / CUDA 12.8 上的实际运行与 4 卡 SFT。
- Linux CPU 实例联网安装和 CPU → GPU 实例的共享环境迁移。
- GRPO：本轮尚未实现；当前交付是任务环境、评测和自采成功 SFT。

## 规则搬运示例

![拿起试管](examples/picked.png)

[完整规则搬运视频](examples/scripted_transport.mp4)
