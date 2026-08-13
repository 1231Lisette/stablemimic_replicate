# stablemimic_replicate

Unitree G1 + 公开 LAFAN1 重定向数据的 StableMimic 复现工程，依据
arXiv:2608.02385 构建。

## 当前状态

代码栈已经覆盖：

- 严格的 36 列 LAFAN1 CSV loader，8 个 `dance` 与 6 个 `fallAndGetUp` 文件分库；
- 将 102--168 秒、包含重复事件的 recovery 文件切为原子“持续倒地→持续直立”片段，
  每个片段拥有自己的 terminal reference；
- 30 FPS reference 在 50 Hz policy clock 上的线性插值、四元数 SLERP 和速度派生；
- Isaac Lab G1 29 个 body joint 到运行时 43 joint articulation 的显式映射；
- Tracking / Recovery / 1.5 秒 Transition 三阶段状态机与水平 reference realignment；
- Tracking 中达到论文 fall criterion 后，用高度、projected gravity 和 29 关节姿态匹配
  最近 recovery 帧并进入 Recovery 监督，不重置物理状态；
- 论文对应的六类 whole-body Gaussian tracking error；
- 4 帧 Actor / Gate / Critic history，维度严格为 884 / 372 / 1428；
- 两个 512-256-128 ELU Expert、proprioceptive softmax Gate、29-D soft-fused mean；
- shared learned scalar policy std、privileged Critic、normalization；
- rollout storage、GAE、PPO、adaptive learning rate 与 Gate/transition auxiliary supervision；
- tracking-only、recovery-only、50/50 joint training、续训与 checkpoint；
- 确定性 Isaac 评估、论文形式的 100 次 matched push schedule；
- 单一 deployable Actor ONNX 导出、metadata 与 PyTorch/ONNX 数值对齐检查。

代码完整不等于已经训练完成。目前仓库不包含收敛权重，也不宣称已经达到论文指标。
正式长训练应在下述 smoke test 全部通过后再开始。

本次代码完整性验收记录见
[`docs/CODE_COMPLETE_VALIDATION.md`](docs/CODE_COMPLETE_VALIDATION.md)。
历史错误、修正依据和全部已运行实验见
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)。

## 仓库结构

```text
stablemimic_replicate/
├── configs/
│   ├── stablemimic_g1.yaml       # 历史 v5 工程配置与实验证据
│   ├── stablemimic_g1_gmr_single_baseline.yaml # 第一版干净 NPZ 基线
│   ├── stablemimic_g1_upstream_v7.yaml # 官方参考实现的可归因奖励 A/B
│   └── motion/lafan1_g1.yaml     # LAFAN1/G1 数据约定
├── src/stablemimic/
│   ├── config.py                 # YAML → 强类型配置及合法性检查
│   ├── motion/                   # CSV loader、原子 recovery 切片、GPU 插值/采样
│   ├── core/                     # 四元数、observation/history、phase 状态机
│   ├── envs/                     # Isaac Lab G1 环境、reset/reward/fall curriculum
│   ├── rewards/                  # 六类 whole-body tracking kernel
│   ├── models/                   # Motion Expert、Recovery Expert、Gate、Critic
│   ├── rl/                       # rollout、GAE、PPO、checkpoint、训练指标
│   ├── eval/                     # matched-push protocol
│   ├── export/                   # ONNX 导出边界
│   └── sim/                      # G1 29→43 joint mapping、关闭 watchdog
├── scripts/
│   ├── audit_lafan1.py           # 数据/原子 recovery clip 审计
│   ├── retarget_lafan1_gmr.py    # raw BVH→GMR G1-29DoF→36列 CSV + QA
│   ├── train_stablemimic.py      # tracking/recovery/joint 训练与续训
│   ├── evaluate_stablemimic.py   # Isaac 确定性/推倒评估
│   ├── evaluate_mujoco.py        # MuJoCo adapter 边界
│   ├── export_onnx.py            # 导出 deployable Actor
│   ├── verify_onnx.py            # PyTorch/ONNX 数值一致性
│   └── visualize_lafan1_g1.py    # reference 可视化
├── tests/                        # loader、切片、phase、PPO、评估协议回归测试
└── docs/                         # 验收、视觉证据、实验记录
```

## 论文事实与复现选择

论文明确公布了网络宽度、历史长度、PPO 主要参数、50 Hz、4096 env、20 秒
horizon、50/50 reset、1.5 秒 transition 和 auxiliary loss 系数，但没有公开训练
simulator、逐元素 observation schema、PD/action 参数、六个 kernel 的数值以及完整
randomization。

本项目选择 Isaac Lab 作为训练 simulator。所有缺失参数集中在
[`configs/stablemimic_g1.yaml`](configs/stablemimic_g1.yaml)，并标注为
`reproduction choice`，不能当作论文原始参数引用。

实现严格用执行后的 `s_(t+1)` 对齐 hidden reference successor `k+1` 计算奖励。
Gate CE 系数为 `0.1`，transition 样本权重为 `4`；相邻稳态样本使用 `0.01`
routing consistency，transition 样本使用 `0.01` expert-output alignment。论文包含
`r_success` 但没有公布其系数，因此配置采用 `reward.success_bonus: 1.0`。Recovery reset
若初始高度不低于 `0.8 * command_height`，仅把 Gate 监督改为 Tracking；它仍保留
Recovery reset、Recovery reward 与 privileged recovery reference，避免改变训练分布。

Recovery 的三个相似度用途已经明确分开，不能互相替代：

- `active reference similarity`：当前状态对正在播放的 recovery reference successor 的
  六类 whole-body Gaussian reward 加权归一值，越接近 `1` 表示越像当前目标帧；它用于
  imitation reward，并以 `recovery_failure_similarity_threshold: 0.05` 判断是否连续严重
  偏离目标，持续 2 秒才判 Recovery 失败；
- `terminal reference similarity`：当前状态与所采样 get-up 序列最后一帧的相似度；达到
  `recovery_terminal_similarity_threshold: 0.70` 才进入 1.5 秒 Transition；
- `tracking_resumption_similarity_threshold: 0.70`：只用于确定性评估中的“恢复后重新跟踪”
  统计，不参与训练状态机。

论文公开了“持续 active-target error”和“terminal get-up reference match”这两个不同事件，
但没有公开上述阈值，因此 `0.05 / 0.70 / 0.70` 均为可配置的 reproduction choice。
策略 log-probability 仍对 29 个动作维度求和；论文给出的 entropy 系数 `0.05` 则作用在
每动作维平均 entropy 上，避免相同系数被动作维数额外放大 29 倍。论文没有公布 entropy
的维度 reduction，因此这一 reduction 也明确属于 reproduction choice。

公开 recovery CSV 不是六条 get-up trajectory，而是六段包含多次倒地/起身的长录像。
`recovery_segmentation` 用论文 fall threshold 与显式的直立/持续时间阈值切片，并丢弃超过
20 秒 episode horizon 的片段。切片和最近帧匹配阈值均是公开代码缺失后的复现选择。
Actor/Gate 看不到所匹配的 motion id、frame、phase 或 hidden recovery successor。

## 信息边界

| 信息 | Expert | Gate | Critic | Reset/Reward | 部署 |
|---|---:|---:|---:|---:|---:|
| 正常 motion command | 是 | 否 | 是 | 是 | 是 |
| noisy deployable proprioception | 是 | 是 | 是 | 否 | 是 |
| uncorrupted proprioception | 否 | 否 | 是 | 是 | 否 |
| recovery sequence/frame/successor | 否 | 否 | 是 | 是 | 否 |
| phase/gate supervision label | 否 | 否 | 否 | loss only | 否 |

每帧 Gate observation 为 `3+3+29+29+29=93`。Actor 在此基础上加入
128-D 正常运动命令得到 221-D。Critic 再加入 93-D uncorrupted proprioception
和 43-D hidden recovery successor 得到 357-D。四帧历史分别为
`372 / 884 / 1428`。Recovery reference 从不进入 Expert、Gate 或 ONNX。

## 服务器目录

```text
代码: /root/gpufree-share/stablemimic_replicate
数据: /root/gpufree-data/stablemimic_replicate
旧 CSV : /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1
GMR CSV: /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected/csv
GMR NPZ: /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected/npz
运行输出: /root/gpufree-data/stablemimic_replicate/runs
```

首次部署：

```bash
cd /root/gpufree-share
git clone https://github.com/1231Lisette/stablemimic_replicate.git
mkdir -p /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1
mkdir -p /root/gpufree-data/stablemimic_replicate/runs
```

把官方 [LAFAN1 Retargeting Dataset 的 G1 CSV](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1)
中的 8 个 `dance*.csv` 和 6 个 `fallAndGetUp*.csv` 放入上述 CSV 目录。仓库不复制大数据或
checkpoint；数据和运行输出全部留在 `gpu_data`。服务器镜像需要已有
`/workspace/isaaclab/isaaclab.sh` 与 `/isaac-sim/python.sh`。

以后更新代码：

```bash
cd /root/gpufree-share/stablemimic_replicate
git pull --ff-only origin main
```

请始终从代码目录运行：

```bash
cd /root/gpufree-share/stablemimic_replicate
export PYTHONPATH=src
```

## 0. 从 raw LAFAN1 用 GMR 重定向（训练前必做）

旧的公开 G1 CSV 保留用于 A/B，不覆盖。正式 reference 从 Ubisoft raw LAFAN1 BVH 使用
冻结的官方 [YanjieZe/GMR](https://github.com/YanjieZe/GMR) `unitree_g1` 29-DoF 配置生成。
当前冻结 revision 为 `bb1bbe40774794fceb2a7c579a3464a28e68c844`。

该 revision 的上游 `bvh_to_robot_dataset.py` 与同 revision 库接口不一致：脚本导入不存在的
`load_lafan1_file`，并使用错误的 `src_human="bvh"` key。仓库 adapter 不修改冻结上游，
而是调用实际存在的 `load_bvh_file(..., format="lafan1")` 与
`src_human="bvh_lafan1"`，逐项验证关节顺序并输出 pickle、36 列 CSV 和 manifest：

```bash
env PYTHONPATH=src:/root/gpufree-data/stablemimic_replicate/tools/gmr_py311 \
  /workspace/isaaclab/isaaclab.sh -p scripts/retarget_lafan1_gmr.py \
  --src-folder /root/gpufree-data/stablemimic_replicate/datasets/lafan1_raw/bvh \
  --output-root /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected \
  --gmr-root /root/gpufree-data/stablemimic_replicate/tools/GMR-bb1bbe4 \
  --gmr-revision bb1bbe40774794fceb2a7c579a3464a28e68c844 \
  --joint-velocity-limit 9.42477796076938 \
  --ground-clearance 0.002 \
  --ground-offset-speed-limit 0.5
```

默认只选择 8 个 `dance*.bvh` 与 6 个 `fallAndGetUp*.bvh`。后处理使用双向对称 rate
projection 约束真正的 IK 帧跳；随后按 MuJoCo floor contact distance 生成平滑、非穿透的
root-Z offset。它不会修改 root XY、root quaternion 或冻结的 GMR checkout。

生成后必须检查 `manifest.json`，14 条动作均须满足：

- `velocity_elements_over_limit == 0`；
- `maximum_floor_penetration_m == 0`；
- quaternion norm 接近 1，29 关节顺序与 CSV loader 完全一致；
- 视觉检查没有明显悬空、肢体畸变或左右映射错误。

任何一项失败都不得切换训练配置或启动 PPO。

### 后台运行与第二天检查

先确认没有同名任务，避免两个进程同时写一个输出目录：

```bash
pgrep -af "scripts/retarget_lafan1_gmr.py" || true
```

没有输出时，才可从头在后台生成。`--overwrite` 表示一致地重建这个专用生成目录；不会覆盖
旧的 `/datasets/lafan1/g1`：

```bash
cd /root/gpufree-share/stablemimic_replicate
OUT=/root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected
mkdir -p "$OUT"

nohup env PYTHONPATH=src:/root/gpufree-data/stablemimic_replicate/tools/gmr_py311 \
  /workspace/isaaclab/isaaclab.sh -p scripts/retarget_lafan1_gmr.py \
  --src-folder /root/gpufree-data/stablemimic_replicate/datasets/lafan1_raw/bvh \
  --output-root "$OUT" \
  --gmr-root /root/gpufree-data/stablemimic_replicate/tools/GMR-bb1bbe4 \
  --gmr-revision bb1bbe40774794fceb2a7c579a3464a28e68c844 \
  --joint-velocity-limit 9.42477796076938 \
  --ground-clearance 0.002 \
  --ground-offset-speed-limit 0.5 \
  --overwrite > "$OUT/retarget.log" 2>&1 < /dev/null &

echo $! > "$OUT/retarget.pid"
echo "background PID=$(cat "$OUT/retarget.pid")"
```

第二天查看状态和已完成数量：

```bash
OUT=/root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected
PID=$(cat "$OUT/retarget.pid")
ps -p "$PID" -o pid,etime,stat,pcpu,pmem,cmd
tail -n 30 "$OUT/retarget.log"
find "$OUT/csv" -maxdepth 1 -name "*.csv" -type f | wc -l
```

正常完成时，进程已经退出、CSV 数量为 14、日志末尾出现 `Wrote 14 motions`，并生成
`manifest.json`：

```bash
grep "Wrote 14 motions" "$OUT/retarget.log"
test -f "$OUT/manifest.json"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("/root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected")
manifest = json.loads((root / "manifest.json").read_text())
motions = manifest["motions"]
assert len(motions) == 14, len(motions)
assert all(m["velocity_elements_over_limit"] == 0 for m in motions)
assert all(m["maximum_floor_penetration_m"] <= 1e-9 for m in motions)
assert all(abs(m["quaternion_norm_min"] - 1.0) < 1e-6 for m in motions)
assert all(abs(m["quaternion_norm_max"] - 1.0) < 1e-6 for m in motions)
print("[PASS] 14/14 GMR references passed manifest QA")
print("max raw/final joint velocity:",
      max(m["raw_maximum_abs_joint_velocity_rad_s"] for m in motions),
      max(m["maximum_abs_joint_velocity_rad_s"] for m in motions))
print("max post-joint penetration / final penetration:",
      max(m["post_joint_limit_maximum_floor_penetration_m"] for m in motions),
      max(m["maximum_floor_penetration_m"] for m in motions))
PY
```

如果 `ps` 仍显示进程，就只继续等待，不要再次启动。如果进程已退出但没有
`Wrote 14 motions`/`manifest.json`，先查看 `tail -n 100 "$OUT/retarget.log"`，不要启动训练。

### 转成标准 50 Hz NPZ 并查看

转换器使用训练环境同一个 `isaaclab_assets.G1_29DOF_CFG` 做 FK，输出 BeyondMimic 标准
字段 `fps/joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`；额外
保存 joint/body names、root reference、sample time 和 SHA-256 manifest。它不会上传 W&B，
也不会切换训练配置：

```bash
cd /root/gpufree-share/stablemimic_replicate
env PYTHONPATH=src /workspace/isaaclab/isaaclab.sh -p scripts/convert_lafan1_npz.py \
  --input-dir /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected/csv \
  --output-dir /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected/npz \
  --output-fps 50 --overwrite --headless --device cuda:0

env PYTHONPATH=src /workspace/isaaclab/isaaclab.sh -p scripts/audit_lafan1_npz.py \
  --npz-dir /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected/npz
```

在服务器 Desktop 的 Terminal 中查看一段起身动作。不要添加 `--headless`：

```bash
cd /root/gpufree-share/stablemimic_replicate
env PYTHONPATH=src /workspace/isaaclab/isaaclab.sh -p scripts/replay_lafan1_npz.py \
  --file /root/gpufree-data/stablemimic_replicate/datasets/lafan1_gmr_bb1bbe4_corrected/npz/fallAndGetUp1_subject1.npz \
  --start-time 4.0 --end-time 12.0 --loop --follow-camera --device cuda:0
```

这是标准 kinematic NPZ replay：每帧直接写 reference root/joint state，日志会明确打印
`physics_step=False`。它用于检查格式、关节映射和视觉动作，不是策略或电机动力学测试；
关闭窗口或按 `Ctrl-C` 即可结束循环。

## 1. 检查环境与数据

运行全部单元/真实数据测试：

```bash
LAFAN1_G1_ROOT=/root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1 \
  /isaac-sim/python.sh -m unittest discover -s tests -v
```

单独审计 CSV：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/audit_lafan1.py \
  --data-root /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1
```

当前 14 文件数据应输出 `recovery_atomic_clips: 86`，时长范围约 `1.6--17.1 s`、中位数
约 `4.93 s`。状态机先检测到 89 个原始 cycle，其中 3 个超过 20 秒 episode horizon，按
配置明确排除。如果输出仍为 `6`，说明仍在错误地按整文件训练；数量不一致或脚本报错时，
应先检查阈值/数据版本，不能启动训练。

检查 G1、29→43 mapping 与 50 Hz physics step：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/visualize_lafan1_g1.py \
  --file /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1/dance1_subject1.csv \
  --headless --max-steps 5
```

视觉验收结果见
[`docs/PHASE1_VISUAL_VALIDATION.md`](docs/PHASE1_VISUAL_VALIDATION.md)。

## 2. 必做的端到端 smoke test

该命令只运行 4 个环境、24 个 policy steps 和一次 PPO update，不是正式训练：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --mode joint \
  --num-envs 4 \
  --iterations 1 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/smoke \
  --headless
```

成功标准：输出包含有限的 reward/loss/KL/std、生成 `latest.pt`，并出现：

```text
[PASS] Training run completed.
```

## 3. 训练

论文式主实验是 `joint`：Motion Expert、Recovery Expert、proprioceptive Gate 与 Critic
在同一次 PPO update 中一起训练。`tracking` 和 `recovery` 模式只用于诊断/消融，不是必须
先后预训练的三个模型；proprioceptive 部分是 Gate，不是第三个 Expert。

### 3.1 Tracking-only（诊断）

只从 `dance*.csv` reset：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --mode tracking \
  --num-envs 512 \
  --iterations 1000 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/tracking_v1 \
  --headless
```

先观察 `metrics.jsonl` 中 reward、policy std、KL 和 loss 是否有限、稳定，再增加
环境数和迭代数。配置保留论文的 4096 env，但单张 24 GB RTX 4090 上本实现同时维护
controlled/reference 两套 articulation，应从 256 或 512 env 起逐级测试显存。

### 3.2 Recovery-only（诊断）

只从 `fallAndGetUp*.csv` 的均匀/失败自适应混合分布 reset：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --mode recovery \
  --num-envs 512 \
  --iterations 1000 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/recovery_v1 \
  --headless
```

Recovery Actor 看不到 get-up reference；hidden successor 只用于 Critic 和 reward。

### 3.3 50/50 joint training（主实验）

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --mode joint \
  --num-envs 1024 \
  --iterations 10000 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/joint_v1 \
  --headless
```

`joint` 使用配置中的 `tracking_reset_probability: 0.5`。两个 Expert 始终输出
29-D mean，Gate 只看 proprioceptive history，并连续 soft fusion，不存在硬编码
`if fallen` policy switch。环境中的 fall→Recovery 只为训练 reward、Critic hidden reference
和 Gate label 建立正确监督；部署 Actor 的输入和输出不含 phase。

50/50 指 reset 分布，不保证随机策略的 rollout phase 仍为 50/50。为避免初期 Tracking
一摔倒就切走、让 Motion Expert 缺少样本，默认 curriculum 在 iterations 1--100 保持
fall-switch probability 为 0，在 101--200 从 0.01 线性升到 1.0；Recovery reset 从 iteration 1
起始终存在，因此 Motion Expert、Recovery Expert、Gate、Critic 全程仍是联合训练。只有物理
机器人达到 fall criterion、同时当前 Tracking reference 本身不属于低姿态/大倾斜动作时，
才算 `tracking_fall_candidate`。评估不使用 warmup，始终以 probability 1.0 测试恢复。

当前 RTX 4090 的有效资格规模是 `1024` env。`2048` 虽然显存够，但曾触发 PhysX
patch-buffer overflow，因此在明确调大并重新验证 PhysX buffer 前不能用；显存占用低不等于
simulator capacity 已通过。`4096` 更不能仅根据 4.7 GiB 的 1024-env 观测线性推断。

Actor 按论文描述使用共享标量方差的高斯动作。`environment.action_clip` 是论文未公布的
复现选项，默认 `100.0`，在正常范围内等效为不截断，确保 PPO 保存的采样动作与 simulator
实际执行动作一致；部署端仍按 metadata 中的同一阈值执行安全裁剪。
论文同样没有公布方差初始化；配置采用 `model.initial_std: 0.2`，避免随机初始化阶段大比例
动作越过单位幅值并使正则惩罚淹没 imitation reward。训练中该共享标量仍由 PPO 自由学习。

### 3.4 第一版：单 Tracking 的 GMR NPZ 干净基线

第一版不是继续此前 checkpoint，而是一次独立的随机初始化实验：Tracking 只使用视觉确认过的
`dance1_subject2.npz`，Recovery 使用六个 `fallAndGetUp*.npz` 长录像切出的全部原子片段。
配置中的文件白名单是实际训练数据边界，不需要复制数据或创建软链接。

这一版保留论文式 50/50 Tracking/Recovery reset、两个 Expert、Gate、Critic 联合 PPO、六类
imitation reward、Recovery `2.5` 倍权重和 50% uniform + 50% failure-adaptive Recovery phase
采样；暂时关闭此前加入的 live Tracking fall→Recovery、静止最低点 reset、`0.40--0.75`
phase window、recovery progress bonus 和外力。它恢复 fresh-run 的 `initial_std: 1.0`、
`learning_rate: 0.001`，且禁止 `--resume`/`--initialize-from`。

先做 16 环境和 1024 环境的一次 update smoke：

```bash
cd /root/gpufree-share/stablemimic_replicate

/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1_gmr_single_baseline.yaml \
  --mode joint --num-envs 16 --iterations 1 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/gmr_single_tracking_paper_v1_smoke16 \
  --headless

/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1_gmr_single_baseline.yaml \
  --mode joint --num-envs 1024 --iterations 1 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/gmr_single_tracking_paper_v1_smoke1024 \
  --headless
```

两档都出现 `[PASS]` 且 reward/loss/KL/std 全部有限后，第一段只训练 100 iterations：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1_gmr_single_baseline.yaml \
  --mode joint --num-envs 1024 --iterations 100 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/gmr_single_tracking_paper_v1 \
  --headless
```

100 iterations 是数值健康和学习方向的决策点，不是“已经学会起身”的终点。先检查分 phase
reward/success、Gate 路由、KL、std、动作幅值和 NaN/PhysX 错误，再决定是否继续到 500、
1000 或只加回一个 curriculum trick。

## 4. 续训

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --mode joint \
  --num-envs 512 \
  --iterations 2000 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/joint_v1 \
  --resume /root/gpufree-data/stablemimic_replicate/runs/joint_v1/latest.pt \
  --headless
```

Checkpoint 包含 Actor、Gate、Critic、三个 normalizer、optimizer、iteration 和配置快照。

注意：v4 Recovery 切片从每个周期内满足倒地判据的最低点开始，并以 25% 概率从该状态
近零速度 reset；当前库为 80 条。该变化改变了 motion id、failure histogram 尺寸、terminal target 与
phase 语义。`joint_ab_std_020`、`joint_paper_aligned_v1`、`joint_terminal_entropy_v1`
、`joint_atomic_recovery_v1` 与 `joint_atomic_warmup_v1` 全部只保留作诊断证据，严禁续训。
新实验必须从随机初始化建立新目录，例如 `joint_static_recovery_v4`。只有同一 commit、同一配置和同一 curriculum
语义的 checkpoint 才可续训。
runner 会检查 `training_semantics_version`，对旧 checkpoint 明确报错，避免误续训。

历史 v5 使用 `--initialize-from` 从旧 checkpoint 只载入 Agent 和 normalizer，但重置 optimizer、
iteration 和 failure histogram。它与 `--resume` 互斥，适合奖励或 curriculum 语义改变后的
显式 warm start：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml --mode joint \
  --num-envs 1024 --iterations 100 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/joint_recovery_frontier_v5 \
  --initialize-from /root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/checkpoint_001000.pt \
  --headless
```

v5 的非静止 Recovery reset 限制在归一化 phase `0.40--0.75`，同时保留 25% Recovery
reset 从静止最低倒地点开始。新增势函数只奖励相邻 policy step 的真实高度/直立度进步，
不会直接奖励参考 root 的运动，也不会修改策略观察。

第一版干净 NPZ 基线的冻结 checkpoint 属于 training semantics v6。当前代码加入新的
tracking body/recovery shaping 边界后属于 v7，因此 v6 checkpoint 不能 `--resume`；仍可用
`--initialize-from` 显式只加载 Agent/normalizer，optimizer、iteration 和 failure histogram
都会重置。

### v7：移植官方参考实现、一次只改奖励

`configs/stablemimic_g1_upstream_v7.yaml` 是独立 A/B，不覆盖 v6 配置或 run：

- 从 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 固定 14 个 G1
  关键刚体以及六类 tracking kernel 的公开权重/sigma；
- 从 [HumanUP](https://github.com/RunpeiDong/HumanUP) Stage I 移植 base-height、body-up、
  双 ankle-roll 承重三个奖励及公开权重；
- 保持 v6 的 G1 asset、PD、uniform `action_scale: 0.5`、reset/data/PPO 不变；
- 不启用 HumanUP 的 drag force、不施加 push，也不把 net contact force 冒充真正的
  self-collision pair 检测。

先跑 16/1024 环境 smoke，再从冻结 v6 checkpoint 做显式 warm start 的 100-iteration
决策段：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1_upstream_v7.yaml --mode joint \
  --num-envs 16 --iterations 1 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/gmr_upstream_v7_smoke16 \
  --headless

/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1_upstream_v7.yaml --mode joint \
  --num-envs 1024 --iterations 1 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/gmr_upstream_v7_smoke1024 \
  --headless

/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1_upstream_v7.yaml --mode joint \
  --num-envs 1024 --iterations 100 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/gmr_upstream_reward_v7 \
  --initialize-from /root/gpufree-data/stablemimic_replicate/runs/gmr_single_tracking_paper_v1/checkpoint_001000.pt \
  --headless
```

这一步不同时移植 BeyondMimic 的逐关节 action scale/PD/self-collision asset。它们会改变
动作到力矩的物理含义，应在 reward A/B 有结论后另开语义版本测试，不能混在本轮里。

实测记录（v7 warm-start，1024 env，total 1000 iterations）：

| Recovery reset | v6 physical success | v7 physical success | v7 terminal success |
|---|---:|---:|---:|
| static | 0/256 | 19/256 | 0/256 |
| early | 5/256 | 11/256 | 0/256 |
| middle | 4/256 | 17/256 | 0/256 |
| late | 27/256 | 48/256 | 0/256 |

四组物理成功都提高，说明上游奖励移植有正向作用；但训练 success 目前仍要求
`terminal_similarity >= 0.70`，iteration 1000 的 mean terminal similarity 仅约 `0.070`，所以
从未进入 Transition。这不是“完全没有起身”，而是物理站稳判据与全身末帧相似度判据错配。
下一版本应把 Recovery success 改成可审计的物理稳定持续条件，并继续保留 terminal
similarity 作诊断；不能仅把 `0.70` 随意调低后宣称成功。

## 5. 确定性评估

普通评估：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/evaluate_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_v1/latest.pt \
  --num-envs 64 \
  --steps 1000 \
  --output /root/gpufree-data/stablemimic_replicate/runs/joint_v1/eval.json \
  --headless
```

无外力标准起身评估会从每条合格 Recovery 轨迹中“倾角至少 60°且高度不超过 0.5m”的
最低帧启动，清零初速度和 reset 噪声，并分别统计仰卧、俯卧、左右侧卧的物理起身率：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/evaluate_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/latest.pt \
  --num-envs 256 \
  --steps 1000 \
  --standard-recovery \
  --output /root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/standard_recovery.json \
  --headless
```

物理成功定义为高度至少 0.7m、倾角不超过 30°并持续 0.5 秒，不要求关节姿态恰好等于
示范末帧。`--reference-reset-velocity` 只用于诊断示范动量依赖，不是标准协议。

物理诊断可记录前 2 秒的支撑接触、逐关节力矩/限位、分 reward 和高度/倾角进展；
`--reference-actions` 会绕过策略，用 privileged 下一帧参考关节角测试 retarget/PD/contact：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/evaluate_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/checkpoint_001000.pt \
  --num-envs 256 --steps 1000 --standard-recovery \
  --physical-diagnostics --reference-actions \
  --output /root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/physical.json \
  --headless
```

若要判断训练内 success 是否只来自接近站立的参考帧，可分别从 Recovery 轨迹的前、中、后
三分之一随机 reset；该诊断保留训练时的 reset 噪声和参考速度，但不施加外力：

```bash
for PHASE in early middle late; do
  /workspace/isaaclab/isaaclab.sh -p scripts/evaluate_stablemimic.py \
    --config configs/stablemimic_g1.yaml \
    --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/latest.pt \
    --num-envs 256 \
    --steps 1000 \
    --recovery-phase-bin "${PHASE}" \
    --output "/root/gpufree-data/stablemimic_replicate/runs/joint_static_recovery_v4/recovery_${PHASE}.json" \
    --headless
done
```

训练日志还会把新产生的 Recovery success 按 reset 来源精确拆成
`recovery_success_static_count`、`recovery_success_early_count`、
`recovery_success_middle_count`、`recovery_success_late_count` 和
`recovery_success_dynamic_fall_count`。旧 checkpoint 可以评估，但旧的 `metrics.jsonl`
没有保存 reset 来源，不能事后精确拆分历史 success。

论文形式的 matched pushes：100 个并行环境，`±x/±y` 每方向 25 次，
525–575 N、持续 0.2 秒：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/evaluate_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_v1/latest.pt \
  --num-envs 100 \
  --steps 1000 \
  --matched-pushes \
  --output /root/gpufree-data/stablemimic_replicate/runs/joint_v1/push_eval.json \
  --headless
```

该模式强制所有环境从正常 Tracking 状态开始，再在第 50 个 policy step 同时施加
matched push；不会混入 Recovery reset。

评估默认关闭训练期的 early fall/failure reset；Recovery 片段到末端但尚未成功时也会保持
terminal reference，而不会偷偷 reset 被推倒的机器人，使机器人摔倒后仍能继续执行策略。
JSON 中的 `paper_fall_count` 使用论文明确给出的判据：root/pelvis 高度低于 0.5 m，
或 root tilt 大于 60°。论文没有公开“恢复后重新跟踪”的精确阈值，因此本复现把
`tracking_resumption_count` 定义为连续 0.5 秒同时满足：高度至少为命令高度的 80%、
tilt 不超过 30°、active tracking similarity 达到
`tracking_resumption_similarity_threshold`、且 Tracking Gate 权重至少 0.5。
该定义也会原样写入 metrics JSON。若只想复现训练期重置行为，可额外传入
`--enable-early-termination`。

论文的统一评测 simulator 是 MuJoCo，但公开材料没有给出可直接复用的精确 G1 MJCF、
PD 和 observation adapter。本仓库当前命令在 Isaac 中复现相同 push schedule；在精确
MuJoCo asset/adapter 固定前，不把它称为论文 MuJoCo 指标。

仓库也提供了严格 joint-name/actuator 校验的 MuJoCo adapter。安装 `mujoco>=3.2` 并
提供你确认过的 G1 MJCF 后，可运行完整 motion：

```bash
/isaac-sim/python.sh -m pip install 'mujoco>=3.2'
/isaac-sim/python.sh scripts/evaluate_mujoco.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_v1/latest.pt \
  --mjcf /path/to/verified_g1.xml \
  --motion /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1/dance1_subject1.csv \
  --output /root/gpufree-data/stablemimic_replicate/runs/joint_v1/mujoco_dance1.json
```

该 adapter 固定 MuJoCo `dt=0.005`、每 4 个 physics step 执行一次确定性 action，
并拒绝缺少 29 个同名 joint 或 position actuator 的模型。由于当前服务器尚未提供
经确认的 MJCF，此入口完成了代码边界但尚未计入本次运行通过项。

## 6. 导出 ONNX

```bash
/isaac-sim/python.sh scripts/export_onnx.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_v1/latest.pt \
  --output /root/gpufree-data/stablemimic_replicate/runs/joint_v1/actor.onnx
```

导出结果只有 `actor_observation` 和 `gate_observation` 两个部署输入，输出
`joint_target_mean` 与 `gate_weights`。旁边的 `actor.json` 固化 joint order、维度、
50 Hz 和 config SHA-256。

数值一致性检查：

```bash
/isaac-sim/python.sh scripts/verify_onnx.py \
  --config configs/stablemimic_g1.yaml \
  --checkpoint /root/gpufree-data/stablemimic_replicate/runs/joint_v1/latest.pt \
  --onnx /root/gpufree-data/stablemimic_replicate/runs/joint_v1/actor.onnx
```

## 7. 输出文件

```text
RUN_DIR/
├── metrics.jsonl          # 每次 PPO iteration 的指标
├── checkpoint_XXXXXX.pt   # 按配置间隔保存
├── latest.pt              # 本次命令结束时保存
├── eval.json              # 普通评估
├── push_eval.json         # matched-push 评估
├── actor.onnx             # 部署 Actor
└── actor.json             # 部署 metadata / joint order / config hash
```

数据集、训练输出和模型均放在 `gpu_data` 数据盘；Git 仓库只保存代码、配置、测试和
小型视觉验收证据。

## 8. 如何判断训练有没有真的变好

不要只看 total reward 或某一时刻的 `transition_sample_fraction`。Transition 是固定 1.5 秒
的短暂 occupancy，比例下降可能只是 reset/事件频率改变，不能直接说明 Recovery 变差。
至少联合检查：

- `tracking_fall_entered_recovery_count` 是否在推倒/训练中非零；
- `tracking_fall_candidate_count` 与配置生成的 `fall_recovery_probability`；warmup 中前者可
  非零，但后者和 entered count 应为 0；
- `recovery_success_count` 与 `transition_completed_count`；
- 分来源的 `recovery_success_{static,early,middle,late,dynamic_fall}_count`，防止后段接近
  站立的 reset 掩盖静止倒地失败；
- 每 1000 Recovery step 的 success/failure；
- matched-push 的 `paper_fall_count`、`tracking_resumption_count` 和恢复延迟；
- KL、policy std、action clipping、NaN/Inf 与 PhysX/CUDA 错误。

`reference similarity` 是物理状态和参考姿态/速度经过六类 Gaussian kernel 后的归一化相似度，
不是图像相似度，也不是 Gate 概率。active similarity 用于当前恢复帧 imitation/严重偏离判断；
terminal similarity 只判断是否到达该原子片段末端的稳定站立姿态。

## 9. 已知环境行为

当前 Isaac Sim 5.1 容器偶尔会在 `SimulationApp.close()` 阻塞。CLI 在成功输出
`[PASS]` 后使用 15 秒 watchdog 释放进程；异常路径会先打印 traceback 并返回非零，
不会再被 Isaac 的退出状态掩盖。
