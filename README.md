# stablemimic_replicate

Unitree G1 + 公开 LAFAN1 重定向数据的 StableMimic 复现工程，依据
arXiv:2608.02385 构建。

## 当前状态

代码栈已经覆盖：

- 严格的 36 列 LAFAN1 CSV loader，8 个 `dance` 与 6 个 `fallAndGetUp` 序列分库；
- 30 FPS reference 在 50 Hz policy clock 上的线性插值、四元数 SLERP 和速度派生；
- Isaac Lab G1 29 个 body joint 到运行时 43 joint articulation 的显式映射；
- Tracking / Recovery / 1.5 秒 Transition 三阶段状态机与水平 reference realignment；
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
CSV : /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1
运行输出: /root/gpufree-data/stablemimic_replicate/runs
```

请始终从代码目录运行：

```bash
cd /root/gpufree-share/stablemimic_replicate
export PYTHONPATH=src
```

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

## 3. 分阶段训练

建议不要一上来直接运行 joint long training。

### 3.1 Tracking-only

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

### 3.2 Recovery-only

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

### 3.3 50/50 joint training

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/train_stablemimic.py \
  --config configs/stablemimic_g1.yaml \
  --mode joint \
  --num-envs 512 \
  --iterations 10000 \
  --run-dir /root/gpufree-data/stablemimic_replicate/runs/joint_v1 \
  --headless
```

`joint` 使用配置中的 `tracking_reset_probability: 0.5`。两个 Expert 始终输出
29-D mean，Gate 只看 proprioceptive history，并连续 soft fusion，不存在硬编码
`if fallen` policy switch。

Actor 按论文描述使用共享标量方差的高斯动作。`environment.action_clip` 是论文未公布的
复现选项，默认 `100.0`，在正常范围内等效为不截断，确保 PPO 保存的采样动作与 simulator
实际执行动作一致；部署端仍按 metadata 中的同一阈值执行安全裁剪。
论文同样没有公布方差初始化；配置采用 `model.initial_std: 0.2`，避免随机初始化阶段大比例
动作越过单位幅值并使正则惩罚淹没 imitation reward。训练中该共享标量仍由 PPO 自由学习。

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

注意：奖励/phase/entropy 语义修改后，旧的 `joint_ab_std_020` 与
`joint_paper_aligned_v1` checkpoint 只保留作诊断证据，不应续训。请从随机初始化建立新的
run directory。

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

评估默认关闭训练期的 early fall/failure reset，使机器人摔倒后仍能继续执行策略。
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

## 8. 已知环境行为

当前 Isaac Sim 5.1 容器偶尔会在 `SimulationApp.close()` 阻塞。CLI 在成功输出
`[PASS]` 后使用 15 秒 watchdog 释放进程；异常路径会先打印 traceback 并返回非零，
不会再被 Isaac 的退出状态掩盖。
