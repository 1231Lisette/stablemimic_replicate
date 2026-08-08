# StableMimic G1 + LAFAN1 Stage 0 差距分析

## 结论摘要

本次仅完成 Stage 0：仓库审计、公开 LAFAN1 G1 数据审计和差距分析；未修改训练代码，未实现 PPO、MoE、Recovery Expert、Gate 或 T800 适配。

最关键结论如下：

1. **当前工作区不是一个可审计的 StableMimic/机器人训练代码库。** Git 仓库没有有效 `HEAD`、提交、分支、远端或项目源文件。除本次审计产生的数据与文档外，仓库为空。
2. 因此，当前仓库中不存在 simulator、G1 模型、action 定义、PD 参数、control loop、motion loader、FK、tracking reward、PPO、ONNX、MuJoCo 或 sim2sim 实现。
3. 已从公开数据集下载第一阶段所需的全部 **8 个 `dance*.csv`** 和 **6 个 `fallAndGetUp*.csv`**，保存到 `data/lafan1/g1/`。
4. 14 个 CSV 的实际 schema 完全一致：**无表头、无时间戳，每行 36 个浮点数 = root position[3] + root quaternion xyzw[4] + G1 joint position[29]**。
5. 数据集官方文档明确标注 **30 FPS**。所有文件共 73,733 帧；全量检查未发现 NaN、Inf、列数错误、相邻四元数符号翻转或相对于数据集官方 G1 URDF 的关节限位越界。
6. CSV 只包含 configuration，不包含速度、body pose、contact 或 recovery phase。`MotionReference` 的速度和 body state 必须通过有限差分/四元数差分/FK 计算；这是**复现工程选择**，不是论文公开的原始存储格式。
7. 论文明确用 MuJoCo 作为统一 G1 仿真评测环境，但**没有说明训练 simulator**。当前阶段不能把 “Isaac Lab 训练 -> MuJoCo sim2sim” 写成论文原始流程。
8. Phase 1 不能在现有代码上做“小改动”；它实际上需要先选定/导入基础机器人训练框架，再新建最小 CSV loader、统一 MotionReference、G1 FK adapter、时间采样器、可视化播放器和测试。

---

## 审计范围与证据

### 论文来源

- `StableMimic: Smooth Human-Like Recovery for Humanoid Motion Tracking`
- arXiv:2608.02385v1，2026-08-03，8 页
- 本地来源：`/Users/tortoise/Downloads/2608.02385v1.pdf`
- 已对全部 8 页进行文本提取和 PNG 渲染检查；公式、表格、图注和 Appendix Table V 均清晰可读。

### 数据来源

- 官方数据集：[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- G1 目录：[g1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1)
- schema/FPS/顺序依据：[官方 README](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/blob/main/README.md)
- 关节限位与 FK 交叉检查依据：数据集提供的 `g1_29dof_rev_1_0.urdf` 和 `rerun_visualize.py`

### 事实标签

- **论文事实**：论文正文、表格或 Appendix 明确给出。
- **数据集事实**：公开数据集 README、原始 CSV、元数据、URDF 或官方可视化代码明确给出。
- **仓库事实**：本地工作区实际检查结果。
- **复现建议**：为完成 baseline 建议采用的工程方案，不代表论文原始实现。

---

## A. 当前仓库架构

### A.1 Git 与文件状态

| 项目 | 审计结果 |
|---|---|
| Git worktree | 是 |
| 有效 `HEAD` | 无，`fatal: Not a valid object name HEAD` |
| commit history | 无 |
| branch/remote | 无可用项目分支或远端 |
| 项目源文件 | 无 |
| `AGENTS.md` | 无 |
| 用户已有未提交代码 | 未发现 |

当前新增内容仅来自本次 Stage 0：

- `data/lafan1/g1/*.csv`：公开数据集的第一阶段子集；
- `STABLEMIMIC_G1_LAFAN1_GAP_ANALYSIS.md`：本报告；
- `task_plan.md`、`findings.md`、`progress.md`：审计过程记录。

### A.2 功能矩阵

| 领域 | 当前仓库状态 | 可复用性 |
|---|---|---|
| Simulator | 不存在 | 无 |
| G1 robot config/model | 不存在 | 无 |
| URDF/MJCF/USD | 不存在 | 无 |
| Joint/action mapping | 不存在 | 无 |
| PD/actuator config | 不存在 | 无 |
| Motion loader | 不存在 | 无 |
| FK/body state | 不存在 | 无 |
| Environment/reset/contact | 不存在 | 无 |
| Rewards | 不存在 | 无 |
| Actor/Critic/history | 不存在 | 无 |
| PPO/GAE/storage | 不存在 | 无 |
| ONNX export | 不存在 | 无 |
| MuJoCo/sim2sim | 不存在 | 无 |
| Real G1 interface | 不存在 | 无 |

### A.3 Simulator 结论

- **当前仓库使用什么 simulator：没有 simulator。**
- **论文训练 simulator：未公开。**
- **论文评测 simulator：MuJoCo。** 论文 §IV-A 明确说明所有策略在一个统一的 nominal MuJoCo G1 environment 中评测，simulation step 为 0.005 s，确定性 action 频率为 50 Hz。
- 后续选择 Isaac Lab、Isaac Gym、MuJoCo 或其他训练平台，都必须标注为**复现工程选择**。

---

## B. LAFAN1 G1 CSV 实际 schema

### B.1 文件分类

#### Tracking Library

自动匹配 `dance*.csv`，共 8 个：

| 文件组 | 文件数 | 每文件帧数 | 单文件采样跨度 `(N-1)/30` |
|---|---:|---:|---:|
| `dance1_subject{1,2,3}.csv` | 3 | 3,945 | 131.4667 s |
| `dance2_subject{1,2,3,4,5}.csv` | 5 | 6,771 | 225.6667 s |

Tracking 合计：45,690 帧。

#### Recovery Library

自动匹配 `fallAndGetUp*.csv`，共 6 个：

| 文件组 | 文件数 | 每文件帧数 | 单文件采样跨度 `(N-1)/30` |
|---|---:|---:|---:|
| `fallAndGetUp1_subject{1,4,5}.csv` | 3 | 5,047 | 168.2000 s |
| `fallAndGetUp2_subject{2,3}.csv` | 2 | 4,918 | 163.9000 s |
| `fallAndGetUp3_subject1.csv` | 1 | 3,066 | 102.1667 s |

Recovery 合计：28,043 帧。

### B.2 单帧结构

CSV **没有 header**。每行固定 36 列：

```text
Frame[36]
├── root_pos[3]        # x, y, z
├── root_quat[4]       # qx, qy, qz, qw
└── joint_pos[29]      # 下列固定顺序
```

单位说明：README 未逐字段书写单位；结合官方 URDF、Pinocchio configuration 约定与数值范围，root position 按米、joint position 按弧度解释，quaternion 无量纲。实现时应将这些单位作为 loader contract 显式断言，而不是依赖隐式约定。

### B.3 G1 29-DoF joint order

| Index | Joint | Index | Joint |
|---:|---|---:|---|
| 0 | `left_hip_pitch_joint` | 15 | `left_shoulder_pitch_joint` |
| 1 | `left_hip_roll_joint` | 16 | `left_shoulder_roll_joint` |
| 2 | `left_hip_yaw_joint` | 17 | `left_shoulder_yaw_joint` |
| 3 | `left_knee_joint` | 18 | `left_elbow_joint` |
| 4 | `left_ankle_pitch_joint` | 19 | `left_wrist_roll_joint` |
| 5 | `left_ankle_roll_joint` | 20 | `left_wrist_pitch_joint` |
| 6 | `right_hip_pitch_joint` | 21 | `left_wrist_yaw_joint` |
| 7 | `right_hip_roll_joint` | 22 | `right_shoulder_pitch_joint` |
| 8 | `right_hip_yaw_joint` | 23 | `right_shoulder_roll_joint` |
| 9 | `right_knee_joint` | 24 | `right_shoulder_yaw_joint` |
| 10 | `right_ankle_pitch_joint` | 25 | `right_elbow_joint` |
| 11 | `right_ankle_roll_joint` | 26 | `right_wrist_roll_joint` |
| 12 | `waist_yaw_joint` | 27 | `right_wrist_pitch_joint` |
| 13 | `waist_roll_joint` | 28 | `right_wrist_yaw_joint` |
| 14 | `waist_pitch_joint` |  |  |

这个顺序与数据集官方 README 和其 G1 URDF 中的 29 个 revolute joints 顺序一致。

### B.4 CSV 未提供的字段

| 字段 | CSV 是否提供 | 后续取得方式（复现建议） |
|---|---:|---|
| frame index / timestamp | 否 | `t_i = i / 30` |
| root linear velocity | 否 | 连续时间位置导数/有限差分 |
| root angular velocity | 否 | 最短弧 quaternion difference + log map |
| joint velocity | 否 | 插值轨迹导数或中心有限差分 |
| body position/orientation | 否 | 使用完全匹配的 G1 模型做 FK |
| body linear/angular velocity | 否 | FK/Jacobian 或连续 body pose 差分 |
| contacts | 否 | simulator contact state；不能从 CSV 静默伪造 |
| recovery phase/label | 否 | 文件类别和训练期 rollout/reset 状态管理 |

### B.5 数据完整性结果

对全部 14 个文件、73,733 行进行检查：

| 检查项 | 结果 |
|---|---:|
| 非 36 列的行 | 0 |
| NaN | 0 |
| Inf | 0 |
| quaternion norm 范围 | 0.99999915 - 1.00000089 |
| 相邻 quaternion 负点积（符号翻转） | 0 / 73,719 对 |
| 官方 URDF joint-limit 越界 | 0 / 2,138,257 个 joint samples |

Loader 仍应对 quaternion 归一化，并在 SLERP 前执行 `dot < 0 -> q1 = -q1` 的 shortest-path 保护；当前数据没有发现翻转不代表未来输入永远不会出现。

### B.6 坐标系

数据集官方可视化脚本将场景声明为 **right-handed Z-up**，并直接把每行 36-D configuration 传入 Pinocchio free-flyer G1 模型。脚本/README 没有完整定义训练环境需要的所有局部坐标、heading、body velocity 表达约定，因此后续 adapter 必须明确：

- world frame 与 simulator world frame 的轴向映射；
- quaternion 的 xyzw/wxyz API 转换边界；
- angular velocity 是 world frame 还是 body frame；
- body tracking error 是否 pelvis-relative、yaw-aligned；
- 左右关节和 body name 映射。

---

## C. 当前仓库能否直接读取这些 CSV

**不能。** 当前仓库没有任何 loader 或训练代码。

第一版 loader 至少必须做到：

1. 只按文件名构建两个 sequence-disjoint library：
   - `dance*.csv` -> `tracking_motion_library`
   - `fallAndGetUp*.csv` -> `recovery_motion_library`
2. 拒绝列数不等于 36、非有限值、未知 joint mapping 的文件；
3. 显式绑定 30 FPS、xyzw quaternion 和 29-joint order；
4. 保留 sequence 边界，不能把所有文件直接拼成一条连续轨迹；
5. 内部可构建缓存，但缓存是**复现工程实现**，不能称为论文原始 NPZ 格式；
6. 输出统一 `MotionReference`，并记录所有派生字段的方法和 frame convention。

---

## D. G1 joint mapping 是否匹配

结论分两层：

- **CSV 与公开数据集自带 G1 URDF：匹配。** 29 个关节名称、顺序和限位一致；全量数据无关节限位越界。
- **CSV 与当前仓库 G1：无法比较。** 当前仓库没有 G1 model 或 joint order。

后续引入训练框架后，必须生成显式 permutation：

```text
csv_joint_index -> simulator_joint_index -> policy_action_index -> deployment_joint_index
```

四个顺序即使名字集合相同，也不能假设 index 相同。启动时应断言 29 个名字一一匹配且无重复/遗漏；ONNX metadata 或部署配置中也应固化同一映射。

---

## E. 30 FPS 数据与 50 Hz policy 的时间处理方案

### E.1 论文事实与数据事实

- 数据集事实：CSV 为 30 FPS。
- 论文事实：policy/control frequency 为 50 Hz；MuJoCo 评测 simulation step 为 0.005 s，即每个 policy action 对应 4 个 simulation steps。
- 论文未公开其内部 motion file format 或具体插值代码。

### E.2 推荐接口

```python
reference = motion.sample(t_seconds)
```

不能使用 `frame += 1`，因为那会把 30 FPS 数据错误地按 50 FPS 播放，速度放大为 5/3。

建议采用：

```text
source_dt  = 1 / 30 s
policy_dt  = 1 / 50 s
t          = motion_start_time + policy_step * policy_dt
u          = t / source_dt
i0         = floor(u)
i1         = min(i0 + 1, N - 1)
alpha      = u - i0
```

采样规则：

- root position：线性插值；
- root quaternion：归一化、shortest-path SLERP；
- joint position：线性插值；
- velocity：由连续采样定义或稳定的有限差分/FK/Jacobian 计算；
- sequence end：训练与完整序列评测分别明确 terminate/clamp/wrap 策略，默认不能跨文件插值。

30 Hz 与 50 Hz 的相位每 0.1 s 对齐一次；时间驱动采样会自然形成正确的 `0, 0.6, 1.2, ...` source-frame coordinate。

### E.3 官方可视化脚本的注意点

公开脚本使用 `time.sleep(0.03)`，对应约 33.33 Hz，而不是严格 30 Hz。它可以作为 joint/FK 参考，但不能直接作为精确 playback timing 实现。Phase 1 应使用 monotonic clock 和目标时间戳，或离线按 30 FPS/50 Hz 确定性推进。

---

## F. 当前 motion tracking pipeline

当前仓库中不存在 motion tracking pipeline。下面各项均缺失：

- motion library discovery；
- CSV/NPZ loader；
- random sequence/frame reset；
- motion time manager；
- interpolation；
- root/heading alignment；
- FK/body-state calculation；
- Actor motion command observation；
- whole-body tracking reward；
- episode termination/logging；
- reference/ghost visualization。

数据集自身提供一个基于 Pinocchio + Rerun 的离线可视化脚本，但它不是当前仓库的训练 pipeline，也不提供 50 Hz reference sampling、reward 或 simulator integration。

---

## G. StableMimic 缺失模块

由于仓库为空，除了已下载原始数据外，StableMimic baseline 的所有运行模块都缺失。

### G.1 基础设施缺失

- 可运行的 Python package/build/dependency 管理；
- 训练 simulator 与 G1 scene；
- G1 29-DoF model、collision、inertial、actuator、joint limits；
- position target action、action scale、PD gains、effort/velocity limits；
- 50 Hz policy loop 与 simulator substeps；
- vectorized environments、4096-env 能力和 logging/checkpoint。

### G.2 Motion/Tracking 缺失

- LAFAN1 CSV loader 与两类 motion library；
- `MotionReference`；
- 30 -> 50 Hz continuous-time sampling；
- quaternion/FK/body velocities；
- reset near arbitrary dance frame；
- root/heading alignment；
- six-family whole-body tracking kernels；
- tracking metrics与 reference visualization。

### G.3 RL 缺失

- deployable Actor observation/history；
- privileged Critic observation；
- normalization；
- Gaussian policy、learned std；
- PPO、GAE、rollout storage、adaptive LR、minibatches；
- reward/termination/config/assertions。

### G.4 StableMimic 特有模块缺失

- recovery library/reset/perturbation；
- hidden successor-state reward；
- recovery-only validation task；
- tracking/recovery dual experts；
- proprioceptive four-frame Gate；
- soft expert-mean fusion；
- information-boundary assertions；
- training-only gate targets与 auxiliary losses；
- 1:1 reset curriculum；
- recovery -> 1.5 s transition -> tracking state machine；
- command horizontal realignment；
- failure-adaptive frame sampling；
- recovery success/failure phase metrics。

### G.5 Evaluation/Deployment 缺失

- nominal MuJoCo G1 paper-style evaluation；
- full dance sequence metrics；
- 100-trial push protocol；
- post-fall motion/load/energy metrics；
- gate-weight traces；
- one-Actor ONNX export；
- real G1 observation/action/safety interface。

---

## H. 需要修改或新建的具体文件

当前没有既有源文件可修改；以下是**推荐新建的最小文件布局**，属于复现工程建议，不代表论文源码结构。

### H.1 Phase 1 最小范围

| 建议路径 | 职责 |
|---|---|
| `pyproject.toml` | 最小依赖和可执行入口 |
| `configs/motion/lafan1_g1.yaml` | 30 FPS、glob、joint order、坐标/单位、路径 |
| `src/stablemimic/data/lafan1_csv.py` | 严格 36-column CSV loader、分类与校验 |
| `src/stablemimic/motion/reference.py` | 统一 `MotionReference` 数据结构 |
| `src/stablemimic/motion/sampling.py` | 时间采样、linear interpolation、SLERP、边界策略 |
| `src/stablemimic/robot/g1_kinematics.py` | G1 joint mapping、FK、body states/velocities |
| `src/stablemimic/visualization/lafan1_player.py` | 30 FPS 和 50 Hz 插值播放、ghost/reference robot |
| `tests/test_lafan1_schema.py` | 文件分类、36 列、finite、29-joint mapping |
| `tests/test_motion_sampling.py` | endpoint、SLERP、30->50 Hz、velocity 测试 |
| `tests/test_g1_kinematics.py` | FK、左右映射、T-pose、joint limits 测试 |

### H.2 后续阶段候选文件

在 simulator/framework 选定后再确定真实路径；不应在 Stage 0 先实现：

- `configs/robot/g1.yaml`
- `configs/tasks/g1_dance_tracking.yaml`
- `configs/tasks/g1_recovery.yaml`
- `src/stablemimic/envs/g1_tracking_env.py`
- `src/stablemimic/envs/g1_recovery_env.py`
- `src/stablemimic/rewards/tracking.py`
- `src/stablemimic/models/stablemimic_actor.py`
- `src/stablemimic/rl/ppo.py`
- `src/stablemimic/eval/mujoco_protocol.py`
- `src/stablemimic/export/onnx.py`

首先必须决定是导入一个现成上游训练仓库，还是从零搭建。若有预期的上游仓库，应先把它正确 clone/checkout 到本工作区，再重新执行一次 Repository Audit；否则本报告中的“具体文件”只能是建议新建路径，不能假装是对现有代码的定位。

---

## I. 推荐实施阶段

### Gate 0：补齐基础仓库（新增前置阶段）

1. 明确并导入基础机器人训练框架；
2. 固定 commit、依赖、G1 model provenance 与许可证；
3. 确定 reproduction training simulator；
4. 重新审计真实 action/PD/control/obs/reward/PPO/deployment pipeline；
5. 在报告中把 simulator 选择标注为 reproduction choice。

Gate 0 未完成前，不能可靠进入论文复现。

### Phase 1：LAFAN1 reference visualization

- 实现严格 CSV loader 和两个 library discovery；
- 构建 `MotionReference`；
- 使用准确 G1 FK；
- 以 30 FPS 原速和 50 Hz 插值采样播放多条 dance/get-up；
- 检查 root、quaternion、left/right、joint order、FK、速度与插值。

### Phase 2：Dance-only tracking baseline

- 只使用 `dance*.csv`；
- 完成 reset、observations、action/PD、tracking kernels、PPO smoke test；
- tracking 不稳定时停止，不进入 recovery。

### Phase 3-8

按用户指定顺序继续：recovery loader/reset -> recovery-only MVP -> dual-expert Gate -> 50/50 joint training -> transition/realignment -> MuJoCo paper-style evaluation -> ONNX/G1 deployment。

---

## J. 每阶段验证标准

### Gate 0 验证

- 仓库存在有效 commit 和可复现实验环境；
- G1 model、29 joint order、limits、actuator、PD、action scale、sim dt、policy dt 均可定位到具体文件；
- 一个 G1 环境可以 reset/step；
- 明确区分训练 simulator 与论文 MuJoCo evaluation。

### Phase 1 验证

- 14/14 文件分类正确、schema 校验通过；
- root quaternion 归一化且方向正确；
- reference G1 无左右反转、扭曲或明显脚滑之外的 FK 错误；
- 30 FPS 原速播放持续时间与 `(N-1)/30` 一致；
- 50 Hz sampling 在 source-frame endpoint 上精确一致；
- SLERP 单元测试覆盖相同、近 180 度和 quaternion 双覆盖；
- velocity 与 finite-difference sanity check 一致；
- 所有 joint/body mapping assertion 通过。

### Phase 2 验证

- dance-only 环境可随机 sequence/time reset；
- perfect reference state 的 reward 显著高于扰动 state；
- Actor/Critic/history/action shape 全部记录且无 NaN/Inf；
- 短 rollout + 一次 PPO update 的 loss/KL/grad/std finite；
- 长训前记录 episode length、root/body/joint errors、action rate、fall rate。

### Phase 3-4 验证

- recovery reset 能复现 prone、supine 和 intermediate frames；
- perturbation 全部配置化并可逐项关闭；
- hidden next-reference 不进入 Actor；
- recovery-only success、time、failure phase、speed/effort 可记录；
- recovery-only 学不会则停止。

### Phase 5-7 验证

- 两个 expert 均输出 29-D mean；
- Gate 只接收 proprioceptive history；
- softmax weights finite、非负、和为 1；
- fused mean 与显式加权结果一致；
- privileged feature leak assertion 通过；
- 50/50 reset 比例统计正确；
- recovery 成功后进入约 1.5 s transition，不直接 done；
- transition 后重新获得 tracking，且不追 pre-fall world XY。

### Phase 8/部署验证

- MuJoCo nominal evaluation 配置固定并可重复；
- 完整 dance sequence 从头到尾评测；
- 100 次 matched pushes：4 方向 x 25，0.2 s，525-575 N；
- tracking/recovery/load/energy/gate metrics 齐全；
- ONNX 与训练 Actor 数值对齐；
- export 不包含 Critic、get-up library、phase/frame ID、reward 或 gate label。

---

## K. 风险清单

| 优先级 | 风险 | 影响 | 缓解措施 |
|---:|---|---|---|
| P0 | 当前仓库为空/可能打开了错误目录 | 无法审计或实现任何已有 pipeline | 确认并导入预期上游仓库，固定 commit 后重审 |
| P0 | 训练 simulator 未由论文公开 | 容易把复现选择误报为论文事实 | 文档和 config 显式标注 reproduction choice；MuJoCo 仅按论文事实作为统一评测环境 |
| P0 | 公开 retarget 数据只考虑运动学，不含 dynamics/actuator limits | 某些 reference 在物理 G1 上不可跟踪或恢复 | Phase 1 可视化后先做 dance tracking；对 reference feasibility 做速度/加速度/力矩诊断 |
| P0 | 引入的 G1 model 与数据集模型不一致 | FK、joint order、contact geometry、body reward 全部可能错误 | 名称映射 + model hash/provenance + T-pose/FK regression test |
| P1 | 30 FPS 被按 50 FPS 逐帧推进 | reference 快 5/3，速度/reward 错误 | 全部改为 `sample(t)`；禁止 `frame += 1` |
| P1 | xyzw/wxyz API 混用 | root 姿态完全错误但可能不立即报错 | 单一内部 convention，API 边界显式转换和 known-rotation tests |
| P1 | CSV 没有速度/body states | reward/critic 字段可能被静默伪造 | 派生过程有明确 provenance、坐标系和数值测试 |
| P1 | tracking/recovery sequence 边界被拼接 | 产生非物理 successor target | library 保留独立 sequence；禁止跨文件插值 |
| P1 | privileged recovery reference 泄漏到 Actor/Gate | 得到不可部署的“假复现” | observation schema 分型、零 mask、运行时 assertions、export input audit |
| P1 | 论文未公开 PD gains、action scale、kernel weights/sigmas等 | 结果对超参数敏感且无法精确复现 | 全部配置化并标注 reproduction choice；做 ablation/敏感性记录 |
| P1 | recovery world XY 对齐错误 | 机器人追旧位置或拖地平移 | recovery 用 root height；成功后 horizontal realignment |
| P2 | 公开脚本 `sleep(0.03)` 不是严格 30 FPS | 人眼播放速度略快，时间验证误差累积 | 使用目标时间戳/离线 deterministic timeline |
| P2 | 数据集许可证为 CC BY-NC-ND 4.0（README 声明） | 商用、再分发、衍生数据处理可能受限 | 在发布模型/缓存/转换数据前单独核验许可与归属；本报告不构成法律意见 |

### 最大技术风险

**最大风险是基础仓库缺失。** 这不仅是“少几个 StableMimic 模块”，而是 simulator、robot、control、motion、RL、evaluation 五条基础链路都不存在，无法进行真实的 file-level reuse 或 gap closure。其次是公开 CSV 仅有运动学配置，且数据集作者明确说明 retargeting 未考虑 dynamic constraints 或 actuator limitations；即使 FK 可视化正确，也不保证物理 G1 可跟踪。

---

## 论文公开参数与未知项边界

### 论文明确给出的实现/训练参数

| 类别 | 论文公开值 |
|---|---|
| Actor history | 4 frames |
| Actor/Gate/Critic input dims | 884 / 372 / 1428（Appendix Table V） |
| Expert MLP | 512-256-128, ELU |
| Action mean | 29-D joint target mean |
| Policy std | shared learned scalar standard deviation |
| PPO | gamma 0.99, GAE lambda 0.95, clip 0.2, adaptive LR 0.001 |
| PPO update | 24 steps, 5 epochs, 4 minibatches, grad norm <= 1.0 |
| Entropy | 0.05 |
| Auxiliary loss | CE 0.1, transition 4.0, consistency 0.01, alignment 0.01 |
| Training | 4096 envs, 50 Hz, 20 s, equal tracking/recovery sampling |
| Recovery | 1.5 s transition, 2.0 s error tolerance, recovery coefficient 2.5x nominal |
| Reset noise | Appendix Table V 的 root pose/velocity/angular velocity/joint noise 范围 |
| Evaluation | MuJoCo dt 0.005 s, deterministic 50 Hz |

### 论文未明确给出/本仓库也无法补充

- training simulator；
- G1 asset 的精确版本/hash；
- 29 joints 的论文内部排列；
- PD stiffness/damping；
- action scale/default pose；
- simulator substep/training physics 参数；
- six kernel 的逐项 weights/sigmas；
- regularization reward 的全部 weights；
- observation 221-D、gate 93-D、Critic 1428-D 的逐元素顺序；
- body set/body order（评测明确说 14 common key bodies，但训练 body 列表未完整展开）；
- observation corruption 和 dynamics randomization 的数值；
- adaptive learning-rate schedule 的具体 KL rule；
- reference horizontal/yaw alignment 的完整算法细节。

这些参数后续只能通过作者代码/补充材料获得，或作为复现选择配置化；不得标成论文原始参数。

---

## 第一轮 19 项汇报速查

1. **当前仓库 simulator**：无；论文训练 simulator 未公开，MuJoCo 是明确的统一评测环境。
2. **当前 G1 model 在哪里**：当前仓库无 G1 model；公开数据集另有官方 G1 URDF，但它不是当前仓库训练模型。
3. **当前 action definition**：无。论文公开的是 29-D joint target mean，经 position-control/PD 接口执行。
4. **当前 joint order**：无。CSV 的 29-joint order 见 B.3。
5. **当前 PD 参数**：无；论文未给出具体 stiffness/damping。
6. **当前 control frequency**：无。论文为 50 Hz；MuJoCo eval dt=0.005 s。
7. **当前 motion loader 格式**：无 loader。公开数据是无表头 CSV，不是 NPZ。
8. **LAFAN1 CSV schema**：36-D = XYZ + QXQYQZQW + joint_pos[29]。
9. **CSV FPS**：30 FPS。
10. **CSV joint order 与当前 G1 是否对齐**：无法比较，因为当前 G1 不存在；与数据集官方 G1 URDF 对齐。
11. **当前是否已有 FK**：仓库无；数据集官方可视化脚本有 Pinocchio FK 示例。
12. **当前是否已有 motion interpolation**：无。
13. **当前是否已有 whole-body tracking reward**：无。
14. **当前 Actor/Critic observation**：无；论文只公开总维度和信息边界，未公开完整逐元素 schema。
15. **当前 PPO 实现**：无；论文 Appendix 参数已整理，但实现细节仍不完整。
16. **当前是否已有 MuJoCo/sim2sim**：无。
17. **StableMimic 缺失模块**：除原始 CSV 外全部缺失，详见 G。
18. **Phase 1 最小修改方案**：实际上是新建 loader、MotionReference、sampler、G1 FK adapter、viewer 与 tests；在此之前先导入基础仓库/G1 model。
19. **最大技术风险**：空仓库导致无法复用任何基础链路；其次是 kinematic-only 数据不保证动力学/执行器可行。

---

## Stage 0 停止点

本报告完成后停止。下一步不应直接实现 StableMimic；应先由用户确认：

1. 当前目录是否就是预期仓库；
2. 若不是，应提供/clone 哪个上游仓库与 commit；
3. 若确实从零开始，选择哪个训练 simulator/framework 作为明确的 reproduction engineering choice。
