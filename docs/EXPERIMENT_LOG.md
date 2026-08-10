# StableMimic G1 实验与错误记录

本文件保存失败实验和修正依据。不同代码语义下的 reward 绝对值不能直接横向比较，所有旧
checkpoint 都保留在 `gpu_data` 作证据，不放入 Git。

## 最终定位到的主要错误

1. **Recovery 数据粒度错误（当前首要根因）**：6 个 `fallAndGetUp*.csv` 各长
   102--168 秒，内部反复倒地/起身；旧 loader 把每个整文件当一条 motion，并始终把文件
   最后一帧当 terminal。20 秒 episode 大部分时候根本到不了这个 terminal，Recovery Expert
   接收到的是歧义且通常不可达的目标。持续状态诊断在原始文件中识别出 89 个倒地→直立
   cycle；其中 86 个不超过 20 秒 horizon 并进入训练库，3 个超长 cycle 被明确排除。新代码
   为每段使用本地最后一帧。
2. **训练/评估 phase 不一致**：旧代码只有 Recovery reset 才进入 Recovery。Tracking 被推倒
   后，训练在高度低于 0.18 m 才 reset；关闭 early termination 的 matched-push 评估则永远
   留在 Tracking，Recovery phase fraction 和 Recovery Gate target 都严格为 0。新代码达到
   论文 fall criterion 后保持物理状态，用当前高度、projected gravity 与关节姿态匹配最近
   recovery 帧，再进入 Recovery 监督。
3. **success 与 failure 共用错误信号**：早期代码用 active-reference similarity 和同一个
   0.82 阈值同时判 Recovery 失败/成功，导致中间帧偶然相似会“成功”，正常误差又会累计失败。
   现已分为 active imitation、persistent failure、terminal get-up 三个信号。
4. **奖励时间错一拍**：旧 reward 将执行后的 `s_(t+1)` 和 reference `k` 比较；现改为 hidden
   successor `k+1`。
5. **entropy 放大 29 倍**：旧实现把 29-D entropy 求和后乘论文系数 0.05，std 从约 0.25
   增至 0.51，动作和 Expert disagreement 变大。现使用每动作维平均 entropy；log probability
   仍按 29 维求和。
6. **动作裁剪破坏 on-policy**：第一轮有约 56--58% sampled action 超过 `[-1,1]`，PPO 记录的
   动作/log-prob 与 simulator 实际动作不同。现默认 clip=100（观测范围内等效不裁剪）并记录
   clip fraction。
7. **错误地把显存当作容量结论**：2048 env 数值和显存都能运行，但 PhysX 报
   patch-buffer overflow；所以当前只资格化 1024 env。4096 不能依据 1024 env 约 4.7 GiB
   直接外推。
8. **把 Transition 占比当成成功率**：Transition 只持续 1.5 秒，sample fraction 受 episode
   reset 和相位占用影响。现在同时记录 recovery success、transition completion、fall entry、
   failure 及每 1000 Recovery steps 的事件率。

## 已保存实验

| Run | 训练结果 | 确定性评估 | 结论 |
|---|---|---|---|
| `joint_ab_std_020` | 100→500；std 0.248→0.393，KL 有限 | mean termination 3.70→4.44；transition 2→1 | action/std 修正后可训练，但 tracking 稳定性变差 |
| `joint_paper_aligned_v1` | 100→500；只有第 100 前 1 次 success/transition；std 最终 0.506 | matched push 两次均 100 falls / 0 resumptions | similarity 混用与 entropy 尺度导致 Recovery 信号近乎失效 |
| `joint_terminal_entropy_v1` | 1024 env，500 iter；前 100 有 427 successes/419 transitions，101--500 有 466/458；最终 std 0.209、KL 0.0155 | iter500 mixed 1 success/1 transition；matched push 100 falls / 0 resumptions | 数值/终端信号修好，但鲁棒恢复仍失败 |

## 最后一次只读诊断

- 相同 128 个 Recovery reset：iteration 100 的 fused/tracking-only/recovery-only success 为
  `8/8/7`；iteration 500 为 `5/4/1`。Recovery Expert 没有形成优势。
- iteration 500 动作平均绝对值约 tracking `0.144`、recovery `0.215--0.220`，Expert RMS
  disagreement 约 `0.33`。
- matched push 后 100/100 跌倒，Recovery phase fraction `0`，Recovery Gate target `0`；
  Gate 的软 Recovery 权重约 `0.473`，但环境仍用 Tracking reward/reference。
- 6 个 recovery 文件共有 28,043 帧。持续 0.5 秒的 fallen/upright 状态机识别出 89 个原始
  cycle，96.6% 不超过 20 秒；实际训练库保留 86 个，时长 1.6--17.1 秒，中位数 4.93 秒。

## 当前实验边界

原子切片和 Tracking-fall curriculum 改变了训练数据语义与 failure histogram 形状，因此旧
checkpoint 不能续训。正确顺序是：真实数据审计 → 全测试 → 16/128/1024 CUDA smoke → 从
随机初始化启动新的 `joint_atomic_recovery_v1`，先训练 100 iterations 并跑同协议 mixed 与
100-trial matched-push 评估，再决定是否到 500。不得仅凭训练 success count 自动续训。

## 原子 Recovery 补丁验证（2026-08-10）

- 真实数据与 PyTorch 完整测试 `26/26` 通过，`compileall` 和 `git diff --check` 通过。
- 真实数据审计：8 个 Tracking 文件、6 个 Recovery 文件，训练时 Recovery 库为 86 个原子
  clip，时长 `1.6--17.1 s`、中位数 `4.93 s`。
- 16-env joint smoke 从随机初始化完成一次 PPO update，并首次在真实 Isaac 路径触发 1 次
  `tracking_fall_entered_recovery`；checkpoint 的 `training_semantics_version=2`，failure
  histogram 为 `86×64`。
- 最终向量化 sampler 的 1024-env joint smoke 完成一次有限 update：wall time `2.46 s`，
  std `0.2008`，67 次 Tracking-fall→Recovery，5 次 Recovery success，action clip fraction
  `0`，无 CUDA/PhysX/NaN 错误。首次随机 update 的 KL 为 `0.198`，高于正式目标，只能作为
  接口 smoke；需要观察后续 adaptive-LR 迭代，不能据此声称已训练稳定或收敛。
- 100-trial、300-step matched-push 随机策略负对照：100/100 fall、100/100 进入 Recovery，
  Recovery target/sample fraction `86.66%`，sequence termination `0`、总 termination `0`、
  tracking resumption `0`。这证明新评估不再把推倒状态留在 Tracking，也不会在 recovery
  clip 末端偷偷 reset；随机策略仍不能伪造恢复成功。

## 原子 Recovery iteration-100 结果与 warmup 修正

- `joint_atomic_recovery_v1` 在 1024 env、100 iterations 下数值健康：final/max std
  `0.20568/0.20581`、final KL `0.01831`、last-20 KL mean `0.01768`，无裁剪、CUDA、PhysX
  或非有限错误。训练累计 9,094 次 fall entry、287 successes、216 transitions。
- 但 last-20 Tracking exposure 只有 `21.92%`：50/50 reset 后，随机 Tracking 策略很快跌倒并
  切入 Recovery，导致 Motion Expert 样本不足。mixed 评估 64/64 fall、0 resumption；matched
  push 100/100 fall 且正确进入 Recovery，但 0/100 resumption。该 checkpoint 停在 100。
- Tracking 数据中仅 220/45,690 帧（0.4815%）满足 raw fall criterion，但存在最长 97 帧的
  commanded floor-motion 区间，证明 fall switch 还需要 reference-aware guard。
- 新的 version-3 curriculum 保留从 iteration 1 开始的 50/50 Tracking/Recovery resets；仅把
  Tracking fall switch 在 1--100 设为 0、101--200 线性 ramp 到 1，并要求 Tracking reference
  本身非 fallen。这样不是顺序预训练：两个 Expert、Gate 和 Critic 仍在每次 joint PPO update
  中一起优化。version-2 checkpoint 不可续训。
- version-3 的 1024-env iteration-1 CUDA smoke 中，fall switch probability 为 `0`，观察到
  207 个 reference-aware fall candidates、0 个 fall entry；Tracking/Recovery exposure 分别为
  `58.26%/41.41%`，证明 warmup 阻止随机 Tracking 策略迅速吞掉 Motion 样本。首次随机 update
  KL `0.19683`，仅用于接口验证。
- 同一 checkpoint 的 100-trial matched-push 评估强制 probability 为 `1`：100/100 fall、
  100/100 candidates、100/100 进入 Recovery，Recovery target/sample fraction `86.65%`，
  termination `0`、resumption `0`。这同时验证训练 warmup 与完整评估路径，仍不把随机策略的
  负对照误报为恢复成功。

## version-3 warmup iteration-100 决策点（2026-08-10）

- `joint_atomic_warmup_v1` 从随机初始化完成 1024-env、100 iterations；训练期间 fall switch
  probability 始终为 `0`。337,717 个 reference-aware fall candidates 中进入 Recovery 的数量
  严格为 `0`，而 Recovery resets 从第一轮起仍产生训练样本；累计 411 recovery successes、
  315 transitions。
- last-20 Tracking/Recovery exposure 为 `41.80%/57.05%`，相比 version-2 同期 Tracking
  `21.92%` 明显改善。last-20 KL mean `0.01790`、final std `0.20627`、action clip fraction
  `0`，训练数值健康。
- 64-env、500-step mixed 评估：64/64 fall、24 次 reference-aware fall entry、1 recovery
  success/1 transition、0 tracking resumptions、0 terminations。
- 100-trial、500-step matched-push：100/100 fall、100/100 candidates、100/100 进入 Recovery，
  0 recovery success、0 transition、0 tracking resumption、0 termination。说明 curriculum 的
  路由修复有效，但 iteration 100 的策略尚未学会外力推倒后的恢复；按预定决策门停止，不把
  “训练数值健康”误写成“恢复能力已经获得”。
