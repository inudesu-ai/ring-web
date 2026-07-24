# 面向电子机械狗的指环 IMU 手势识别路线

更新日期：2026-07-24

## 结论

机械狗控制不是“给滑动窗口做一次十分类”这么简单。合理的实时链路是：

```text
Ring BLE
  -> 数据质量/断连检测
  -> 静止与动作起止检测（spotting）
  -> 预分段时序分类器
  -> ZUPT 轨迹/圆形/前后物理特征融合
  -> 置信度拒识 + 连续确认 + 冷却 + 重置
  -> 机械狗命令适配器
```

当前最合适的分类骨干不是大模型，而是 **轻量 1D 时序 CNN
（depthwise convolution）+ 时间聚合**。在真实未标注数据足够多以后，再增加
masked autoencoder 自监督预训练；在收集到多用户标签以后，再做跨用户训练和少量
个性化微调。

## 主流方法及适用性

### 1. 轻量时序 CNN / CNN-RNN

- [DeepConvLSTM](https://doi.org/10.3390/s16010115) 将卷积局部特征和
  LSTM 时间建模结合，是穿戴式传感识别的经典基线。
- [TinyHAR](https://publikationen.bibliothek.kit.edu/1000150216) 面向资源受限
  HAR，强调轻量的跨通道与时间融合。
- [TinierHAR](https://arxiv.org/abs/2507.07949) 进一步采用 depthwise
  separable residual convolution、GRU 和时间聚合，在多个数据集上压缩参数量和
  运算量。
- 单个六轴 IMU 的手势任务已有
  [RNN 连续推断实验](https://www.mdpi.com/1424-8220/21/4/1404)，说明六轴输入
  本身适合时序网络，但部署效果仍依赖设备和用户域是否匹配。

**本项目决策：**采用 10,019 参数的 depthwise temporal CNN，加 attentive
mean/max pooling。模型导出为 NumPy `.npz`，实时推理不依赖 PyTorch。

### 2. Transformer

- [HART](https://arxiv.org/abs/2209.11750) 针对移动 IMU 设计 sensor-wise
  lightweight transformer，并研究跨设备和佩戴位置泛化。
- [TASKED](https://arxiv.org/abs/2209.09092) 将 Transformer、域对抗、
  MMD 和自蒸馏组合，用于跨用户泛化。

Transformer 适合更大、更多用户/设备的数据。当前只有仿真标签和少量未标注实机
记录，直接上 Transformer 会增加模型和调参复杂度，却不能解决主要的
sim-to-real 数据缺口，因此暂不作为第一部署模型。

### 3. 自监督预训练

- [LIMU-BERT](https://doi.org/10.1145/3485730.3485937) 使用 masked IMU
  建模学习无标签表示。
- 2024 年的系统比较发现，在其穿戴式 HAR 设置下，
  [masked autoencoder 优于所比较的 SimCLR 等方法](https://arxiv.org/abs/2404.15331)。
- [MaskCAE](https://doi.org/10.1109/JBHI.2024.3373019) 用高效全卷积网络重建
  被遮挡传感器数据，以避免较重的 Transformer encoder。
- 指环场景中，
  [ssLOTR](https://doi.org/10.1145/3534587) 已证明可用自监督方法减少 3D
  手指运动跟踪的标注需求。
- 2026 年预印本
  [UniMotion](https://arxiv.org/abs/2603.12218) 继续沿用“大量无标签动作预训练
  + 少量目标手势微调”的跨设备路线。

**本项目决策：**先连续采集真实戒指无标签流，再做 time-span/channel masking
预训练。当前一百万条仿真记录不能替代真实无标签分布。

### 4. 连续手势 spotting

- [GestureKeeper](https://arxiv.org/abs/1903.06643) 将连续惯性流中的动作起点
  检测与后续分类分开。
- 用于工业机器人控制的工作也明确把任务定义为
  [实时 segmentation + recognition](https://arxiv.org/abs/1309.2084)。
- 用 IMU 控制辅助设备的
  [sequence-matching 接口](https://doi.org/10.3390/signals2040043) 说明少量
  个性化样本与序列模式也能构成实用控制接口。

**本项目决策：**训练数据改为“一次动作一个 session”，运行时由静止/ZUPT
检测动作边界。旧仿真把一个动作无限周期化并随机截相位，使 `up/down`、
`rotate_front/rotate_back` 在统计上部分不可辨，这是数据生成问题，不是堆更大
网络能解决的问题。

### 5. 机器人控制

- 2023 年的 wearable-IMU telemanipulation 工作采用
  [Bi-LSTM 做模式分类，同时估计方向和强度](https://www.mdpi.com/2227-7390/11/16/3514)。
- 2025 年 smartwatch 控制机械臂实验把
  [X/Y/Z 方向手势映射为机器人运动](https://doi.org/10.3390/app15158297)。
- 连续机器人命令必须把非交流动作和命令动作区分开；因此仅看分类 accuracy
  不够，还需要 idle 误触发、拒识覆盖率、连续确认和断连行为。

本项目已增加 `RobotCommandGate`：

- 普通命令要求先经过 idle 重新武装；
- 置信度至少 0.85；
- 连续两个窗口一致才发出一次命令；
- 发出后锁存，必须回到 idle 才能再次触发；
- `double_tap -> stop` 为单窗口优先命令；
- BLE 数据间断会清空并解除武装。

## 数据审计

### 旧一百万条仿真

- 1,000,000 行，100 个模拟用户，10 类。
- 70/15/15 用户 train/validation/test，无用户泄漏。
- MLP 测试 macro-F1 为 `0.7810`。
- 主要错误来自 `up/down` 和 `rotate_front/rotate_back`。
- 根因是周期动作的随机相位截窗；相反方向可通过半周期平移得到近似相同信号。
- 旧仿真还把静止重力放在传感器 Z 轴，而本地实机数据静止时约为
  `accel=[+1g, 0, 0]`，存在固定 90° 封装安装差异。

### 新一百万条 episodic 仿真

- 文件：`ml/data/simulated-episodic-1m/`
- 1,000,000 行，25,000 个独立 episode，100 个模拟用户，10 类。
- 每个 episode 为 1.6 秒，包含随机 neutral prefix、动作和 suffix。
- 固定封装安装方向按真实戒指 `+X ~= gravity` 对齐。
- 随机化动作起点、速度、幅度、传感器放置、增益、偏置、随机游走、
  带宽、白噪声、时钟和量化。
- 70/15/15 用户隔离；test 用户增加放置、噪声和偏置域偏移。

真实文件 `lr_capture*.jsonl` 和 `rotation_capture.jsonl` 没有可靠的动作类别时间
标注，不能拿来声称真实分类准确率；它们目前只能验证静止误触发。

## 本次训练结果

报告：`ml/results/temporal-cnn-episodic-1m/experiment_report.json`

| 指标 | 结果 |
|---|---:|
| 参数量 | 10,019 |
| 模型文件 | 53,108 bytes |
| NumPy 单窗口推理 | 约 0.033 ms |
| held-out test macro-F1 | 1.0000 |
| stress test macro-F1 | 0.9989 |
| test ECE | 0.0222 |
| stress test ECE | 0.0302 |
| train/validation/test gap | 0 / 0 |

stress test 额外加入未见过的 18° 安装旋转、时间平移、轴增益、偏置和噪声。
同数据训练的一层 MLP clean test 也是 1.0，但 stress macro-F1 只有 `0.9416`，
说明 temporal CNN 的结构和旋转增强确实提高了扰动鲁棒性。

这些高分只证明仿真内部没有明显过拟合/欠拟合，**不等于真实机械狗控制准确率**。

## 实机未标注静止检查

报告：`ml/results/temporal-cnn-episodic-1m/real_idle_evaluation.json`

- 3 个真实 capture，共 292 个滑动窗口；
- 其中 131 个窗口至少 80% 样本被在线 AHRS 标为 stationary；
- 新 temporal CNN 对这 131 个窗口的 idle recall 为 `1.0`；
- 在 0.85 命令阈值下，静止误命令率为 `0.0`。

旧 MLP 对相同 131 个窗口的 argmax idle recall 为 `0.0`，几乎全部猜成
`rotate_back`；虽然其置信度阈值拦住了这些命令，但这证明修正真实安装方向和
episodic 数据生成是必要的。

## 下一批真实数据的最低要求

要得到可用于机械狗的真实指标，需要：

1. 至少 5 人，最好 10 人；
2. 每人每类至少 30 次独立动作；
3. 每次动作保留动作前后各 0.5 秒 idle；
4. 单独采集不少于 20 分钟“日常乱动但不发命令”的 hard-negative 流；
5. 保存 `subject_id/session_id/label/start/end/ring_firmware`；
6. 按用户隔离 train/validation/test，禁止随机拆相邻窗口；
7. 主要指标为 macro-F1、每类 recall、每分钟误命令数、拒识覆盖率和端到端延迟；
8. 最终机械狗测试必须包含 BLE 断连、丢包、动作中止和连续重复动作。

在这批数据到位前，新模型适合作为采集和集成基线，不应把仿真 100% 当作最终
完成标准。
