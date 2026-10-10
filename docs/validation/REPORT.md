# 原版核心对应与无 VLM 复制复现

2026-10-10。对照版本为本地 ToGo6.13；外部策略同时对照 SimToolReal 官方公开下载包。所有运行不调用 VLM，不训练或微调策略。

## 核心对应

| 部分 | 核查与结论 |
| --- | --- |
| 预训练权重 | 本地原版与官方 pretrained_policy.zip 的 model.pth SHA256 完全相同：4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9。 |
| 策略配置 | 原版 config.yaml 比官方下载配置只多 resetWhenTableForceTooHigh:true 和 parttoolgenBenchmarkExtraActors:[] 两个环境字段，网络与训练配置未变。交接继续保留原版配置。 |
| 策略推理 | 同一 GPU、同一权重和连续 128 步固定观测；原版 RlPlayer 与精简加载器的动作、LSTM 状态最大差值均为 0。测试含零输入、正常随机输入及放大输入，不是全场景数学等价证明。 |
| 观测与动作协议 | observation_action_utils_sharpa.py 与原版逐字节相同，含 140 维观测、29 维动作、FK 和关节目标转换。保留额外固定标量50、归一化与确定性均值动作。 |
| 工具几何 | 八工具的面标签、部件几何、已选接触点与 bbox 动作轴与原版相同；八工具均可用现有标签重新构建先验。 |
| 编译与路径求解 | 原版 SE(3) 求解主干、残差、阶段编译及路径检查保留。完成下述约束对齐后，九组相同几何与 ProgramIR 的全部固定目标位姿逐项相同，最大差值0。 |
| 仿真环境 | 保留原版机器人/工具物理资源和动力学、动作步进及 fixed-goal 推进。精简训练、统计和录像依赖，增加统一场景资源入口；并非官方完整训练仓库的逐字复制。 |

官方来源：https://github.com/tylerlum/simtoolreal 。公开下载地址：https://download.cs.stanford.edu/juno/simtoolreal/pretrained_policy.zip 。官方 config.yaml 的 SHA256 为147717cfdf1d19c06d5d443e428d84f8c2adb0b00abc7c159f6a3fa1ea7becea；原版含上述两字段的配置为af8c711dc252a14837a28ae3f4f02b78d90ea2a4aeb3bbde61f5163b4ec36779。详见official_weight_parity.json、policy_config.diff。

## 本轮恢复原版行为

整理时曾给编译器接触阶段添加6 mm支持面穿入放宽，并在入口加0.01 mm接触误差拒绝门槛。逐项对照后均移除：编译器恢复原版零穿桌约束，路径检查继续使用原版已有的allowed_contact_penetration_m参数，入口沿用原版编译与路径状态。接触残差及求解警告仍保存，厘米级偏差仍是实际问题。

对齐前的九组目标与原版最大分量差为0至6 mm，见compiler_parity.json；对齐后全部为0，见compiler_parity_aligned.json。去掉了专门测试新增拒绝门槛的用例，现有软件回归为107项，全部通过；没有删掉原版几何与路径回归。

保留已明确的方法整理差异：统一公共参数、相机射线与交互平面求交、保留显式轴绑定、修正端点计算、去掉评价控制回退和首次渲染同步修复。以上意味着完整旧runner与新入口不保证轨迹、任务结果逐位相同。同输入编译对照的输入已在compiler_inputs/保存，两版输出仅用于诊断，不直接执行。

## 实际复制复现

新建Python3.8虚拟环境，仅根据requirements安装依赖，不复用历史site-packages；安装版本记录在fresh_requirements.lock，随包requirements-tested.txt可安装同一组版本。

将交接目录完整复制到core_parity/copied_handoff，从/tmp作为工作目录运行，移除PYTHONPATH以及外部SDK、策略和VLM配置。复制版使用自身isaacgym/python与pretrained_policy，首次编译gymtorch，重新GPU渲染，然后用配对新观测、缓存人工关键点和缓存工具引用运行完整plan+execute。锤子完成26/26目标，445步。面标签prepare-tool同时重新生成先验与16视图缓存。新环境软件回归107 passed（1.95秒）。

本地完整交接目录中策略和SDK是独立副本；SDK的内部相对库链接均有效，无指向历史项目的外部链接。它们被Git忽略，因此复制本地完整目录与仅git clone源码的资源范围不同。操作见交接包NON_VLM_REPRO.md。

## 对齐后的九场景真实执行

以下使用原固定人工像素输入、原选定工具引用和统一settings，均从正式run.py入口生成并执行目标；没有用诊断投影文件执行。

| 场景 | 控制状态 | 目标进度 | 步数 |
| --- | --- | --- | --- |
| blue_brush_sweep_block | completed | 36/36 | 971 |
| claw_hammer_strike | completed | 25/25 | 358 |
| handle_eraser_tabletop | completed | 42/42 | 543 |
| handle_eraser_whiteboard | completed | 30/30 | 730 |
| long_screwdriver_insert | completed | 21/21 | 299 |
| mallet_hammer_strike | completed | 26/26 | 396 |
| marker_whiteboard | completed | 30/30 | 745 |
| red_brush_sweep_block | timeout | 33/36 | 1500 |
| short_screwdriver_insert | timeout | 21/23 | 1500 |

九场景均可编译并启动真实策略，七个控制序列完成，红刷子与短螺丝刀超时。桌面擦除在对齐后本次完成，但单次结果不能证明稳定成功。completed依然只表示目标推进完成，没有把扫落、插入、擦除等实际效果判定加入交接包。

## 复现边界

已验证的范围是现有八种准备好的工具：标签到几何、工具视图、固定输入lifting/编译、真实GPU渲染和预训练策略执行。新网格从零进行P3SAM分割需要独立Python/CUDA环境及P3SAM/Sonata权重；虽已定位本地模型，当前候选环境缺fpsample、spconv等依赖，尚未通过从零安装与实际推理验收。不能把八工具的准备后复现说成新工具分割也已全部验证。

新工具先验生成到包外后再单独搬迁的旧路径问题尚未修复；本次重建放在包内outputs，生成引用相对于包根，整体复制可用。此前任务/工具目标绑定和重用失败输出的问题也未在本轮扩展实现。推荐每次使用新输出目录，避免混用任务输入和目标。

随仓库提供的证据：policy_parity.json、geometry_parity.json、compiler_parity_aligned.json、prior_rebuild.json、reproduction_result.json及gpu_aligned/execute_summary.json。完整输入、两版诊断输出、策略观测样本和各阶段日志保存在本地验收目录，不包含在源码发布中；本页涉及core_parity/下的目录名为本地验收记录的位置。原无VLM实验记录和其snapshot_manifest保留为旧版本结果。
