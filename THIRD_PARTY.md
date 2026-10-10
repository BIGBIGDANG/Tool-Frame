# 第三方代码与资源

本页记录交接包中的来源及保留文件，不代表全部资产已经完成公开分发核查。项目尚未选择根级许可证。

| 内容 | 来源与现有说明 |
| --- | --- |
| SimToolReal 仿真与策略代码 | 从本地 ToGo6.13 整理；上游为 [tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal)。保留 [MIT 许可证](simtoolreal/LICENSE)。整理版精简训练依赖并调整场景入口。 |
| IsaacGymEnvs 代码 | 保留 [BSD 3-Clause 许可证](simtoolreal/isaacgymenvs/LICENSE.txt)。该文件另要求关注资产各自的许可证，不能将代码许可证直接视为所有模型资产的许可。 |
| rl_games 代码 | 保留 [MIT 许可证](simtoolreal/rl_games/LICENSE)。 |
| 桌面纹理 | Poly Haven wood_table；保留 [来源与 CC0 说明](simtoolreal/assets/urdf/scene/tabletop_polyhaven_wood_table/LICENSE.md)。 |
| KUKA iiwa 资源 | 保留原版 [BSD 2-Clause 许可证](simtoolreal/assets/licenses/kukaiiwa-LICENSE.txt)。该说明不覆盖组合模型中的 Sharpa 手部资源。 |
| 机器人、八种工具的网格、材质与物理资源 | 从本地 ToGo6.13 的 simtoolreal/assets 中选取。当前没有完整的逐资产来源及许可对应表，公开打包前需补齐；不从上层代码许可证推断资产许可。 |
| 工具先验与缓存视图 | 基于上述工具网格及原有 P3SAM 面标签生成；资产来源随底层工具网格一起核对。交接包不包含 P3SAM/Sonata 模型与完整运行环境。 |
| 预训练策略 | 本地完整目录在 pretrained_policy/ 保存原版 config.yaml 与 model.pth；权重与官方公开下载包逐字节相同。被 Git 忽略，源码使用者自行获取；下载来源与哈希见 [NON_VLM_REPRO.md](NON_VLM_REPRO.md)。 |
| IsaacGym SDK | 本地完整目录在 isaacgym/python/ 保存 SDK 与原许可证，仅用于已测试的运行环境。被 Git 忽略，不纳入 Git 发布；源码使用者自行准备 NVIDIA SDK。 |

本次按项目所有者要求公开研究预览源码，未替项目选择根许可证，也未将待核查的资产标为开放许可。后续仍需确定自有代码的根许可证，完成机器人及工具资产的来源和分发范围核对，并继续保留以上第三方声明。本地完整交接目录与 Git 源码的资源范围不同。
