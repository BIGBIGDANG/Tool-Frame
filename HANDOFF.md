# 维护说明

运行与输入格式见 [README.md](README.md)。所有任务使用 `run.py` 和共享的 `settings.json`，通过指令、工具几何、观测及场景物体表达任务差别。

## 方法流程

1. 根据网格面分割标签构建工具部件图和接触、轴候选。
2. VLM 先选择工作部件，再选择接触面和工具局部动作轴。
3. 在配对的 RGB-D 观测中标注场景关键点及方向。
4. 将适用关键点沿标定相机射线 lift 到交互平面，其余点使用深度 lifting。
5. 编译 approach、contact、action 阶段，细分路径，生成固定工具位姿目标。
6. 加载外部控制策略，依次执行目标。

## 代码位置

| 位置 | 职责 |
| --- | --- |
| `pipeline/handoff.py` | 命令入口和流程编排 |
| `pipeline/prepare_tool.py` | 工具分割与先验准备 |
| `pipeline/lifting.py` | 标定配对和交互平面 lifting |
| `cli/tool_prior/` | 工具证据视图与引用选择 |
| `cli/stages/` | 场景 RGB-D 观测与关键点标注 |
| `image_grounded/staged_mainline_adapter.py` | 工具、场景引用到分阶段程序的转换 |
| `compiler/`、`checks/`、`adapters/` | 位姿求解、路径细分和支持面几何处理 |
| `program/`、`constraints/`、`validation/` | 程序表示、模型调用与输入检查 |
| `providers/` | 工具几何先验与视图缓存 |
| `execution/` | 固定目标控制与 SDK 适配 |

以上路径相对于 `ToGo/src/ToGo/`。`simtoolreal/` 提供仿真环境、机器人资源和策略推理依赖；对应许可证随目录保留。`ToGo/tests/` 用于软件回归检查。

维护代码时优先保持直接、简洁：外部输入在入口检查一次，内部使用已确定的数据格式；避免重复校验、猜测性 fallback、只转发参数的包装和为未来需求预建抽象。几何求解与控制所需约束保留，注释解释不直观的原因。

## 数据与坐标

工具网格、分割标签、部件图、工具引用及 URDF 必须使用一致的工具局部坐标系和米制单位。面标签与网格三角面逐一对应。`assets/tool_prior/` 中的视图缓存用于选择工作部件和接触面。

RGB、深度及相机标定必须来自同一视图。存在交互平面时，在 `interaction_plane` 中声明平面及适用点角色。插入操作可仅将入口起点 lift 到平面，深入终点保留深度。显式绑定的工具动作轴沿路径保持同一绑定。

编译器使用最终选定的工具引用，保留原 6.13 的支持面约束与路径检查。接触误差通过原有残差和求解警告呈现；入口不追加接触误差拒绝条件。`allowed_contact_penetration_m` 用于原版路径检查，不改变编译器的零穿桌约束。

## 扩展与运行资源

新增任务从 `examples/*.json` 的输入格式开始，共用同一编译和执行流程。新增工具先运行 `prepare-tool`；进行仿真执行时，还需提供 URDF，并在 `simtoolreal/dextoolbench/objects.py` 登记物理资源和握持尺寸。

控制策略保留 140 维观测、29 维动作及其内部输入编码。`goalTolerance` 对应共享的目标到达容差，`successes` 是后端推进固定目标的计数器；`sceneActors` 接收实际场景物体。

本地完整交接目录含独立的 IsaacGym SDK 与策略权重副本，二者被 Git 忽略。Git 源码的使用者需另行准备这些资源；来源及复现说明见 [NON_VLM_REPRO.md](NON_VLM_REPRO.md)。VLM 配置与凭据由环境提供。输出路径建议放在包外。

核心几何与编译支持 CPU 运行。完整仿真需要兼容的 Python 3.8、IsaacGym 和 NVIDIA GPU，并在允许访问 GPU 的环境中执行。当前主机在沙箱外可识别八张 NVIDIA RTX 4090 D；沙箱内的驱动访问失败不能据此判断主机没有可用 GPU。已核对原版核心和官方预训练权重，并验证新环境及目录复制后的无 VLM 运行。九个示例均可编译并启动策略，本次七个控制序列完成、两个超时；完整在线 VLM 流程尚未验证。控制目标序列完成不等于实际任务效果已完成。
