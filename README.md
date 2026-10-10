# Tool-Frame / ToGo 6.13

独立运行的工具操作代码，包含工具分割先验、VLM 工具引用选择、场景关键点 grounding、几何 lifting、分阶段位姿编译和固定目标策略执行。所有任务使用同一入口和同一组参数。

目录可以整体移动。包含八种工具的几何输入和九份物理场景示例；代码结构与维护约定见 [HANDOFF.md](HANDOFF.md)。

不调用 VLM 的复现命令、预训练策略来源和复制所需资源见 [NON_VLM_REPRO.md](NON_VLM_REPRO.md)。

当前为研究预览版：全新 Python 3.8 环境中软件回归 107 项通过，整体复制后重新渲染与策略执行已跑通。九场景均能编译并启动原生预训练策略，本次七个控制序列完成，红刷子和短螺丝刀超时；完成控制序列不代表任务效果成功。在线 VLM 和新网格从零 P3SAM 分割尚未验收。证据见 [验收报告](docs/validation/REPORT.md)。根许可证尚未选择，部分模型资产的来源说明待补齐，见 [LICENSES.md](LICENSES.md) 和 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 安装

核心编译、工具视图生成和软件测试使用 Python 3.8：

```bash
git clone https://github.com/BIGBIGDANG/Tool-Frame.git
cd Tool-Frame
python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
python run.py doctor
```

`doctor` 列出当前解释器中的依赖和外部资源配置，不执行 GPU 检查。

完整仿真需要 NVIDIA GPU、与 Python 3.8 匹配的 IsaacGym SDK，以及原 6.13 控制策略的 `config.yaml` 和 `model.pth`。本机交接目录已在被 Git 忽略的 `isaacgym/python/` 和 `pretrained_policy/` 放置独立副本；只获取 Git 源码时需要另行准备：

```bash
python -m pip install -r requirements-runtime.txt
export TOGO_ISAACGYM=/absolute/path/to/isaacgym/python
export TOGO_PRETRAINED_POLICY_ROOT=/absolute/path/to/pretrained_policy
```

在线标注时，由运行环境设置 `VLM_MODEL` 和 `OPENAI_API_KEY`（也接受 `VLM_API_KEY`）；模型需要支持图像输入及 JSON 输出。兼容服务可通过 `VLM_API_BASE` 配置。交接目录不存放凭据或个人接口配置，也不保存原始模型回答。

如核心代码与 IsaacGym 使用不同环境，设置 `TOGO_RUNTIME_PYTHON=/absolute/path/to/runtime/python`，或传入 `--runtime-python`。`doctor` 检查的是调用它的解释器。

## 统一运行入口

下面所有命令的输出目录都应放在交接包之外，或放在被忽略的 `outputs/` 中。

完整流程：读取场景配置，渲染一幅配对 RGB-D，选择工具引用并标注场景点，编译固定目标，再执行。

```bash
python run.py run \
  --task examples/mallet_hammer_strike.json \
  --output /tmp/togo_run \
  --device cuda:0
```

只做规划：

```bash
python run.py plan \
  --task examples/mallet_hammer_strike.json \
  --output /tmp/togo_plan \
  --device cuda:0
```

有自己的相机观测和关键点时，可以离线编译，不需要 VLM 或 GPU：

```bash
python run.py plan \
  --task examples/mallet_hammer_strike.json \
  --scene-evidence /absolute/path/to/scene_evidence.json \
  --scene-keypoints /absolute/path/to/scene_keypoints.json \
  --tool-refs assets/tool_refs/mallet_hammer.json \
  --output /tmp/togo_plan
```

缓存引用必须与当前工具及用途匹配。随包的 `assets/tool_refs/` 仅保存已选定的工具几何，不替代新场景的 grounding，也不保证与任意接触平面兼容。

执行已经编译的目标：

```bash
python run.py execute \
  --task examples/mallet_hammer_strike.json \
  --fixed-goals /tmp/togo_plan/fixed_goals.json \
  --output /tmp/togo_execution \
  --device cuda:0
```

规划输出为 `constraint_program.json`、`tool_pose_subgoals.json`、`fixed_goals.json`。执行输出 `execution_state.json`，其 `completed` 表示控制器完成目标序列，`timeout` 表示步数耗尽。

## 工具先验

已有的八种工具可以直接使用 `assets/tool_prior/`。为新网格生成先验：

```bash
python run.py prepare-tool \
  --tool-id my_tool \
  --mesh /absolute/path/to/my_tool.obj \
  --p3sam-root /absolute/path/to/P3SAM \
  --p3sam-python /absolute/path/to/p3sam/python \
  --output outputs/my_tool_prior
```

也可用 `--face-labels /path/to/labels.npy` 替代 P3SAM 调用。标签必须是一维整数数组，逐面对应原网格；网格以米为单位，坐标系必须与执行使用的工具模型一致。P3SAM 调用关闭网格清理，以保持面编号。

结果在 `outputs/my_tool_prior/my_tool/`，包括 `tool_prior/`、分割标签、网格副本和工具视图。场景配置的 `tool.tool_prior_dir` 指向生成的 `tool_prior/`。先验放在包内时可随整个目录移动；包外生成的先验含绝对路径，单独移动后需重新生成。

独立选择工具引用：

```bash
python run.py tool-ref \
  --task examples/mallet_hammer_strike.json \
  --output /tmp/togo_tool_refs
```

工具端统一采用“选择工作部件，再选择接触面与工具局部动作轴”的两阶段 VLM 流程。添加新工具做仿真执行时，还需要 URDF 和 `simtoolreal/dextoolbench/objects.py` 中的物理资源登记及策略握持尺寸；`prepare-tool` 只负责几何先验。

## 输入约定

`examples/*.json` 给出完整场景配置。`task_id` 只用于标识；方法代码不按任务名分支。`instruction` 描述操作意图，`tool` 指定工具和先验，`execution` 指定工具初始位姿、物理资源及场景物体。位姿格式为 `[x,y,z,qx,qy,qz,qw]`，长度单位为米。

`metadata.public_support_surface_point_world` 和 `metadata.public_support_surface_normal_world` 描述支持面几何。场景物体初始位置仅用于搭建仿真，不作为 VLM 目标点的替代来源。

如果存在交互平面，通过 `interaction_plane` 指定：

```json
{
  "point_world": [0, 0, 0.53],
  "normal_world": [0, 0, 1],
  "point_roles": ["start", "end", "path"],
  "source": "provided_scene_geometry"
}
```

适用点通过相机射线与平面求交得到三维坐标。插入操作可设 `point_roles:["start"]`，入口落在平面上，深入终点保留 RGB-D 深度。`anchor_from_depth:"start"` 可替代显式 `point_world`，以起点深度确定平面锚点，并结合提供的法向估计局部平面。

外部 `scene_evidence.json` 至少包含如下配对关系；文件路径相对于该 JSON，也可以使用绝对路径：

```json
{
  "instruction": "Use this tool to act on the visible target.",
  "items": [
    {"id":"scene_rgb", "kind":"scene_rgb", "path":"rgb.png"},
    {"id":"scene_depth", "kind":"scene_depth", "path":"depth.npy"},
    {"id":"scene_camera", "kind":"camera", "path":"camera.json"}
  ],
  "metadata":{"prompt_images":["rgb.png"], "grounding_image_index":0}
}
```

深度为正的相机 Z 深度，形状为 `[height,width]`，单位为米。`camera.json` 包含 `width,height,fx,fy,cx,cy,camera_to_world`；相机坐标约定为 X 向右、Y 向下、Z 向前。深度、相机和被标注 RGB 必须来自同一视图。

`scene_keypoints.json` 使用 `start_point:{u,v}`、可选 `end_point:{u,v}` 和有序 `path_points:[{u,v},…]`。方向可由 `action_direction:{start:{u,v},end:{u,v}}` 给出，或由 `direction_decision.selected_default_axis_id` 选择 `support_surface_normal` / `opposite_support_surface_normal`。也可从非零路径段推导方向；单点操作必须给出方向或平面法向选择。

共享参数在 `settings.json`，所有任务使用同一组默认值，也可通过 `--settings` 传入另一份全局配置。输入无效或原版路径检查失败时会报错；接触残差与求解警告保存在 `tool_pose_subgoals.json`，不另加接触误差门槛。
