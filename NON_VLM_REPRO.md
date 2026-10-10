# 无 VLM 复现

本页复现已准备工具的几何、位姿编译与真实 GPU 策略执行。示例关键点是对实际渲染图像的固定人工标注，明确记录在 `fixture_source`，用于替代在线 VLM 输入。它验证运行流程，不表示锤击效果已通过评价。

## 复制与安装

本机完整交接目录含以下普通文件副本，没有指向历史项目的符号链接：

- `pretrained_policy/config.yaml` 与 `model.pth`：原 6.13 实际使用的 SimToolReal 预训练策略。
- `isaacgym/python/`：本机实际运行的 IsaacGym Python 3.8 SDK。
- `assets/tool_prior/`：八种工具的 P3SAM 面标签、部件图及视图。
- `simtoolreal/assets/`：机器人、工具及九场景的物理资源。
- `examples/observations/mallet_hammer/`：配对 RGB-D、相机标定及固定关键点。

可用 `cp -a ToGo_handoff_clean /your/destination/` 复制本地完整目录。`git clone` 不包含被忽略的权重和 SDK；权重可按 [SimToolReal 官方项目](https://github.com/tylerlum/simtoolreal) 的下载流程准备，SDK 使用 NVIDIA 的 IsaacGym 安装包。本地 SDK 保留原许可证，不纳入 Git 发布。

使用 Python 3.8、NVIDIA 驱动和 C++ 编译器，在新环境中安装：

```bash
cd /your/destination/ToGo_handoff_clean
python -m pip install -r requirements-tested.txt
export TOGO_ISAACGYM="$PWD/isaacgym/python"
export TOGO_PRETRAINED_POLICY_ROOT="$PWD/pretrained_policy"
export TOGO_RUNTIME_PYTHON="$(command -v python)"
```

IsaacGym 需要可加载的 `libpython3.8.so`。若 Python 安装未将其目录注册到动态链接器，将该安装的 `lib/` 加入 `LD_LIBRARY_PATH`。SDK 首次运行会编译 gymtorch，需要 C++ 编译器和 ninja。

## 固定观测直接运行

以下命令给齐三个离线输入，不调用工具选择或场景 grounding 的 VLM：

```bash
python run.py run \
  --task examples/mallet_hammer_strike.json \
  --scene-evidence examples/observations/mallet_hammer/scene_evidence.json \
  --scene-keypoints examples/observations/mallet_hammer/scene_keypoints.json \
  --tool-refs assets/tool_refs/mallet_hammer.json \
  --output /tmp/togo_non_vlm \
  --device cuda:0
```

若要重跑 GPU 渲染，先生成新观测：

```bash
python ToGo/src/ToGo/cli/stages/render_simtoolreal_rgbd_scene_evidence.py \
  --task examples/mallet_hammer_strike.json \
  --output-dir /tmp/togo_observation --device cuda:0
```

再将上面的 `--scene-evidence` 改为 `/tmp/togo_observation/scene_evidence.json`。相机及场景使用示例配置；固定像素只适用于该配置，移动相机或物体后需要重新标注。

## 从已有面标签重建先验

八种工具的面标签已经包含在交接目录，使用原版几何生成器，无需再次加载分割模型：

```bash
python run.py prepare-tool \
  --tool-id mallet_hammer \
  --mesh simtoolreal/assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.obj \
  --face-labels assets/tool_prior/mallet_hammer/face_labels/p3sam_face_labels.npy \
  --output outputs/rebuilt_prior
```

本例输出放在包内，生成的资源引用相对于包根目录，整体复制仍有效。新网格的 P3SAM 分割需要独立的 P3SAM Python/CUDA 环境、P3SAM 与 Sonata 权重；它不属于 IsaacGym Python 3.8 环境，尚未通过本轮从零安装验收。现有八工具复现不依赖它。

## 原版对应关系

从官方公开下载包独立核对后，`model.pth` 与本地原版完全相同。原版配置仅比官方配置多两个环境字段：`resetWhenTableForceTooHigh:true` 和 `parttoolgenBenchmarkExtraActors:[]`；网络配置相同。

原版资源 SHA256：

```text
config.yaml  af8c711dc252a14837a28ae3f4f02b78d90ea2a4aeb3bbde61f5163b4ec36779
model.pth    4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9
```

保留原生 140 维观测、29 维动作、额外固定标量 50、观测归一化、LSTM 状态和确定性动作推理。整理版策略加载器精简了训练相关依赖；与原版连续 128 步固定观测对照，动作和 LSTM 状态最大差值均为 0。

原版核心接收同一份几何与 ProgramIR 的九场景对照，所有固定目标位姿逐项相同。完整旧 runner 与新入口存在已明确的差异：共享参数替代逐任务参数，平面 lifting 改为相机射线求交，折线路径保持显式动作轴，修正端点计算，并移除评价影响执行的回退逻辑。因此不承诺旧实验轨迹和成功率逐位复现。

本轮在全新 Python 3.8 环境安装并整体复制到另一目录后，重新 GPU 渲染、编译和策略执行完成26/26目标（445步）；标签重建先验与工具视图也通过。软件回归107项通过。九场景复跑有七个控制序列完成，红刷子与短螺丝刀超时。
