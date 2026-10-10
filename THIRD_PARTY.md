# Third-party components and resources

| Component | Source and retained notice |
| --- | --- |
| Simulation backend and policy integration | Adapted from [SimToolReal](https://github.com/tylerlum/simtoolreal); [MIT license](simtoolreal/LICENSE). Training dependencies are trimmed and scene/fixed-goal integration is adapted. |
| IsaacGymEnvs code | [BSD 3-Clause license](simtoolreal/isaacgymenvs/LICENSE.txt). Code licensing does not cover every asset automatically. |
| rl_games code | [MIT license](simtoolreal/rl_games/LICENSE). |
| KUKA iiwa assets | [BSD 2-Clause notice](simtoolreal/assets/licenses/kukaiiwa-LICENSE.txt). This notice does not cover the combined model's Sharpa hand assets. |
| Table texture | Poly Haven wood_table; [source and CC0 notice](simtoolreal/assets/urdf/scene/tabletop_polyhaven_wood_table/LICENSE.md). |
| Robot/tool meshes, materials, and physical models | Included under `simtoolreal/assets/`. Complete per-asset source and redistribution attribution is not yet available, including Sharpa hand and tool model attribution. These assets are not designated as covered by the project code license. |
| Prepared tool priors and views | Derived from the included tool meshes and P3SAM face labels. Underlying mesh attribution applies. |

External runtime resources are obtained separately:

- **SimToolReal policy:** official checkpoint and configuration; download source,
  model hash, and configuration differences are in [runtime setup](docs/reproducibility.md).
- **IsaacGym SDK:** obtain the NVIDIA SDK separately under its supplied terms.
- **P3SAM/Sonata:** models and a separate segmentation environment are required
  only for new-mesh segmentation, not for the prepared-tool examples.

SDK binaries, pretrained weights, credentials, and generated runs are excluded
from source commits. The project-level license remains unselected; see
[LICENSES.md](LICENSES.md). Asset attribution gaps remain unresolved.
