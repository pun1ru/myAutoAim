# EKF 网页调参说明

网页入口：`EKF Tuning`。每一行修改后点击该行的 `Apply`。

网页显示名称与代码接口名称不同。接口通过 `/api/control?action=ekf-param:<name>:<value>`
提交；当前值通过 `/api/status` 的 `ekf_tuning` 返回。

## 生效规则

- `Initial uncertainty` 五项只影响下一次初始化。修改后点击 `Reset EKF Track` 才能立即重新初始化。
- 其余 Q、R、关联、几何和 NIS 参数在下一帧开始生效，不会清空当前轨迹。
- `Yaw valid` 表示约束重投影 yaw 解算成功；`Yaw used` 表示该 yaw 通过关联、门控和 NIS 检查，并实际进入 EKF 更新。
- 控制 `Yaw used` 是否容易出现的网页项是 `Yaw used gate (EKF update)`，接口名仍为
  `maximum_yaw_update_innovation_rad`。
- 控制 `Yaw valid` 的四项位于网页的 `Yaw validity` 分组，修改后下一帧重新解算即可生效。

## Initial uncertainty

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Initial position std | `initial_position_std_m` | 初始车体中心位置不确定度 | 初始更容易被观测修正 | 初始更依赖首帧推算 |
| Initial velocity std | `initial_velocity_std_mps` | 初始平移速度不确定度 | 更快估计速度，可能抖 | 速度更稳但启动慢 |
| Initial theta std | `initial_theta_std_rad` | 初始 E0 连续 yaw 不确定度 | 更容易修正初始相位 | 初始相位更固定 |
| Initial omega std | `initial_omega_std_rad_s` | 初始自转角速度不确定度 | 更快响应转速变化 | 转速估计更平滑 |
| Initial geometry std | `initial_geometry_std_m` | 初始半径、半径差、高度差不确定度 | 几何更容易修正 | 几何更稳定但更难修正 |

## Process noise Q

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Linear acceleration density | `q_linear_acceleration` | 车体中心平移加速度噪声 | 移动跟随更快，中心更抖 | 中心更平滑但滞后 |
| Angular acceleration density | `q_angular_acceleration` | 自转角速度变化噪声 | 更快跟随变速旋转 | 匀速旋转更稳但变速滞后 |
| Geometry random walk | `q_geometry` | 半径和高度差随机游走 | 几何会随时间漂移 | 几何保持固定；静态车建议为 0 |

## Measurement noise R

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Position XY std | `position_std_xy_m` | PnP 横向位置标准差 | 更不信当前观测，平滑但滞后 | 更信当前观测，响应快但抖 |
| Position Z std | `position_std_z_m` | PnP 深度位置标准差 | 深度更平滑 | 深度更快但噪声大 |
| Yaw std | `yaw_std_rad` | yaw 基础测量标准差 | yaw 对 EKF 影响变弱 | yaw 更强，错误 yaw 影响更大 |
| Reprojection RMS scale | `reprojection_rms_scale` | 重投影 RMS 对 R 的放大系数 | 差角点自动降权更明显 | RMS 对权重影响变弱 |
| Range noise scale | `range_noise_scale_per_m` | 距离增加时的噪声增长率 | 远处观测降权更多 | 远近观测权重差异小 |
| Minimum quality | `minimum_quality` | 检测质量下限 | 更容易拒绝低质量观测 | 接受更多观测，也更易进坏数据 |

## Multi-armor weighting

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Single-plate variance scale | `single_armor_position_variance_scale` | 单板位置观测额外降权倍率 | 单板不易拉偏中心，响应变慢 | 单板修正快，但中心易漂 |
| Association position scale | `association_position_variance_scale` | 关联阶段位置门控放宽倍率 | 更容易找回目标，也更易错关联 | 关联更严格，可能漏关联 |
| Max multi-plate residual | `maximum_multi_armor_position_residual_m` | 双板共同更新允许的最大位置残差 | 双板更新更容易触发，但坏数据风险大 | 双板更严格，几何更可靠 |

## Slot association

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| **Yaw used gate (EKF update)** | `maximum_yaw_update_innovation_rad` | yaw 创新多大时仍允许进入 EKF | `Yaw used` 更容易为 true，但错误 yaw 风险更大 | yaw 更严格，常变成 position-only |
| Yaw association gate | `maximum_yaw_association_innovation_rad` | 候选槽位关联允许的最大 yaw 差 | 更容易切到新槽位，也更易错槽 | 关联严格，可能卡在 E0 |
| Yaw phase cost std | `yaw_phase_cost_std_rad` | yaw 差异在槽位代价中的权重尺度 | 降低 yaw 差异影响 | 更依赖 yaw 匹配 |
| Adjacent slot penalty | `adjacent_slot_penalty` | 切换到相邻槽位的额外代价 | 不易切 E0/E1 | 更容易切换 |
| Opposite slot penalty | `opposite_slot_penalty` | 切换到对面槽位的额外代价 | 更禁止跳对面 | 更容易跳槽但错配风险大 |
| Minimum visibility cosine | `minimum_visibility_cosine` | 槽位朝向相机的可见性门限 | 只接受正面板 | 接受更多侧面板 |

相邻装甲板 yaw 间隔约为 `pi/2 = 1.57 rad`。如果旋转时 E0 不切换，通常先检查：

```text
Yaw association gate = 1.8
Yaw phase cost std = 0.7
Adjacent slot penalty = 0.1
Opposite slot penalty = 1.0
```

`Yaw used gate` 只控制 yaw 是否进入 EKF 更新，不负责单独决定槽位；槽位选择主要由
`Yaw association gate`、位置 NIS 和槽位切换惩罚共同决定。

## Geometry gate

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Geometry yaw consistency | `geometry_yaw_consistency_rad` | 双板 yaw 相位一致性门限 | 更容易进行双板几何更新 | 几何更新更严格 |
| Minimum plate baseline | `geometry_minimum_baseline_m` | 双板间最小有效空间基线 | 只用分离明显的双板 | 更多双板可触发，但可观测性差 |
| Geometry confirming frames | `geometry_confirming_frames` | 连续多少帧才确认双板几何 | 半径更稳定但修正慢 | 半径修正快但易受坏帧影响 |

## NIS gate

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Position NIS gate | `nis_gate_3d` | position-only 三维观测门限 | 接受更多观测，也更易收坏数据 | 更严格，可能只预测 |
| Position+yaw NIS gate | `nis_gate_4d` | position+yaw 四维观测门限 | `Yaw used` 更容易通过 NIS | 四维观测更容易被拒绝 |

## Yaw validity

这四项在 EKF 之前由约束重投影 yaw solver 使用，决定 `Yaw valid` 是否为 true。

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Yaw max reprojection RMS | `yaw_max_reprojection_rms_px` | yaw 候选四角重投影 RMS 上限 | 更容易判定 yaw 有效，但可能接受错误姿态 | 更严格，可能使 yaw valid 变少 |
| Yaw max standard deviation | `yaw_max_std_rad` | yaw 曲率估计标准差上限 | 接受不确定度更大的 yaw | 只接受更稳定的 yaw |
| Yaw minimum facing cosine | `yaw_min_facing_cosine` | 装甲板朝向相机的最低余弦值 | 只接受更正面的板，valid 变少 | 接受更多侧向板，但偏差更大 |
| Yaw opposite-solution margin | `yaw_min_opposite_margin_px` | 正反候选 RMS 至少要拉开的像素差 | 要求正反解更明确，valid 变少 | 更容易接受正反不够分明的 yaw |

这四项只影响 yaw solver 的 `valid` 判定，不会直接改变 EKF 的 `used` 判定。
例如 `Yaw valid=true` 但 `Yaw used=false`，还要继续检查 `Yaw used gate (EKF update)`、
`Yaw association gate` 和 `Position+yaw NIS gate`。

## Projection Debug

`Projection Debug` 是独立于 EKF 的投影验证模式，不会写入 EKF 状态。

- 点击 `Enable` 后，画面中的橙色 `D0-D3` 是手动几何模型投影，白色 `PnP ref` 是当前观测板的 PnP 中心。
- 默认 `Anchor: observed`：当前第一块带可靠 yaw 的观测板定义为 E0，使用它的 PnP 中心和 yaw 计算 `cx/cy/cz/theta`，网页主要调 `Debug r0/dr/dz`。
- 点击 `Anchor: manual` 后，`Debug center cx/cy/cz` 和 `Debug theta` 也完全由网页输入，适合验证固定世界坐标投影。
- `Debug r0`、`Debug dr`、`Debug dz` 分别对应偶数板半径、奇数板半径差和奇数板高度差。
- 当前运行配置固定 `dr=0` 和 `dz=0`，因此 `Debug dr`、`Debug dz` 都保持为 0，不用于修改实际 EKF 几何。
- `r0` 默认值已统一为 `0.15 m`；单次调试可在网页修改 `Debug r0`，但重启后回到该默认值。
- 这个模式只验证“给定 T/odom 三维点能否正确投影回图像”，不参与数据关联、EKF 更新或火控。

投影调试接口命令为：

```text
projection-debug-toggle
projection-debug-anchor-toggle
projection-debug-param:<name>:<value>
```

## Angular limits

| 网页名称 | 接口名称 | 意义 | 调大 | 调小 |
|---|---|---|---|---|
| Maximum angular speed | `maximum_angular_speed_rad_s` | 限制 EKF 的最大车辆自转角速度 | 允许更快旋转，减少速度饱和 | 抑制异常角速度，但真实高速旋转会滞后 |
| Maximum omega correction/frame | `maximum_omega_correction_rad_s` | 每帧允许 omega 修正的最大幅度 | 旋转变速跟随更快 | omega 更平滑但更滞后 |

## Yaw pitch constraint

当前 yaw solver 从 PnP 的 `rotation_camera_from_armor` 提取装甲板法向的 pitch，
只把该 pitch 作为 yaw 搜索的姿态约束；PnP 的 Rodrigues yaw 仍不会直接写入 EKF yaw。

## 推荐排查顺序

1. 先看 `Yaw valid` 是否为 true；若为 false，调 EKF 参数没有意义，应检查 yaw 解算和重投影。
2. `Yaw valid=true` 但 `Yaw used=false` 时，先看 `Yaw used gate (EKF update)` 和 `Position+yaw NIS gate`。
3. 旋转时 E0 不切换时，调 `Yaw association gate`、`Yaw phase cost std` 和槽位惩罚。
4. 车体中心滞后时，再调 `Linear acceleration density`、`Position XY std` 和单板倍率。
