# 整车 EKF 数学模型与网页调参对应表

本文严格描述当前 `WholeVehicleEkf` 的实际实现。跟踪坐标系 `T` 在一条
轨迹内固定为 ROS odom；位置单位为 m，角度单位为 rad。

本文所有运行时网页/API 参数均通过下列形式提交：

```text
/api/control?action=ekf-param:<参数名>:<数值>
```

例如 `ekf-param:q_angular_acceleration:2.0` 修改角加速度过程噪声谱密度。
以 `yaw_max_` 开头的参数属于上游约束重投影 yaw 求解器，不属于 EKF 本身。

## 1. 状态与装甲几何模型

第 $k$ 帧的 11 维后验状态与协方差为：

```math
\mathbf{x}_k =
\begin{bmatrix}
c_x & v_x & c_y & v_y & c_z & v_z & \theta & \omega & r_0 & \Delta r & \Delta z
\end{bmatrix}^{T},
\qquad
\mathbf{P}_k = \operatorname{Cov}(\mathbf{x}_k).
```

| 物理符号 | 实际含义 | 程序状态下标 |
|---|---|---|
| $c_x,c_y,c_z$ | 车辆旋转中心在 `T` 系的位置 | `CenterX`, `CenterY`, `CenterZ` |
| $v_x,v_y,v_z$ | 车辆中心在 `T` 系的速度 | `VelocityX`, `VelocityY`, `VelocityZ` |
| $\theta$ | 物理槽位 E0 的连续 inward yaw | `Theta` |
| $\omega$ | 自转角速度 | `Omega` |
| $r_0$ | 偶数槽位 E0/E2 的旋转半径 | `RadiusEven` |
| $\Delta r$ | 奇偶槽位半径差 | `RadiusOddDelta` |
| $\Delta z$ | 奇偶槽位高度差 | `HeightOddDelta` |

对物理装甲槽位 $i\in\{0,1,2,3\}$，定义：

```math
p_i=i\bmod2,\qquad
\phi_i=\theta+i\frac{\pi}{2},\qquad
r_i=r_0+p_i\Delta r.
```

该槽位装甲中心和 inward yaw 的预测值为：

```math
\mathbf{p}_{A,i}(\mathbf{x})=
\begin{bmatrix}
c_x-r_i\cos\phi_i\\
c_y-r_i\sin\phi_i\\
c_z+p_i\Delta z
\end{bmatrix},
\qquad
\psi_{A,i}(\mathbf{x})=\phi_i.
```

这对应 `observe(state, armor_slot)`。虽然 $\Delta r$ 与 $\Delta z$ 在状态中，
当前更新路径会在每次更新后将二者及其协方差行列清零。因此目前真正可估计的
几何量只有 $r_0$。

## 2. 预测模型

令 $\Delta t=\texttt{dt_s}$，它由相邻图像曝光时间戳计算。均值采用匀速、匀角速度
模型：

```math
\mathbf{x}_{k|k-1}=\mathbf{F}(\Delta t)\mathbf{x}_{k-1|k-1},
```

```math
\mathbf{F}(\Delta t)=
\begin{bmatrix}
1&\Delta t&0&0&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0&0&0&0\\
0&0&1&\Delta t&0&0&0&0&0&0&0\\
0&0&0&1&0&0&0&0&0&0&0\\
0&0&0&0&1&\Delta t&0&0&0&0&0\\
0&0&0&0&0&1&0&0&0&0&0\\
0&0&0&0&0&0&1&\Delta t&0&0&0\\
0&0&0&0&0&0&0&1&0&0&0\\
0&0&0&0&0&0&0&0&1&0&0\\
0&0&0&0&0&0&0&0&0&1&0\\
0&0&0&0&0&0&0&0&0&0&1
\end{bmatrix}.
```

即：

```math
c_x^-=c_x+v_x\Delta t,\quad c_y^-=c_y+v_y\Delta t,\quad
c_z^-=c_z+v_z\Delta t,\quad \theta^-=\theta+\omega\Delta t.
```

对一个位置-速度对，连续白噪声加速度谱密度为 $q$ 时，程序使用精确离散化的
匀速模型过程噪声：

```math
\mathbf{Q}_{cv}(q,\Delta t)=q
\begin{bmatrix}
\frac{\Delta t^3}{3}&\frac{\Delta t^2}{2}\\
\frac{\Delta t^2}{2}&\Delta t
\end{bmatrix}.
```

完整过程噪声矩阵 $\mathbf Q$ 是稀疏分块矩阵，非零块为：

```text
Q[{CenterX, VelocityX}] = Qcv(options_.q_linear_acceleration, dt_s)
Q[{CenterY, VelocityY}] = Qcv(options_.q_linear_acceleration, dt_s)
Q[{CenterZ, VelocityZ}] = Qcv(options_.q_linear_acceleration, dt_s)
Q[{Theta, Omega}]       = Qcv(options_.q_angular_acceleration, dt_s)
Q[RadiusEven, RadiusEven] = options_.q_geometry * dt_s
Q[RadiusOddDelta, RadiusOddDelta] = 0
Q[HeightOddDelta, HeightOddDelta] = 0
```

协方差预测：

```math
\mathbf{P}_{k|k-1}=
\mathbf{F}\mathbf{P}_{k-1|k-1}\mathbf{F}^{T}+\mathbf{Q},
\qquad
\mathbf{P}_{k|k-1}\leftarrow
\frac{\mathbf{P}_{k|k-1}+\mathbf{P}_{k|k-1}^{T}}{2}.
```

| 物理量 | 网页/API 参数 | 程序变量 |
|---|---|---|
| 平移加速度谱密度 | `q_linear_acceleration` | `options_.q_linear_acceleration` |
| 角加速度谱密度 | `q_angular_acceleration` | `options_.q_angular_acceleration` |
| 半径随机游走谱密度 | `q_geometry` | `options_.q_geometry` |

增大 $q$ 会使预测协方差增长更快，从而更相信新观测；它不会直接改变均值预测。

## 3. 观测模型与 Jacobian

对已关联到槽位 $i$ 的装甲，四维观测为：

```math
\mathbf z_i=
\begin{bmatrix}x_{m,i}&y_{m,i}&z_{m,i}&\psi_{m,i}\end{bmatrix}^{T}
=
\begin{bmatrix}
\texttt{measurement.position\_T\_m.x()}\\
\texttt{measurement.position\_T\_m.y()}\\
\texttt{measurement.position\_T\_m.z()}\\
\texttt{measurement.inward\_yaw\_T\_rad}
\end{bmatrix}.
```

非线性预测观测：

```math
\mathbf h_i(\mathbf x)=
\begin{bmatrix}
c_x-r_i\cos\phi_i\\
c_y-r_i\sin\phi_i\\
c_z+p_i\Delta z\\
\phi_i
\end{bmatrix}.
```

在第 1 节状态顺序下，解析 Jacobian 为：

```math
\mathbf H_i=
\begin{bmatrix}
1&0&0&0&0&0&r_i\sin\phi_i&0&-\cos\phi_i&-p_i\cos\phi_i&0\\
0&0&1&0&0&0&-r_i\cos\phi_i&0&-\sin\phi_i&-p_i\sin\phi_i&0\\
0&0&0&0&1&0&0&0&0&0&p_i\\
0&0&0&0&0&0&1&0&0&0&0
\end{bmatrix}.
```

yaw 创新采用环绕角度：

```math
\operatorname{wrapToPi}(a)=((a+\pi)\bmod2\pi)-\pi,
\qquad
\nu_{\psi,i}=\operatorname{wrapToPi}(\psi_{m,i}-\phi_i).
```

故名义创新为：

```math
\boldsymbol\nu_i=
\begin{bmatrix}
\mathbf p_{m,i}-\mathbf p_{A,i}(\mathbf x^-)\\
\nu_{\psi,i}
\end{bmatrix}.
```

若 `measurement.has_inward_yaw == false`，只使用 $\mathbf z_i$、
$\mathbf h_i$、$\mathbf H_i$ 的前三个位置维度。

## 4. 自适应测量协方差

先定义当前帧质量缩放系数：

```math
s=\frac{
\left(1+\texttt{options\_.reprojection\_rms\_scale}
\max(0,\texttt{measurement.reprojection\_rms\_px})\right)
\left(1+\texttt{options\_.range\_noise\_scale\_per\_m}
\texttt{measurement.camera\_range\_m}\right)
}{
\max(\texttt{options\_.minimum\_quality},\texttt{measurement.confidence})
\max(\texttt{options\_.minimum\_quality},\texttt{measurement.keypoint\_quality})
\max(\texttt{options\_.minimum\_quality},\texttt{measurement.view\_quality})
}.
```

相机系位置协方差：

```math
\mathbf R_{C,p}=\operatorname{diag}\left(
(\texttt{options\_.position\_std\_xy\_m}s)^2,
(\texttt{options\_.position\_std\_xy\_m}s)^2,
(\texttt{options\_.position\_std\_z\_m}s)^2
\right).
```

若 `measurement.has_exposure_camera_geometry=true`，转换到 `T` 系：

```math
\mathbf R_{T,p}=
\texttt{measurement.R\_TC}\;\mathbf R_{C,p}\;
\texttt{measurement.R\_TC}^{T}.
```

否则 $\mathbf R_{T,p}=\mathbf R_{C,p}$。yaw 标准差为：

```math
\sigma_\psi=\max\left(
\texttt{options\_.yaw\_std\_rad}\;s,
\texttt{measurement.yaw\_std\_rad}
\right),
```

前提是存在正的 `measurement.yaw_std_rad`；否则
$\sigma_\psi=\texttt{options\_.yaw\_std\_rad}\;s$。最终单板测量协方差：

```math
\mathbf R_i=
\begin{bmatrix}
\mathbf R_{T,p}&\mathbf0\\
\mathbf0^T&\sigma_\psi^2
\end{bmatrix}.
```

| 物理量 | 网页/API 参数 |
|---|---|
| 横向位置基础标准差 | `position_std_xy_m` |
| 深度位置基础标准差 | `position_std_z_m` |
| yaw 基础标准差 | `yaw_std_rad` |
| 重投影 RMS 到噪声的放大系数 | `reprojection_rms_scale` |
| 距离到噪声的放大系数 | `range_noise_scale_per_m` |
| 检测质量的最小分母 | `minimum_quality` |

## 5. 槽位关联、可见性与 NIS 门控

`number_id`/`color_id` 只用于车辆身份门控，不决定 E0--E3。实际槽位通过一对一
最小关联代价选出。

关联阶段仅放宽位置协方差：

```math
\mathbf R_{\mathrm{gate},p}=
\texttt{options\_.association\_position\_variance\_scale}\;\mathbf R_{T,p}.
```

yaw 可以参与槽位关联的条件：

```math
|\nu_{\psi,i}|\le
\texttt{options\_.maximum\_yaw\_association\_innovation\_rad}.
```

yaw 可以实际进入 EKF 更新的条件：

```math
|\nu_{\psi,i}|\le
\texttt{options\_.maximum\_yaw\_update\_innovation\_rad}.
```

对含 yaw 的候选：

```math
\mathbf S_i=\mathbf H_i\mathbf P^-\mathbf H_i^T+
\mathbf R_{\mathrm{gate},i},
\qquad
\mathrm{NIS}_i=\boldsymbol\nu_i^T\mathbf S_i^{-1}\boldsymbol\nu_i,
\qquad
\mathrm{NIS}_i\le\texttt{options\_.nis\_gate\_4d}.
```

对 position-only 候选，使用对应的前三维矩阵以及 `options_.nis_gate_3d`。

当 yaw 可关联但不能更新时，附加相位代价为：

```math
J_{\mathrm{phase}}=
\left(
\frac{\nu_{\psi,i}}
{\texttt{options\_.yaw\_phase\_cost\_std\_rad}}
\right)^2.
```

候选总成本：

```math
J_i=\mathrm{NIS}_i+J_{\mathrm{transition},i}+J_{\mathrm{phase}}.
```

`associateAll()` 优先选择关联数量最大的方案；数量相同则选择 $\sum_iJ_i$ 最小者。

| 关联规则 | 网页/API 参数 |
|---|---|
| 关联时位置协方差放大倍率 | `association_position_variance_scale` |
| yaw 允许用于槽位关联的最大创新 | `maximum_yaw_association_innovation_rad` |
| yaw 允许用于 EKF 更新的最大创新 | `maximum_yaw_update_innovation_rad` |
| yaw 相位代价标准差 | `yaw_phase_cost_std_rad` |
| 相邻槽位切换惩罚 | `adjacent_slot_penalty` |
| 对侧槽位切换惩罚 | `opposite_slot_penalty` |
| 预测装甲可见性余弦下限 | `minimum_visibility_cosine` |
| 三维位置 NIS 门限 | `nis_gate_3d` |
| 位置+yaw 四维 NIS 门限 | `nis_gate_4d` |

## 6. 多装甲几何可观测性

仅当连续 `geometry_confirming_frames` 帧满足以下条件，`update_geometry` 才为真：
至少有两块关联装甲、至少一块使用 yaw，并且任意两块有：

```math
\|\mathbf p_{m,a}-\mathbf p_{m,b}\|\ge
\texttt{options\_.geometry\_minimum\_baseline\_m}.
```

若两块都使用 yaw，还必须满足槽位相位一致：

```math
\left|\operatorname{wrapToPi}\left(
(\psi_{m,a}-\psi_{m,b})-(a-b)\frac{\pi}{2}
\right)\right|
\le\texttt{options\_.geometry\_yaw\_consistency\_rad}.
```

| 几何条件 | 网页/API 参数 |
|---|---|
| 双板 yaw 相位一致性门限 | `geometry_yaw_consistency_rad` |
| 双板最小空间基线 | `geometry_minimum_baseline_m` |
| 连续确认帧数 | `geometry_confirming_frames` |

## 7. 多板联合 EKF 更新

设同一曝光帧内接受 $m$ 个关联，程序虽然固定分配 16 维矩阵，但数学上只有前
$4m$ 行有效。堆叠为：

```math
\mathbf H=
\begin{bmatrix}\mathbf H_1\\\vdots\\\mathbf H_m\end{bmatrix},
\qquad
\boldsymbol\nu=
\begin{bmatrix}\boldsymbol\nu_1\\\vdots\\\boldsymbol\nu_m\end{bmatrix},
\qquad
\mathbf R=\operatorname{blockdiag}(\mathbf R_1,\ldots,\mathbf R_m).
```

更新前，代码定义：

```text
position_observable = (m >= 2) || (yaw_observation_count == 1)
```

下列情况会使该观测块的位置三行失效：位置不可观测；单板且未使用 yaw；或在
非几何更新时双板位置残差超过 `maximum_multi_armor_position_residual_m`。实现为：

```math
\mathbf H_i[0:2,:]\leftarrow0,\qquad
\boldsymbol\nu_i[0:2]\leftarrow0.
```

单板位置更新有效时，其位置协方差会被额外放大：

```math
\mathbf R_{i,p}\leftarrow
\texttt{options\_.single\_armor\_position\_variance\_scale}
\begin{cases}
4\mathbf R_{i,p},&\texttt{motion\_observed}=\mathrm{false},\\
\mathbf R_{i,p},&\texttt{motion\_observed}=\mathrm{true}.
\end{cases}
```

当 `update_geometry == false` 时，$\mathbf H$ 的 `RadiusEven`、
`RadiusOddDelta`、`HeightOddDelta` 三列归零；未使用 yaw 时，相应观测块的 yaw 行归零。

创新协方差与 Kalman 增益为：

```math
\mathbf S=\mathbf H\mathbf P^-\mathbf H^T+\mathbf R,
\qquad
\mathbf K=\mathbf P^-\mathbf H^T\mathbf S^{-1}.
```

代码通过 LDLT 解线性方程获得 $\mathbf K$，不显式计算 $\mathbf S^{-1}$。

随后会直接修改增益：若槽位切换，或已使用 yaw 的残差超过
`0.5 * maximum_yaw_update_innovation_rad`，则：

```text
K[Omega, :] = 0
```

位置不可观测时，`CenterX..VelocityZ` 的增益行归零。无论是否更新几何，
`RadiusOddDelta` 与 `HeightOddDelta` 的增益行最终都归零。

令 $\delta\mathbf x_{raw}=\mathbf K\boldsymbol\nu$。角速度修正限幅：

```math
\delta\omega\leftarrow
\operatorname{clip}(\delta\omega_{raw},
-\texttt{options\_.maximum\_omega\_correction\_rad\_s},
\texttt{options\_.maximum\_omega\_correction\_rad\_s}),
```

并限制：

```math
|\omega^+|\le\texttt{options\_.maximum\_angular\_speed\_rad\_s}.
```

此外，单帧 $\delta\theta$ 被固定裁剪到 $[-0.45,0.45]$ rad。代码通过缩放
$\mathbf K$ 相应行来实现这两个裁剪。

最终更新使用 Joseph 形式：

```math
\mathbf x^+=\mathbf x^-+\mathbf K\boldsymbol\nu,
\qquad
\mathbf A=\mathbf I-\mathbf K\mathbf H,
\qquad
\mathbf P^+=\mathbf A\mathbf P^-\mathbf A^T+
\mathbf K\mathbf R\mathbf K^T.
```

最后强制 `RadiusOddDelta=0`、`HeightOddDelta=0`，并清空二者协方差行列。

| 联合更新行为 | 网页/API 参数 |
|---|---|
| 单板位置协方差额外倍率 | `single_armor_position_variance_scale` |
| 非几何双板位置残差上限 | `maximum_multi_armor_position_residual_m` |
| 角速度绝对值上限 | `maximum_angular_speed_rad_s` |
| 单帧角速度修正上限 | `maximum_omega_correction_rad_s` |

## 8. 初始化与初始协方差

初始化必须满足 `measurement.has_inward_yaw == true`。第一块有效装甲定义为 E0：

```math
\theta_0=\psi_m,\qquad
r_{0,0}=\texttt{options\_.radius\_prior\_m},
```

```math
c_{x,0}=x_m+r_{0,0}\cos\theta_0,\qquad
c_{y,0}=y_m+r_{0,0}\sin\theta_0,\qquad
c_{z,0}=z_m.
```

并令 $v_x=v_y=v_z=\omega=\Delta r=\Delta z=0$。初始协方差的非零对角项：

```text
P[CenterX, CenterX] = P[CenterY, CenterY] = P[CenterZ, CenterZ]
                     = options_.initial_position_std_m^2
P[VelocityX, VelocityX] = P[VelocityY, VelocityY] = P[VelocityZ, VelocityZ]
                         = options_.initial_velocity_std_mps^2
P[Theta, Theta] = options_.initial_theta_std_rad^2
P[Omega, Omega] = options_.initial_omega_std_rad_s^2
P[RadiusEven, RadiusEven] = options_.initial_geometry_std_m^2
```

| 初始不确定度 | 网页/API 参数 |
|---|---|
| 车辆中心位置标准差 | `initial_position_std_m` |
| 车辆中心速度标准差 | `initial_velocity_std_mps` |
| E0 初相位标准差 | `initial_theta_std_rad` |
| 初始角速度标准差 | `initial_omega_std_rad_s` |
| 初始半径标准差 | `initial_geometry_std_m` |

`radius_prior_m`、半径边界、丢失阈值及 `confirming_hits` 在当前网页控制实现中
没有对应的 `ekf-param` 接口。

## 9. 上游 yaw 有效性参数

下列参数决定约束重投影 yaw 求解器是否产生
`measurement.has_inward_yaw`；它们不是 EKF 的 $\mathbf F$、$\mathbf Q$、
$\mathbf H$ 或 $\mathbf R$ 的元素：

| 求解器条件 | 网页/API 参数 |
|---|---|
| 约束 yaw 最大重投影 RMS | `yaw_max_reprojection_rms_px` |
| 重投影曲率估计的最大 yaw 标准差 | `yaw_max_std_rad` |
| 最小正面朝向余弦 | `yaw_min_facing_cosine` |
| 正反候选解最小 RMS 差 | `yaw_min_opposite_margin_px` |

它们使用相同的请求形式，例如：

```text
/api/control?action=ekf-param:yaw_max_std_rad:0.35
```
