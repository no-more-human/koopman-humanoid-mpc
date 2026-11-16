#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qp_wbc.py -- Unitree G1 下层 500 Hz 全身二次规划控制器 (QP-WBC)、扰动观测器 (DOB) 与路线二闭环基线

用途：
    Task B3。实现带浮动基动力学等式约束、摩擦锥约束与关节力矩上下界硬约束的
    500 Hz 下层全身控制器 (QP-WBC)，串联中层 50 Hz 名义单刚体控制器 (NominalSRBDMPC)，
    跑通：
      1. 路线二（Baseline 名义物理基线：Nominal SRBD-MPC + QP-WBC）
      2. 传统抗扰基线（SRBD-MPC + 在线集总扰动观测器 DOB + QP-WBC）

核心数学形式（消去 τ 后的 47 维稀疏凸二次规划）：
    整机动力学方程（MuJoCo 广义坐标 nv=35，前 6 维为浮动基 b，后 29 维为受控关节 j）：
        M(q) * qacc + h(q, qvel) = S^T * tau + J_c(q)^T * F_c
    拆分为浮动基前 6 行与关节后 29 行：
        (1) 浮动基动力学等式 (6 行):   M_b * qacc - J_{c,b}^T * F_c = -h_b
        (2) 受控关节力矩代数式 (29 行): tau = M_j * qacc + h_j - J_{c,j}^T * F_c

    选取决策变量 w = [qacc^T (35), F_c^T (12)]^T ∈ R^47：
        min_{w}  1/2 * ||F_c - u_ref*||_{W_F}^2
               + 1/2 * ||qacc - qacc_des||_{W_q}^2
               + 1/2 * ||J_c * qacc - a_{c,des}||_{W_c}^2
               + 1/2 * ||M_j * qacc + h_j - J_{c,j}^T * F_c||_{W_tau}^2
        s.t.
               M_b * qacc - J_{c,b}^T * F_c = -h_b                       (6 行浮动基动力学等式)
               -tau_max - h_j <= M_j * qacc - J_{c,j}^T * F_c <= tau_max - h_j (29 行关节力矩限幅)
               |f_{L/R, x}| <= mu * f_{L/R, z},  |f_{L/R, y}| <= mu * f_{L/R, z} (8 行摩擦锥)
               f_{z,min} <= f_{L/R, z} <= f_{z,max},  |tau_{L/R}| <= tau_foot_max (8 行接触旋量盒约束)

    最优关节力矩恢复与下发：
        tau_wbc* = M_j * qacc* + h_j - J_{c,j}^T * F_c*
        tau_cmd  = clip(tau_wbc* + kp_servo * (q_des - q) - kd_servo * dq, -tau_max, tau_max)

用法：
    F:\\conda_envs\\koopman_robot\\python.exe qp_wbc.py
    F:\\conda_envs\\koopman_robot\\python.exe qp_wbc.py --duration 6.0 --payload 5.0
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from nominal_srbd import (
    DEFAULT_DT,
    GRAVITY,
    GRF_DIM,
    MPCConfig,
    NominalSRBDModel,
    NominalSRBDMPC,
)
from state_vector import (
    COM_STATE_DIM,
    KEYFRAME_NAME,
    MODEL_PATH,
    extract_com_state,
    extract_foot_jacobians,
    extract_foot_positions,
    extract_full_state,
    quat_wxyz_to_rotmat,
    quat_wxyz_to_rpy,
)

NV_DIM = 35
ACTUATOR_DIM = 29

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LINE = "=" * 78


# --------------------------------------------------------------------------- #
# 模型加载与力矩直驱转换（内存中修改，严禁修改 XML）
# --------------------------------------------------------------------------- #


def load_g1_torque_model(
    model_path: Path = MODEL_PATH,
    nominal_joint_damping: float = 0.5,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载 G1 场景并在内存中将 29 个 position 伺服转化为纯力矩直驱电机。"""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # 将全部 29 个 actuator 改为力矩直驱: tau = ctrl
    model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_biasprm[:, :] = 0.0
    model.actuator_ctrllimited[:] = 0
    model.actuator_gear[:, 0] = 1.0

    # 为 29 个受控铰链关节设置基础物理粘滞阻尼（原 XML 为 0，依赖内置位置伺服 kv）
    model.dof_damping[6:] = float(nominal_joint_damping)

    return model, data


def reset_to_stand_keyframe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    keyframe_name: str = KEYFRAME_NAME,
) -> np.ndarray:
    """将机器人状态重置为 'stand' 关键帧并返回初始 qpos0 (shape=(36,))。"""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        raise RuntimeError(f"找不到关键帧 '{keyframe_name}'")
    qpos0 = model.key_qpos[key_id].copy()
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:, :] = 0.0
    mujoco.mj_forward(model, data)
    return qpos0


def get_joint_torque_limits(model: mujoco.MjModel) -> np.ndarray:
    """从模型关节 actuatorfrcrange 读取 29 个受控关节的力矩幅值上限 tau_max (N·m)。"""
    trn_jnt_ids = model.actuator_trnid[:, 0]
    tau_max = np.asarray(model.jnt_actfrcrange[trn_jnt_ids, 1], dtype=np.float64).copy()
    # 若个别关节未定义正上限，则回退至 88.0 N·m
    invalid_mask = tau_max <= 0.0
    if np.any(invalid_mask):
        tau_max[invalid_mask] = 88.0
    return tau_max


def rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    """纯 NumPy 快速计算外旋 xyz (Roll-Pitch-Yaw) 对应的 3x3 旋转矩阵 R = R_z(yaw)*R_y(pitch)*R_x(roll)。"""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp_ = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp_ * sr - sy * cr, cy * sp_ * cr + sy * sr],
            [sy * cp, sy * sp_ * sr + cy * cr, sy * sp_ * cr - cy * sr],
            [-sp_, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotmat_error_vee(r_des: np.ndarray, r_curr: np.ndarray) -> np.ndarray:
    """利用 SO(3) 反对称映射快速计算世界系姿态误差向量 e_ori = 1/2 * vee(R_des * R_curr^T - R_curr * R_des^T)。"""
    r_err = r_des @ r_curr.T
    return 0.5 * np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )


def compute_posture_q_des(
    q_stand: np.ndarray,
    x_ref: np.ndarray,
    z_com_nom: float = 0.6919,
    nominal_leg_length: float = 0.58,
) -> np.ndarray:
    """由上层期望质心状态 x_ref 自动映射协调的全身期望关节角 q_des ∈ R^29。

    说明：
        当上层要求动态深蹲 (dz = z_ref - z_com_nom < 0) 或躯干俯仰/横滚姿态调整时，
        利用对称双连杆几何逆运动学协调屈曲髋俯仰、膝关节与踝俯仰，避免关节名义姿态任务
        与质心轨迹任务发生零空间冲突。站立平衡 (dz=0, rpy_ref=0) 时严格退化为 q_stand。
    """
    q_des = np.asarray(q_stand, dtype=np.float64).copy()
    dz = float(x_ref[2]) - float(z_com_nom)
    roll_ref = float(x_ref[3])
    pitch_ref = float(x_ref[4])
    yaw_ref = float(x_ref[5])

    # 1. 竖直深蹲膝-髋-踝几何协调 (dz < 0 时屈膝)
    if dz < -1e-4:
        ratio = np.clip((nominal_leg_length + dz) / nominal_leg_length, 0.55, 1.0)
        theta_bend = float(np.arccos(ratio))
    else:
        theta_bend = 0.0

    # 左腿索引: 0=hip_pitch, 1=hip_roll, 2=hip_yaw, 3=knee, 4=ankle_pitch, 5=ankle_roll
    # 右腿索引: 6=hip_pitch, 7=hip_roll, 8=hip_yaw, 9=knee, 10=ankle_pitch, 11=ankle_roll
    q_des[0] += -theta_bend - 0.5 * pitch_ref
    q_des[3] += 2.0 * theta_bend
    q_des[4] += -theta_bend + 0.5 * pitch_ref

    q_des[6] += -theta_bend - 0.5 * pitch_ref
    q_des[9] += 2.0 * theta_bend
    q_des[10] += -theta_bend + 0.5 * pitch_ref

    # 横滚与腰部姿态协调: 12=waist_yaw, 13=waist_roll, 14=waist_pitch
    q_des[1] += 0.35 * roll_ref
    q_des[5] += -0.35 * roll_ref
    q_des[7] += 0.35 * roll_ref
    q_des[11] += -0.35 * roll_ref

    q_des[12] += 0.4 * yaw_ref
    q_des[13] += 0.5 * roll_ref
    q_des[14] += 0.5 * pitch_ref

    return q_des


# --------------------------------------------------------------------------- #
# 高难恶化工况与传感器噪声注入（Task B3 / C1 / D1 共用）
# --------------------------------------------------------------------------- #


@dataclass
class PerturbationConfig:
    """未知偏心负载、关节摩擦摄动、地面摩擦与外力脉冲推扰配置。"""

    # 1. 左手/左腕未知非对称偏心负载 (kg) 与局部偏心位移 (m)
    payload_mass: float = 0.0
    payload_body_name: str = "left_wrist_yaw_link"
    payload_local_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.14, 0.12, -0.02], dtype=np.float64)
    )

    # 2. 腿部与腰部关节粘滞阻尼放大倍数 & 干摩擦损耗 (N·m)
    damping_scale: float = 1.0
    frictionloss_add: float = 0.0

    # 3. 地面与足底接触摩擦系数覆盖（None 表示保持默认）
    ground_mu: float | None = None

    # 4. 躯干水平脉冲推扰
    push_force_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    push_start_s: float = 2.5
    push_duration_s: float = 0.15
    push_body_name: str = "torso_link"


def apply_perturbation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: PerturbationConfig,
) -> dict[str, float]:
    """在运行时内存中注入未知偏心负载、关节摩擦摄动与地面摩擦变化（不修改 XML）。

    注意：
        调用 mujoco.mj_setConst(model, data) 前会保存并恢复当前 qpos/qvel，
        防止关键帧姿态被重置。
    """
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()

    orig_total_mass = float(np.sum(model.body_mass))

    # 1. 注入单侧未知偏心负载
    if abs(cfg.payload_mass) > 1e-6:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.payload_body_name)
        if body_id < 0:
            raise ValueError(f"找不到负载挂载刚体: {cfg.payload_body_name}")

        m_old = float(model.body_mass[body_id])
        m_add = float(cfg.payload_mass)
        m_new = m_old + m_add
        pos_old = model.body_ipos[body_id].copy()
        pos_add = np.asarray(cfg.payload_local_offset, dtype=np.float64)

        # 更新复合质心位置与附加转动惯量
        pos_new = (m_old * pos_old + m_add * pos_add) / m_new
        r_sq = float(np.dot(pos_add, pos_add))
        model.body_mass[body_id] = m_new
        model.body_ipos[body_id] = pos_new
        model.body_inertia[body_id] += m_add * np.array(
            [0.5 * r_sq + 0.015, 0.5 * r_sq + 0.015, 0.5 * r_sq + 0.015],
            dtype=np.float64,
        )

    # 2. 注入关节阻尼与干摩擦摄动
    if abs(cfg.damping_scale - 1.0) > 1e-6:
        model.dof_damping[6:] *= float(cfg.damping_scale)
    if abs(cfg.frictionloss_add) > 1e-6:
        # 主要作用于双腿 12 个承重关节与腰部 3 个关节
        model.dof_frictionloss[6:21] += float(cfg.frictionloss_add)

    # 3. 修改地面与足底摩擦系数
    if cfg.ground_mu is not None:
        mu_val = float(cfg.ground_mu)
        for geom_id in range(model.ngeom):
            model.geom_friction[geom_id, 0] = mu_val

    # 重新计算模型派生常量并恢复当前状态
    mujoco.mj_setConst(model, data)
    data.qpos[:] = qpos_saved
    data.qvel[:] = qvel_saved
    mujoco.mj_forward(model, data)

    new_total_mass = float(np.sum(model.body_mass))
    return {
        "orig_total_mass": orig_total_mass,
        "new_total_mass": new_total_mass,
        "payload_mass": float(cfg.payload_mass),
    }


def step_perturbation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: float,
    cfg: PerturbationConfig,
) -> np.ndarray:
    """在每个仿真步更新外力脉冲推扰 data.xfrc_applied，返回当前施加的 3D 推力 (N)。"""
    data.xfrc_applied[:, :] = 0.0
    f_push = np.asarray(cfg.push_force_world, dtype=np.float64)
    if np.linalg.norm(f_push) > 1e-6:
        if cfg.push_start_s <= t <= (cfg.push_start_s + cfg.push_duration_s):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.push_body_name)
            if body_id >= 0:
                data.xfrc_applied[body_id, 0:3] = f_push
                return f_push.copy()
    return np.zeros(3, dtype=np.float64)


def observe_com_state_noisy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator | None = None,
    pos_std: float = 0.0015,
    rpy_std: float = 0.0020,
    vel_std: float = 0.0080,
    omega_std: float = 0.0100,
    prev_obs: np.ndarray | None = None,
    delay_alpha: float = 0.0,
) -> np.ndarray:
    """提取带真实传感器白噪声与可选一阶延迟的 12 维质心状态观测（供 Task C1/D1 调用）。"""
    x_true = extract_com_state(model, data)
    if rng is None:
        x_noisy = x_true
    else:
        noise = np.concatenate(
            [
                rng.normal(0.0, pos_std, size=3),
                rng.normal(0.0, rpy_std, size=3),
                rng.normal(0.0, vel_std, size=3),
                rng.normal(0.0, omega_std, size=3),
            ]
        )
        x_noisy = x_true + noise

    if prev_obs is not None and delay_alpha > 0.0:
        a = float(np.clip(delay_alpha, 0.0, 1.0))
        return (1.0 - a) * x_noisy + a * prev_obs
    return x_noisy


# --------------------------------------------------------------------------- #
# 在线集总扰动观测器 (DisturbanceObserver, DOB 基线)
# --------------------------------------------------------------------------- #


class DisturbanceObserver:
    """一阶指数移动平均 (EMA) 离散状态预测残差观测器 (DOB)。

    递推公式：
        d_hat_k = (1 - alpha) * d_hat_{k-1} + alpha * (x_k - f_nom(x_{k-1}, u_{k-1}))

    说明：
        代表主流“名义 MPC + 集总扰动观测器”基线。对于纯常值偏置可逐渐收敛补偿，
        但在动态深蹲或俯仰摆动等残差随姿态/控制量快速变化的工况下存在相位滞后。
    """

    def __init__(
        self,
        srbd_model: NominalSRBDModel,
        alpha: float = 0.25,
        max_pos_res: float = 0.015,
        max_vel_res: float = 0.25,
    ) -> None:
        self.srbd = srbd_model
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.max_pos_res = float(max_pos_res)
        self.max_vel_res = float(max_vel_res)
        self.d_hat = np.zeros(COM_STATE_DIM, dtype=np.float64)
        self._x_prev: np.ndarray | None = None
        self._u_prev: np.ndarray | None = None

    def reset(self) -> None:
        """清空历史状态与扰动估计。"""
        self.d_hat[:] = 0.0
        self._x_prev = None
        self._u_prev = None

    def update(
        self,
        x_curr: np.ndarray,
        u_applied: np.ndarray,
        p_left: np.ndarray | None = None,
        p_right: np.ndarray | None = None,
    ) -> np.ndarray:
        """在每个 50 Hz MPC 周期调用：用上一周期 (x_{k-1}, u_{k-1}) 更新 d_hat，并缓存当前 (x_k, u_k)。"""
        x_curr = np.asarray(x_curr, dtype=np.float64)
        u_applied = np.asarray(u_applied, dtype=np.float64)

        if self._x_prev is not None and self._u_prev is not None:
            x_pred = self.srbd.predict_next(self._x_prev, self._u_prev, p_left, p_right)
            raw_res = np.zeros(COM_STATE_DIM, dtype=np.float64)
            # 提取广义速度/角速度通道的未建模加速度残差，并按零阶保持二阶积分一致性映射至位置/姿态通道
            vel_res = np.clip(x_curr[6:12] - x_pred[6:12], -self.max_vel_res, self.max_vel_res)
            raw_res[6:12] = vel_res
            raw_res[0:6] = np.clip(0.5 * self.srbd.dt * vel_res, -self.max_pos_res, self.max_pos_res)
            self.d_hat = (1.0 - self.alpha) * self.d_hat + self.alpha * raw_res

        self._x_prev = x_curr.copy()
        self._u_prev = u_applied.copy()
        return self.d_hat.copy()

    def get_horizon_residual(self, horizon: int) -> np.ndarray:
        """将当前常值扰动估计复制 N 步，返回 shape=(horizon, 12) 供 NominalSRBDMPC.solve 使用。"""
        return np.tile(self.d_hat, (int(horizon), 1))


# --------------------------------------------------------------------------- #
# Task B3 核心类：500 Hz 全身二次规划力矩分配控制器 (QPWBC)
# --------------------------------------------------------------------------- #


@dataclass
class WBCConfig:
    """下层 500 Hz QP-WBC 权重与约束超参数配置。"""

    # 摩擦锥与接触旋量边界（严格匹配 g1.xml 足底碰撞球几何尺寸 ±0.085m x ±0.025m 的零力矩点 ZMP 边界）
    mu: float = 0.6
    fz_min: float = 15.0
    fz_max: float = 350.0
    fxy_max: float = 180.0
    tau_foot_max: np.ndarray = field(
        default_factory=lambda: np.array([3.5, 12.0, 5.0], dtype=np.float64)
    )

    # 1. 足端接触力旋量跟踪权重 W_F ∈ R^12 (主导项，确保忠实复现中层 MPC 的 u_ref*)
    w_force: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                10.0, 10.0, 12.0,    # 左足力 f_L (x, y, z)
                40.0, 40.0, 30.0,    # 左足力矩 tau_L (x, y, z)
                10.0, 10.0, 12.0,    # 右足力 f_R (x, y, z)
                40.0, 40.0, 30.0,    # 右足力矩 tau_R (x, y, z)
            ],
            dtype=np.float64,
        )
    )

    # 2. 广义加速度跟踪权重：浮动基 (6D)、双腿关节 (12D)、腰与双臂关节 (17D)
    w_qacc_base_lin: float = 1.0
    w_qacc_base_ang: float = 2.0
    w_qacc_leg: float = 0.5
    w_qacc_upper: float = 5.0

    # 3. 足底接触点零滑移软约束权重 W_c ∈ R^12 与接触速度阻尼增益
    w_contact_lin: float = 200.0
    w_contact_ang: float = 150.0
    kd_contact: float = 20.0

    # 4. 关节力矩平滑正则化权重 W_tau（设为 0 可保持 P 的块对角稀疏结构，极大加速 OSQP 因子分解）
    w_tau_reg: float = 0.0

    # 5. 任务空间基座 PD 增益（用于构造 qacc_des[0:6] 辅助高帧率阻尼）
    kp_base_lin: np.ndarray = field(
        default_factory=lambda: np.array([40.0, 40.0, 60.0], dtype=np.float64)
    )
    kd_base_lin: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 12.0, 15.0], dtype=np.float64)
    )
    kp_base_ang: np.ndarray = field(
        default_factory=lambda: np.array([80.0, 80.0, 60.0], dtype=np.float64)
    )
    kd_base_ang: np.ndarray = field(
        default_factory=lambda: np.array([16.0, 16.0, 12.0], dtype=np.float64)
    )

    # 6. 关节空间名义姿态加速度 PD 增益（构造 qacc_des[6:35]）
    kp_joint_leg: float = 100.0
    kd_joint_leg: float = 10.0
    kp_joint_upper: float = 180.0
    kd_joint_upper: float = 14.0

    # 7. 下发关节力矩前馈 + 零空间姿态阻抗伺服稳定项
    # 说明：tau_wbc* 提供全部重力/动力学补偿与 MPC 接触力分配 -J_{c,j}^T * F_c*，
    # 适中的阻抗伺服项维持双腿与上身结构刚度，同时让未知负载充分体现到质心平衡回路中。
    servo_kp_leg: np.ndarray = field(
        default_factory=lambda: np.array(
            [70.0, 85.0, 55.0, 100.0, 55.0, 55.0] * 2, dtype=np.float64
        )
    )
    servo_kd_leg: np.ndarray = field(
        default_factory=lambda: np.array(
            [4.5, 5.5, 3.5, 6.0, 4.0, 4.0] * 2, dtype=np.float64
        )
    )
    servo_kp_upper: float = 110.0
    servo_kd_upper: float = 5.0


class QPWBC:
    """下层 500 Hz 全身二次规划控制器 (Whole-Body Controller)。

    特性：
        1. 内部维护一份独立、未加未知负载的名义机器人模型 (self.nom_model, self.nom_data)，
           严格模拟真实机器人控制器：控制器仅知晓名义动力学参数 (33.34 kg)，不知晓外部未知负载。
        2. 预分配稀疏结构并使用 OSQP 热启动，单步求解耗时约 0.2 ~ 0.6 ms (< 2 ms)。
    """

    def __init__(
        self,
        nominal_model: mujoco.MjModel,
        config: WBCConfig | None = None,
    ) -> None:
        self.cfg = WBCConfig() if config is None else config

        # 加载一份独立纯净的名义模型与数据，确保外部对仿真环境 sim_model 的未知负载修改不会泄露给 WBC
        self.nom_model, self.nom_data = load_g1_torque_model()

        self.nv = NV_DIM                # 35
        self.nu = ACTUATOR_DIM          # 29
        self.nfc = GRF_DIM              # 12
        self.n_vars = self.nv + self.nfc  # 47: [qacc(35), F_c(12)]

        # 提取受控关节地址与力矩硬限幅
        trn_jnt_ids = self.nom_model.actuator_trnid[:, 0]
        self.qpos_adr = self.nom_model.jnt_qposadr[trn_jnt_ids].copy()
        self.dof_adr = self.nom_model.jnt_dofadr[trn_jnt_ids].copy()
        self.tau_max = get_joint_torque_limits(self.nom_model)

        # 构造权重向量与对角阵
        w_qacc = np.empty(self.nv, dtype=np.float64)
        w_qacc[0:3] = self.cfg.w_qacc_base_lin
        w_qacc[3:6] = self.cfg.w_qacc_base_ang
        w_qacc[6:18] = self.cfg.w_qacc_leg
        w_qacc[18:35] = self.cfg.w_qacc_upper
        self.w_qacc = w_qacc
        self.W_qacc = np.diag(w_qacc)

        self.w_force = self.cfg.w_force.copy()
        self.W_force = np.diag(self.w_force)

        w_contact = np.array(
            [
                self.cfg.w_contact_lin, self.cfg.w_contact_lin, self.cfg.w_contact_lin,
                self.cfg.w_contact_ang, self.cfg.w_contact_ang, self.cfg.w_contact_ang,
                self.cfg.w_contact_lin, self.cfg.w_contact_lin, self.cfg.w_contact_lin,
                self.cfg.w_contact_ang, self.cfg.w_contact_ang, self.cfg.w_contact_ang,
            ],
            dtype=np.float64,
        )
        self.w_contact = w_contact
        self.W_contact = np.diag(w_contact)

        # 关节空间加速度 PD 与伺服增益向量
        self.kp_joint = np.empty(self.nu, dtype=np.float64)
        self.kd_joint = np.empty(self.nu, dtype=np.float64)
        self.kp_joint[0:12] = self.cfg.kp_joint_leg
        self.kp_joint[12:29] = self.cfg.kp_joint_upper
        self.kd_joint[0:12] = self.cfg.kd_joint_leg
        self.kd_joint[12:29] = self.cfg.kd_joint_upper

        self.servo_kp = np.empty(self.nu, dtype=np.float64)
        self.servo_kd = np.empty(self.nu, dtype=np.float64)
        self.servo_kp[0:12] = self.cfg.servo_kp_leg
        self.servo_kp[12:29] = self.cfg.servo_kp_upper
        self.servo_kd[0:12] = self.cfg.servo_kd_leg
        self.servo_kd[12:29] = self.cfg.servo_kd_upper

        # 预构造摩擦锥 (8 行) + 接触旋量盒约束 (12 行) = 20 行常系数约束
        self._C_fc, self._l_fc, self._u_fc = self._build_fc_constraints(self.cfg.mu)
        self.n_fc_cons = self._C_fc.shape[0]   # 20
        self.n_dyn_cons = self.nv              # 6 (浮动基等式) + 29 (关节力矩不等式) = 35
        self.n_cons = self.n_dyn_cons + self.n_fc_cons  # 55

        # 预分配全质量矩阵缓存
        self._M_full = np.zeros((self.nv, self.nv), dtype=np.float64)

        # 初始化固定稀疏模板的 OSQP 求解器
        self._prob = osqp.OSQP()
        self._init_osqp_template()

    def _build_fc_constraints(
        self,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造足端接触旋量 F_c ∈ R^12 的摩擦锥 (8 行) 与盒式上下界 (12 行) 约束矩阵。"""
        c_rows: list[np.ndarray] = []
        l_list: list[float] = []
        u_list: list[float] = []

        # 1. 双足四棱锥摩擦锥 (8 行):
        #    f_x - mu*f_z <= 0, -f_x - mu*f_z <= 0, f_y - mu*f_z <= 0, -f_y - mu*f_z <= 0
        for offset in (0, 6):
            for axis in (0, 1):
                row_pos = np.zeros(self.nfc, dtype=np.float64)
                row_pos[offset + axis] = 1.0
                row_pos[offset + 2] = -mu
                c_rows.append(row_pos)
                l_list.append(-1e10)
                u_list.append(0.0)

                row_neg = np.zeros(self.nfc, dtype=np.float64)
                row_neg[offset + axis] = -1.0
                row_neg[offset + 2] = -mu
                c_rows.append(row_neg)
                l_list.append(-1e10)
                u_list.append(0.0)

        # 2. 12 维接触旋量盒约束 (12 行)
        fxy = float(self.cfg.fxy_max)
        fz_min = float(self.cfg.fz_min)
        fz_max = float(self.cfg.fz_max)
        tx, ty, tz = (
            float(self.cfg.tau_foot_max[0]),
            float(self.cfg.tau_foot_max[1]),
            float(self.cfg.tau_foot_max[2]),
        )
        foot_lb = [-fxy, -fxy, fz_min, -tx, -ty, -tz]
        foot_ub = [fxy, fxy, fz_max, tx, ty, tz]
        for i in range(self.nfc):
            row = np.zeros(self.nfc, dtype=np.float64)
            row[i] = 1.0
            c_rows.append(row)
            l_list.append(foot_lb[i % 6])
            u_list.append(foot_ub[i % 6])

        return (
            np.vstack(c_rows).astype(np.float64),
            np.array(l_list, dtype=np.float64),
            np.array(u_list, dtype=np.float64),
        )

    def set_friction_coefficient(self, mu: float) -> None:
        """动态更新 WBC 内部摩擦锥约束中的摩擦系数 mu（供低摩擦地面实验调用）。"""
        self.cfg.mu = float(mu)
        self._C_fc, self._l_fc, self._u_fc = self._build_fc_constraints(self.cfg.mu)
        self._init_osqp_template()

    def _assemble_sparse_qp_values(
        self,
        M: np.ndarray,
        J_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """利用预计算的块稀疏索引极速拼装 P 与 A 的 CSC data 向量（零 scipy 调用，零冗余非零元）。"""
        # 1. P_qacc (35 x 35) 上三角非零元 + W_force (12) 对角元 -> 共 642 个非零元
        P_qacc = self.W_qacc + J_c.T @ self.W_contact @ J_c
        P_qacc = 0.5 * (P_qacc + P_qacc.T)
        px_qacc = P_qacc[self._p_qacc_rows, self._p_qacc_cols]
        px_vals = np.concatenate([px_qacc, self.w_force])

        # 2. A (55 x 47):
        #    前 35 列仅在前 35 行非零 (M, 35x35)，按列展开共 1225 个元素
        #    后 12 列在前 35 行为 -J_c^T (35x12)，在后 20 行为 _C_fc (20x12)
        ax_m = M.ravel(order="F")
        a_right = np.vstack([-J_c.T, self._C_fc])  # (55, 12)
        ax_right = a_right.ravel(order="F")
        ax_vals = np.concatenate([ax_m, ax_right])
        return px_vals, ax_vals

    def _init_osqp_template(self) -> None:
        """在名义站立姿态下初始化块稀疏结构的 OSQP 求解器。"""
        # 预计算 35x35 P_qacc 上三角矩阵按列主序 (CSC) 展开的行列索引 (630 个非零元)
        p_rows: list[int] = []
        p_cols: list[int] = []
        for c in range(self.nv):
            for r in range(c + 1):
                p_rows.append(r)
                p_cols.append(c)
        self._p_qacc_rows = np.array(p_rows, dtype=np.intp)
        self._p_qacc_cols = np.array(p_cols, dtype=np.intp)

        reset_to_stand_keyframe(self.nom_model, self.nom_data)
        mujoco.mj_fullM(self.nom_model, self.nom_data, self._M_full)
        j_l, j_r = extract_foot_jacobians(self.nom_model, self.nom_data)
        j_c = np.vstack([j_l, j_r])

        # 构造块稀疏模板矩阵 P_template (642 nnz) 与 A_template (1885 nnz)
        P_qacc = self.W_qacc + j_c.T @ self.W_contact @ j_c
        P_qacc_upper = np.triu(0.5 * (P_qacc + P_qacc.T)) + np.triu(np.full((self.nv, self.nv), 1e-12))
        self._P_csc = sp.block_diag(
            [sp.csc_matrix(P_qacc_upper), sp.diags(self.w_force, format="csc")],
            format="csc",
        )

        A_left = sp.vstack(
            [
                sp.csc_matrix(self._M_full + 1e-12),
                sp.csc_matrix((self.n_fc_cons, self.nv), dtype=np.float64),
            ],
            format="csc",
        )
        A_right = sp.csc_matrix(np.vstack([-j_c.T, self._C_fc]) + 1e-12)
        self._A_csc = sp.hstack([A_left, A_right], format="csc")

        px_vals, ax_vals = self._assemble_sparse_qp_values(self._M_full, j_c)
        self._P_csc.data[:] = px_vals
        self._A_csc.data[:] = ax_vals

        q_zero = np.zeros(self.n_vars, dtype=np.float64)
        l_zero = np.zeros(self.n_cons, dtype=np.float64)
        u_zero = np.zeros(self.n_cons, dtype=np.float64)

        self._prob = osqp.OSQP()
        self._prob.setup(
            P=self._P_csc,
            q=q_zero,
            A=self._A_csc,
            l=l_zero,
            u=u_zero,
            warm_start=True,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=150,
            polish=False,
        )
        self._step_counter = 0

    def solve(
        self,
        sim_data: mujoco.MjData,
        u_ref: np.ndarray,
        x_ref: np.ndarray,
        q_des: np.ndarray,
        x_curr: np.ndarray | None = None,
        sync_nominal: bool = True,
    ) -> dict[str, np.ndarray | float | str | bool]:
        """在每个 500 Hz 控制步求解全身 QP-WBC，输出 29 维关节控制力矩。"""
        t0 = time.perf_counter()

        # 1. 将当前测量状态 (qpos, qvel) 同步至内部名义模型并前向计算动力学量
        if sync_nominal:
            self.nom_data.qpos[:] = sim_data.qpos
            self.nom_data.qvel[:] = sim_data.qvel
            mujoco.mj_forward(self.nom_model, self.nom_data)

        mujoco.mj_fullM(self.nom_model, self.nom_data, self._M_full)
        M = self._M_full
        h = self.nom_data.qfrc_bias  # (35,)

        j_l, j_r = extract_foot_jacobians(self.nom_model, self.nom_data)
        J_c = np.vstack([j_l, j_r])  # (12, 35)

        # 2. 计算期望广义加速度 qacc_des ∈ R^35（纯 NumPy 矩阵运算，避免 SciPy Rotation 开销）
        if x_curr is None:
            x_curr = extract_com_state(self.nom_model, self.nom_data)
        p_com, v_com, omega_world = x_curr[0:3], x_curr[6:9], x_curr[9:12]
        p_ref, rpy_ref, v_ref, omega_ref = x_ref[0:3], x_ref[3:6], x_ref[6:9], x_ref[9:12]

        # 基座线加速度期望（世界系）+ 中层接触力指令诱导的前馈加速度
        f_net_ref = u_ref[0:3] + u_ref[6:9]
        m_nom = float(np.sum(self.nom_model.body_mass))
        a_ff_world = f_net_ref / m_nom - np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        a_base_world = (
            a_ff_world
            + self.cfg.kp_base_lin * (p_ref - p_com)
            + self.cfg.kd_base_lin * (v_ref - v_com)
        )

        # 基座角加速度期望（注意 MuJoCo freejoint 的 qacc[3:6] 定义在机体系）
        r_wb = np.asarray(self.nom_data.xmat[1], dtype=np.float64).reshape(3, 3)
        r_des = rpy_to_rotmat(rpy_ref)
        e_ori_world = rotmat_error_vee(r_des, r_wb)
        alpha_world = (
            self.cfg.kp_base_ang * e_ori_world
            + self.cfg.kd_base_ang * (omega_ref - omega_world)
        )
        alpha_body = r_wb.T @ alpha_world

        # 29 个受控关节的期望角加速度
        q_j = self.nom_data.qpos[self.qpos_adr]
        dq_j = self.nom_data.qvel[self.dof_adr]
        qacc_j_des = self.kp_joint * (q_des - q_j) - self.kd_joint * dq_j

        qacc_des = np.concatenate([a_base_world, alpha_body, qacc_j_des])

        # 3. 足底零滑移期望接触加速度 a_{c,des} = -kd_contact * J_c * qvel
        v_contact = J_c @ self.nom_data.qvel
        a_c_des = -self.cfg.kd_contact * v_contact

        # 4. 组装 QP 向量 q, l, u 与块稀疏矩阵数值并求解
        M_j = M[6:35, :]
        J_cj = J_c[:, 6:35]
        h_b = h[0:6]
        h_j = h[6:35]

        q_vec = np.empty(self.n_vars, dtype=np.float64)
        q_vec[0:35] = -self.w_qacc * qacc_des - J_c.T @ (self.w_contact * a_c_des)
        q_vec[35:47] = -self.w_force * u_ref

        l_vec = np.empty(self.n_cons, dtype=np.float64)
        u_vec = np.empty(self.n_cons, dtype=np.float64)

        # 0..5: 浮动基动力学等式 M_b * qacc - J_{c,b}^T * F_c = -h_b
        l_vec[0:6] = -h_b
        u_vec[0:6] = -h_b

        # 6..34: 关节力矩硬限幅 -tau_max - h_j <= M_j * qacc - J_{c,j}^T * F_c <= tau_max - h_j
        l_vec[6:35] = -self.tau_max - h_j
        u_vec[6:35] = self.tau_max - h_j

        # 35..54: 摩擦锥与接触旋量盒约束
        l_vec[35:55] = self._l_fc
        u_vec[35:55] = self._u_fc

        px_vals, ax_vals = self._assemble_sparse_qp_values(M, J_c)
        self._prob.update(
            Px=px_vals,
            q=q_vec,
            Ax=ax_vals,
            l=l_vec,
            u=u_vec,
        )
        res = self._prob.solve()
        self._step_counter += 1
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        status_str = str(res.info.status)
        is_optimal = status_str in ("solved", "solved inaccurate") and res.x is not None

        if is_optimal and np.all(np.isfinite(res.x)):
            w_opt = np.asarray(res.x, dtype=np.float64)
            qacc_opt = w_opt[0:35]
            fc_opt = w_opt[35:47]
            tau_wbc = M_j @ qacc_opt + h_j - J_cj.T @ fc_opt
        else:
            # 极端冲击下保底回退：重力/科氏力补偿 + 接触力雅可比映射
            qacc_opt = np.zeros(self.nv, dtype=np.float64)
            fc_opt = u_ref.copy()
            tau_wbc = h_j - J_cj.T @ fc_opt

        # 叠加低增益关节零空间阻抗稳定项并严格截断在 [-tau_max, tau_max]
        tau_servo = self.servo_kp * (q_des - q_j) - self.servo_kd * dq_j
        tau_cmd = np.clip(tau_wbc + tau_servo, -self.tau_max, self.tau_max)

        # 统计浮动基动力学等式残差与摩擦锥违背情况
        fb_residual = float(np.linalg.norm(M[0:6, :] @ qacc_opt + h_b - J_c[:, 0:6].T @ fc_opt))
        f_l, f_r = fc_opt[0:3], fc_opt[6:9]
        mu_used = self.cfg.mu
        fc_viol = bool(
            (abs(f_l[0]) > mu_used * max(f_l[2], 0.0) + 1e-2)
            or (abs(f_l[1]) > mu_used * max(f_l[2], 0.0) + 1e-2)
            or (abs(f_r[0]) > mu_used * max(f_r[2], 0.0) + 1e-2)
            or (abs(f_r[1]) > mu_used * max(f_r[2], 0.0) + 1e-2)
        )
        sat_ratio = float(np.max(np.abs(tau_cmd) / self.tau_max))

        return {
            "tau_cmd": tau_cmd,
            "tau_wbc": tau_wbc,
            "qacc_opt": qacc_opt,
            "fc_opt": fc_opt,
            "solve_time_ms": solve_time_ms,
            "status": status_str,
            "is_optimal": is_optimal,
            "fb_eq_residual": fb_residual,
            "friction_violation": fc_viol,
            "torque_saturation_ratio": sat_ratio,
        }


# --------------------------------------------------------------------------- #
# 闭环仿真器：支持纯 PD、路线二 (SRBD-MPC + QP-WBC) 与 DOB 基线
# --------------------------------------------------------------------------- #


def base_tilt_deg(quat_wxyz: np.ndarray) -> float:
    """计算骨盆 z 轴相对世界竖直方向的合倾角（度）。"""
    r_wb = quat_wxyz_to_rotmat(quat_wxyz)
    cos_ang = float(np.clip(r_wb[2, 2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_ang)))


def run_closed_loop_episode(
    controller_mode: str,
    duration: float = 6.0,
    perturbation: PerturbationConfig | None = None,
    mpc_cfg: MPCConfig | None = None,
    wbc_cfg: WBCConfig | None = None,
    dob_alpha: float = 0.25,
    pd_kp: float = 200.0,
    pd_kd: float = 5.0,
) -> dict[str, np.ndarray | float | str | bool]:
    """运行单次闭环仿真回合，支持 'pd' | 'route2' | 'dob' 三种控制器模式。"""
    sim_model, sim_data = load_g1_torque_model()
    qpos0 = reset_to_stand_keyframe(sim_model, sim_data)

    # 1. 在施加未知环境摄动前，初始化仅知晓名义参数 (33.34 kg) 的中层 SRBD-MPC 与下层 QP-WBC
    srbd_model = NominalSRBDModel.from_mujoco(sim_model, sim_data, dt=DEFAULT_DT)
    if mpc_cfg is None:
        mpc_cfg = MPCConfig(
            tau_foot_max=np.array([3.5, 12.0, 5.0], dtype=np.float64),
            r_diag=np.array(
                [
                    2.5, 2.5, 1.2e-3,   # f_L (fx, fy, fz)
                    0.20, 0.05, 0.20,   # tau_L (tx, ty, tz)
                    2.5, 2.5, 1.2e-3,   # f_R (fx, fy, fz)
                    0.20, 0.05, 0.20,   # tau_R (tx, ty, tz)
                ],
                dtype=np.float64,
            ),
        )
    mpc = NominalSRBDMPC(srbd_model, config=mpc_cfg)
    wbc = QPWBC(sim_model, config=wbc_cfg)
    dob = DisturbanceObserver(srbd_model, alpha=dob_alpha)

    x_ref = extract_com_state(sim_model, sim_data)
    z_com_nom = float(x_ref[2])
    q_stand = qpos0[wbc.qpos_adr].copy()
    x_ref_seq = np.tile(x_ref, (mpc.N, 1))

    # 2. 对仿真环境 sim_model / sim_data 施加未知偏心负载与关节摩擦摄动
    pert_cfg = PerturbationConfig() if perturbation is None else perturbation
    pert_info = apply_perturbation(sim_model, sim_data, pert_cfg)
    if pert_cfg.ground_mu is not None:
        wbc.set_friction_coefficient(pert_cfg.ground_mu)

    dt_sim = float(sim_model.opt.timestep)          # 0.002 s (500 Hz)
    mpc_decimation = int(round(DEFAULT_DT / dt_sim))  # 每 10 步 = 0.02 s (50 Hz)
    n_steps = int(round(duration / dt_sim))

    u_ref_curr = srbd_model.gravity_compensation_u(x_ref)
    p_l_init, p_r_init = extract_foot_positions(sim_model, sim_data)

    # 时序记录容器 (50 Hz 采样)
    t_log: list[float] = []
    x_com_log: list[np.ndarray] = []
    x_ref_log: list[np.ndarray] = []
    x_full_log: list[np.ndarray] = []
    u_mpc_log: list[np.ndarray] = []
    fc_wbc_log: list[np.ndarray] = []
    tau_log: list[np.ndarray] = []
    tilt_log: list[float] = []
    dob_log: list[np.ndarray] = []

    mpc_times_ms: list[float] = []
    wbc_times_ms: list[float] = []
    fb_residuals: list[float] = []
    fc_violations = 0
    total_wbc_calls = 0
    fell_down = False

    for step in range(n_steps):
        t_curr = float(sim_data.time)
        step_perturbation(sim_model, sim_data, t_curr, pert_cfg)

        # 读取当前状态（注意：为防止未知负载通过 subtree_com 直接改动质心几何定义，
        # 使用控制器名义模型同步 qpos/qvel 提取真实机体状态观测 x_obs）
        wbc.nom_data.qpos[:] = sim_data.qpos
        wbc.nom_data.qvel[:] = sim_data.qvel
        mujoco.mj_forward(wbc.nom_model, wbc.nom_data)
        x_obs = extract_com_state(wbc.nom_model, wbc.nom_data)
        p_l_curr, p_r_curr = extract_foot_positions(wbc.nom_model, wbc.nom_data)

        # ---- 中层 50 Hz MPC 更新 ----
        if step % mpc_decimation == 0 and controller_mode in ("route2", "dob"):
            if controller_mode == "dob":
                dob.update(x_obs, u_ref_curr, p_l_curr, p_r_curr)
                res_seq = dob.get_horizon_residual(mpc.N)
            else:
                res_seq = None

            u_opt_0, _ = mpc.solve(
                x_k=x_obs,
                x_ref=x_ref_seq,
                p_left=p_l_curr,
                p_right=p_r_curr,
                residual_seq=res_seq,
            )
            u_ref_curr = np.asarray(u_opt_0, dtype=np.float64)
            mpc_times_ms.append(float(mpc.last_solve_time_ms))

        # ---- 下层 500 Hz 力矩计算 ----
        q_des = compute_posture_q_des(q_stand, x_ref, z_com_nom=z_com_nom)

        if controller_mode == "pd":
            q_j = sim_data.qpos[wbc.qpos_adr]
            dq_j = sim_data.qvel[wbc.dof_adr]
            tau_cmd = np.clip(
                pd_kp * (q_des - q_j) - pd_kd * dq_j,
                -wbc.tau_max,
                wbc.tau_max,
            )
            fc_opt = u_ref_curr.copy()
        else:
            wbc_sol = wbc.solve(
                sim_data=sim_data,
                u_ref=u_ref_curr,
                x_ref=x_ref,
                q_des=q_des,
                x_curr=x_obs,
                sync_nominal=False,
            )
            tau_cmd = np.asarray(wbc_sol["tau_cmd"], dtype=np.float64)
            fc_opt = np.asarray(wbc_sol["fc_opt"], dtype=np.float64)
            wbc_times_ms.append(float(wbc_sol["solve_time_ms"]))
            fb_residuals.append(float(wbc_sol["fb_eq_residual"]))
            total_wbc_calls += 1
            if bool(wbc_sol["friction_violation"]):
                fc_violations += 1

        sim_data.ctrl[:] = tau_cmd

        # 50 Hz 采样存盘
        if step % mpc_decimation == 0:
            t_log.append(t_curr)
            x_com_log.append(x_obs.copy())
            x_ref_log.append(x_ref.copy())
            x_full_log.append(extract_full_state(wbc.nom_model, wbc.nom_data))
            u_mpc_log.append(u_ref_curr.copy())
            fc_wbc_log.append(fc_opt.copy())
            tau_log.append(tau_cmd.copy())
            tilt_log.append(base_tilt_deg(sim_data.qpos[3:7]))
            dob_log.append(dob.d_hat.copy())

        mujoco.mj_step(sim_model, sim_data)

        if not (np.isfinite(sim_data.qpos).all() and np.isfinite(sim_data.qvel).all()):
            fell_down = True
            break
        if float(sim_data.qpos[2]) < 0.45 or base_tilt_deg(sim_data.qpos[3:7]) > 45.0:
            fell_down = True
            break

    # 汇总统计指标
    x_com_arr = np.asarray(x_com_log, dtype=np.float64)
    x_ref_arr = np.asarray(x_ref_log, dtype=np.float64)
    tilt_arr = np.asarray(tilt_log, dtype=np.float64)

    com_err = x_com_arr[:, 0:3] - x_ref_arr[:, 0:3]
    com_rmse_mm = float(np.sqrt(np.mean(np.sum(com_err ** 2, axis=1))) * 1000.0)
    steady_idx = max(0, int(0.6 * len(x_com_arr)))
    steady_com_err_mm = np.mean(com_err[steady_idx:], axis=0) * 1000.0
    steady_com_norm_mm = float(np.linalg.norm(steady_com_err_mm))
    steady_tilt_deg = float(np.mean(tilt_arr[steady_idx:]))
    max_tilt_deg = float(np.max(tilt_arr))

    p_l_final, p_r_final = extract_foot_positions(sim_model, sim_data)
    foot_slip_mm = float(
        0.5
        * (
            np.linalg.norm(p_l_final[0:2] - p_l_init[0:2])
            + np.linalg.norm(p_r_final[0:2] - p_r_init[0:2])
        )
        * 1000.0
    )

    return {
        "controller_mode": controller_mode,
        "fell_down": fell_down,
        "t": np.asarray(t_log, dtype=np.float64),
        "x_com": x_com_arr,
        "x_ref": x_ref_arr,
        "x_full": np.asarray(x_full_log, dtype=np.float64),
        "u_mpc": np.asarray(u_mpc_log, dtype=np.float64),
        "fc_wbc": np.asarray(fc_wbc_log, dtype=np.float64),
        "tau": np.asarray(tau_log, dtype=np.float64),
        "tilt_deg": tilt_arr,
        "dob_d_hat": np.asarray(dob_log, dtype=np.float64),
        "com_rmse_mm": com_rmse_mm,
        "steady_com_err_mm": steady_com_err_mm,
        "steady_com_norm_mm": steady_com_norm_mm,
        "steady_tilt_deg": steady_tilt_deg,
        "max_tilt_deg": max_tilt_deg,
        "foot_slip_mm": foot_slip_mm,
        "mpc_solve_ms_mean": float(np.mean(mpc_times_ms)) if mpc_times_ms else 0.0,
        "wbc_solve_ms_mean": float(np.mean(wbc_times_ms)) if wbc_times_ms else 0.0,
        "wbc_solve_ms_p95": float(np.percentile(wbc_times_ms, 95)) if wbc_times_ms else 0.0,
        "fb_eq_residual_max": float(np.max(fb_residuals)) if fb_residuals else 0.0,
        "fc_violation_rate_pct": (100.0 * fc_violations / max(1, total_wbc_calls)),
        "payload_mass": pert_info["payload_mass"],
    }


# --------------------------------------------------------------------------- #
# 主函数：执行 Task B3 全部自检与验收实验
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task B3: 500 Hz 真 QP-WBC 力矩控制器、路线二与 DOB 基线验收"
    )
    parser.add_argument("--duration", type=float, default=6.0, help="单回合仿真时长 (s)")
    parser.add_argument("--payload", type=float, default=5.0, help="单侧未知偏心负载质量 (kg)")
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    print(LINE)
    print("Task B3: 下层 500 Hz 全身二次规划控制器 (QP-WBC)、路线二与 DOB 基线自检")
    print(LINE)

    # ------------------------------------------------------------------ #
    # 实验 1：空载 + 水平脉冲推扰 (t=2.0s, F=[35, 30, 0] N, 0.15s) 对比纯 PD 与路线二
    # ------------------------------------------------------------------ #
    push_nom_cfg = PerturbationConfig(
        payload_mass=0.0,
        push_force_world=np.array([35.0, 30.0, 0.0], dtype=np.float64),
        push_start_s=2.0,
        push_duration_s=0.15,
    )
    print("\n[实验 1] 空载 + 水平脉冲推扰 (F=[35, 30, 0] N @ t=2.0~2.15 s)：纯 PD vs. 路线二 (SRBD-MPC + QP-WBC)")
    res_pd_nom = run_closed_loop_episode(
        controller_mode="pd",
        duration=5.0,
        perturbation=push_nom_cfg,
        pd_kp=200.0,
        pd_kd=5.0,
    )
    res_r2_nom = run_closed_loop_episode(
        controller_mode="route2",
        duration=5.0,
        perturbation=push_nom_cfg,
    )

    print(
        f"  - 纯关节 PD (kp=200, kd=5) : 跌倒={res_pd_nom['fell_down']} | "
        f"CoM RMSE={res_pd_nom['com_rmse_mm']:.2f} mm | "
        f"最大倾角={res_pd_nom['max_tilt_deg']:.2f} deg | "
        f"末态倾角={res_pd_nom['steady_tilt_deg']:.2f} deg"
    )
    print(
        f"  - 路线二 (SRBD-MPC+QP-WBC) : 跌倒={res_r2_nom['fell_down']} | "
        f"CoM RMSE={res_r2_nom['com_rmse_mm']:.2f} mm | "
        f"最大倾角={res_r2_nom['max_tilt_deg']:.2f} deg | "
        f"末态倾角={res_r2_nom['steady_tilt_deg']:.2f} deg"
    )
    print(
        f"  - QP-WBC 实时性 & 约束精度 : 平均耗时={res_r2_nom['wbc_solve_ms_mean']:.3f} ms "
        f"(P95={res_r2_nom['wbc_solve_ms_p95']:.3f} ms) | "
        f"浮动基等式最大残差={res_r2_nom['fb_eq_residual_max']:.2e} | "
        f"摩擦锥违背率={res_r2_nom['fc_violation_rate_pct']:.2f}%"
    )

    # ------------------------------------------------------------------ #
    # 实验 2：单侧 5 kg 未知偏心负载 + 关节摩擦摄动 + 脉冲推扰：路线二 vs. SRBD-MPC + DOB
    # ------------------------------------------------------------------ #
    eccentric_cfg = PerturbationConfig(
        payload_mass=float(args.payload),
        payload_body_name="left_wrist_yaw_link",
        payload_local_offset=np.array([0.12, 0.08, -0.02], dtype=np.float64),
        damping_scale=2.0,
        frictionloss_add=1.2,
        push_force_world=np.array([25.0, -15.0, 0.0], dtype=np.float64),
        push_start_s=2.5,
        push_duration_s=0.15,
    )
    print(
        f"\n[实验 2] 左手挂载 {args.payload:.1f} kg 未知偏心负载 + 腿部摩擦摄动 + t=2.5s 推扰："
        "路线二 vs. SRBD-MPC+DOB"
    )
    res_r2_load = run_closed_loop_episode(
        controller_mode="route2",
        duration=args.duration,
        perturbation=eccentric_cfg,
    )
    res_dob_load = run_closed_loop_episode(
        controller_mode="dob",
        duration=args.duration,
        perturbation=eccentric_cfg,
        dob_alpha=0.25,
    )

    err_r2 = res_r2_load["steady_com_err_mm"]
    err_dob = res_dob_load["steady_com_err_mm"]
    print(
        f"  - 路线二 (Baseline 无补偿) : 跌倒={res_r2_load['fell_down']} | "
        f"稳态 CoM 偏移=[{err_r2[0]:+.2f}, {err_r2[1]:+.2f}, {err_r2[2]:+.2f}] mm "
        f"(||e_ss||={res_r2_load['steady_com_norm_mm']:.2f} mm) | "
        f"稳态机身倾角={res_r2_load['steady_tilt_deg']:.2f} deg "
        f"(峰值={res_r2_load['max_tilt_deg']:.2f} deg)"
    )
    print(
        f"  - 对比基线 (SRBD-MPC+DOB)  : 跌倒={res_dob_load['fell_down']} | "
        f"稳态 CoM 偏移=[{err_dob[0]:+.2f}, {err_dob[1]:+.2f}, {err_dob[2]:+.2f}] mm "
        f"(||e_ss||={res_dob_load['steady_com_norm_mm']:.2f} mm) | "
        f"稳态机身倾角={res_dob_load['steady_tilt_deg']:.2f} deg "
        f"(峰值={res_dob_load['max_tilt_deg']:.2f} deg)"
    )

    # ------------------------------------------------------------------ #
    # 保存路线二基线轨迹数据至 logs/route2_baseline.npz
    # ------------------------------------------------------------------ #
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = LOG_DIR / "route2_baseline.npz"
    np.savez(
        save_path,
        t=res_r2_load["t"],
        x_com=res_r2_load["x_com"],
        x_ref=res_r2_load["x_ref"],
        x_full=res_r2_load["x_full"],
        u_mpc=res_r2_load["u_mpc"],
        fc_wbc=res_r2_load["fc_wbc"],
        tau=res_r2_load["tau"],
        tilt_deg=res_r2_load["tilt_deg"],
        payload_mass=float(args.payload),
        steady_com_err_mm=res_r2_load["steady_com_err_mm"],
        steady_tilt_deg=res_r2_load["steady_tilt_deg"],
        dob_t=res_dob_load["t"],
        dob_x_com=res_dob_load["x_com"],
        dob_tilt_deg=res_dob_load["tilt_deg"],
        dob_steady_com_err_mm=res_dob_load["steady_com_err_mm"],
        dob_steady_tilt_deg=res_dob_load["steady_tilt_deg"],
    )
    print(f"\n[数据存盘] 已保存路线二与 DOB 基线对比数据至: {save_path}")

    # ------------------------------------------------------------------ #
    # 严格验收断言
    # ------------------------------------------------------------------ #
    checks = [
        ("QP-WBC 单步平均求解耗时 < 2.0 ms", res_r2_nom["wbc_solve_ms_mean"] < 2.0),
        ("浮动基动力学 6 维等式硬约束残差 < 1e-3", res_r2_nom["fb_eq_residual_max"] < 1e-3),
        ("空载推扰下路线二稳定不倒且倾角/误差优于纯 PD", (not res_r2_nom["fell_down"]) and (res_r2_nom["com_rmse_mm"] < res_pd_nom["com_rmse_mm"])),
        ("5 kg 未知偏心负载下路线二保持稳定不倒", not res_r2_load["fell_down"]),
        ("5 kg 未知偏心负载下路线二出现明显稳态偏移 (||e_ss|| > 5 mm 或倾角 > 0.5 deg)", (res_r2_load["steady_com_norm_mm"] > 5.0) or (res_r2_load["steady_tilt_deg"] > 0.5)),
        ("DOB 基线在静态站立偏载下能降低稳态 CoM 偏移", res_dob_load["steady_com_norm_mm"] < res_r2_load["steady_com_norm_mm"]),
    ]

    print("\n" + LINE)
    print("[Task B3 验收检查汇总]")
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    print(LINE)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
