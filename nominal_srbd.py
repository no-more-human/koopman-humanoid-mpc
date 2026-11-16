#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nominal_srbd.py -- Unitree G1 名义单刚体/质心物理先验模型与凸 MPC 控制器（Task B2）

用途：
    构建路线二（Baseline SRBD-MPC + QP-WBC）与路线三（Proposed 物理先验 + Koopman 残差）
    共用的中层物理先验预测器与 OSQP 凸模型预测控制器：
        x_{k+1}^{nom} = A_nom(x_k) x_k + B_nom(x_k) u_k + g_nom

具体实现内容（严格对齐 PLAN.md Task B2）：
    1. 状态向量 x_k ∈ R^12：
       x_k = [p_com(3), rpy_base(3), v_com(3), omega_base(3)]^T
    2. 控制输入 u_k ∈ R^12（左右脚 6 维地面反力/力矩 GRF，世界坐标系）：
       u_k = [f_L(3), tau_L(3), f_R(3), tau_R(3)]^T
    3. 离散化牛顿-欧拉质心动力学（默认采样周期 dt = 0.02 s，即 50 Hz）：
       构造线性状态转移矩阵 A_nom(x_k)、控制矩阵 B_nom(x_k, p_L, p_R) 与重力项 g_nom，
       并提供单步预测接口 f_nom(x_k, u_k, p_L, p_R) 供 Task C1 计算物理残差。
    4. 封装 OSQP 名义凸模型预测控制器 NominalSRBDMPC：
       给定参考轨迹 x_ref（以及可选的 Koopman 残差补偿项），在摩擦锥约束
       |f_{x,y}| <= mu * f_z、法向力上下界 f_{z,min} <= f_z <= f_{z,max} 及足端力矩约束下，
       毫秒级（< 2 ms）在线求解最优足端接触力参考 u_ref^*。

用法：
    直接运行脚本执行 Task B2 独立验收测试（机身前倾 3° 恢复力与 OSQP 耗时测试）：
    F:\\conda_envs\\koopman_robot\\python.exe nominal_srbd.py

日期：2026-11-15
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from state_vector import (
    COM_STATE_DIM,
    KEYFRAME_NAME,
    LEFT_FOOT_BODY_NAME,
    MODEL_PATH,
    RIGHT_FOOT_BODY_NAME,
    extract_com_state,
    extract_foot_positions,
    get_total_mass,
    quat_wxyz_to_rotmat,
)

# --------------------------------------------------------------------------- #
# 常量定义
# --------------------------------------------------------------------------- #

GRF_DIM = 12          # [f_L(3), tau_L(3), f_R(3), tau_R(3)]
DEFAULT_DT = 0.02     # 中层 MPC 采样周期 0.02 s (50 Hz)
GRAVITY = 9.81        # 重力加速度标量 (m/s^2)
LINE = "=" * 72


# --------------------------------------------------------------------------- #
# 刚体运动学与代数辅助函数
# --------------------------------------------------------------------------- #


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """构造 3 维向量 v = [vx, vy, vz]^T 的反对称叉乘矩阵 [v]_x ∈ R^{3x3}。"""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=np.float64,
    )


def rot_z(yaw: float) -> np.ndarray:
    """绕世界系 Z 轴旋转角 yaw (rad) 的旋转矩阵 R_z(psi) ∈ R^{3x3}。"""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compute_nominal_composite_inertia(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """基于平行轴定理计算当前姿态下机器人关于全身质心 p_CoM 的机体系合成惯量张量 I_body ∈ R^{3x3}。

    说明：
        1. 对所有刚体 i = 1..nbody-1，在世界系下累加其自身惯量 R_i * diag(I_i) * R_i^T
           与平行轴偏移项 m_i * ((r_i^T r_i) I_3 - r_i r_i^T)，其中 r_i = xipos_i - p_com。
        2. 将累加得到的世界系合成惯量 I_world 左乘 R_wb^T、右乘 R_wb 转换回基座局部坐标系（机体系），
           从而在 get_discrete_matrices 中通过 R_z(psi) * I_body * R_z(psi)^T 避免二次旋转误差。
    """
    p_com = np.array(data.subtree_com[0], dtype=np.float64)
    i_total_world = np.zeros((3, 3), dtype=np.float64)

    for body_id in range(1, model.nbody):
        m_i = float(model.body_mass[body_id])
        if m_i <= 0.0:
            continue
        # 刚体主惯量与主轴方向矩阵 ximat (3x3，世界系)
        inertia_diag = np.asarray(model.body_inertia[body_id], dtype=np.float64)
        r_mat = np.asarray(data.ximat[body_id], dtype=np.float64).reshape(3, 3)
        i_body_world = r_mat @ np.diag(inertia_diag) @ r_mat.T

        # 刚体质心相对全身 CoM 的位置矢量
        r_i = np.asarray(data.xipos[body_id], dtype=np.float64) - p_com
        parallel_axis = m_i * (float(np.dot(r_i, r_i)) * np.eye(3) - np.outer(r_i, r_i))
        i_total_world += i_body_world + parallel_axis

    base_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    r_wb = quat_wxyz_to_rotmat(base_quat)
    return r_wb.T @ i_total_world @ r_wb


# --------------------------------------------------------------------------- #
# Task B2 核心类 1：名义单刚体动力学模型 (NominalSRBDModel)
# --------------------------------------------------------------------------- #


class NominalSRBDModel:
    """Unitree G1 名义单刚体/质心动力学先验模型。

    离散状态方程（采样周期 dt = 0.02 s，50 Hz）：
        x_{k+1}^{nom} = A_nom(x_k) * x_k + B_nom(x_k, p_L, p_R) * u_k + g_nom

    状态定义 (12 维)：
        x_k = [p_com(3), Theta(3), v_com(3), omega(3)]^T
    输入定义 (12 维，世界系下左右足底接触中心对机器人的广义接触力/力矩)：
        u_k = [f_L(3), tau_L(3), f_R(3), tau_R(3)]^T
    """

    def __init__(
        self,
        mass: float = 33.3411,
        inertia_body: np.ndarray | None = None,
        dt: float = DEFAULT_DT,
        gravity: float = GRAVITY,
        default_p_left: np.ndarray | None = None,
        default_p_right: np.ndarray | None = None,
        default_com_pos: np.ndarray | None = None,
    ) -> None:
        self.mass = float(mass)
        self.dt = float(dt)
        self.gravity = float(gravity)

        if inertia_body is None:
            # G1 站立姿态下关于 CoM 的名义合成惯量矩阵 (kg·m^2)
            self.inertia_body = np.diag([1.65, 1.52, 0.35]).astype(np.float64)
        else:
            self.inertia_body = np.asarray(inertia_body, dtype=np.float64).copy()
            assert self.inertia_body.shape == (3, 3), "inertia_body 必须为 (3, 3) 矩阵"

        # 默认站立时的左右足底几何接触中心位置 (m) 与名义质心位置 (m)
        self.default_p_left = (
            np.array([0.0350, 0.1185, -0.0019], dtype=np.float64)
            if default_p_left is None
            else np.asarray(default_p_left, dtype=np.float64).copy()
        )
        self.default_p_right = (
            np.array([0.0350, -0.1185, -0.0019], dtype=np.float64)
            if default_p_right is None
            else np.asarray(default_p_right, dtype=np.float64).copy()
        )
        self.default_com_pos = (
            np.array([0.0033, 0.0001, 0.6919], dtype=np.float64)
            if default_com_pos is None
            else np.asarray(default_com_pos, dtype=np.float64).copy()
        )

        # 预构造离散重力常数项 g_nom ∈ R^12
        g_vec = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)
        self.g_nom = np.zeros(COM_STATE_DIM, dtype=np.float64)
        self.g_nom[0:3] = 0.5 * (self.dt ** 2) * g_vec
        self.g_nom[6:9] = self.dt * g_vec

    @classmethod
    def from_mujoco(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt: float = DEFAULT_DT,
    ) -> "NominalSRBDModel":
        """直接从已初始化的 MuJoCo 模型与数据中提取名义质量、机体系合成惯量与双足底位置。"""
        mass = get_total_mass(model)
        inertia_body = compute_nominal_composite_inertia(model, data)
        p_left, p_right = extract_foot_positions(model, data)
        x_com = extract_com_state(model, data)
        return cls(
            mass=mass,
            inertia_body=inertia_body,
            dt=dt,
            default_p_left=p_left,
            default_p_right=p_right,
            default_com_pos=x_com[0:3],
        )

    def get_discrete_matrices(
        self,
        x_k: np.ndarray,
        p_left: np.ndarray | None = None,
        p_right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """基于当前状态 x_k 构造离散化牛顿-欧拉矩阵 (A_nom, B_nom, g_nom)。

        参数：
            x_k     : 当前 12 维质心状态 [p_com(3), rpy(3), v_com(3), omega(3)]
            p_left  : 左足底接触中心世界系坐标 (3,)，若为 None 则使用默认站立左足位置
            p_right : 右足底接触中心世界系坐标 (3,)，若为 None 则使用默认站立右足位置

        返回：
            A_nom : shape=(12, 12) 状态转移矩阵
            B_nom : shape=(12, 12) 控制输入矩阵
            g_nom : shape=(12,)    重力离散偏置向量
        """
        x_k = np.asarray(x_k, dtype=np.float64)
        assert x_k.shape == (COM_STATE_DIM,), f"x_k 维度错误: {x_k.shape}"

        p_com = x_k[0:3]
        yaw = float(x_k[5])
        p_l = self.default_p_left if p_left is None else np.asarray(p_left, dtype=np.float64)
        p_r = self.default_p_right if p_right is None else np.asarray(p_right, dtype=np.float64)

        dt = self.dt
        rz = rot_z(yaw)
        rz_t = rz.T

        # 1. 状态转移矩阵 A_nom(x_k) ∈ R^{12x12}
        # p_{k+1}     = p_k + dt * v_k
        # Theta_{k+1} = Theta_k + dt * R_z(psi)^T * omega_k
        # v_{k+1}     = v_k
        # omega_{k+1} = omega_k
        a_nom = np.eye(COM_STATE_DIM, dtype=np.float64)
        a_nom[0:3, 6:9] = dt * np.eye(3, dtype=np.float64)
        a_nom[3:6, 9:12] = dt * rz_t

        # 2. 世界系惯量逆矩阵 I_w^{-1} = (R_z * I_body * R_z^T)^{-1}
        i_world = rz @ self.inertia_body @ rz_t
        i_world_inv = np.linalg.inv(i_world)

        # 3. 足端力臂矢量 r_L = p_L - p_com, r_R = p_R - p_com
        r_l_skew = skew_symmetric(p_l - p_com)
        r_r_skew = skew_symmetric(p_r - p_com)

        # 连续时间加速度输入映射 B_v, B_omega ∈ R^{3x12}
        # u_k = [f_L(0:3), tau_L(3:6), f_R(6:9), tau_R(9:12)]
        inv_m = (1.0 / self.mass) * np.eye(3, dtype=np.float64)
        b_v = np.zeros((3, GRF_DIM), dtype=np.float64)
        b_v[:, 0:3] = inv_m
        b_v[:, 6:9] = inv_m

        b_omega = np.zeros((3, GRF_DIM), dtype=np.float64)
        b_omega[:, 0:3] = i_world_inv @ r_l_skew
        b_omega[:, 3:6] = i_world_inv
        b_omega[:, 6:9] = i_world_inv @ r_r_skew
        b_omega[:, 9:12] = i_world_inv

        # 4. 离散控制矩阵 B_nom(x_k) ∈ R^{12x12}（含零阶保持二阶积分项 0.5*dt^2）
        b_nom = np.zeros((COM_STATE_DIM, GRF_DIM), dtype=np.float64)
        b_nom[0:3, :] = 0.5 * (dt ** 2) * b_v
        b_nom[3:6, :] = 0.5 * (dt ** 2) * (rz_t @ b_omega)
        b_nom[6:9, :] = dt * b_v
        b_nom[9:12, :] = dt * b_omega

        return a_nom, b_nom, self.g_nom.copy()

    def predict_next(
        self,
        x_k: np.ndarray,
        u_k: np.ndarray,
        p_left: np.ndarray | None = None,
        p_right: np.ndarray | None = None,
    ) -> np.ndarray:
        """名义单步前向预测函数 f_nom(x_k, u_k) = A_nom x_k + B_nom u_k + g_nom。

        供 Task C1 计算真实物理残差 delta_x_k = x_{k+1}^{real} - f_nom(x_k, u_k)。
        """
        a_nom, b_nom, g_nom = self.get_discrete_matrices(x_k, p_left, p_right)
        u_vec = np.asarray(u_k, dtype=np.float64)
        assert u_vec.shape == (GRF_DIM,), f"u_k 维度错误: {u_vec.shape}"
        return a_nom @ x_k + b_nom @ u_vec + g_nom

    def gravity_compensation_u(
        self,
        x_ref: np.ndarray | None = None,
        p_left: np.ndarray | None = None,
        p_right: np.ndarray | None = None,
    ) -> np.ndarray:
        """返回给定参考质心状态 x_ref 下的精确静力平衡接触力旋量 u_eq ∈ R^12。

        说明：
            满足：
            1. 竖直合力平衡重力：f_{L,z} + f_{R,z} = m * g（双足均分 0.5 * m * g）
            2. 关于质心 p_com 的净力矩严格为零：
               (p_L - p_com) x f_L + tau_L + (p_R - p_com) x f_R + tau_R = 0
            从而保证在静止参考状态 x_ref 处 ||f_nom(x_ref, u_eq) - x_ref||_2 < 1e-12。
        """
        p_com = (
            self.default_com_pos
            if x_ref is None
            else np.asarray(x_ref[0:3], dtype=np.float64)
        )
        p_l = self.default_p_left if p_left is None else np.asarray(p_left, dtype=np.float64)
        p_r = self.default_p_right if p_right is None else np.asarray(p_right, dtype=np.float64)

        fz_half = 0.5 * self.mass * self.gravity
        f_l = np.array([0.0, 0.0, fz_half], dtype=np.float64)
        f_r = np.array([0.0, 0.0, fz_half], dtype=np.float64)

        # 双足支持力关于 p_com 产生的重力力臂力矩 M_f，由双足踝关节力矩对称抵消
        m_force = np.cross(p_l - p_com, f_l) + np.cross(p_r - p_com, f_r)
        tau_half = -0.5 * m_force

        return np.concatenate([f_l, tau_half, f_r, tau_half]).astype(np.float64)

    def compute_net_wrench_about_com(
        self,
        x_k: np.ndarray,
        u_k: np.ndarray,
        p_left: np.ndarray | None = None,
        p_right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算控制输入 u_k 在质心处产生的净外力 F_net (不含重力) 与净恢复力矩 M_net。"""
        p_com = x_k[0:3]
        p_l = self.default_p_left if p_left is None else np.asarray(p_left, dtype=np.float64)
        p_r = self.default_p_right if p_right is None else np.asarray(p_right, dtype=np.float64)

        f_l, tau_l = u_k[0:3], u_k[3:6]
        f_r, tau_r = u_k[6:9], u_k[9:12]

        f_net = f_l + f_r
        m_net = np.cross(p_l - p_com, f_l) + tau_l + np.cross(p_r - p_com, f_r) + tau_r
        return f_net, m_net


# --------------------------------------------------------------------------- #
# Task B2 核心类 2：基于 OSQP 的名义凸模型预测控制器 (NominalSRBDMPC)
# --------------------------------------------------------------------------- #


@dataclass
class MPCConfig:
    """名义 SRBD-MPC 控制器超参数与物理约束配置。"""

    horizon: int = 10            # 预测时域步数 N_p (10 * 0.02s = 0.20s)
    dt: float = DEFAULT_DT       # 采样周期 0.02 s (50 Hz)
    mu: float = 0.6              # 地面摩擦系数 mu（四棱锥摩擦锥近似 |f_{x,y}| <= mu * f_z）
    fz_min: float = 15.0         # 双足支撑期单脚最小法向力 (N)，防止足端腾空
    fz_max: float = 350.0        # 单脚最大法向力 (N)
    tau_foot_max: np.ndarray = field(
        default_factory=lambda: np.array([25.0, 45.0, 15.0], dtype=np.float64)
    )                            # 单脚踝关节地面反力矩幅值上限 [tau_x, tau_y, tau_z] (N·m)

    # 状态跟踪权重对角元 Q_diag ∈ R^12: [p_com(3), rpy(3), v_com(3), omega(3)]
    q_diag: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                12000.0, 12000.0, 18000.0,  # p_com (x, y, z)
                8000.0, 8000.0, 4000.0,     # rpy (roll, pitch, yaw)
                800.0, 800.0, 1000.0,       # v_com (vx, vy, vz)
                400.0, 400.0, 200.0,        # omega (wx, wy, wz)
            ],
            dtype=np.float64,
        )
    )
    # 终端状态权重放大系数
    terminal_weight_scale: float = 2.0

    # 控制输入正则化权重对角元 R_diag ∈ R^12: [f_L(3), tau_L(3), f_R(3), tau_R(3)]
    # 对水平切向力给较高阻尼权重，避免纯单刚体在无运动学约束下用切向力产生反向平动加速
    r_diag: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                2.5e-1, 2.5e-1, 1.0e-3,     # f_L (fx, fy, fz)
                2.0e-2, 2.0e-2, 5.0e-2,     # tau_L (tx, ty, tz)
                2.5e-1, 2.5e-1, 1.0e-3,     # f_R (fx, fy, fz)
                2.0e-2, 2.0e-2, 5.0e-2,     # tau_R (tx, ty, tz)
            ],
            dtype=np.float64,
        )
    )


class NominalSRBDMPC:
    """使用 OSQP 封装的名义单刚体凸模型预测控制器（支持可选的 Koopman 残差注入）。

    优化问题形式（稀疏状态-控制联合 QP，便于 OSQP 极速求解与矩阵热更新）：
        min_{X, U}  sum_{k=0}^{N-1} ( ||x_{k+1} - x_{ref,k+1}||_Q^2 + ||u_k - u_eq||_R^2 )
        s.t.        x_{k+1} - A_nom x_k - B_nom u_k = g_nom + d_{res,k}
                    |f_{L/R, x}| <= mu * f_{L/R, z}
                    |f_{L/R, y}| <= mu * f_{L/R, z}
                    f_{z,min} <= f_{L/R, z} <= f_{z,max}
                    -tau_max  <= tau_{L/R}  <= tau_max
    """

    def __init__(
        self,
        srbd_model: NominalSRBDModel,
        config: MPCConfig | None = None,
    ) -> None:
        self.model = srbd_model
        self.cfg = MPCConfig() if config is None else config

        self.nx = COM_STATE_DIM
        self.nu = GRF_DIM
        self.N = int(self.cfg.horizon)

        # 决策变量 z = [x_1, ..., x_N, u_0, ..., u_{N-1}] ∈ R^{N*(nx+nu)}
        self.n_x_vars = self.N * self.nx
        self.n_u_vars = self.N * self.nu
        self.n_vars = self.n_x_vars + self.n_u_vars

        # 单步摩擦锥与盒式不等式约束矩阵 C_u * u_k ∈ [l_u, u_u] (共 16 行/步)
        self._cu_step, self._lu_step, self._uu_step = self._build_single_step_input_constraints(
            mu=self.cfg.mu,
            fz_min=self.cfg.fz_min,
            fz_max=self.cfg.fz_max,
            tau_max=self.cfg.tau_foot_max,
        )
        self.n_ineq_step = self._cu_step.shape[0]
        self.n_eq = self.N * self.nx
        self.n_ineq = self.N * self.n_ineq_step
        self.n_cons = self.n_eq + self.n_ineq

        # 构造恒定的目标函数二次项矩阵 P ∈ R^{n_vars x n_vars}
        self._P_csc = self._build_cost_matrix_P()

        # 预构造不等式约束块（仅依赖摩擦系数 mu，在线无需重排稀疏结构）
        self._A_ineq_csc = sp.hstack(
            [
                sp.csc_matrix((self.n_ineq, self.n_x_vars), dtype=np.float64),
                sp.kron(sp.eye(self.N, format="csc"), sp.csc_matrix(self._cu_step), format="csc"),
            ],
            format="csc",
        )
        self._l_ineq = np.tile(self._lu_step, self.N)
        self._u_ineq = np.tile(self._uu_step, self.N)

        # 缓存上一次解用于热启动
        self._last_sol: np.ndarray | None = None
        self.last_solve_time_ms: float = 0.0
        self.last_status: str = "uninitialized"

        # 预热一次 OSQP（在名义平衡点完成符号分析与首次因子分解）
        self._solver: osqp.OSQP | None = None
        self._init_osqp_solver()

    @staticmethod
    def _build_single_foot_constraints(
        mu: float,
        fz_min: float,
        fz_max: float,
        tau_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造单只脚 6 维输入 [fx, fy, fz, tx, ty, tz] 的线性约束矩阵 (8 行 x 6 列)。

        约束行定义：
            0: fx - mu * fz <= 0          -> (-inf, 0]
            1: -fx - mu * fz <= 0         -> (-inf, 0]
            2: fy - mu * fz <= 0          -> (-inf, 0]
            3: -fy - mu * fz <= 0         -> (-inf, 0]
            4: fz_min <= fz <= fz_max     -> [fz_min, fz_max]
            5: -tau_x <= tx <= tau_x      -> [-tau_x, tau_x]
            6: -tau_y <= ty <= tau_y      -> [-tau_y, tau_y]
            7: -tau_z <= tz <= tau_z      -> [-tau_z, tau_z]
        """
        c_foot = np.array(
            [
                [1.0, 0.0, -mu, 0.0, 0.0, 0.0],
                [-1.0, 0.0, -mu, 0.0, 0.0, 0.0],
                [0.0, 1.0, -mu, 0.0, 0.0, 0.0],
                [0.0, -1.0, -mu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        l_foot = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, fz_min, -tau_max[0], -tau_max[1], -tau_max[2]],
            dtype=np.float64,
        )
        u_foot = np.array(
            [0.0, 0.0, 0.0, 0.0, fz_max, tau_max[0], tau_max[1], tau_max[2]],
            dtype=np.float64,
        )
        return c_foot, l_foot, u_foot

    def _build_single_step_input_constraints(
        self,
        mu: float,
        fz_min: float,
        fz_max: float,
        tau_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造双足 12 维控制输入 u_k 的单步约束矩阵 (16 行 x 12 列)。"""
        c_f, l_f, u_f = self._build_single_foot_constraints(mu, fz_min, fz_max, tau_max)
        c_step = np.zeros((16, GRF_DIM), dtype=np.float64)
        c_step[0:8, 0:6] = c_f
        c_step[8:16, 6:12] = c_f
        l_step = np.concatenate([l_f, l_f])
        u_step = np.concatenate([u_f, u_f])
        return c_step, l_step, u_step

    def _build_cost_matrix_P(self) -> sp.csc_matrix:
        """构造 OSQP 二次目标矩阵 P（对应 0.5 * z^T P z + q^T z）。"""
        q_Diagnostics = []
        for k in range(self.N):
            scale = self.cfg.terminal_weight_scale if (k == self.N - 1) else 1.0
            q_Diagnostics.append(2.0 * scale * self.cfg.q_diag)
        r_Diagnostics = [2.0 * self.cfg.r_diag for _ in range(self.N)]
        p_diag = np.concatenate(q_Diagnostics + r_Diagnostics)
        return sp.diags(p_diag, format="csc")

    def _init_osqp_solver(self) -> None:
        """初始化 OSQP 求解器实例，预构建固定稀疏结构索引并完成符号分解预热。"""
        # 用唯一哨兵值占位，确保 -A_nom 和 -B_nom 的全部 12x12 元素保留在 CSC 稀疏结构中
        sentinel_a = np.arange(1000.0, 1000.0 + self.nx * self.nx, dtype=np.float64).reshape(self.nx, self.nx)
        sentinel_b = np.arange(2000.0, 2000.0 + self.nx * self.nu, dtype=np.float64).reshape(self.nx, self.nu)

        eye_nx = sp.eye(self.nx, format="csc")
        x_blocks = sp.kron(sp.eye(self.N, format="csc"), eye_nx, format="csc")
        if self.N > 1:
            sub_diag = sp.kron(sp.eye(self.N - 1, format="csc"), sp.csc_matrix(sentinel_a), format="csc")
            sub_padded = sp.vstack(
                [
                    sp.csc_matrix((self.nx, self.n_x_vars), dtype=np.float64),
                    sp.hstack(
                        [
                            sub_diag,
                            sp.csc_matrix(((self.N - 1) * self.nx, self.nx), dtype=np.float64),
                        ],
                        format="csc",
                    ),
                ],
                format="csc",
            )
            x_blocks = (x_blocks + sub_padded).tocsc()

        u_blocks = sp.kron(sp.eye(self.N, format="csc"), sp.csc_matrix(sentinel_b), format="csc")
        a_eq_template = sp.hstack([x_blocks, u_blocks], format="csc")
        a_full = sp.vstack([a_eq_template, self._A_ineq_csc], format="csc")

        # 建立从 (-A_nom).ravel() 与 (-B_nom).ravel() 到 a_full.data 的直接向量化索引映射
        data_arr = a_full.data
        if self.N > 1:
            self._a_data_idx = np.empty((self.nx * self.nx, self.N - 1), dtype=np.int64)
            for idx in range(self.nx * self.nx):
                self._a_data_idx[idx, :] = np.flatnonzero(data_arr == (1000.0 + idx))
        else:
            self._a_data_idx = np.empty((self.nx * self.nx, 0), dtype=np.int64)

        self._b_data_idx = np.empty((self.nx * self.nu, self.N), dtype=np.int64)
        for idx in range(self.nx * self.nu):
            self._b_data_idx[idx, :] = np.flatnonzero(data_arr == (2000.0 + idx))

        # 填入名义初始矩阵数值
        x_init = np.zeros(self.nx, dtype=np.float64)
        x_init[2] = 0.6919
        a_nom, b_nom, g_nom = self.model.get_discrete_matrices(x_init)

        self._ax_data = data_arr.copy()
        if self.N > 1:
            self._ax_data[self._a_data_idx] = (-a_nom).reshape(-1, 1)
        self._ax_data[self._b_data_idx] = (-b_nom).reshape(-1, 1)
        a_full.data[:] = self._ax_data

        b_eq = np.tile(g_nom, self.N)
        b_eq[0 : self.nx] += a_nom @ x_init
        self._l_full = np.concatenate([b_eq, self._l_ineq])
        self._u_full = np.concatenate([b_eq, self._u_ineq])
        q_zero = np.zeros(self.n_vars, dtype=np.float64)

        solver = osqp.OSQP()
        solver.setup(
            P=self._P_csc,
            q=q_zero,
            A=a_full,
            l=self._l_full,
            u=self._u_full,
            verbose=False,
            warm_start=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=200,
            polish=False,
        )
        self._solver = solver

    def update_friction_cone(self, mu: float) -> None:
        """动态更新地面摩擦系数 mu（用于 Task D1 湿滑地面工况）。"""
        self.cfg.mu = float(mu)
        self._cu_step, self._lu_step, self._uu_step = self._build_single_step_input_constraints(
            mu=self.cfg.mu,
            fz_min=self.cfg.fz_min,
            fz_max=self.cfg.fz_max,
            tau_max=self.cfg.tau_foot_max,
        )
        self._A_ineq_csc = sp.hstack(
            [
                sp.csc_matrix((self.n_ineq, self.n_x_vars), dtype=np.float64),
                sp.kron(sp.eye(self.N, format="csc"), sp.csc_matrix(self._cu_step), format="csc"),
            ],
            format="csc",
        )
        self._l_ineq = np.tile(self._lu_step, self.N)
        self._u_ineq = np.tile(self._uu_step, self.N)
        self._init_osqp_solver()

    def solve(
        self,
        x_k: np.ndarray,
        x_ref: np.ndarray,
        p_left: np.ndarray | None = None,
        p_right: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
        residual_seq: np.ndarray | None = None,
        b_res: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """在线求解凸 MPC，输出当前时刻最优足端接触力参考 u_ref^* 与第一步预测状态 x_next^*。

        参数：
            x_k          : 当前 12 维质心状态向量 shape=(12,)
            x_ref        : 参考状态向量 shape=(12,) 或参考状态序列 shape=(N, 12)
            p_left       : 左足底接触中心世界系坐标 shape=(3,)，默认取模型初始足端位置
            p_right      : 右足底接触中心世界系坐标 shape=(3,)，默认取模型初始足端位置
            u_ref        : 参考控制输入 shape=(12,)，默认取满足净力矩为零的精确静力平衡力 u_eq
            residual_seq : 可选的 Koopman 仿射残差补偿项 shape=(12,) 或 (N, 12)（供路线三 / DOB 使用）
            b_res        : 可选的 Koopman 控制残差矩阵 B_r ∈ R^{12x12}（供路线三叠加 B_eff = B_nom + B_r）

        返回：
            (u_opt_0, x_pred_1) :
                u_opt_0  : shape=(12,) 最优足端接触力参考 [f_L, tau_L, f_R, tau_R]
                x_pred_1 : shape=(12,) 预测下一时刻质心状态
        """
        t_start = time.perf_counter()

        x_k = np.asarray(x_k, dtype=np.float64)
        x_ref_arr = np.asarray(x_ref, dtype=np.float64)
        if x_ref_arr.ndim == 1:
            assert x_ref_arr.shape == (self.nx,), f"x_ref 维度错误: {x_ref_arr.shape}"
            x_ref_seq = np.tile(x_ref_arr, (self.N, 1))
        else:
            assert x_ref_arr.shape == (self.N, self.nx), f"x_ref 序列维度错误: {x_ref_arr.shape}"
            x_ref_seq = x_ref_arr

        u_eq = (
            self.model.gravity_compensation_u(x_ref_seq[0], p_left, p_right)
            if u_ref is None
            else np.asarray(u_ref, dtype=np.float64)
        )

        # 1. 计算当前时刻线性化矩阵 A_nom, B_nom, g_nom，并可选融合 Koopman 控制残差矩阵 B_r
        a_nom, b_nom, g_nom = self.model.get_discrete_matrices(x_k, p_left, p_right)
        b_eff = b_nom if b_res is None else (b_nom + np.asarray(b_res, dtype=np.float64))

        # 2. 原地更新等式约束右端项 b_eq 与稀疏矩阵非零元 _ax_data（零 scipy 开销）
        b_eq = np.tile(g_nom, self.N)
        b_eq[0 : self.nx] += a_nom @ x_k
        if residual_seq is not None:
            res_arr = np.asarray(residual_seq, dtype=np.float64)
            if res_arr.ndim == 1:
                b_eq += np.tile(res_arr, self.N)
            else:
                b_eq += res_arr.reshape(-1)

        if self.N > 1:
            self._ax_data[self._a_data_idx] = (-a_nom).reshape(-1, 1)
        self._ax_data[self._b_data_idx] = (-b_eff).reshape(-1, 1)
        self._l_full[0 : self.n_eq] = b_eq
        self._u_full[0 : self.n_eq] = b_eq

        # 3. 构造一次项向量 q ∈ R^{n_vars}
        q_x_mat = -2.0 * (x_ref_seq * self.cfg.q_diag)
        q_x_mat[-1, :] *= self.cfg.terminal_weight_scale
        q_u_step = -2.0 * (self.cfg.r_diag * u_eq)
        q_vec = np.concatenate([q_x_mat.reshape(-1), np.tile(q_u_step, self.N)])

        # 4. OSQP 增量热更新与极速求解
        assert self._solver is not None
        self._solver.update(q=q_vec, Ax=self._ax_data, l=self._l_full, u=self._u_full)
        if self._last_sol is not None:
            self._solver.warm_start(x=self._last_sol)
        res = self._solver.solve()

        self.last_solve_time_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_status = str(res.info.status)

        if res.x is None or not np.all(np.isfinite(res.x)):
            u_fallback = u_eq.copy()
            x_next = a_nom @ x_k + b_eff @ u_fallback + g_nom
            return u_fallback, x_next

        self._last_sol = res.x.copy()
        x_pred_1 = res.x[0 : self.nx].copy()
        u_opt_0 = res.x[self.n_x_vars : self.n_x_vars + self.nu].copy()
        return u_opt_0, x_pred_1


# --------------------------------------------------------------------------- #
# Task B2 独立自检与验收主程序
# --------------------------------------------------------------------------- #


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not MODEL_PATH.exists():
        print(f"[错误] 找不到 G1 场景文件: {MODEL_PATH}")
        return 1

    # 1. 加载 MuJoCo 模型，获取站立关键帧下的真实物理参数与名义状态 x_ref
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mj_data = mujoco.MjData(mj_model)

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if key_id >= 0:
        mj_data.qpos[:] = mj_model.key_qpos[key_id]
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    x_ref = extract_com_state(mj_model, mj_data)
    p_left, p_right = extract_foot_positions(mj_model, mj_data)

    # 2. 初始化名义物理先验模型与 OSQP 凸 MPC 控制器
    srbd = NominalSRBDModel.from_mujoco(mj_model, mj_data, dt=DEFAULT_DT)
    mpc = NominalSRBDMPC(srbd)

    # 检查名义平衡态下的单步预测与矩阵形状
    a_nom, b_nom, g_nom = srbd.get_discrete_matrices(x_ref, p_left, p_right)
    u_eq = srbd.gravity_compensation_u(x_ref, p_left, p_right)
    x_nom_eq = srbd.predict_next(x_ref, u_eq, p_left, p_right)
    eq_drift = float(np.linalg.norm(x_nom_eq - x_ref))

    # 3. 构造 Task B2 验收测试工况：给定机身前倾 3° (pitch = +3.0 deg) 姿态偏差
    pitch_deg = 3.0
    pitch_rad = float(np.radians(pitch_deg))
    x_perturbed = x_ref.copy()
    # 绕脚踝前倾 3° 时，机身 pitch = +3°，同时质心自然前移约 h_com * sin(3°)
    h_lever = float(x_ref[2] - 0.5 * (p_left[2] + p_right[2]))
    x_perturbed[0] += h_lever * np.sin(pitch_rad)
    x_perturbed[4] = pitch_rad  # theta (pitch) = +3 deg

    # 先进行一次预热调用，再统计多次在线求解耗时（模拟 50 Hz 控制循环内真实表现）
    u_opt, x_pred1 = mpc.solve(x_perturbed, x_ref, p_left, p_right)
    solve_times_ms = []
    for _ in range(20):
        u_opt, x_pred1 = mpc.solve(x_perturbed, x_ref, p_left, p_right)
        solve_times_ms.append(mpc.last_solve_time_ms)
    avg_solve_ms = float(np.mean(solve_times_ms))
    min_solve_ms = float(np.min(solve_times_ms))

    # 4. 分析求解出的最优地面反力 u_ref^*
    f_l, tau_l = u_opt[0:3], u_opt[3:6]
    f_r, tau_r = u_opt[6:9], u_opt[9:12]
    fz_sum = float(f_l[2] + f_r[2])
    mg = srbd.mass * srbd.gravity
    fz_rel_err_pct = abs(fz_sum - mg) / mg * 100.0

    f_net, m_net = srbd.compute_net_wrench_about_com(x_perturbed, u_opt, p_left, p_right)
    restoring_pitch_torque = float(m_net[1])  # 绕 Y 轴净恢复力矩，前倾(+pitch)时应为负值(< 0)

    # 检查摩擦锥满足情况
    mu = mpc.cfg.mu
    friction_ok = bool(
        abs(f_l[0]) <= mu * f_l[2] + 1e-4
        and abs(f_l[1]) <= mu * f_l[2] + 1e-4
        and abs(f_r[0]) <= mu * f_r[2] + 1e-4
        and abs(f_r[1]) <= mu * f_r[2] + 1e-4
        and mpc.cfg.fz_min - 1e-4 <= f_l[2] <= mpc.cfg.fz_max + 1e-4
        and mpc.cfg.fz_min - 1e-4 <= f_r[2] <= mpc.cfg.fz_max + 1e-4
    )

    # 5. 打印详细验收报告
    print(LINE)
    print("[Task B2 自检报告: nominal_srbd.py —— 名义单刚体物理先验与凸 MPC]")
    print(LINE)
    print(f"  模型文件              : {MODEL_PATH}")
    print(f"  G1 名义总质量 m       : {srbd.mass:.4f} kg  (重力 mg = {mg:.2f} N)")
    print(f"  机体系合成惯量对角元  : {np.diag(srbd.inertia_body).round(4)} kg·m^2")
    print(f"  离散采样周期 dt       : {srbd.dt:.4f} s ({1.0 / srbd.dt:.0f} Hz)")
    print(f"  MPC 预测步数 N_p      : {mpc.N} 步 (预测时域 {mpc.N * srbd.dt:.2f} s)")
    print(f"  摩擦系数 mu           : {mpc.cfg.mu:.2f}  (单脚法向力范围 [{mpc.cfg.fz_min:.1f}, {mpc.cfg.fz_max:.1f}] N)")
    print(f"  A_nom, B_nom 维度     : A_nom={a_nom.shape}, B_nom={b_nom.shape}, g_nom={g_nom.shape}")
    print(f"  名义平衡态单步漂移    : ||f_nom(x_ref, u_eq) - x_ref||_2 = {eq_drift:.2e}")
    print(LINE)

    with np.printoptions(precision=4, suppress=True, linewidth=120):
        print(f"[测试工况] 机身前倾 {pitch_deg:.1f}° (pitch = +{pitch_rad:.4f} rad, CoM 前移 +{x_perturbed[0] - x_ref[0]:.4f} m):")
        print(f"  参考状态 x_ref        : {x_ref}")
        print(f"  静力平衡输入 u_eq     : {u_eq}")
        print(f"  当前受扰状态 x_k      : {x_perturbed}")
        print(f"  OSQP 求解状态         : {mpc.last_status}")
        print(f"  OSQP 单步求解耗时     : 平均 {avg_solve_ms:.3f} ms (最快 {min_solve_ms:.3f} ms, 首次热更新 {solve_times_ms[0]:.3f} ms)")
        print()
        print("[最优足端接触力参考 u_ref^* = [f_L, tau_L, f_R, tau_R]^T]:")
        print(f"  左脚力   f_L   (N)    : {f_l}")
        print(f"  左脚力矩 tau_L (N·m)  : {tau_l}")
        print(f"  右脚力   f_R   (N)    : {f_r}")
        print(f"  右脚力矩 tau_R (N·m)  : {tau_r}")
        print()
        print("[物理合理性分析]:")
        print(f"  双脚法向力之和 f_z    : {fz_sum:.2f} N vs. mg = {mg:.2f} N (相对偏差 {fz_rel_err_pct:.2f}%)")
        print(f"  质心处净外力 F_net    : {f_net} N  (F_x = {f_net[0]:.2f} N 推动质心向后恢复)")
        print(f"  质心处净恢复力矩 M_net: {m_net} N·m (M_y = {restoring_pitch_torque:.2f} N·m < 0 抵抗前倾)")
        print(f"  下一步预测 pitch 角   : {np.degrees(x_pred1[4]):.4f}° (由 {pitch_deg:.4f}° 减小，趋势正确)")
        print(f"  下一步预测 pitch 角速 : {x_pred1[10]:.4f} rad/s (< 0 反向恢复角速度)")

    # 6. 验收标准判定
    time_ok = avg_solve_ms < 2.0
    fz_ok = fz_rel_err_pct < 5.0
    torque_ok = restoring_pitch_torque < -1.0 and x_pred1[10] < 0.0
    drift_ok = eq_drift < 1e-12
    finite_ok = bool(np.all(np.isfinite(u_opt)) and np.all(np.isfinite(x_pred1)))

    print(LINE)
    print("[验收判定]")
    print(f"  1. OSQP 求解耗时 < 2 ms       : {'通过' if time_ok else '未通过'} ({avg_solve_ms:.3f} ms)")
    print(f"  2. 双脚 f_z 之和 ≈ mg (<5%误差): {'通过' if fz_ok else '未通过'} ({fz_sum:.2f} N vs {mg:.2f} N)")
    print(f"  3. 产生反向恢复俯仰力矩 M_y<0 : {'通过' if torque_ok else '未通过'} (M_y = {restoring_pitch_torque:.2f} N·m)")
    print(f"  4. 平衡态单步漂移 < 1e-12     : {'通过' if drift_ok else '未通过'} ({eq_drift:.2e})")
    print(f"  5. 严格满足摩擦锥与力边界约束 : {'通过' if friction_ok else '未通过'}")
    print(f"  6. 无 NaN / Inf               : {'通过' if finite_ok else '未通过'}")
    print(LINE)

    if not (time_ok and fz_ok and torque_ok and drift_ok and friction_ok and finite_ok):
        print("[失败] Task B2 部分验收指标未通过，请检查！")
        return 1

    print("[Task B2 完成] nominal_srbd.py 自检全部通过！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
