#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""state_vector.py -- Unitree G1 统一状态向量与足底接触雅可比提取接口（Task B1）

用途：
    为四条实验对比路线提供标准化、无副作用的状态与运动学特征提取函数：
    1. 全部路线统一的 12 维质心/浮动基核心状态：
       - extract_com_state(model, data) -> shape=(12,)
         提取 12 维核心状态：[p_com(3), rpy_base(3), v_com(3), omega_base(3)]
    2. 全身状态记录与诊断接口：
       - extract_full_state(model, data) -> shape=(71,)
         提取完整 71 维全身状态：[base_pos(3), base_quat(4), q_joints(29), base_linvel(3), base_angvel(3), dq_joints(29)]
    3. 下层全身控制（QP-WBC）与足底接触力映射：
       - extract_foot_positions(model, data) -> (p_L, p_R), shape=(3,)
         提取左右足底几何接触中心（4 个碰撞球中心均值下移球半径 0.005m，局部偏移 [0.035, 0.0, -0.035]m）的世界系坐标。
       - extract_foot_jacobians(model, data) -> (J_L, J_R), shape=(6, nv)
         调用 mujoco.mj_jac 在左右足底接触中心 p_L, p_R 处提取 6 维空间接触雅可比矩阵 [J_p; J_r]。

用法：
    直接运行进行 Task B1 自检与验收：
    F:\\conda_envs\\koopman_robot\\python.exe state_vector.py

日期：2026-11-15
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------- #
# 常量与默认配置
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "assets" / "g1" / "scene.xml"
KEYFRAME_NAME = "stand"

LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"

# 足底 4 个碰撞球 ([-0.05, ±0.025, -0.03], [0.12, ±0.03, -0.03], 半径 0.005m) 的几何接触中心
# 相对于 ankle_roll_link 原点的局部偏移量 (m)
FOOT_SOLE_LOCAL_OFFSET = np.array([0.035, 0.0, -0.035], dtype=np.float64)

COM_STATE_DIM = 12
FULL_STATE_DIM = 71

LINE = "=" * 72


# --------------------------------------------------------------------------- #
# 辅助物理属性查询函数
# --------------------------------------------------------------------------- #


def get_total_mass(model: mujoco.MjModel) -> float:
    """获取机器人全身总质量 m (kg)。

    注意：worldbody (body 0) 质量为 0，全身总质量等于所有刚体质量之和
    （也等于 model.body_subtreemass[1] 或 model.body_subtreemass[0]）。
    """
    return float(np.sum(model.body_mass))


def quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """将 MuJoCo 四元数 (w, x, y, z) 转换为欧拉角 RPY [roll(phi), pitch(theta), yaw(psi)] (rad)。

    采用标准外旋 xyz（等价于内旋 ZYX：R = R_z(psi) * R_y(theta) * R_x(phi)）。
    """
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return np.asarray(rpy, dtype=np.float64)


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """将 MuJoCo 四元数 (w, x, y, z) 转换为 3x3 旋转矩阵 R_wb（机体坐标系 -> 世界坐标系）。"""
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)


# --------------------------------------------------------------------------- #
# Task B1 核心接口 1：12 维质心状态提取（供路线二 / 路线三使用）
# --------------------------------------------------------------------------- #


def extract_com_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """提取 12 维质心与浮动基状态向量 x_com，返回 shape=(12,)。

    状态定义：
        x_com = [p_com(3), rpy_base(3), v_com(3), omega_base(3)]^T ∈ R^12
    其中：
        - p_com(3)      : 全身真实质心位置 (m)，取自 data.subtree_com[0]（世界系）
        - rpy_base(3)   : 浮动基姿态欧拉角 [phi, theta, psi]^T (rad)
        - v_com(3)      : 全身真实质心线速度 (m/s)，由质心雅可比 mj_jacSubtreeCom * qvel 计算（世界系）
        - omega_base(3) : 浮动基角速度 (rad/s)，由机体系角速度 data.qvel[3:6] 转换至世界系
    """
    # 1. 全身真实质心位置 p_com (3,)
    # data.subtree_com[0] 为以 worldbody 为根的整棵运动学树质心（即 G1 全身质心）
    p_com = np.array(data.subtree_com[0], dtype=np.float64).copy()

    # 2. 浮动基姿态四元数转欧拉角 Theta = [phi, theta, psi]^T (3,)
    base_quat = data.qpos[3:7]
    rpy_base = quat_wxyz_to_rpy(base_quat)

    # 3. 全身真实质心线速度 v_com = J_com * qvel (3,)
    jac_com = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSubtreeCom(model, data, jac_com, 0)
    v_com = jac_com @ data.qvel

    # 4. 浮动基角速度 omega_base (3,)
    # MuJoCo freejoint 的 qvel[3:6] 定义在局部基座（pelvis）坐标系下，左乘 R_wb 转到世界系
    r_wb = quat_wxyz_to_rotmat(base_quat)
    omega_base = r_wb @ data.qvel[3:6]

    x_com = np.concatenate([p_com, rpy_base, v_com, omega_base]).astype(np.float64)
    assert x_com.shape == (COM_STATE_DIM,), (
        f"extract_com_state 维度错误：期望 ({COM_STATE_DIM},)，实际 {x_com.shape}"
    )
    return x_com


# --------------------------------------------------------------------------- #
# Task B1 核心接口 2：71 维全身状态提取（全状态记录与诊断接口）
# --------------------------------------------------------------------------- #


def extract_full_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """提取完整 71 维全身状态向量 x_full，返回 shape=(71,)。

    状态定义（按 PLAN.md 规范）：
        [base_pos(3), base_quat(4), q_joints(29), base_linvel(3), base_angvel(3), dq_joints(29)]
    说明：
        对于 29 自由度 G1 模型（nq=36, nv=35）：
        - base_pos(3)    : data.qpos[0:3]
        - base_quat(4)   : data.qpos[3:7] (w, x, y, z)
        - q_joints(29)   : data.qpos[7:36]（全部 29 个受控关节角）
        - base_linvel(3) : data.qvel[0:3]
        - base_angvel(3) : data.qvel[3:6]
        - dq_joints(29)  : data.qvel[6:35]（全部 29 个受控关节角速度，总维度 36 + 35 = 71 维）
    """
    base_pos = np.asarray(data.qpos[0:3], dtype=np.float64)
    base_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    q_joints = np.asarray(data.qpos[7:], dtype=np.float64)

    base_linvel = np.asarray(data.qvel[0:3], dtype=np.float64)
    base_angvel = np.asarray(data.qvel[3:6], dtype=np.float64)
    dq_joints = np.asarray(data.qvel[6:], dtype=np.float64)

    x_full = np.concatenate(
        [base_pos, base_quat, q_joints, base_linvel, base_angvel, dq_joints]
    ).astype(np.float64)
    assert x_full.shape == (FULL_STATE_DIM,), (
        f"extract_full_state 维度错误：期望 ({FULL_STATE_DIM},)，实际 {x_full.shape}"
    )
    return x_full


# --------------------------------------------------------------------------- #
# Task B1 核心接口 3：左右足底接触中心坐标与 6 维空间接触雅可比矩阵提取
# --------------------------------------------------------------------------- #


def extract_foot_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_body_name: str = LEFT_FOOT_BODY_NAME,
    right_body_name: str = RIGHT_FOOT_BODY_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    """提取左右足底几何接触中心在世界坐标系下的位置向量 (p_L, p_R)，每个 shape=(3,)。

    说明：
        g1.xml 中的 <site name="left_foot"/> 未指定 pos 属性，默认位于 ankle_roll_link 原点
        （距地面高度约 0.0331m）。此处根据足底 4 个碰撞球中心均值下移球半径 0.005m，
        采用局部偏移 FOOT_SOLE_LOCAL_OFFSET = [0.035, 0.0, -0.035]m 计算真实足底接触中心。
    """
    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_body_name)
    right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_body_name)
    if left_body_id < 0 or right_body_id < 0:
        raise ValueError(f"找不到左右脚刚体: '{left_body_name}' 或 '{right_body_name}'")

    r_left = np.asarray(data.xmat[left_body_id], dtype=np.float64).reshape(3, 3)
    r_right = np.asarray(data.xmat[right_body_id], dtype=np.float64).reshape(3, 3)

    p_left = np.asarray(data.xpos[left_body_id], dtype=np.float64) + r_left @ FOOT_SOLE_LOCAL_OFFSET
    p_right = np.asarray(data.xpos[right_body_id], dtype=np.float64) + r_right @ FOOT_SOLE_LOCAL_OFFSET
    return p_left.copy(), p_right.copy()


def extract_foot_jacobians(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_body_name: str = LEFT_FOOT_BODY_NAME,
    right_body_name: str = RIGHT_FOOT_BODY_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    """调用 mujoco.mj_jac 在左右足底几何接触中心 p_L, p_R 处提取 6 维空间雅可比矩阵 J_L, J_R。

    参数：
        model           : MuJoCo 模型实例
        data            : MuJoCo 数据实例（需已执行 mj_forward 或 mj_step）
        left_body_name  : 左脚刚体名称，默认 'left_ankle_roll_link'
        right_body_name : 右脚刚体名称，默认 'right_ankle_roll_link'

    返回：
        (J_L, J_R) : tuple[np.ndarray, np.ndarray]
        每个矩阵形状为 (6, nv)，前 3 行为足底接触点线速度雅可比 J_p，后 3 行为角速度雅可比 J_r，
        与作用在足底接触中心 p_L, p_R 的 6 维广义接触力旋量 F = [f_x, f_y, f_z, tau_x, tau_y, tau_z]^T 严格对应。
    """
    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_body_name)
    if left_body_id < 0:
        raise ValueError(f"模型中找不到左脚刚体: '{left_body_name}'")

    right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_body_name)
    if right_body_id < 0:
        raise ValueError(f"模型中找不到右脚刚体: '{right_body_name}'")

    p_left, p_right = extract_foot_positions(
        model, data, left_body_name=left_body_name, right_body_name=right_body_name
    )
    nv = model.nv

    jacp_l = np.zeros((3, nv), dtype=np.float64)
    jacr_l = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp_l, jacr_l, p_left, left_body_id)
    j_left = np.vstack([jacp_l, jacr_l])

    jacp_r = np.zeros((3, nv), dtype=np.float64)
    jacr_r = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp_r, jacr_r, p_right, right_body_id)
    j_right = np.vstack([jacp_r, jacr_r])

    assert j_left.shape == (6, nv), f"J_L 形状异常: {j_left.shape}"
    assert j_right.shape == (6, nv), f"J_R 形状异常: {j_right.shape}"
    return j_left, j_right


# --------------------------------------------------------------------------- #
# 通用包装接口（兼容 mode="com_12" / "full_71" 与控制量提取）
# --------------------------------------------------------------------------- #


def extract_x(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mode: str = "com_12",
) -> np.ndarray:
    """根据指定模式提取状态向量 x_k。

    参数：
        mode : "com_12"（默认，12 维质心状态）或 "full_71"（71 维全身状态，兼容旧名 "full_59"）
    """
    if mode == "com_12":
        return extract_com_state(model, data)
    if mode in ("full_71", "full_59"):
        return extract_full_state(model, data)
    raise ValueError(f"不支持的状态提取模式 mode='{mode}'，可选 'com_12' 或 'full_71'")


def extract_u(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """提取当前控制输入向量 u = data.ctrl.copy()，返回 shape=(nu,)。"""
    u = np.array(data.ctrl, dtype=np.float64).copy()
    assert u.shape == (model.nu,), f"extract_u 维度错误：期望 ({model.nu},)，实际 {u.shape}"
    return u


# --------------------------------------------------------------------------- #
# 自检与验收主程序
# --------------------------------------------------------------------------- #


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not MODEL_PATH.exists():
        print(f"[错误] 找不到模型文件: {MODEL_PATH}")
        return 1

    # 1. 加载 G1 场景并初始化为 'stand' 关键帧
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # 2. 提取各项物理量与状态向量
    total_mass = get_total_mass(model)
    x_com = extract_com_state(model, data)
    x_full = extract_full_state(model, data)
    j_left, j_right = extract_foot_jacobians(model, data)
    p_left, p_right = extract_foot_positions(model, data)
    u_ctrl = extract_u(model, data)

    p_com = x_com[0:3]
    rpy_base = x_com[3:6]
    v_com = x_com[6:9]
    omega_base = x_com[9:12]

    # 3. 数值健康检查（无 NaN / Inf）
    all_arrays = [x_com, x_full, j_left, j_right, p_left, p_right, u_ctrl]
    has_nan = not all(np.all(np.isfinite(arr)) for arr in all_arrays)

    # 4. 打印验收报告
    print(LINE)
    print("[Task B1 自检报告: state_vector.py —— 统一状态与足底接触雅可比提取接口]")
    print(LINE)
    print(f"  模型文件              : {MODEL_PATH}")
    print(f"  模型维度              : nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")
    print(f"  初始化关键帧          : '{KEYFRAME_NAME}' (key_id={key_id})")
    print(f"  G1 全身总质量 m       : {total_mass:.4f} kg")
    print(f"  全身质心坐标 p_CoM    : [{p_com[0]:.4f}, {p_com[1]:.4f}, {p_com[2]:.4f}] m")
    print(f"  基座骨盆坐标 p_base   : [{data.qpos[0]:.4f}, {data.qpos[1]:.4f}, {data.qpos[2]:.4f}] m")
    print(f"  左足底接触中心 p_L    : [{p_left[0]:.4f}, {p_left[1]:.4f}, {p_left[2]:.4f}] m ({LEFT_FOOT_BODY_NAME})")
    print(f"  右足底接触中心 p_R    : [{p_right[0]:.4f}, {p_right[1]:.4f}, {p_right[2]:.4f}] m ({RIGHT_FOOT_BODY_NAME})")
    print(LINE)

    with np.printoptions(precision=4, suppress=True, linewidth=120):
        print("[1] 12 维质心状态向量 extract_com_state(model, data):")
        print(f"  shape                 : {x_com.shape}")
        print(f"  p_com (0:3) [m]       : {p_com}")
        print(f"  rpy_base (3:6) [rad]  : {rpy_base}")
        print(f"  v_com (6:9) [m/s]     : {v_com}")
        print(f"  omega_base (9:12)[r/s]: {omega_base}")
        print(f"  完整 12 维向量 x_com  : {x_com}")
        print()

        print("[2] 71 维全身状态向量 extract_full_state(model, data):")
        print(f"  shape                 : {x_full.shape}")
        print(f"  前 7 维 (浮动基位姿)  : {x_full[0:7]}")
        print(f"  关节角 q_joints (29维): {x_full[7:36]}")
        print(f"  速度部分 qvel (35维)  : {x_full[36:71]}")
        print()

        print("[3] 左右足底接触中心 6 维空间雅可比矩阵 extract_foot_jacobians(model, data):")
        print(f"  J_L shape             : {j_left.shape}  (范数 ||J_L||_F = {np.linalg.norm(j_left):.4f})")
        print(f"  J_R shape             : {j_right.shape}  (范数 ||J_R||_F = {np.linalg.norm(j_right):.4f})")

    print(LINE)
    print("[验收判定]")
    print(f"  总质量 m > 0          : {'通过' if total_mass > 0 else '未通过'} ({total_mass:.4f} kg)")
    print(f"  x_com 维度 == (12,)   : {'通过' if x_com.shape == (12,) else '未通过'} {x_com.shape}")
    print(f"  x_full 维度 == (71,)  : {'通过' if x_full.shape == (71,) else '未通过'} {x_full.shape}")
    print(f"  J_L, J_R 维度==(6,nv) : {'通过' if j_left.shape == (6, model.nv) and j_right.shape == (6, model.nv) else '未通过'} ({j_left.shape}, {j_right.shape})")
    print(f"  无 NaN / Inf          : {'通过' if not has_nan else '未通过'}")
    print(LINE)

    if has_nan:
        print("[失败] 存在 NaN 或 Inf！")
        return 1

    print("[Task B1 完成] state_vector.py 自检全部通过！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
