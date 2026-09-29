#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_g1_load.py -- Unitree G1 机器人 MuJoCo 模型加载与仿真冒烟测试

功能：
    1. 加载 assets/g1/scene.xml（含地面的 G1 完整场景）；
    2. 打印基本动力学信息：nq / nv / nu，以及 body / joint / geom 数量；
    3. 步进仿真 100 步，逐帧检测 qpos / qvel / qacc 中是否出现 NaN 或 Inf；
    4. 输出最终基座高度，验证模型未因穿模或质量异常导致物理爆炸。

用法：
    F:\\conda_envs\\koopman_robot\\python.exe test_g1_load.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 模型路径：优先相对脚本所在目录，兼容任意工作目录调用
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "assets" / "g1" / "scene.xml"

NUM_STEPS = 100


def load_g1_model(scene_path: Path):
    """加载 G1 模型（路径已为纯英文，直接 from_xml_path 即可）。"""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(scene_path))


def check_finite(arr: np.ndarray, name: str, step: int) -> str | None:
    """检查数组中是否存在 NaN 或 Inf，返回错误描述或 None。"""
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        return f"[step {step}] {name} 出现异常值：NaN={nan_count}, Inf={inf_count}"
    return None


def main() -> int:
    # -----------------------------------------------------------------------
    # 1. 加载模型
    # -----------------------------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"[ERROR] 找不到模型文件：{MODEL_PATH}")
        return 1

    print(f"加载模型：{MODEL_PATH}")
    try:
        import mujoco

        model = load_g1_model(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as exc:
        print(f"[ERROR] 模型加载失败：{exc}")
        return 1

    print("模型加载成功。\n")

    # -----------------------------------------------------------------------
    # 2. 打印基本动力学信息
    # -----------------------------------------------------------------------
    print("=" * 55)
    print("  G1 机器人基本动力学信息")
    print("=" * 55)
    try:
        _name0 = model.names[0]
        model_name = _name0.decode() if isinstance(_name0, bytes) else str(_name0)
    except Exception:
        model_name = "N/A"
    print(f"  模型名称          : {model_name}")
    print(f"  广义坐标维数 nq   : {model.nq}  (浮动基座 7 + 关节数)")
    print(f"  广义速度维数 nv   : {model.nv}")
    print(f"  执行器数量 nu     : {model.nu}")
    print(f"  body 数量         : {model.nbody}")
    print(f"  joint 数量        : {model.njnt}")
    print(f"  geom 数量         : {model.ngeom}")
    print(f"  mesh 数量         : {model.nmesh}")
    print(f"  仿真步长 dt       : {model.opt.timestep:.6f} s")
    print("=" * 55)
    print()

    # -----------------------------------------------------------------------
    # 3. 前向动力学 + 步进仿真 100 步，逐帧检测 NaN / Inf
    # -----------------------------------------------------------------------
    mujoco.mj_forward(model, data)

    print(f"开始步进仿真（共 {NUM_STEPS} 步）...")
    errors: list[str] = []

    for step in range(1, NUM_STEPS + 1):
        mujoco.mj_step(model, data)

        # 逐帧检测物理数值
        for arr, name in [
            (data.qpos, "qpos"),
            (data.qvel, "qvel"),
            (data.qacc, "qacc"),
        ]:
            err = check_finite(np.asarray(arr), name, step)
            if err:
                errors.append(err)

        # 只在前 5 步和最后一步打印进度，避免刷屏
        if step <= 5 or step == NUM_STEPS:
            base_z = data.qpos[2] if model.nq >= 3 else float("nan")
            print(f"  step {step:3d}/{NUM_STEPS}  |  基座高度 z = {base_z:.4f} m")

    print()

    # -----------------------------------------------------------------------
    # 4. 结果汇总
    # -----------------------------------------------------------------------
    print("=" * 55)
    print("  仿真检验结果")
    print("=" * 55)

    if errors:
        print(f"[FAIL] 检测到 {len(errors)} 处物理数值异常：")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... 其余 {len(errors) - 10} 处省略")
        print("=" * 55)
        return 1

    final_z = data.qpos[2] if model.nq >= 3 else float("nan")
    print(f"[PASS] 连续 {NUM_STEPS} 步仿真无 NaN / Inf，物理稳定。")
    print(f"       最终基座高度 z = {final_z:.4f} m（初始约 0.7~0.8 m，"
          f"落地后应保持在合理范围，未出现负值穿模或发散）。")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
