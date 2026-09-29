#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_env.py -- 人形机器人数据驱动控制开发环境自检脚本

项目：人形机器人数据驱动控制（MuJoCo 仿真 + Koopman 算子 / 凸优化预测控制）
环境：conda env "koopman_robot" (Python 3.10)
      路径 F:\\conda_envs\\koopman_robot

用法：
    conda activate koopman_robot
    python verify_env.py

    或直接指定解释器：
    F:\\conda_envs\\koopman_robot\\python.exe verify_env.py

功能：
    1. 打印运行环境信息（解释器、Python 版本、平台、是否为目标 conda 环境）；
    2. 导入并打印全部核心依赖库的版本号；
    3. 对每个库执行功能性冒烟测试（不只看"能不能 import"）；
    4. 汇总结果，并以退出码反映成功(0) / 失败(1)。
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import traceback
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

ENV_NAME = "koopman_robot"

# 需要检查的核心依赖: (import 名, PyPI 发行包名, 用途说明)
REQUIRED_PACKAGES = [
    ("numpy", "numpy", "科学计算基础"),
    ("scipy", "scipy", "科学计算 / 线性代数 / ODE"),
    ("matplotlib", "matplotlib", "绘图与可视化"),
    ("torch", "torch", "深度学习 / 自动微分"),
    ("mujoco", "mujoco", "机器人物理仿真引擎"),
    ("cvxpy", "cvxpy", "凸优化建模（DCP）"),
    ("osqp", "osqp", "二次规划求解器（ADMM）"),
    ("qpsolvers", "qpsolvers", "QP 求解器统一接口"),
]

LINE = "=" * 72


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


def banner(title: str) -> None:
    """打印分节标题。"""
    print()
    print(LINE)
    print(title)
    print(LINE)


def get_version(module, dist_name: str) -> str:
    """优先取模块的 __version__，取不到则用 importlib.metadata 兜底。"""
    ver = getattr(module, "__version__", None)
    if ver:
        return str(ver)
    try:
        return pkg_version(dist_name)
    except PackageNotFoundError:
        return "unknown"


def describe_os() -> str:
    """给出更准确的系统名称。

    Python 的 platform 模块在 Windows 11 上依旧只报告 "Windows 10"，
    这里改用 NT 内部版本号（Win11 build >= 22000）来区分。
    """
    if platform.system() == "Windows":
        try:
            _release, version, _csd, _ptype = platform.win32_ver()
            build = int(version.split(".")[-1])
            name = "Windows 11" if build >= 22000 else "Windows 10"
            return f"{name} (build {version})"
        except Exception:
            return f"Windows {platform.release()}"
    return platform.platform()


def print_platform_info() -> None:
    """打印运行环境信息，确认跑在目标 conda 环境里。"""
    banner("[1/4] 运行环境信息")
    prefix = sys.prefix.replace("\\", "/")
    in_target_env = ENV_NAME in prefix.lower()
    # conda 环境没有 pyvenv.cfg，所以 sys.prefix == sys.base_prefix，
    # 用 conda-meta 目录来判断是否为 conda 环境更可靠。
    is_conda_env = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))

    print(f"操作系统        : {describe_os()} ({platform.machine()})")
    print(f"Python 版本     : {sys.version.split()[0]}")
    print(f"解释器路径      : {sys.executable}")
    print(f"环境根目录      : {sys.prefix}")
    print(f"是否为 conda 环境 : {'是 (存在 conda-meta)' if is_conda_env else '否'}")
    print(f"是否 {ENV_NAME} 环境 : {'是' if in_target_env else '否  << 注意：当前未使用目标环境的解释器'}")


def check_imports():
    """逐个导入核心依赖并打印版本。返回 (模块字典, 是否全部成功)。"""
    banner("[2/4] 核心依赖导入与版本")
    modules = {}
    all_ok = True
    width = max(len(dist) for _, dist, _ in REQUIRED_PACKAGES)

    for mod_name, dist_name, purpose in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - 自检脚本需要兜住一切异常
            all_ok = False
            print(f"[FAIL] {dist_name:<{width}} : 导入失败 -> {type(exc).__name__}: {exc}")
            continue
        modules[mod_name] = module
        version = get_version(module, dist_name)
        print(f"[ OK ] {dist_name:<{width}} : {version:<14} ({purpose})")

    imported = sum(1 for mod_name, _, _ in REQUIRED_PACKAGES if mod_name in modules)
    print()
    print(f"共需检查 {len(REQUIRED_PACKAGES)} 个依赖，成功导入 {imported} 个。")
    return modules, all_ok


# --------------------------------------------------------------------------- #
# 冒烟测试：不只是"能 import"，而是要真的算出一个结果
# --------------------------------------------------------------------------- #

# 一个最小可用的单摆模型：水平起步，靠重力自然摆动
MUJOCO_SMOKE_XML = """
<mujoco model="pendulum_smoke_test">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" density="1000"/>
      <geom name="bob" type="sphere" pos="0.5 0 0" size="0.06" density="5000"/>
    </body>
  </worldbody>
</mujoco>
"""

# 标准 QP 问题（cvxpy 与 qpsolvers 共用，用于交叉验证）
QP_P = None  # 在 _qp_problem() 中初始化，避免重复构造
QP_Q = None
QP_REFERENCE = {}  # 存放 cvxpy 求得的解，供 qpsolvers 测试交叉验证


def _qp_problem(mods):
    """构造一个带箱型约束的 QP：min 0.5*x'Px + q'x, s.t. lb <= x <= ub。

    解析最优解为 x* = [-0.5, 0.5]（两个约束均起作用）。
    """
    global QP_P, QP_Q
    np = mods["numpy"]
    if QP_P is None:
        QP_P = np.array([[2.0, 0.5], [0.5, 1.0]])
        QP_Q = np.array([1.0, -1.0])
    lb = -0.5 * np.ones(2)
    ub = 0.5 * np.ones(2)
    return QP_P, QP_Q, lb, ub


def test_mujoco(mods):
    """MuJoCo：加载模型 -> 建 MjData -> 前向动力学 -> 步进仿真。"""
    mujoco = mods["mujoco"]
    model = mujoco.MjModel.from_xml_string(MUJOCO_SMOKE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q_start = float(data.qpos[0])

    for _ in range(500):  # dt=0.002 -> 仿真 1.0 s
        mujoco.mj_step(model, data)
    q_end = float(data.qpos[0])

    if abs(q_end - q_start) < 1e-6:
        raise RuntimeError("仿真 500 步后关节角没有变化，物理引擎可能未生效")
    return (
        f"单摆模型加载成功 (nq={model.nq}, nv={model.nv}, nbody={model.nbody})；"
        f"仿真 500 步后 qpos 由 {q_start:+.4f} 摆动到 {q_end:+.4f} rad"
    )


def test_scipy(mods):
    """SciPy：矩阵指数 + ODE 积分（人形机器人常用的两种数值工具）。"""
    np, scipy = mods["numpy"], mods["scipy"]
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    # expm(A * pi/2) 应为 90 度旋转矩阵 [[0,1],[-1,0]]
    rot = expm(np.array([[0.0, 1.0], [-1.0, 0.0]]) * np.pi / 2.0)
    err_expm = float(np.max(np.abs(rot - np.array([[0.0, 1.0], [-1.0, 0.0]]))))

    # 简谐振子 x'' = -x，积分一个完整周期后应回到初值
    sol = solve_ivp(
        lambda t, y: np.array([y[1], -y[0]]),
        [0.0, 2.0 * np.pi],
        [1.0, 0.0],
        rtol=1e-10,
        atol=1e-12,
    )
    err_ivp = float(np.max(np.abs(sol.y[:, -1] - np.array([1.0, 0.0]))))

    if err_expm > 1e-12 or err_ivp > 1e-6:
        raise RuntimeError(f"数值精度不达标 (expm={err_expm:.2e}, solve_ivp={err_ivp:.2e})")
    return f"expm(90deg 旋转) 误差={err_expm:.2e}；solve_ivp 简谐振子一周期回归误差={err_ivp:.2e}"


def test_matplotlib(mods):
    """Matplotlib：用无界面 Agg 后端完成一次真实渲染。"""
    np, matplotlib = mods["numpy"], mods["matplotlib"]
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.linspace(0.0, 2.0 * np.pi, 256)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(t, np.sin(t), label="sin(t)")
    ax.plot(t, np.cos(t), label="cos(t)")
    ax.set_title("matplotlib smoke test")
    ax.legend()
    fig.canvas.draw()  # 真正触发一次渲染
    nbytes = len(fig.canvas.buffer_rgba())
    plt.close(fig)
    return f"Agg 后端渲染成功，画布 RGBA 缓冲 {nbytes} 字节，当前后端={matplotlib.get_backend()}"


def test_torch(mods):
    """PyTorch：张量运算 + 自动微分 + CUDA 可用性。"""
    torch = mods["torch"]
    a = torch.randn(4, 5)
    b = torch.randn(5, 3)
    c = a @ b
    if tuple(c.shape) != (4, 3):
        raise RuntimeError(f"矩阵乘法结果维度异常: {tuple(c.shape)}")

    x = torch.tensor([2.0], requires_grad=True)
    y = (x ** 3).sum()
    y.backward()
    if abs(float(x.grad) - 12.0) > 1e-5:
        raise RuntimeError(f"自动微分结果异常: dy/dx={float(x.grad)}，期望 12.0")

    return (
        f"(4x5)@(5x3) -> {tuple(c.shape)}；反传 d(x^3)/dx={float(x.grad):.4f} (期望 12)；"
        f"线程数={torch.get_num_threads()}；CUDA 可用={torch.cuda.is_available()}"
    )


def test_cvxpy_qp(mods):
    """CVXPY + OSQP：求解带箱型约束的 QP，并与解析解比对。"""
    np, cp = mods["numpy"], mods["cvxpy"]
    P, q, lb, ub = _qp_problem(mods)

    x = cp.Variable(2, name="x")
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + cp.sum(cp.multiply(q, x)))
    problem = cp.Problem(objective, [x >= lb, x <= ub])
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"求解状态异常: {problem.status}")
    if x.value is None:
        raise RuntimeError("求解器未返回最优解 (x.value is None)")

    sol = np.asarray(x.value).ravel()
    QP_REFERENCE["x_cvxpy"] = sol  # 供 qpsolvers 测试交叉验证

    expected = np.array([-0.5, 0.5])
    if not np.allclose(sol, expected, atol=1e-4):
        raise RuntimeError(f"最优解 {sol} 与解析解 {expected} 不一致")
    return (
        f"状态={problem.status}，最优解 x={np.array2string(sol, precision=6)}"
        f"（解析解 [-0.5, 0.5]），目标值={problem.value:.6f}"
    )


def test_qpsolvers(mods):
    """qpsolvers：用统一接口求解同一个 QP，并与 cvxpy 结果交叉验证。"""
    np, qpsolvers = mods["numpy"], mods["qpsolvers"]
    P, q, lb, ub = _qp_problem(mods)

    used_solver = "osqp"
    try:
        # qpsolvers 直接调用 OSQP 时用的是 OSQP 默认精度（eps ~1e-3），
        # 这里把精度收紧到 1e-8，才能和 cvxpy 的结果做严格交叉验证。
        sol = qpsolvers.solve_qp(P, q, lb=lb, ub=ub, solver="osqp",
                                 eps_abs=1e-8, eps_rel=1e-8)
    except Exception:
        used_solver = "auto"
        sol = qpsolvers.solve_qp(P, q, lb=lb, ub=ub)

    if sol is None:
        raise RuntimeError("qpsolvers 未找到可行解 (返回 None)")

    sol = np.asarray(sol).ravel()
    reference = QP_REFERENCE.get("x_cvxpy")
    if reference is None:
        reference = np.array([-0.5, 0.5])
        ref_note = "解析解 [-0.5, 0.5]"
    else:
        ref_note = "cvxpy 结果"

    # OSQP 是一阶(ADMM)求解器，即使收紧精度也保留 1e-3 的工程容差
    if not np.allclose(sol, reference, atol=1e-3):
        raise RuntimeError(f"解 {sol} 与{ref_note} {reference} 不一致")

    backends = getattr(qpsolvers, "available_solvers", None)
    backends = "[" + ", ".join(backends) + "]" if backends else "未知"
    deviation = float(np.max(np.abs(sol - reference)))
    return (
        f"使用求解器={used_solver}，解 x={np.array2string(sol, precision=8)}，"
        f"与{ref_note}的最大偏差={deviation:.2e}；qpsolvers 可用后端={backends}"
    )


# --------------------------------------------------------------------------- #
# 测试调度与汇总
# --------------------------------------------------------------------------- #

SMOKE_TESTS = [
    ("MuJoCo 物理仿真", test_mujoco, "mujoco"),
    ("SciPy 数值计算", test_scipy, "scipy"),
    ("Matplotlib 绘图", test_matplotlib, "matplotlib"),
    ("PyTorch 张量计算", test_torch, "torch"),
    ("CVXPY + OSQP 凸优化", test_cvxpy_qp, "cvxpy"),
    ("qpsolvers 二次规划", test_qpsolvers, "qpsolvers"),
]


def run_test(name, func, mods) -> bool:
    """执行单个冒烟测试并捕获所有异常，保证脚本不会中途崩掉。"""
    try:
        detail = func(mods)
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要兜住一切异常
        print(f"[FAIL] {name}")
        print(f"       原因: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3, file=sys.stdout)
        return False
    print(f"[PASS] {name}")
    print(f"       {detail}")
    return True


def main() -> int:
    # 统一用 UTF-8 输出，避免 Windows 控制台/重定向时的编码问题
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(LINE)
    print("人形机器人数据驱动控制 -- 开发环境自检")
    print("MuJoCo 仿真  +  Koopman 算子 / 凸优化预测控制")
    print(LINE)

    print_platform_info()
    modules, import_ok = check_imports()

    banner("[3/4] 功能性冒烟测试")
    results = {}
    for name, func, required in SMOKE_TESTS:
        if required not in modules:
            results[name] = False
            print(f"[SKIP] {name}  <-- 依赖 {required} 未成功导入")
            continue
        results[name] = run_test(name, func, modules)

    banner("[4/4] 验证汇总")
    for name, _, _ in SMOKE_TESTS:
        mark = "[PASS]" if results.get(name) else "[FAIL]"
        print(f"  {mark}  {name}")

    passed = sum(1 for value in results.values() if value)
    imported = sum(1 for mod_name, _, _ in REQUIRED_PACKAGES if mod_name in modules)
    print()
    print(f"依赖导入情况: {imported}/{len(REQUIRED_PACKAGES)} 个依赖导入成功")
    print(f"冒烟测试情况: {passed}/{len(SMOKE_TESTS)} 项测试通过")
    print()

    if import_ok and passed == len(SMOKE_TESTS):
        print(">>> 环境验证成功：全部依赖可用，MuJoCo 仿真与凸优化 QP 求解均已跑通。")
        print(">>> 现在可以开始人形机器人数据驱动控制（Koopman 算子 + 凸优化预测控制）的开发。")
        return 0

    print(">>> 环境验证失败：请根据上面的 [FAIL] / [SKIP] 信息补齐依赖或修复问题。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

