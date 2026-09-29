# Koopman-Humanoid-MPC

> Data-driven predictive control for humanoid robots using **Koopman operator theory** and **convex MPC**, developed and validated in **MuJoCo** on the **Unitree G1**.

This is a work-in-progress research project (2026.10 – 2027.02). The goal is to learn a *globally linear* model of a legged/humanoid robot directly from data and to solve whole-body control as a fast quadratic program instead of a nonlinear NMPC.

## Highlights

- **Physics-informed residual Koopman** — a low-order nominal prior (LIP / SRB) plus Koopman lifting of the unmodeled residual ($\Delta f(x,u)$), which avoids the divergence of purely black-box models under contact impacts.
- **Koopman-based linear MPC** — the nonlinear whole-body problem is recast as a standard QP, targeting a control rate $\ge 50\ \text{Hz}$ and a solver latency $\le 5\ \text{ms}$.
- **Adaptive / incremental learning** — recursive EDMD updates that correct the Koopman matrices when ground stiffness, external pushes or payload change (sim-to-real domain shift).

## Repository layout

| Path | Description |
| --- | --- |
| `verify_env.py` | Environment self-check: reports interpreter/OS, imports every core dependency and runs a functional smoke test for each (MuJoCo, SciPy, Matplotlib, PyTorch, CVXPY + OSQP, qpsolvers). |
| `test_g1_load.py` | Loads `assets/g1/scene.xml`, prints the model dimensions (`nq`/`nv`/`nu`, bodies, joints, geoms) and steps the simulation 100 frames while checking for `NaN` / `Inf`. |
| `run_verify.bat` | One-click Windows runner for `verify_env.py`; writes `verify_env.log` and removes the temporary files produced during setup. |
| `assets/g1/` | Unitree G1 MJCF model (29 DoF) — XML description plus STL meshes. |
| `RESEARCH_PLAN_Koopman_Humanoid_Robot.md` | Full research plan: background, literature, prerequisites, four-phase roadmap, schedule and risk mitigation. |
| `verify_env.log` | Captured output of the latest successful environment check. |

## Environment

Built and verified with the conda environment `koopman_robot` (Python 3.10):

| Package | Version | Role |
| --- | --- | --- |
| numpy | 2.2.6 | numerical computing |
| scipy | 1.15.3 | linear algebra / ODE integration |
| matplotlib | 3.10.9 | plotting and visualisation |
| torch | 2.14.0+cpu | deep learning / automatic differentiation |
| mujoco | 3.14.0 | physics simulation |
| cvxpy | 1.7.5 | convex optimisation modelling (DCP) |
| osqp | 1.1.3 | QP solver (ADMM) |
| qpsolvers | 4.13.0 | unified QP solver interface |

## Quick start

```bash
conda activate koopman_robot

# 1) Full environment check (imports + functional smoke tests)
python verify_env.py      # or simply: run_verify.bat

# 2) G1 model smoke test (load MJCF, step 100 frames, check for NaN/Inf)
python test_g1_load.py
```

Both scripts exit with code `0` on success and `1` on failure, so they can be wired into CI as-is.

## Roadmap

| Phase | Weeks | Objective |
| --- | --- | --- |
| 1 — Simulation & nominal baseline | 1 – 3 | MuJoCo + G1 pipeline, joint-level PD/impedance control, nominal QP foot-force allocation. |
| 2 — Data pipeline & Koopman fitting | 4 – 6 | PRBS excitation satisfying persistence of excitation, 5k – 20k transition pairs, EDMD / Deep-Koopman identification, $k$-step prediction error. |
| 3 — Closed-loop MPC & ablation | 7 – 10 | Koopman-MPC in OSQP at 10 – 20 ms, robustness tests (mass/friction variation, 100 – 200 N pushes) and comparison against nominal MPC, NMPC and PPO. |
| 4 — Writing & submission | 11 – 14 | IEEE-style figures and video, hardware validation if available, RA-L / ICRA / IROS submission. |

See `RESEARCH_PLAN_Koopman_Humanoid_Robot.md` for the complete plan (in Chinese).

## Third-party assets

`assets/g1/` is the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) description of the [Unitree G1](https://www.unitree.com/g1/), released under the **BSD-3-Clause** licence — see `assets/g1/LICENSE` and `assets/g1/README.md`. The full Menagerie checkout used during development is intentionally *not* tracked in this repository (`_temp_menagerie/`, ignored by `.gitignore`).

## License

Released under the MIT License — see `LICENSE`.
