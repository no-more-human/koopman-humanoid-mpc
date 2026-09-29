# 人形机器人黑盒/未建模动力学的数据驱动库普曼预测控制与 Sim-to-Real 研究规划报告

- **规划周期**：2026年10月 —— 2027年2月  
- **目标愿景**：面向 2027 Fall 美国名校留学申请，产出一篇完整、具创新性且有高保真仿真/真机验证的控制工程高水平论文（目标投稿：IEEE RA-L / ICRA / ACC / IROS）。

---

## 1\. 课题定位与核心创新点提炼

### 1.1 课题背景与术语界定

针对多自由度人形机器人（如 Unitree G1）在复杂动态行走时难以建立精确显式解析动力学模型（拉格朗日方程复杂、参数不准、地面冲击/接触存在非光滑切换）的难题，本研究采用\*\*纯数据驱动的库普曼算子（Koopman Operator）\*\*方法。

- **非限制性/黑盒未建模动态（Black-box / Unmodeled Dynamics）**：不依赖对机械臂/腿部各连杆质量、惯量矩阵、非线性摩擦的机理推导，完全从传感器观测数据与控制输入中提取动力学特征。

### 1.2 创新点设计（三层递进架构）

1. **轻量名义先验 \+ 库普曼残差线性化（Physics-Informed Residual Koopman）**：  
   - 彻底避免纯黑盒端到端学习在接触冲击下的发散问题。以低自由度几何/简化倒立摆（LIP/SRB）为名义基础，将未建模的腿部惯量、非线性摩擦、执行器迟滞建模为残差项 \$\\Delta f(x, u)\$。  
   - 使用 Koopman 提升函数（Lifting Functions）将非线性残差映射至高维线性可观测空间，构建全局线性增广预测模型。  
2. **极速求解的凸优化预测控制（Koopman-based Linear MPC）**：  
   - 将非线性高维整身控制转化为标准的二次规划（QP）问题，保证控制频率 \$\\ge 50 \\text{ Hz}\$，大幅领先传统非线性 NMPC（通常需十几到数十毫秒）。  
3. **面向 Sim-to-Real 的增量自适应机制（Adaptive/Incremental Learning）**：  
   - 引入在线/微批次递归更新（Recursive EDMD），在地面刚度突变、外力推扰、负载变化时快速修正 Koopman 矩阵，解决域漂移（Domain Shift）。

---

## 2\. 必读核心参考文献清单

| 类别 | 推荐必读书目 | 核心学习内容与参考价值 |
| :---- | :---- | :---- |
| **最直接参考** | Feihan Li et al. (CMU, 2025\) *Continual Learning and Lifting of Koopman Dynamics for Linear Control of Legged Robots* (L4DC 2025\) | 重点学习其状态向量构建、损失函数设计（预测误差+重构误差加权）、隐空间增量升维策略。参考其代码开源架构。 |
| **数据驱动与机器人** | Randall T. Fawcett et al. (Caltech & VT, 2023\) *Distributed Data-Driven Predictive Control for Multi-Agent Collaborative Legged Locomotion* (ICRA 2023\) | 学习高层数据驱动预测控制与底层 QP-WBC 控制器的分层架构，以及 Sim-to-Real 的扰动注入实验设计。 |
| **课题组理论依托** | Jialei Cai, Dong Liang et al. *Data-driven Cooperative Output Regulation of Singular Linear Multi-agent Systems* (2025/2026) | 学习导师团队在基于 Willems 基本引理与数据驱动 LMI 求解控制增益的数学推导规范，便于论文理论部分的撰写与请教。 |
| **经典控制理论** | Haojie Shi & Max Meng (2022) *Deep Koopman Operator with Control for Nonlinear Systems* (IEEE RA-L) | 学习带有外部控制输入 \$u\$ 的深度库普曼网络结构（Deep Koopman with Control）。 |

---

## 3\. 前置知识与技能储备清单

### 3.1 数学与控制理论

1. **Koopman Operator 基础**：  
   - 掌握动态模态分解（DMD）和扩展动态模态分解（EDMD）。  
   - 理解可观测函数（Observables）与提升函数（Lifting Function，如径向基 RBF、多项式、全连接神经网络）的原理。  
2. **模型预测控制（MPC）**：  
   - 掌握离散状态空间方程的预测时域展开（Prediction Horizon \$N\$）。  
   - 将带约束状态跟踪表述为标准二次规划（Quadratic Program, QP）：\$\\min \\frac{1}{2} U^T H U \+ g^T U \\quad \\text{s.t.} \\quad C U \\le b\$。

### 3.2 软件与工程工具

1. **仿真**：`MuJoCo`（Python 绑定）或 `IsaacLab`，掌握 URDF/MJCF 机器人模型加载与传感器数据读取。  
2. **优化求解器**：Python `cvxpy`、`osqp`、`qpsolvers`。  
3. **数据处理与学习**：`PyTorch`（构建轻量提升网络）、`NumPy`、`SciPy`。

---

## 4\. 全流程四阶段执行路线

\[Phase 1: 仿真与基线\] ──\> \[Phase 2: 数据与Koopman拟合\] ──\> \[Phase 3: 闭环MPC与消融\] ──\> \[Phase 4: 写作与投稿\]

### Phase 1：环境极速就位与名义控制器搭建（第 1 \~ 3 周）

* **任务 1**：在本地搭建 Python 3.10 \+ MuJoCo 仿真环境，导入 Unitree G1（或 Go2）官方模型。  
* **任务 2**：跑通机器人的关节级阻抗/PD控制，能使机器人在悬挂或支撑状态下保持预定姿势。  
* **任务 3**：搭建最基础的名义跟踪控制器（例如基于逆运动学或基础 QP 分配足端力）。

### Phase 2：数据采集管线与 Koopman 算子拟合（第 4 \~ 6 周）

* **任务 1（数据采集）**：让机器人在仿真中做多种步态运动（Walk/Squat/Trot），在控制指令中注入伪随机扰动（PRBS）以满足**持续激励条件（Persistence of Excitation, PE）**，采集 5,000 \~ 20,000 帧状态转移对 \$\\mathcal{D} \= {(x\_k, u\_k, x\_{k+1})}\$。  
* **任务 2（EDMD 算子求解）**：  
  - 提取状态 \$x \= \[p\_{base}, \\dot{p}*{base}, \\theta*{base}, \\omega\_{base}, q\_{joints}, \\dot{q}\_{joints}\]^T\$；  
  - 编写 EDMD 求解脚本，利用正规方程（Normal Equation）或 SVD 求解矩阵 \$(A, B)\$： \$\$z\_{k+1} \= A z\_k \+ B u\_k, \\quad z\_k \= \\phi(x\_k)\$\$  
* **任务 3（预测精度评估）**：绘制 1\~15 步的预测误差折线图（\$k\$-step prediction error），对比线性自回归（ARX）与名义物理模型。

### Phase 3：闭环预测控制与鲁棒性消融实验（第 7 \~ 10 周）

* **任务 1（Koopman-MPC 闭环）**：  
  - 将拟合好的高维线性模型导入 OSQP 求解器，构建有限时域 MPC，控制周期控制在 \$10 \\sim 20\\text{ ms}\$。  
  - 在 MuJoCo 中闭环运行，实现机器人的稳态平衡与速度指令跟踪。  
* **任务 2（鲁棒性与 Sim-to-Real 压力测试）**：  
  - **参数摄动**：躯干质量增加 20%\~40%，关节摩擦/阻尼增加 50%；  
  - **环境扰动**：地面摩擦系数突变（如 0.8 骤降至 0.2，模拟结冰/打滑）；  
  - **外力推扰**：施加持续 0.2s 的脉冲冲击力（100N\~200N）。  
* **任务 3（消融对比基线）**：与名义线性 MPC、纯非线性 NMPC、传统强化学习（如 PPO）对比**跟踪误差（ISE/RMSE）、求解延迟（Latency）、能耗（Control Effort）与抗跌倒失稳率**。

### Phase 4：论文撰写、真机验证与投稿（第 11 \~ 14 周）

* **任务 1（实验图表与录像）**：输出符合 IEEE 顶会标准的矢量图（学习曲线、相平面图、相轨迹对比、推扰响应曲线）；录制仿真实验演示视频。  
* **任务 2（真机录制 \- 若有条件）**：去公司借调 Unitree G1 或四足机器人，测试实机步态跟踪与外力推扰，录制实物视频（大幅加分项）。  
* **任务 3（论文写作与打磨）**：使用 IEEE 双栏 LaTeX 模板撰写，重点打磨 Abstract、Introduction 与 Problem Formulation，邀请梁栋老师审定批改并定稿。

---

## 5\. 倒排时间表（2026.10 \- 2027.02）

| 时间区间 | 核心产出目标 | 检视要点 |
| :---- | :---- | :---- |
| **2026.10.01 \- 2026.10.20** | 跑通 MuJoCo G1/Go2 仿真与控制接口，确定状态空间向量维度 | 能否稳定读取传感器、下发力矩？ |
| **2026.10.21 \- 2026.11.15** | 完成数据采集管线，写出 EDMD / Deep-Koopman 模型并完成预测精度收敛 | 5步预测误差是否显著低于名义模型？ |
| **2026.11.16 \- 2026.12.15** | 完成 Koopman-MPC 闭环控制，完成所有扰动抗性实验与消融对比表格 | MPC 单次求解耗时是否在 5ms 内？ |
| **2026.12.16 \- 2027.01.15** | 汇总所有数据、绘制高质量图表、完成论文初稿全英文撰写 | 论文引言逻辑、公式推导是否严谨完整？ |
| **2027.01.16 \- 2027.02.15** | 导师审阅修改、完善真机/演示视频、完成投稿与挂 ArXiv 预印本 | 确保申请季网申时已有成型论文与链接支撑。 |

---

## 6\. 关键风险与应对策略

1. **若借不到 G1 人形真机**：  
   - **预案**：立即切换到四足机器人（Unitree Go1/Go2）或机械臂平台，**算法结构百分之百通用**，真机验证一样成立；若真机皆不可得，则把 MuJoCo 与 IsaacGym/IsaacLab 做交叉验证（Sim-to-Sim 迁移），配合严苛的外力突变，学术认可度同样充分。  
2. **全维度状态升维后维数过大导致 MPC 求解慢**：  
   - **预案**：采用双层控制架构——上层基于质心动力学（CoM/浮动基）提取 12 维关键状态做 Koopman 升维与 MPC 规划足端力；下层用解析 QP-WBC 分配关节力矩，兼顾全局线性与极速求解。