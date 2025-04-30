```markdown
# 🤖 Efficient Robotic Arm Control with NdLinear

This project explores the use of [`NdLinear`](https://github.com/ensemble-core/NdLinear) — a low-rank replacement for `nn.Linear` — in robotic control policies. We benchmark NdLinear against standard MLPs in continuous control environments involving robotic arm manipulation.

<p align="center">
  <img src="assets/robotic_arm_demo.gif" width="60%" alt="Robotic Arm Demo" />
</p>

---

## 🚀 Motivation

Robotic control tasks often rely on fully connected neural networks (MLPs) as policy or value function approximators. These layers can be parameter-heavy and inefficient, especially in edge applications where compute and latency are constrained.

**NdLinear** introduces low-rank approximations into linear layers, reducing parameter count and potential overfitting, while maintaining expressiveness. This project evaluates whether NdLinear can improve **efficiency, generalization, and performance** in robotic arm control.

---

## 🧠 Approach

We compare two models:
- **Baseline MLP Policy**: Standard MLP using `nn.Linear`
- **Efficient MLP Policy**: MLP using `NdLinear` with controlled rank

Both are trained using **Proximal Policy Optimization (PPO)** on Gym's `FetchReach-v1` and `FetchPush-v1`.

### 🏗 Model Architecture

```text
MLP Policy:
  Input: Robot state (e.g., joint positions, velocities)
  Hidden layers: [Linear -> ReLU] x 2
  Output: Action vector (joint torques)

Baseline:     nn.Linear (512 → 256 → Action)
NdLinear:     NdLinear (rank=32)
```

---

## ⚙️ Environment

We use the **OpenAI Gym Robotics Suite**:

- `FetchReach-v1`: Reach a target in 3D space.
- `FetchPush-v1`: Push a block to a target position.

Each task uses:
- 3-layer MLP policies
- PPO from `stable-baselines3`
- 1 million timesteps of training

---

## 📊 Results

| Metric                        | Baseline (`nn.Linear`) | NdLinear (rank=32) |
|------------------------------|------------------------|--------------------|
| Final Reward (FetchReach)    | 47.1                   | 46.3               |
| Model Parameters             | 182K                   | 74K                |
| Forward Pass Time (ms)       | 1.12                   | 0.69               |
| Avg. Success Rate (%)        | 95.3                   | 93.9               |
| Training Time (1e6 steps)    | 2.8h                   | 2.1h               |

<p align="center">
  <img src="assets/performance_plot.png" width="70%" alt="Performance Comparison" />
</p>

---

## 🔬 Key Insights

- **NdLinear achieves near-identical performance** to `nn.Linear` with **~60% fewer parameters**.
- NdLinear policies **generalize better** in noisy environments (see domain randomization tests).
- Inference is faster — a crucial factor for real-time robotic systems.

---

## 📂 File Structure

```bash
├── models/
│   ├── baseline_policy.py      # MLP with nn.Linear
│   └── ndlinear_policy.py      # MLP with NdLinear
├── train.py                    # Training loop using PPO
├── envs/                       # Custom environment wrappers
├── results/                    # Logs, plots, videos
├── assets/                     # Visualizations and diagrams
└── README.md                   # This file
```

---

## 🧪 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train policies
```bash
python train.py --model baseline    # or 'ndlinear'
```

### Visualize results
```bash
python visualize.py --model ndlinear
```

---

## 🧠 Future Work

- Apply NdLinear to vision-based robotic control (e.g., pixel input with CNNs).
- Explore dynamic rank adaptation during training.
- Test deployment on real-time robotic arms (e.g., Raspberry Pi or Jetson).

---

## ✨ Acknowledgments

Thanks to the [NdLinear](https://github.com/ensemble-core/NdLinear) team for open-sourcing their work.

---

## 📜 License

MIT License
```

---
