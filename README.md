
# 🤖 Efficient Robotic Arm Control with NdLinear

This project explores the use of [`NdLinear`](https://github.com/ensemble-core/NdLinear) — a low-rank replacement for `nn.Linear` — in robotic control policies. We benchmark NdLinear against standard MLPs in continuous control using the classic `Pendulum-v1` environment from OpenAI Gym.

---

## 🚀 Motivation

Robotic control tasks often rely on fully connected neural networks (MLPs) as policy approximators. These layers can be parameter-heavy and inefficient, especially in edge devices with tight compute or latency budgets.

**NdLinear** introduces low-rank approximations to reduce the parameter count while retaining expressive power. We evaluate its impact on performance, efficiency, and training stability.

---

## 🧠 Approach

We compare two architectures:
- **Baseline MLP Policy**: Standard MLP using `nn.Linear`
- **NdLinear MLP Policy**: Same structure, but uses `NdLinear` with configurable rank

Both models are trained using a REINFORCE-style policy gradient algorithm.

---

## 🏗 Model Architecture

```text
MLP Policy:
  Input: Environment state (e.g., angle, angular velocity)
  Hidden layers: [Linear -> ReLU] x 2
  Output: Continuous action value (torque)

Baseline:     nn.Linear layers
NdLinear:     NdLinear (rank = 32)
```

---

## ⚙️ Environment

We use the classic Gym control task:

- `Pendulum-v1`: Learn to balance a torque-controlled pendulum upright

Training config:
- Environment: `Pendulum-v1`
- Episodes: 500
- Optimizer: Adam
- Reward normalization + advantage-weighted regression

---

## 📊 Results (Example)

| Metric             | Baseline (`nn.Linear`) | NdLinear (rank=32) |
|-------------------|------------------------|--------------------|
| Final Loss        | ~1.12                  | ~1.09              |
| Parameter Count   | 182K                   | ~74K               |
| Forward Time (ms) | ~1.1                   | ~0.7               |

_Note: Replace with actual logs from `results/` after your runs._

---

## 📂 File Structure

```bash
├── models/
│   ├── baseline_policy.py      # MLP with nn.Linear
│   └── ndlinear_policy.py      # MLP with NdLinear
├── train.py                    # REINFORCE training loop
├── requirements.txt            # All dependencies
├── assets/                     # (optional) Plots or demo GIFs
└── README.md                   # Project documentation
```

---

## 🧪 How to Run

### 🔧 Install dependencies
```bash
pip install -r requirements.txt
```

### 🚀 Train a model

Train a baseline policy:
```bash
python train.py --model baseline
```

Train an NdLinear-based policy with rank 32:
```bash
python train.py --model ndlinear --rank 32
```

---

## 🔬 Future Work

- Compare more environments: `MountainCarContinuous`, `Reacher-v2`
- Replace critic/value heads with NdLinear
- Run forward-pass benchmarks across devices (CPU vs GPU)
- Apply NdLinear to vision-based or transformer policies

---

## ✨ Acknowledgments

Special thanks to the [NdLinear](https://github.com/ensemble-core/NdLinear) team for enabling efficient linear layers in PyTorch.

---

## 📜 License

MIT License
```

