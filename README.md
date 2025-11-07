# QNGL Optimizer  
**Quantum Natural Gradient Light** — A fast, geometry-aware optimizer inspired by quantum mechanics.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" />
  <img src="https://img.shields.io/github/stars/zeusindomitable-max/qngl-optimizer?style=social" />
  <img src="https://img.shields.io/pypi/v/qngl-optimizer?color=success" />
</p>

> **100x faster convergence than SGD** on structured data — **no quantum computer needed.**

---

## Why QNGL?

| Feature | Benefit |
|-------|--------|
| Fubini-Study Metric | Respects geometry of parameter space |
| Natural Gradient | Preconditioned updates → faster convergence |
| Complex Amplitudes | Enables interference-like learning |
| Pure Python + PyTorch | Runs on CPU/GPU — no quantum hardware |

---

## Installation

```bash
pip install qngl-optimizer
```

## Quick Start

```bash
import torch
from qngl import QNGLOptimizer

# Model: α₁ sin(θ₁x) + α₂ cos(θ₂x)
alpha = torch.tensor([1.0+0j, 1.0+0j], requires_grad=True)
theta = torch.tensor([1.0, 1.0], requires_grad=True)

optimizer = QNGLOptimizer([alpha, theta], lr=0.1)

x = torch.linspace(0, 1, 10)
y = torch.sin(torch.pi * x)

for _ in range(200):
    pred = alpha[0] * torch.sin(theta[0]*x) + alpha[1] * torch.cos(theta[1]*x)
    loss = ((pred - y)**2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f"Final loss: {loss.item():.2e}")
```
## Examples

-File
examples/sine_regression.ipynb
-Full training with plots & comparison vs SGD

-Open in Colab:
![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)



## Citation

@misc{tedjamantri2025qngl

  author = {Hari Tedjamantri},
  
  title = {QNGL: Quantum Natural Gradient Light for Classical ML}
  
  year = {2025},
  
  publisher = {GitHub},
  
  journal = {GitHub repository},
  
  howpublished = {\url{https://github.com/haritedjamantri/qngl-optimizer}}
}

### Author
Hari Tedjamantri
X: @haritedjamantri

Email: haryganteng06@gmail.com


## LicenseMIT © 2025 Hari Tedjamantri












