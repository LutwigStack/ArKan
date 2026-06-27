# Competitor Setup

Both competitor libraries were installed via pip (no vendoring required):

```
pip install git+https://github.com/Blealtan/efficient-kan.git   # MIT license
pip install git+https://github.com/ZiyaoLi/fast-kan.git          # Apache-2.0 license
```

**efficient-kan** (Blealtan/efficient-kan, MIT):
- Pure PyTorch B-spline KAN with base+spline residual term
- `from efficient_kan import KAN`

**FastKAN** (ZiyaoLi/fast-kan, Apache-2.0):
- Radial Basis Function (RBF) approximation to KAN
- `from fastkan import FastKAN`
- NOTE: FastKAN uses RBF (Gaussian kernels), NOT B-splines. It is NOT a direct
  mathematical equivalent to ArKan or efficient-kan.

If pip install fails (e.g. no internet), vendor the single model file:
- efficient-kan: `efficient_kan/src/efficient_kan/kan.py` → this directory as `efficient_kan.py`
  with header: `# Source: https://github.com/Blealtan/efficient-kan (MIT License)`
- FastKAN: `fastkan/fastkan/fastkan.py` → this directory as `fastkan.py`
  with header: `# Source: https://github.com/ZiyaoLi/fast-kan (Apache-2.0 License)`
