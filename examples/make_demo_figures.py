"""Make two demo figures with plotpack (run: uv run python examples/make_demo_figures.py).

Writes examples/figures/{accuracy-by-size,calibration}.{pdf,png,svg,plotpack}; then try
`uv run plotpack inspect examples/figures/calibration.png`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotpack

OUT = Path(__file__).parent / "figures"
rng = np.random.default_rng(0)

# 1. accuracy vs model size, two benchmarks, with CIs
sizes = np.array([0.5, 1, 3, 7, 14, 32, 70])
acc = pd.DataFrame({
    "params_b": np.tile(sizes, 2),
    "benchmark": ["benchmark A"] * 7 + ["benchmark B"] * 7,
    "accuracy": np.r_[0.41 + 0.09 * np.log(sizes + 1), 0.35 + 0.11 * np.log(sizes + 1)] + rng.normal(0, 0.01, 14),
})
acc["ci_low"], acc["ci_high"] = acc.accuracy - 0.025, acc.accuracy + 0.025
fig, ax = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
for name, g in acc.groupby("benchmark"):
    ax.plot(g.params_b, g.accuracy, marker="o", label=name)
    ax.fill_between(g.params_b, g.ci_low, g.ci_high, alpha=0.2)
ax.set(xscale="log", xlabel="parameters (B)", ylabel="accuracy")
ax.legend(frameon=False)
plotpack.save(fig, "accuracy-by-size", data={"accuracy": acc}, out=OUT,
              params={"seed": 0, "ci": "±0.025 (synthetic)"}, notes="Synthetic demo data.")
plt.close(fig)

# 2. calibration curve plus the raw confidence histogram as an array
conf = rng.beta(5, 2, 2000)
correct = rng.random(2000) < conf ** 1.3
bins = np.linspace(0, 1, 11)
idx = np.digitize(conf, bins) - 1
cal = pd.DataFrame({"bin_mid": (bins[:-1] + bins[1:]) / 2,
                    "accuracy": [correct[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)],
                    "count": np.bincount(idx, minlength=11)[:10]})
fig, ax = plt.subplots(figsize=(3.2, 3), constrained_layout=True)
ax.plot([0, 1], [0, 1], ls=":", c="grey")
ax.plot(cal.bin_mid, cal.accuracy, marker="s")
ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
plotpack.save(fig, "calibration", data={"calibration": cal, "confidences": conf}, out=OUT,
              params={"n": 2000, "bins": 10})
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.iterdir()))
