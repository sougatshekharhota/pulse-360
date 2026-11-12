"""Self-check for PULSE-MMM: run `python tests/test_mmm.py` (~30 s).

Fits the synthetic dataset with short chains and asserts the planted ROIs
land inside the 90% credible intervals — i.e. the sampler actually works.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import mmm  # noqa: E402
from src.mmm_demo import CHANNELS, CONTROLS, TRUTH, write  # noqa: E402

# transforms
x = np.r_[np.zeros(3), 10.0, np.zeros(20)]
assert abs(mmm.adstock(x, 0.6).sum() - 10) < 1e-9, "adstock must preserve total effect"
assert mmm.hill(np.array([2.0]), 2.0)[0] == 0.5, "Hill: half effect at x = K"

# optimizer keeps the total budget and respects bounds
m = np.round(np.arange(0.6, 1.4001, 0.01), 2)
spend = np.array([100.0, 50.0])
curves = np.stack([np.sqrt(m) * 40, m * 80])        # channel 2 has the steeper margin
alloc = mmm.optimize_budget(curves, m, spend, 0.3)
assert abs(alloc.sum() - spend.sum()) < 1e-6 and alloc[1] > spend[1]

# recovery of planted ROIs
with tempfile.TemporaryDirectory() as tmp:
    write(Path(tmp) / "in.csv", Path(tmp) / "truth.json")
    d = mmm.load_csv(Path(tmp) / "in.csv", CHANNELS, CONTROLS)
    f = mmm.fit(d, [[1.5, 1.5], [2.0, 0.4], [1.5, 1.5], [1.5, 1.5]],
                chains=2, warmup=300, draws=300, verbose=False)
th, a, k = (f.flat(v) for v in (f.theta, f.alpha, f.K))
idx = np.linspace(0, len(th) - 1, 120).astype(int)
R = np.array([mmm.counterfactual_roi(f, th[i], a[i], k[i]) for i in idx])
for c, name in enumerate(CHANNELS):
    lo, hi = np.percentile(R[:, c], [5, 95])
    print(f"{name:<12} truth {TRUTH[name]['roi']:.2f}  90% CI [{lo:.2f}, {hi:.2f}]")
    assert lo <= TRUTH[name]["roi"] <= hi, f"{name}: truth outside 90% interval"
print("ok")
