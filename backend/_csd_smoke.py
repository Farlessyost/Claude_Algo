"""Quick sanity tests for backend.csd against synthetic data with known properties.
Run: .\.venv\Scripts\python.exe -m backend._csd_smoke
"""
import math, random
from backend import csd

# 1) Synthetic AR(1) with known phi=0.7
random.seed(7)
phi_true = 0.7
xs = [0.0]
for _ in range(2000):
    xs.append(phi_true * xs[-1] + random.gauss(0, 0.01))
phi_est = csd.phi_ar1(xs)
print("[1] AR(1) phi_true=0.70 phi_est=%.4f  rr=%.4f  (expected rr=%.4f)" % (
    phi_est, csd.recovery_rate(phi_est), -math.log(0.7)))

# 2) Recovery rate edges
print("[2] rr(0)=%.4f rr(1)=%.4f rr(0.99)=%.4f rr(0.5)=%.4f rr(-0.3)=%.4f" % (
    csd.recovery_rate(0.0), csd.recovery_rate(1.0), csd.recovery_rate(0.99),
    csd.recovery_rate(0.5), csd.recovery_rate(-0.3)))

# 3) Low-freq power: slow vs fast sine
N = 128
slow = [math.sin(2*math.pi * 2 * t / N) for t in range(N)]
fast = [math.sin(2*math.pi * 40 * t / N) for t in range(N)]
print("[3] lf_power(slow)=%.4f lf_power(fast)=%.4f" % (
    csd.low_freq_power(slow), csd.low_freq_power(fast)))

# 4) VAR(1) eigenvalue: known A_true with max |lambda|=0.6
random.seed(11)
N = 1000
A_true = [[0.6, 0.1], [0.0, 0.4]]
y = [[0.0, 0.0]]
for _ in range(N):
    prev = y[-1]
    y.append([
        A_true[0][0]*prev[0] + A_true[0][1]*prev[1] + random.gauss(0, 0.01),
        A_true[1][0]*prev[0] + A_true[1][1]*prev[1] + random.gauss(0, 0.01),
    ])
Y = [[row[0] for row in y], [row[1] for row in y]]
A_est = csd.var1_a_matrix(Y)
print("[4] A_est=", A_est, " max_eig=%.4f (true=0.6)" % csd.max_eigenvalue(A_est))

# 5) Well depth: bimodal should be DEEPER (taller histogram peaks) than uniform
random.seed(13)
unimodal = [random.gauss(0, 1) for _ in range(800)]
bimodal = [random.gauss(-2, 0.3) if random.random()<0.5 else random.gauss(2, 0.3) for _ in range(800)]
uniform = [random.uniform(-2, 2) for _ in range(800)]
print("[5] well_depth(unimodal)=%.3f bimodal=%.3f uniform=%.3f" % (
    csd.well_depth(unimodal), csd.well_depth(bimodal), csd.well_depth(uniform)))

# 6) End-to-end compute() on a synthetic walk
random.seed(17)
closes = [100.0]
for _ in range(200):
    closes.append(closes[-1] * (1 + random.gauss(0, 0.001)))
extras = {"spread": [random.gauss(0, 0.0001) for _ in closes]}
b = csd.compute(closes, fv_period=32, window=96, extras=extras)
print("[6] compute bundle:", {k: round(v, 5) for k, v in b.items()})
