# RL vs Industry-Standard Algorithms — Verification Report (fixed networks)
2026-07-28 · scripts/verify_rl_vs_industry.py

Methods: (SIM) 3-arm bandit choice-sequence identity vs independent textbook reference + convergence; (SIM) full PPO/GRPO training on a FIXED softmax-policy network — hand-derived numpy gradients vs PyTorch autograd, 5-update parameter-trajectory identity; (MATH) GAE/pseudo-target/EMA vs torch references.

## [PASS] mean_clip — bandit SIM: 400-step choice sequence + reward vs independent textbook reference (same seeds)
```
identical_choices=True reward ours=124.429364 ref=124.429364 | last-100 best-arm rate=1.00
```
## [PASS] mean_clip — bandit SIM: convergence to the best arm (behavioral validity)
```
best-arm fraction over final 100 steps = 1.00 (world means [0.1, 0.3, 0.22])
```
## [PASS] thompson — bandit SIM: 400-step choice sequence + reward vs independent textbook reference (same seeds)
```
identical_choices=True reward ours=123.789364 ref=123.789364 | last-100 best-arm rate=1.00
```
## [PASS] thompson — bandit SIM: convergence to the best arm (behavioral validity)
```
best-arm fraction over final 100 steps = 1.00 (world means [0.1, 0.3, 0.22])
```
## [PASS] ucb — bandit SIM: 400-step choice sequence + reward vs independent textbook reference (same seeds)
```
identical_choices=True reward ours=124.509364 ref=124.509364 | last-100 best-arm rate=1.00
```
## [PASS] ucb — bandit SIM: convergence to the best arm (behavioral validity)
```
best-arm fraction over final 100 steps = 1.00 (world means [0.1, 0.3, 0.22])
```
## [PASS] eps_greedy — bandit SIM: 400-step choice sequence + reward vs independent textbook reference (same seeds)
```
identical_choices=True reward ours=117.909364 ref=117.909364 | last-100 best-arm rate=0.89
```
## [PASS] thompson — bandit SIM replication seed=7
```
identical_choices=True
```
## [PASS] PPO — FIXED-network SIM: 5 full updates, hand-derived numpy gradients vs PyTorch autograd (industry standard)
```
max |loss diff|=2.13e-14  max |grad diff|=3.55e-15  max |param diff after 5 updates|=5.55e-17
```
## [PASS] GAE — rollout SIM: L0-kernel GAE vs torch recursive reference (H=64)
```
max |diff| = 5.33e-15
```
## [PASS] GRPO — FIXED-network SIM: 5 updates, group-standardized advantages (no value head), numpy vs torch autograd
```
max |grad diff|=1.11e-16  max |param diff|=0.00e+00
```
## [PASS] GRPO — zero-variance group edge -> advantages exactly 0 (never NaN)
```
advantages=[0.0, 0.0, 0.0, 0.0]
```
## [PASS] pseudo-target — y* = h - dL/dh: substrate quadratic-head gradient toward y* == autograd dL/dh (fixed head)
```
autograd=[-0.19, -0.3999999999999999, -0.1]
(h-y*)  =[np.float64(-0.19), np.float64(-0.3999999999999999), np.float64(-0.1)]
```
## [PASS] EMA-stats — 200-sample stream: shipped fold vs torch weighted-tensor reference
```
ours m=0.095745464928374 v=0.296490521410282
torch m=0.095745464928374 v=0.296490521410282
```