# Quasi-Static Full-Pipeline Verification — growable organ vs PURE third-party (PyTorch official) fixed network
2026-07-28 · scripts/verify_quasistatic_full.py

Fixed side uses ONLY official torch components (nn.Linear, F.gelu(approximate='tanh'), distributions.Categorical, optim.SGD, autograd); none of our library code executes on the fixed side.

## [PASS] TIER-1 — t0 pre-growth (trained 400): organ vs PURE-TORCH fixed twin, 257-pt grid
```
max |f_ours - f_torch| = 5.344e-13 (cross-library tanh ulp noise; arbitration below)
```
## [PASS] TIER-1 — t1 POST-DEEPEN instant: organ vs PURE-TORCH fixed twin, 257-pt grid
```
max |f_ours - f_torch| = 5.344e-13 (cross-library tanh ulp noise; arbitration below)
```
## [PASS] TIER-1 — t1 POST-DEEPEN instant: G-1 exactness (function change AT the growth event)
```
max |f_post - f_pre| = 0.000e+00 (quasi-static: growth invisible at the instant)
```
## [PASS] TIER-1 — t2 trained after deepen: organ vs PURE-TORCH fixed twin, 257-pt grid
```
max |f_ours - f_torch| = 1.241e-11 (cross-library tanh ulp noise; arbitration below)
```
## [PASS] TIER-1 — t3 POST-RHO(grow) instant: organ vs PURE-TORCH fixed twin, 257-pt grid
```
max |f_ours - f_torch| = 1.241e-11 (cross-library tanh ulp noise; arbitration below)
```
## [PASS] TIER-1 — t3 POST-RHO(grow) instant: G-1 exactness (function change AT the growth event)
```
max |f_post - f_pre| = 0.000e+00 (quasi-static: growth invisible at the instant)
```
## [PASS] TIER-1 — t4 trained after rho: organ vs PURE-TORCH fixed twin, 257-pt grid
```
max |f_ours - f_torch| = 2.243e-11 (cross-library tanh ulp noise; arbitration below)
```
## [PASS] TIER-1 — ULP arbitration: gelu(0.7391) numpy vs torch vs mpmath-50 truth
```
numpy-path=0.56909797656688066
torch     =0.56909797656744865
mpmath-50 =0.56909797656688066
|numpy-true|=0.00e+00 |torch-true|=5.68e-13
```
## [PASS] TIER-2 — t0: values + GAE + PPO loss (ours=L0/numpy on organ; fixed=official torch on twin)
```
max|V diff|=5.337e-13 max|GAE diff|=5.587e-13 PPO-loss ours=-2.514844948507 torch=-2.514844948507
```
## [PASS] TIER-2 — t1: values + GAE + PPO loss (ours=L0/numpy on organ; fixed=official torch on twin)
```
max|V diff|=5.337e-13 max|GAE diff|=5.587e-13 PPO-loss ours=-2.514844948507 torch=-2.514844948507
```
## [PASS] TIER-2 — t2: values + GAE + PPO loss (ours=L0/numpy on organ; fixed=official torch on twin)
```
max|V diff|=1.235e-11 max|GAE diff|=1.179e-11 PPO-loss ours=-2.648816983320 torch=-2.648816983326
```
## [PASS] TIER-2 — t3: values + GAE + PPO loss (ours=L0/numpy on organ; fixed=official torch on twin)
```
max|V diff|=1.235e-11 max|GAE diff|=1.179e-11 PPO-loss ours=-2.648816983320 torch=-2.648816983326
```
## [PASS] TIER-2 — t4: values + GAE + PPO loss (ours=L0/numpy on organ; fixed=official torch on twin)
```
max|V diff|=1.963e-11 max|GAE diff|=1.963e-11 PPO-loss ours=-2.693970145752 torch=-2.693970145760
```
## [PASS] TIER-3 — FULL PIPELINE 40 updates: our numpy math vs OFFICIAL torch (nn.Linear + Categorical + autograd + optim.SGD) — parameter trajectories
```
max |param diff| across all 40 updates = 1.776e-15
```
## [PASS] TIER-3 — FULL PIPELINE EFFECT: achieved return per update (identical policies => identical effect)
```
returns=['-49.2473', '-55.0479', '-47.7588', '-51.2065', '-49.6254', '-51.7415', '-57.7831', '-53.7233', '-59.2502', '-44.3709', '-35.2495', '-32.5393', '-42.7553', '-44.3866', '-47.6381', '-45.2371', '-39.9330', '-45.4542', '-51.9437', '-44.6443', '-34.3838', '-40.1986', '-37.5109', '-47.9455', '-41.0010', '-42.4445', '-47.3605', '-40.3828', '-40.2380', '-40.1918', '-39.4533', '-34.9010', '-42.7762', '-31.9821', '-29.6446', '-32.9071', '-44.8516', '-33.1356', '-37.1701', '-32.5386']
(both sides share the trajectory because parameters stay identical — verified above at every update)
```
## [PASS] TIER-3 — learning improves the return (behavioral validity; mean of first 10 vs last 10 updates)
```
first10=-51.9755 last10=-35.9360 improvement=+16.0395
```
## [PASS] TIER-3b — KL-ANCHOR pipeline 40 updates: SHIPPED kl_grad_logits vs OFFICIAL torch kl_divergence + autograd — parameter trajectories (fixed reference = transplanted incumbent)
```
max |param diff| across all 40 updates = 2.776e-17
```
## [PASS] TIER-4 — loss-convention arbitration: organ gradient chain vs torch autograd (exact convention identified, not assumed)
```
|organ - autograd(mean-sq)|=1.205e-02  |organ - 0.5*autograd|=2.957e-12  -> 0.5*mean-sq
```
## [PASS] TIER-4 — organ INTERNAL ADAM 50-step pretraining weight trajectory at a GROWN instant vs OFFICIAL torch.optim.Adam on the transplanted twin (all params, every step)
```
max |weight diff| across all 50 steps = 7.098e-12
```