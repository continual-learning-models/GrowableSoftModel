# RL Math Verification — code vs doc 89 vs authoritative algorithms
2026-07-28 · scripts/verify_rl_math.py · authorities: numpy official stats, mpmath-50, sympy exact, libm/numpy/mpmath exp cross-check, numpy Generator, PyTorch autograd, live audit replay

- [PASS] FR-1.1 — credited gain vs numpy.average / mpmath-50 / sympy exact
  data: ours=0.13671428571428573 np=0.13671428571428573 mp=0.13671428571428570 sym=0.13671428571428570
- [PASS] FR-1.1 — advantage = credited - quoted (shipped helper)
  data: advantage=0.04671428571428574 authoritative=0.04671428571428574
- [PASS] FR-1.2 — EMA fold (w,m,v) vs numpy weighted stats + mpmath-50
  data: ours m=-0.01994092106187778 v=0.13927304168389362 | np m=-0.01994092106187774 v=0.13927304168389362 | mp m=-0.01994092106187775
- [PASS] FR-1.2 — variance stability at 1e8 scale (stable recursion vs numpy.var)
  data: ours v=6.666673819226e-05 numpy.var=6.666663885128e-05
- [PASS] FR-1.3 — multiplier raw=0.15 vs libm/numpy/mpmath exp
  data: ours=1.535063009255210 libm=1.535063009255210 (exp agree to 0.0e+00)
- [PASS] FR-1.3 — multiplier raw=0.05 vs libm/numpy/mpmath exp
  data: ours=0.500000000000000 libm=0.500000000000000 (exp agree to 0.0e+00)
- [PASS] FR-1.3 — multiplier raw=0.4 vs libm/numpy/mpmath exp
  data: ours=2.000000000000000 libm=2.000000000000000 (exp agree to 0.0e+00)
- [PASS] FR-1.3 — multiplier raw=0.12 vs libm/numpy/mpmath exp
  data: ours=1.000000000000000 libm=1.000000000000000 (exp agree to 0.0e+00)
- [PASS] FR-1.3 — degenerate reference => exactly 1.0 (tolerance law)
  data: sd=5.5e-17 -> 1.0
- [PASS] FR-2.1 — thompson draw vs numpy Generator.normal(m, se) (official RNG, same seed)
  data: ours=0.21737160795731786 numpy=0.21737160795731786
- [PASS] FR-2.1 — ucb index m + c*sqrt(v/w) (libm vs numpy sqrt)
  data: ours=0.34999999999999998 numpy=0.34999999999999998
- [PASS] FR-2.1 — mean_clip raw = m (identity)
  data: raw==m by construction; multiplier check under FR-1.3
- [PASS] FR-1.4 — quota=2: grant sequence (enumeration referee)
  data: offers=[True, True, False, False] quota_used=2
- [PASS] FR-1.5 — state == deterministic fold of the credit stream (25 random events, rebuild replay)
  data: stats+event_stats identical=True buckets=['deepen|s0', 'grow|s1']
- [PASS] FR-3.1 — GAE via L0 credit_weights/fold vs torch recursive reference
  data: ours=['0.273154959916', '-0.453849059100', '-0.163582200000', '-0.652400000000', '-0.800000000000']
         torch=['0.273154959916', '-0.453849059100', '-0.163582200000', '-0.652400000000', '-0.800000000000']
- [PASS] FR-3.2/O-1 — pseudo-target: (h - y*) == autograd dL/dh (quadratic head)
  data: autograd dL/dh=[0.29999999999999993, -0.30000000000000004] implied=[0.29999999999999993, -0.30000000000000004]
- [PASS] FR-3.1 — PPO clip grad (ratio=1.4, adv=1.0) hand-derived vs torch autograd
  data: hand=0.000000000000 autograd=0.000000000000
- [PASS] FR-3.1 — PPO clip grad (ratio=0.7, adv=1.0) hand-derived vs torch autograd
  data: hand=-0.700000000000 autograd=-0.700000000000
- [PASS] FR-3.1 — PPO clip grad (ratio=1.4, adv=-1.0) hand-derived vs torch autograd
  data: hand=1.400000000000 autograd=1.400000000000
- [PASS] FR-3.0 — GRPO group standardization vs torch mean/std
  data: ours=['-1.069044967650', '0.000000000000', '-0.534522483825', '1.603567451475']
- [PASS] FR-4.3 — both-off byte-identity (committed evidence: TI-02 x3 configs + battery B-1 9/9 archived)
  data: baseline fn_sha=ae1301851b3d0136... dec_sha=f3cc7ce4c19b04d9... (standing suite re-proves on every run)
- [PASS] FR-7 — E2E audit replay: every logged multiplier independently recomputed from the raw event stream via numpy authorities
  data: 10 draw events re-derived, all match < 1e-9