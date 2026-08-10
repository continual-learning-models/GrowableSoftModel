# RL Algorithms vs Third-Party Authoritative Libraries — Multi-Case Comparison Report
2026-07-28 · scripts/verify_vs_authoritative_libs.py

Authorities: Stable-Baselines3 PPO (full library training loop), SB3 official GAE buffer, gymnasium official environments, MABWiser (industry bandit library), torch.optim.Adam. Cases: CartPole-v1 (60k), Acrobot-v1 (120k), 128x16 random MDP (60k), GRPO, 20-arm bandits (20k x 10 seeds x 3 algorithms).

## [PASS] A1-GAE — SB3 RolloutBuffer.compute_returns_and_advantage (official) vs our L0 GAE, 4096 steps
```
max|diff| = 3.15e-06 (SB3 float32 buffer); L0 credit_fold spot-checks at t=0/1000/3000 match truncated-horizon sums
```
## [PASS] A2-Adam — our numpy Adam vs torch.optim.Adam, 200 steps
```
max|param diff| = 2.22e-16
```
## [PASS] B1-CartPole — FULL PIPELINE 60000 steps x 5 seeds: our numpy PPO vs Stable-Baselines3 PPO (means; no tolerance band)
```
OURS mean = 500.00 (runs ['500.0', '500.0', '500.0', '500.0', '500.0'])
SB3  mean = 500.00 (runs ['500.0', '500.0', '500.0', '500.0', '500.0'])
diff = +0.00, combined SE = 0.00  (ours 25s, sb3 105s)
```
## [PASS] B2-Acrobot — FULL PIPELINE 120000 steps x 5 seeds: our numpy PPO vs Stable-Baselines3 PPO (means; no tolerance band)
```
OURS mean = -81.86 (runs ['-74.1', '-88.8', '-81.6', '-83.5', '-81.3'])
SB3  mean = -81.79 (runs ['-84.7', '-78.3', '-81.5', '-80.9', '-83.5'])
diff = -0.07, combined SE = 2.33  (ours 67s, sb3 372s)
```
## [PASS] B3-BigMDP — FULL PIPELINE 60000 steps x 5 seeds: our numpy PPO vs Stable-Baselines3 PPO (means; no tolerance band)
```
OURS mean = 195.51 (runs ['204.9', '201.1', '190.0', '198.8', '182.7'])
SB3  mean = 192.19 (runs ['199.6', '194.8', '191.6', '191.0', '184.0'])
diff = +3.32, combined SE = 4.27  (ours 46s, sb3 162s)
```
## [PASS] B4-GRPO — GRPO (group-standardized, critic-free) on CartPole x 5 seeds; reference standard = SB3 PPO mean above (500.0)
```
GRPO mean = 500.00 (runs ['500.0', '500.0', '500.0', '500.0', '500.0']; 74s)
```
## [PASS] C-thompson — 20-arm Bernoulli bandit, 20000 steps x 10 seeds, stationary config (decay=1, eps rate-matched): OURS vs MABWiser ThompsonSampling (mean cumulative regret; no tolerance factor)
```
regret OURS = 101.3 +/- 17.9   MABWiser = 117.1 +/- 54.2   ratio = 0.865
```
## [PASS] C-eps_greedy — 20-arm Bernoulli bandit, 20000 steps x 10 seeds, stationary config (decay=1, eps rate-matched): OURS vs MABWiser EpsilonGreedy(0.1) (mean cumulative regret; no tolerance factor)
```
regret OURS = 529.1 +/- 167.8   MABWiser = 923.1 +/- 99.2   ratio = 0.573
```
## [PASS] C-ucb — 20-arm Bernoulli bandit, 20000 steps x 10 seeds, stationary config (decay=1, eps rate-matched): OURS vs MABWiser UCB1(1.0) (mean cumulative regret; no tolerance factor)
```
regret OURS = 70.3 +/- 118.8   MABWiser = 913.3 +/- 51.9   ratio = 0.077
```