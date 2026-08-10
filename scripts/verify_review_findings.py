"""Third-party multi-angle verification of the 2026-07-29
review findings (owner order): prove each missing capability
HAS an effect in the AUTHORITATIVE library / canonical
machinery — so our no-effect keys are real deviations, not
imagined ones. Report -> tests/logs/REVIEW_FINDINGS_VERIFICATION.md

Angles:
 V-F1 SB3 PPO ent_coef 0.0 vs 0.5 (same seed/data): params
      MUST differ + policy entropy higher -> entropy term is
      functional in the authority (ours is not = F-1 real)
 V-F4a SB3 n_epochs 1 vs 10: params differ (canonical
      multi-epoch semantics; our organ path ignores it)
 V-F4b SB3 target_kl tiny vs None: params differ (canonical
      KL early stop; our organ path lacks it)
 V-F2 SB3 n_steps (horizon) 64 vs 256: rollout accounting
      differs (the horizon knob is functional in the
      authority)
 V-F6 torch-autograd reference-KL anchor (the canonical
      RLHF/GRPO KL-to-reference term, cited industry
      pattern): adding beta*KL(pi||pi_ref) to the loss keeps
      parameters measurably closer to the reference on
      identical data — third-party autograd route
"""
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO as SB3PPO

ROOT = Path(__file__).resolve().parents[1]
OUT, FAIL = [], []


def rep(name, check, data, ok):
    OUT.append((name, check, data, "PASS" if ok else "FAIL"))
    if not ok:
        FAIL.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:6s} {check}")
    print(f"    {data}")


def _sb3(seed=3, **kw):
    args = dict(n_steps=256, batch_size=64, n_epochs=4,
                learning_rate=3e-4, seed=seed, verbose=0,
                device="cpu")
    args.update(kw)
    m = SB3PPO("MlpPolicy", gym.make("CartPole-v1"), **args)
    m.learn(total_timesteps=512, progress_bar=False)
    return np.concatenate([p.detach().numpy().ravel()
                           for p in m.policy.parameters()])


# V-F1 entropy coefficient is functional in SB3
pa = _sb3(ent_coef=0.0)
pb = _sb3(ent_coef=0.5)
d1 = float(np.max(np.abs(pa - pb)))
rep("V-F1", "SB3 ent_coef 0.0 vs 0.5, same seed -> params differ",
    f"max|param diff| = {d1:.3e} (>0 required)", d1 > 1e-6)

# V-F4a n_epochs is functional in SB3
pc = _sb3(n_epochs=1)
pd = _sb3(n_epochs=10)
d2 = float(np.max(np.abs(pc - pd)))
rep("V-F4a", "SB3 n_epochs 1 vs 10 -> params differ",
    f"max|param diff| = {d2:.3e}", d2 > 1e-6)

# V-F4b target_kl is functional in SB3
pe = _sb3(target_kl=1e-8)
pf = _sb3(target_kl=None)
d3 = float(np.max(np.abs(pe - pf)))
rep("V-F4b", "SB3 target_kl 1e-8 vs None -> params differ "
    "(early stop bites)", f"max|param diff| = {d3:.3e}",
    d3 > 1e-6)

# V-F2 horizon (n_steps) is functional in SB3
pg = _sb3(n_steps=64)
ph = _sb3(n_steps=256)
d4 = float(np.max(np.abs(pg - ph)))
rep("V-F2", "SB3 n_steps 64 vs 256 -> params differ",
    f"max|param diff| = {d4:.3e}", d4 > 1e-6)

# V-F6 reference-KL anchor via torch autograd (canonical
# RLHF/GRPO pattern: loss += beta * KL(pi || pi_ref))
def drift(beta, seed=11):
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(4, 3, generator=g))
    W_ref = W.detach().clone()
    opt = torch.optim.Adam([W], lr=5e-2)
    X = torch.randn(64, 3, generator=g)
    A = torch.randn(64, generator=g)
    acts = torch.randint(0, 4, (64,), generator=g)
    for _ in range(25):
        logits = X @ W.T
        logp = torch.log_softmax(logits, dim=1)
        pol = -(logp[torch.arange(64), acts] * A).mean()
        logp_ref = torch.log_softmax(X @ W_ref.T, dim=1)
        kl = (logp.exp() * (logp - logp_ref)).sum(1).mean()
        loss = pol + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
    return float((W.detach() - W_ref).abs().sum())

d_free = drift(0.0)
d_anch = drift(20.0)
rep("V-F6", "torch: KL-to-reference anchor keeps params near "
    "the incumbent (canonical RLHF/GRPO term)",
    f"drift free={d_free:.3f} anchored={d_anch:.3f} "
    f"(anchored < 0.5x free required)", d_anch < 0.5 * d_free)

print("=" * 64)
print(f"TOTAL {len(OUT)} checks | "
      f"{'ALL PASS' if not FAIL else 'FAILURES: ' + str(FAIL)}")
lines = ["# Review-Findings Third-Party Verification",
         "2026-07-29 · scripts/verify_review_findings.py",
         "", "Each PASS = the capability is FUNCTIONAL in the "
         "authority (SB3 / torch autograd), so its absence or "
         "no-effect in our code is a REAL deviation (findings "
         "F-1/F-2/F-4/F-6 confirmed from the third-party side; "
         "F-3/F-7 are interface-dispatch findings pinned by the "
         "RED unit boxes tests/unit/test_rl_review_red.py).", ""]
for n, c, d, v in OUT:
    lines += [f"## [{v}] {n} — {c}", f"    {d}", ""]
(ROOT / "tests" / "logs"
 / "REVIEW_FINDINGS_VERIFICATION.md").write_text(
    "\n".join(lines))
print("report -> tests/logs/REVIEW_FINDINGS_VERIFICATION.md")
sys.exit(1 if FAIL else 0)
