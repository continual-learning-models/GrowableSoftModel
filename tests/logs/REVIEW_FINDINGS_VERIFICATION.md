# Review-Findings Third-Party Verification
2026-07-29 · scripts/verify_review_findings.py

Each PASS = the capability is FUNCTIONAL in the authority (SB3 / torch autograd), so its absence or no-effect in our code is a REAL deviation (findings F-1/F-2/F-4/F-6 confirmed from the third-party side; F-3/F-7 are interface-dispatch findings pinned by the RED unit boxes tests/unit/test_rl_review_red.py).

## [PASS] V-F1 — SB3 ent_coef 0.0 vs 0.5, same seed -> params differ
    max|param diff| = 5.529e-04 (>0 required)

## [PASS] V-F4a — SB3 n_epochs 1 vs 10 -> params differ
    max|param diff| = 2.950e-02

## [PASS] V-F4b — SB3 target_kl 1e-8 vs None -> params differ (early stop bites)
    max|param diff| = 1.009e-02

## [PASS] V-F2 — SB3 n_steps 64 vs 256 -> params differ
    max|param diff| = 4.740e-03

## [PASS] V-F6 — torch: KL-to-reference anchor keeps params near the incumbent (canonical RLHF/GRPO term)
    drift free=13.248 anchored=0.252 (anchored < 0.5x free required)
