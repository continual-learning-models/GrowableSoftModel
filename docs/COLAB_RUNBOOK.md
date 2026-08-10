# Colab / cloud-GPU runbook

One codebase, one model object; the device is a policy switch.

## 1. Setup (Colab: Runtime -> Change runtime type -> T4 GPU)

    !git clone https://<TOKEN>@github.com/continual-learning-models/<repo>.git
    %cd <repo>
    !pip -q install torch numpy

## 2. Select the backend (P8: user policy, never self-switched)

    import sys; sys.path.insert(0, "modules/Engine")
    from engine.backends import set_compute_policy
    set_compute_policy("torch", "cuda", "float32",
                       acknowledge_f32_precision=True)  # Colab T4
    # (float32 is not accuracy-certified — the flag is the
    #  precision door; "float64" needs no flag)
    # Mac local: ("torch", "mps", "float32",
    #             acknowledge_f32_precision=True)
    # judge / any CPU: ("numpy")  [default]

Models constructed afterwards live on that device; everything —
widen/deepen, body types, substrates (numeric attention host),
SPU on/off — works identically. Since GSM-I2 the categorical and
causal host modes run on the backend kernels as well (BK14
parity box in tests/backend_kit/spec.md).

## 3. Run an exam and fetch results

    !python3 experiments/R2_body_type_scale/driver.py
    # results land in experiments/<exam>/results/*.json — download
    # and adjudicate/commit locally per house discipline.

## 4. Guidance (measured)

- Large hosts (>=100k params): GPU pays — cpu-torch 8.9x, mps
  22x, cuda typically more.
- Small models / serving: plain CPU is fastest (kernel-launch
  overhead dominates tiny tensors); artifacts are device-free and
  SERVE on the numpy judge by default.
- One GPU: run exam arms sequentially (a process pool contends);
  CPU backends: pool freely.
