# Architecture Map (one page)

The system in one sentence: a SOFT-MODEL FACTORY — it produces,
teaches, and governs growable soft models.

## Layers and packages

| Layer | Package | Role (one sentence) |
|---|---|---|
| L0 engine | modules/Engine/engine | Shared computation: math primitives (gelu, LayerNorm pair), numpy/torch kernels (backends/), loop-block kernel math (loop_ops), generic self-processing machinery (spu/). No models here. |
| L1 hosts | core/substrates/ | ALL model hosts behind the ONE contract (base.py; the host catalog lives in substrates/__init__.py): mlp, transformer, sequence, ... Each host implements ITS OWN growth/self-processing surgery. |
| L1 family | modules/ReferenceNet/reference_net | The first-generation recursive multi-scale network family: net, trainer, curriculum, growth policy, bodies, instrument, its own SPU bindings (spu_network, spu_host_walk). |
| L2 shell | core/ | System shell: facade (entry), lifecycle, teaching seam; core/_modules.py is the one module-access point. |
| L2 production | modules/Generator/generator | Model production: create / teach / gate / drift / version. |
| L3 surfaces | cli/, mcp/, standard_methods/ | Outward access; no model logic. |

Dependency law: downward only (surfaces -> shell/production ->
hosts/family -> engine); within a layer public APIs only;
core -> generator, never the reverse.

## Who implements growth / self-processing (surgery lives with
## the body — no directory owns these capabilities)

| Capability | Implementations |
|---|---|
| Growth | reference_net/net.py (node -> inner subnet); core/substrates/mlp.py and transformer.py (host sites via growth_sites/grow_site); growable_attention.py (head_add / head_widen behind the governed grow_attention verb — its own driver, not the site grammar). |
| Growth decision | reference_net/trainer.py (plateau + instability ranking); reference_net/growthpolicy/. |
| Self-processing | engine/spu (generic machinery), driven per model: reference_net/spu bindings for the family; hosts integrate via their own seams. |
