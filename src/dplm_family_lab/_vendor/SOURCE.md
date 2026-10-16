# Selected DPLM-1 source

Upstream: https://github.com/bytedance/dplm
Commit: 8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d
License: Apache-2.0 (included).

Selected from src/byprot/models/dplm/dplm.py, src/byprot/models/dplm/modules/dplm_modeling_esm.py, and sampling functions from src/byprot/models/utils.py. The upstream copyright headers are retained.

Local changes: registry-free relative imports; explicit HF-only loading; Python 3.11 compatible dataclass default factories; no DPLM2, OpenFold, Hydra registries or structural evaluation dependencies. esm.py retains the network/forward only; unused alternative network sampling methods are omitted. Diffusion noise and the DPLM sampler remain upstream code. The application implements zero-mask-safe loss reduction separately and records loading/round-trip checks.

Sampling additionally masks noncanonical amino-acid vocabulary entries (including ESM's null token), consistently for baseline and adapted models. Training logits are unchanged.
