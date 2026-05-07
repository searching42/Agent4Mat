# Dependency lock strategy

This repository uses profile-based, pinned requirement sets:

- `base.in`: minimal runtime requirements
- `cpu.in`: deterministic local/CI profile
- `gpu.in`: Uni-Mol/MinerU capable profile
- `dev.in`: local development + tests

## Installation

```bash
./scripts/install_profile.sh cpu
./scripts/install_profile.sh gpu
./scripts/install_profile.sh dev
```

For GPU profile, `install_profile.sh` installs the correct torch wheel index
according to `TORCH_CUDA_INDEX_URL` (default: cu121).

When using an internal package mirror:

```bash
PIP_EXTRA_ARGS="--index-url https://<your-mirror>/simple" ./scripts/install_profile.sh cpu
```

## Why pinned `.in` files?

- Keeps deployment reproducible across local/dev/CI.
- Makes optional heavy toolchains explicit (`torch`, `unimol-tools`, `magic-pdf`).
- Works offline better than floating installs once a wheelhouse/cache exists.
