# Deployment profile and versioning strategy

## 1) Support matrix (must be explicit)
- Python: 3.9+
- CPU profile: default deterministic pipeline + curation/report stages
- GPU profile: Uni-Mol inference/training adapters (when enabled)

## 1.1) Pinned dependency profiles
- `requirements/base.in`: core runtime pins
- `requirements/cpu.in`: CPU stack pins
- `requirements/gpu.in`: GPU/adapter stack pins
- `requirements/dev.in`: lint/test/build pins

Install command:
```bash
./scripts/install_profile.sh cpu
./scripts/install_profile.sh gpu
./scripts/install_profile.sh dev
```

Lockfile quality gate:
```bash
python3 scripts/validate_lockfiles.py --requirements-dir requirements
```

## 2) Profiles
- CPU:
  - `docker compose --profile cpu up --build`
- GPU:
  - `docker compose --profile gpu up --build`

## 3) Deterministic deployment checklist
1. Run `doctor` and ensure no `fail`.
2. Run `smoke` and ensure `status=success` + `final_output_exists=true`.
3. For GPU environments, confirm `gpu:nvidia_smi` is `pass` in doctor report.

## 4) Optional external tools
- `unimol_tools` (import name: `unimol_tools`)
- MinerU (`magic-pdf`, import name: `magic_pdf`)

These are not hard runtime requirements for the minimal deterministic pipeline, but are checked by `doctor` and should be pinned in environment-specific lockfiles.

## 5) Deployment guarantees and caveats
- For reproducibility, always install via `scripts/install_profile.sh`.
- GPU profile requires host NVIDIA driver compatibility with the selected torch CUDA wheel.
- If your cluster enforces an internal PyPI mirror, mirror all pinned wheels from `requirements/*.in` and set pip index accordingly.
- Some heavy optional packages have Python-version/platform-specific wheel availability:
  - `rdkit` is not included in default `cpu.in` pins
  - `torch`/`numpy`/`pandas`/`scikit-learn` are pinned with Python-version markers
