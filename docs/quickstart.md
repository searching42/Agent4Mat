# Quickstart

## Local (no install mode)
```bash
cd /Users/benton/openclaw-docker/workspace/oled-agent
./scripts/install_profile.sh cpu
PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root .
PYTHONPATH=src python3 -m oled_agent.cli smoke --workspace-root .
```

## Docker CPU
```bash
docker compose --profile cpu up --build
```

## Docker GPU (requires NVIDIA runtime)
```bash
docker compose --profile gpu up --build
```

## Strict CI-style check
```bash
python3 scripts/validate_lockfiles.py --requirements-dir requirements
PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root . --strict
```
