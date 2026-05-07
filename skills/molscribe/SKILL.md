# Agent4Mat MolScribe skill

Use this skill when user requests image/PDF structure-to-SMILES extraction.

## Current scope
- MolScribe is not yet connected as a first-class adapter in current tool registry.
- Nearest compatible insertion points:
  - upstream preprocessing before `generate_candidates`
  - custom `generate_candidates` adapter command via `OLED_AGENT_GENERATE_CMD` or catalog adapter.

## Suggested integration pattern (contract-safe)
1. Build a dedicated adapter script (example: `scripts/adapters/generate_candidates_molscribe_adapter.py`).
2. Accept standard `generate_candidates` JSON payload.
3. Convert source images/figures to SMILES candidates with MolScribe.
4. Write normalized candidate CSV with at least:
   - `candidate_id`
   - `smiles`
5. Return JSON with `status`, `adapter`, `output_csv`, `rows`.

## Validation path
- Validate adapter output contract first:
  - `python3 scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "<your molscribe cmd>" --workspace-root . --json`
- Then run `agent-run-json` using an adapter-enabled model catalog.

## Rules
- Keep MolScribe extraction isolated inside adapter boundary.
- Never change downstream `score_candidates` input schema.
- If OCR/parse fails, return structured adapter failure JSON instead of partial CSV.
