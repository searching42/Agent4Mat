# External scorer error codes

This document defines stable `fallback_error.code` values produced by `score_candidates` when external Uni-Mol scoring fails and local fallback is used.

## Envelope schema

`score_candidates` fallback payload:

```json
{
  "adapter": "local_deterministic_fallback",
  "fallback_reason": "human-readable summary",
  "fallback_error": {
    "code": "external_workspace_missing",
    "message": "External workspace root with scripts/ not found",
    "retryable": false,
    "details": {}
  }
}
```

## Codes

1. `external_workspace_missing`
- Meaning: no external workspace with `scripts/` could be resolved.
- Retryable: `false`
- Typical action: fix workspace layout or configure workspace root.

2. `external_scorer_script_missing`
- Meaning: scoring script path is missing.
- Retryable: `false`
- Typical action: sync `workspace/scripts/score_unimol_property_candidates.py`.

3. `external_scorer_disabled`
- Meaning: `OLED_AGENT_USE_EXTERNAL_SCORER` is not `1`.
- Retryable: `false`
- Typical action: enable env var explicitly when external scoring is desired.

3.1 `external_command_failed` with remote runtime config errors (from scorer script stderr)
- Meaning: remote runtime env is incomplete/missing in `score_unimol_property_candidates.py`.
- Retryable: `false`
- Typical action:
  - set `UNIMOL_REMOTE_HOST`
  - set `UNIMOL_REMOTE_PY`
  - set `UNIMOL_REMOTE_TMP_BASE`
  - optionally set `ALLOW_DEFAULT_UNIMOL_REMOTE=1` only for legacy local compatibility.

4. `invalid_env_config`
- Meaning: retry/timeout env vars are invalid.
- Retryable: `false`
- Typical action: set valid numeric values for:
  - `OLED_AGENT_EXTERNAL_SCORER_TIMEOUT_SEC`
  - `OLED_AGENT_EXTERNAL_SCORER_RETRIES`
  - `OLED_AGENT_EXTERNAL_SCORER_BACKOFF_SEC`

5. `empty_candidate_set`
- Meaning: candidate CSV becomes empty after schema normalization.
- Retryable: `false`
- Typical action: verify candidate generation output and input CSV.

6. `external_timeout`
- Meaning: external command timed out after configured retries.
- Retryable: `true`
- Typical action: increase timeout/retries, reduce batch size, or inspect remote runtime.

7. `external_command_failed`
- Meaning: external command exited non-zero after configured retries.
- Retryable: usually `false`, but becomes `true` for transient transport signatures
  (for example ssh/scp return code 255, connection timeout/refused, DNS transient failure).
- Typical action: inspect command stderr and remote toolchain logs.

8. `unexpected_external_error`
- Meaning: unclassified exception surfaced in external scoring path.
- Retryable: `false`
- Typical action: inspect traceback, add explicit classification in adapter.

## Decision summary integration

`agent-run` now writes `decision_summary.json` with machine-readable score decision fields:
- `score_step.adapter`
- `score_step.used_fallback`
- `score_step.fallback_code`
- `score_step.fallback_retryable`
