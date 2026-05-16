from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_optional_path(raw: Any, workspace_root: Path) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _as_float(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except Exception:
        return None
    return value


def _extract_project_memory_text(request_text: str) -> str:
    text = str(request_text or "")
    marker = "Project memory context:"
    idx = text.find(marker)
    if idx < 0:
        return ""
    return text[idx + len(marker) :].strip()


def _host_counter(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    ctr: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        host = str(urlparse(url).hostname or "").strip().lower()
        if not host:
            continue
        ctr[host] += 1
    return dict(ctr)


def _collect_targets(request_payload: Optional[Dict[str, Any]], plan_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    req_targets = request_payload.get("targets") if isinstance(request_payload, dict) and isinstance(request_payload.get("targets"), list) else []
    for item in req_targets:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "property": str(item.get("property") or ""),
                "objective": str(item.get("objective") or ""),
                "target_value": item.get("target_value"),
                "target_min": item.get("target_min"),
                "target_max": item.get("target_max"),
            }
        )
    if out:
        return out
    design = plan_payload.get("design_spec") if isinstance(plan_payload, dict) and isinstance(plan_payload.get("design_spec"), dict) else {}
    plan_targets = design.get("targets") if isinstance(design.get("targets"), list) else []
    for item in plan_targets:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "property": str(item.get("name") or ""),
                "objective": str(item.get("objective") or ""),
                "target_value": item.get("target_center"),
                "target_min": item.get("target_min"),
                "target_max": item.get("target_max"),
            }
        )
    return out


def _collect_constraints(request_payload: Optional[Dict[str, Any]], plan_payload: Optional[Dict[str, Any]], task_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(request_payload, dict) and isinstance(request_payload.get("constraints"), dict):
        return dict(request_payload.get("constraints") or {})
    if isinstance(plan_payload, dict):
        design = plan_payload.get("design_spec")
        if isinstance(design, dict) and isinstance(design.get("constraints"), dict):
            return dict(design.get("constraints") or {})
    if isinstance(task_payload, dict) and isinstance(task_payload.get("constraints"), dict):
        return dict(task_payload.get("constraints") or {})
    return {}


def _collect_model_choice(request_payload: Optional[Dict[str, Any]], plan_payload: Optional[Dict[str, Any]], task_payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if isinstance(request_payload, dict) and isinstance(request_payload.get("model_preferences"), dict):
        prefs = request_payload.get("model_preferences") or {}
        return {
            "predictor_id": str(prefs.get("predictor_id") or ""),
            "generator_id": str(prefs.get("generator_id") or ""),
        }
    if isinstance(plan_payload, dict):
        design = plan_payload.get("design_spec")
        if isinstance(design, dict) and isinstance(design.get("model_choice"), dict):
            mc = design.get("model_choice") or {}
            return {
                "predictor_id": str(mc.get("predictor_id") or ""),
                "generator_id": str(mc.get("generator_id") or ""),
            }
    if isinstance(task_payload, dict):
        prefs = task_payload.get("model_preferences") if isinstance(task_payload.get("model_preferences"), dict) else {}
        predictor_id = str((prefs or {}).get("predictor_id") or task_payload.get("prediction_model") or "")
        return {"predictor_id": predictor_id, "generator_id": str((prefs or {}).get("generator_id") or "")}
    return {"predictor_id": "", "generator_id": ""}


def _derive_key_facts(
    *,
    targets: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    model_choice: Dict[str, str],
    execution_status: str,
) -> List[str]:
    facts: List[str] = []
    for target in targets[:3]:
        prop = str(target.get("property") or "").strip()
        obj = str(target.get("objective") or "").strip()
        if not prop:
            continue
        low = _as_float(target.get("target_min"))
        high = _as_float(target.get("target_max"))
        center = _as_float(target.get("target_value"))
        if low is not None and high is not None:
            facts.append(f"target:{prop}:{obj}:{low:.3f}-{high:.3f}")
        elif center is not None:
            facts.append(f"target:{prop}:{obj}:{center:.3f}")
        else:
            facts.append(f"target:{prop}:{obj}")
    mw_min = _as_float(constraints.get("mw_min"))
    mw_max = _as_float(constraints.get("mw_max"))
    if mw_min is not None or mw_max is not None:
        facts.append(f"constraint:mw:{mw_min if mw_min is not None else ''}:{mw_max if mw_max is not None else ''}")
    banned = constraints.get("banned_alerts") if isinstance(constraints.get("banned_alerts"), list) else []
    if banned:
        sample = ",".join(str(x).strip() for x in banned[:4] if str(x).strip())
        if sample:
            facts.append(f"constraint:banned_alerts:{sample}")
    predictor = str(model_choice.get("predictor_id") or "").strip()
    generator = str(model_choice.get("generator_id") or "").strip()
    if predictor:
        facts.append(f"model:predictor:{predictor}")
    if generator:
        facts.append(f"model:generator:{generator}")
    facts.append(f"execution_status:{execution_status}")
    return facts


def _tokenize(text: str) -> List[str]:
    raw = str(text or "").lower()
    out: List[str] = []
    current = []
    for ch in raw:
        if ch.isalnum():
            current.append(ch)
            continue
        if current:
            token = "".join(current).strip()
            if len(token) >= 2:
                out.append(token)
            current = []
    if current:
        token = "".join(current).strip()
        if len(token) >= 2:
            out.append(token)
    # Keep order but drop duplicates.
    seen = set()
    uniq: List[str] = []
    for token in out:
        if token in seen:
            continue
        seen.add(token)
        uniq.append(token)
    return uniq


def _memory_entry_score(*, request_tokens: List[str], entry: Dict[str, Any]) -> int:
    if not request_tokens:
        return 0
    hay = " ".join(
        [
            str(entry.get("request_text_head") or ""),
            " ".join(str(x or "") for x in (entry.get("key_facts") or [])),
            str(entry.get("property") or ""),
        ]
    ).lower()
    if not hay:
        return 0
    score = 0
    for token in request_tokens:
        if token and token in hay:
            score += 2
    if str(entry.get("execution_status") or "") == "success":
        score += 1
    return score


def build_memory_context(
    *,
    task_id: str,
    execution_mode: str,
    run_label: str,
    workspace_root: Path,
    execution_payload: Dict[str, Any],
    tool_state: Optional[Dict[str, Any]],
    request_payload: Optional[Dict[str, Any]],
    plan_payload: Optional[Dict[str, Any]],
    task_payload: Optional[Dict[str, Any]],
    web_evidence_path: Optional[Path] = None,
    previous_memory_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    execution = execution_payload if isinstance(execution_payload, dict) else {}
    records = execution.get("records", []) if isinstance(execution.get("records"), list) else []
    records = [r for r in records if isinstance(r, dict)]
    execution_status = str(execution.get("status") or "").strip()
    if execution_status not in ("success", "failed"):
        execution_status = "failed"

    request_payload = request_payload if isinstance(request_payload, dict) else {}
    plan_payload = plan_payload if isinstance(plan_payload, dict) else {}
    task_payload = task_payload if isinstance(task_payload, dict) else {}
    tool_state = tool_state if isinstance(tool_state, dict) else {}

    request_text = str(request_payload.get("request_text") or "")
    if not request_text:
        request_text = str(plan_payload.get("design_spec", {}).get("request_text") or task_payload.get("request_text") or "")

    targets = _collect_targets(request_payload, plan_payload)
    constraints = _collect_constraints(request_payload, plan_payload, task_payload)
    model_choice = _collect_model_choice(request_payload, plan_payload, task_payload)

    web_evidence = _safe_load_json(web_evidence_path)
    web_rows = web_evidence.get("results") if isinstance(web_evidence, dict) and isinstance(web_evidence.get("results"), list) else []
    web_host_counts = _host_counter(web_rows if isinstance(web_rows, list) else [])

    candidate_csv = _resolve_optional_path(tool_state.get("candidate_csv"), workspace_root)
    scored_csv = _resolve_optional_path(tool_state.get("scored_csv"), workspace_root)
    final_output = _resolve_optional_path(tool_state.get("final_output"), workspace_root)
    selected_datasets = tool_state.get("selected_datasets") if isinstance(tool_state.get("selected_datasets"), list) else []

    failed_tools = [str(r.get("name") or "") for r in records if str(r.get("status") or "") != "success" and str(r.get("name") or "").strip()]
    tool_sequence = [str(r.get("name") or "") for r in records if str(r.get("name") or "").strip()]
    adapters = sorted(
        {
            str((r.get("result") or {}).get("adapter") or "").strip()
            for r in records
            if isinstance(r.get("result"), dict) and str((r.get("result") or {}).get("adapter") or "").strip()
        }
    )

    memory_note = _extract_project_memory_text(request_text)
    key_facts = _derive_key_facts(
        targets=targets,
        constraints=constraints,
        model_choice=model_choice,
        execution_status=execution_status,
    )

    prev = previous_memory_context if isinstance(previous_memory_context, dict) else {}
    prev_digest = {
        "exists": bool(prev),
        "generated_at": str(prev.get("generated_at") or ""),
        "execution_status": str(prev.get("execution_status") or ""),
        "key_facts_head": (prev.get("key_facts") or [])[:5] if isinstance(prev.get("key_facts"), list) else [],
    }

    return {
        "schema_version": "1.0.0",
        "generated_at": _now_iso(),
        "task_id": str(task_id or ""),
        "run_label": str(run_label or ""),
        "execution_mode": str(execution_mode or ""),
        "execution_status": execution_status,
        "request_snapshot": {
            "request_text": request_text,
            "project_memory_note": memory_note,
            "mode": str(request_payload.get("mode") or plan_payload.get("design_spec", {}).get("mode") or ""),
            "targets": targets,
            "constraints": constraints,
            "model_choice": model_choice,
            "candidate_data": str(task_payload.get("candidate_data") or ""),
            "train_data": str(task_payload.get("train_data") or ""),
        },
        "evidence_snapshot": {
            "web_evidence_present": bool(web_evidence),
            "web_result_count": len(web_rows) if isinstance(web_rows, list) else 0,
            "web_host_counts": web_host_counts,
            "time_range": str((web_evidence or {}).get("time_range") or "") if isinstance(web_evidence, dict) else "",
            "query_effective": str((web_evidence or {}).get("query_effective") or "") if isinstance(web_evidence, dict) else "",
        },
        "runtime_snapshot": {
            "record_count": len(records),
            "tool_sequence": tool_sequence,
            "failed_tools": failed_tools,
            "adapters": adapters,
            "selected_datasets": [str(x) for x in selected_datasets if str(x).strip()],
            "artifacts": {
                "candidate_csv": str(candidate_csv) if candidate_csv is not None else "",
                "scored_csv": str(scored_csv) if scored_csv is not None else "",
                "final_output": str(final_output) if final_output is not None else "",
            },
        },
        "key_facts": key_facts,
        "carry_over": prev_digest,
    }


def retrieve_memory_hints(
    *,
    workspace_root: Path,
    request_text: str,
    current_task_id: str = "",
    topk: int = 5,
) -> Dict[str, Any]:
    index_path = (workspace_root / "runs" / "agent" / "_memory" / "memory_index.json").resolve()
    topk = max(1, int(topk))
    if not index_path.exists():
        return {
            "status": "missing",
            "index_path": str(index_path),
            "query": str(request_text or ""),
            "matches": [],
            "suggested_candidate_data": "",
            "prompt_context": "",
        }

    payload = _safe_load_json(index_path)
    rows = payload.get("entries") if isinstance(payload, dict) and isinstance(payload.get("entries"), list) else []
    request_tokens = _tokenize(str(request_text or ""))
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("task_id") or "") == str(current_task_id or ""):
            continue
        score = _memory_entry_score(request_tokens=request_tokens, entry=row)
        if score <= 0:
            continue
        item = dict(row)
        item["_score"] = score
        ranked.append(item)

    ranked.sort(
        key=lambda r: (
            int(r.get("_score") or 0),
            str(r.get("generated_at") or ""),
        ),
        reverse=True,
    )
    selected = ranked[:topk]

    candidate_counts: Counter[str] = Counter()
    for row in selected:
        cand = str(row.get("candidate_data") or "").strip()
        if not cand:
            continue
        if str(row.get("execution_status") or "") != "success":
            continue
        candidate_counts[cand] += 1
    suggested_candidate_data = candidate_counts.most_common(1)[0][0] if candidate_counts else ""

    lines: List[str] = []
    for row in selected:
        task_id = str(row.get("task_id") or "")
        run_label = str(row.get("run_label") or "")
        key_facts = row.get("key_facts") if isinstance(row.get("key_facts"), list) else []
        key_text = "; ".join(str(x) for x in key_facts[:3] if str(x).strip())
        candidate_data = str(row.get("candidate_data") or "").strip()
        line = f"- task={task_id} run={run_label}"
        if key_text:
            line += f" facts={key_text}"
        if candidate_data:
            line += f" candidate_data={candidate_data}"
        lines.append(line)

    return {
        "status": "pass",
        "index_path": str(index_path),
        "query": str(request_text or ""),
        "matches": [
            {
                "task_id": str(row.get("task_id") or ""),
                "run_label": str(row.get("run_label") or ""),
                "generated_at": str(row.get("generated_at") or ""),
                "execution_mode": str(row.get("execution_mode") or ""),
                "execution_status": str(row.get("execution_status") or ""),
                "request_text_head": str(row.get("request_text_head") or ""),
                "property": str(row.get("property") or ""),
                "candidate_data": str(row.get("candidate_data") or ""),
                "train_data": str(row.get("train_data") or ""),
                "key_facts": row.get("key_facts") if isinstance(row.get("key_facts"), list) else [],
                "score": int(row.get("_score") or 0),
                "memory_context_path": str(row.get("memory_context_path") or ""),
            }
            for row in selected
        ],
        "suggested_candidate_data": suggested_candidate_data,
        "prompt_context": "\n".join(lines),
    }


def update_memory_index(
    *,
    workspace_root: Path,
    memory_context: Dict[str, Any],
    memory_context_path: Optional[Path] = None,
) -> Path:
    root = (workspace_root / "runs" / "agent" / "_memory").resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "memory_index.json"
    payload: Dict[str, Any] = {"schema_version": "1.0.0", "updated_at": _now_iso(), "entries": []}
    if index_path.exists():
        try:
            old = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if isinstance(old, dict) and isinstance(old.get("entries"), list):
            payload["entries"] = [x for x in old.get("entries", []) if isinstance(x, dict)]

    task_id = str(memory_context.get("task_id") or "")
    run_label = str(memory_context.get("run_label") or "")
    entries = [x for x in payload["entries"] if not (str(x.get("task_id") or "") == task_id and str(x.get("run_label") or "") == run_label)]
    request_snapshot = memory_context.get("request_snapshot") if isinstance(memory_context.get("request_snapshot"), dict) else {}
    targets = request_snapshot.get("targets") if isinstance(request_snapshot.get("targets"), list) else []
    first_target = targets[0] if targets and isinstance(targets[0], dict) else {}
    property_name = str(first_target.get("property") or "")
    row = {
        "task_id": task_id,
        "run_label": run_label,
        "generated_at": str(memory_context.get("generated_at") or ""),
        "execution_mode": str(memory_context.get("execution_mode") or ""),
        "execution_status": str(memory_context.get("execution_status") or ""),
        "request_text_head": str(request_snapshot.get("request_text") or "")[:180],
        "property": property_name,
        "candidate_data": str(request_snapshot.get("candidate_data") or ""),
        "train_data": str(request_snapshot.get("train_data") or ""),
        "key_facts": (memory_context.get("key_facts") or [])[:8] if isinstance(memory_context.get("key_facts"), list) else [],
        "memory_context_path": str(memory_context_path.resolve()) if isinstance(memory_context_path, Path) else "",
    }
    entries.insert(0, row)
    payload["entries"] = entries[:500]
    payload["updated_at"] = _now_iso()
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index_path
