from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_EXECUTION_MODE = "full_pipeline"
DEFAULT_OPERATION = "full_pipeline"


def infer_task_draft(*, request_text: str, task_id: str) -> Dict[str, Any]:
    text = str(request_text or "").strip()
    text_l = text.lower()

    property_name: Optional[str] = None
    if "plqy" in text_l:
        property_name = "plqy"
    elif "lambda" in text_l or "发射" in text:
        property_name = "lambda_em"

    target_range = None
    m = re.search(r"(\d{3})\s*nm", text_l)
    if m:
        center = float(m.group(1))
        target_range = f"{center - 12:.1f}-{center + 12:.1f}nm"
    elif property_name == "plqy":
        target_range = "60-100"

    prediction_model = "unimol_lambda_plqy_v1"
    if property_name == "lambda_em":
        prediction_model = "unimol_lambda_em_v1"

    draft: Dict[str, Any] = {
        "version": "2.0",
        "task_id": task_id,
        "request_text": text,
        "execution_mode": DEFAULT_EXECUTION_MODE,
        "operation": DEFAULT_OPERATION,
        "property": property_name,
        "range": target_range,
        "n_structures": 500,
        "constraints": {
            "mw_min": 150.0,
            "mw_max": 700.0,
            "domain_threshold": 0.2,
            "banned_alerts": [],
        },
        "train_data": None,
        "candidate_data": None,
        "prediction_model": prediction_model,
        "model_preferences": {
            "predictor_id": prediction_model,
            "generator_id": "reinvent4_lambda_em_v2",
        },
        "generation_input": {},
        "provenance": {
            "intake_source": "agent-intake",
            "web_evidence": [],
            "web_evidence_json": "",
        },
        "status": "draft",
        "missing_fields": [],
        "questions": [],
        "compatibility_warnings": [],
    }

    missing, questions = compute_missing_questions(draft)
    draft["missing_fields"] = missing
    draft["questions"] = questions
    return draft


def compute_missing_questions(task_payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    questions: List[str] = []

    if not str(task_payload.get("property") or "").strip():
        missing.append("property")
        questions.append("目标性质是什么？例如 plqy / lambda_em / stability")

    if not str(task_payload.get("range") or "").strip():
        missing.append("range")
        questions.append("目标范围是什么？例如 0.5-1.5eV 或 470±12nm")

    n_structures = task_payload.get("n_structures")
    if not isinstance(n_structures, int) or n_structures < 1:
        missing.append("n_structures")
        questions.append("需要输出多少候选结构？")

    if not str(task_payload.get("prediction_model") or "").strip():
        missing.append("prediction_model")
        questions.append("你希望使用哪个预测模型？")

    candidate_data = str(task_payload.get("candidate_data") or "").strip()
    if not candidate_data:
        missing.append("candidate_data")
        questions.append("候选数据来源是什么？本地CSV路径还是数据库关键词？")

    return missing, questions


def task_v2_to_request_payload(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    prop = str(task_payload.get("property") or "").strip() or "plqy"
    raw_range = str(task_payload.get("range") or "").strip()

    target: Dict[str, Any] = {
        "property": prop,
        "objective": "maximize",
    }

    if prop == "lambda_em":
        target["objective"] = "target_window"

    if raw_range:
        r = raw_range.replace("±", "+-")
        if "+-" in r:
            left, right = r.split("+-", 1)
            try:
                c = float(re.findall(r"[-+]?\d*\.?\d+", left)[0])
                s = float(re.findall(r"[-+]?\d*\.?\d+", right)[0])
                target["target_min"] = c - s
                target["target_max"] = c + s
            except Exception:
                pass
        elif "-" in r:
            nums = re.findall(r"[-+]?\d*\.?\d+", r)
            if len(nums) >= 2:
                try:
                    v1 = float(nums[0])
                    v2 = float(nums[1])
                    target["target_min"] = min(v1, v2)
                    target["target_max"] = max(v1, v2)
                except Exception:
                    pass

    if "target_min" not in target and "target_max" not in target:
        if prop == "plqy":
            target["target_value"] = 60.0
        else:
            target["target_value"] = 470.0

    budget_max = task_payload.get("n_structures")
    if not isinstance(budget_max, int) or budget_max < 1:
        budget_max = 500

    constraints = task_payload.get("constraints") if isinstance(task_payload.get("constraints"), dict) else {}
    model_prefs = task_payload.get("model_preferences") if isinstance(task_payload.get("model_preferences"), dict) else {}

    predictor_id = str(model_prefs.get("predictor_id") or task_payload.get("prediction_model") or "").strip()
    generator_id = str(model_prefs.get("generator_id") or "reinvent4_lambda_em_v2").strip()

    payload: Dict[str, Any] = {
        "task_id": str(task_payload.get("task_id") or "task_default"),
        "request_text": str(task_payload.get("request_text") or ""),
        "mode": "train_then_design"
        if str(task_payload.get("execution_mode") or "full_pipeline") == "full_pipeline" and str(task_payload.get("operation") or "full_pipeline") == "train_predictor"
        else "fast_screen",
        "targets": [target],
        "constraints": constraints,
        "budget": {"max_candidates": budget_max},
        "model_preferences": {
            "predictor_id": predictor_id,
            "generator_id": generator_id,
        },
        "generation_input": task_payload.get("generation_input") if isinstance(task_payload.get("generation_input"), dict) else {},
    }

    candidate_data = str(task_payload.get("candidate_data") or "").strip()
    if candidate_data:
        if payload["constraints"] is None or not isinstance(payload["constraints"], dict):
            payload["constraints"] = {}
        payload["constraints"]["candidate_data"] = candidate_data

    train_data = str(task_payload.get("train_data") or "").strip()
    if train_data:
        if payload["constraints"] is None or not isinstance(payload["constraints"], dict):
            payload["constraints"] = {}
        payload["constraints"]["train_data"] = train_data

    return payload


def legacy_request_to_task_v2(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(request_payload.get("task_id") or "task_default")
    request_text = str(request_payload.get("request_text") or "")
    targets = request_payload.get("targets") if isinstance(request_payload.get("targets"), list) else []
    tgt = targets[0] if targets and isinstance(targets[0], dict) else {}

    prop = str(tgt.get("property") or "plqy")
    target_range = ""
    if isinstance(tgt.get("target_min"), (int, float)) and isinstance(tgt.get("target_max"), (int, float)):
        target_range = f"{float(tgt['target_min'])}-{float(tgt['target_max'])}"
    elif isinstance(tgt.get("target_value"), (int, float)):
        target_range = str(float(tgt["target_value"]))

    budget = request_payload.get("budget") if isinstance(request_payload.get("budget"), dict) else {}
    model_prefs = request_payload.get("model_preferences") if isinstance(request_payload.get("model_preferences"), dict) else {}

    task = {
        "version": "2.0",
        "task_id": task_id,
        "request_text": request_text,
        "execution_mode": "full_pipeline",
        "operation": "full_pipeline",
        "property": prop,
        "range": target_range,
        "n_structures": int(budget.get("max_candidates") or 500),
        "constraints": request_payload.get("constraints") if isinstance(request_payload.get("constraints"), dict) else {},
        "train_data": (
            (request_payload.get("constraints") or {}).get("train_data")
            if isinstance(request_payload.get("constraints"), dict)
            else None
        ),
        "candidate_data": (
            (request_payload.get("constraints") or {}).get("candidate_data")
            if isinstance(request_payload.get("constraints"), dict)
            else None
        ),
        "prediction_model": str(model_prefs.get("predictor_id") or ""),
        "model_preferences": {
            "predictor_id": str(model_prefs.get("predictor_id") or ""),
            "generator_id": str(model_prefs.get("generator_id") or ""),
        },
        "generation_input": request_payload.get("generation_input") if isinstance(request_payload.get("generation_input"), dict) else {},
        "provenance": {
            "compatibility": "legacy_request_mapped_to_task_v2",
            "web_evidence": [],
            "web_evidence_json": "",
        },
        "status": "approved",
        "missing_fields": [],
        "questions": [],
        "compatibility_warnings": [
            "legacy request payload auto-mapped to task.v2",
        ],
    }
    return task


def ensure_task_ready_for_approval(task_payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing, _questions = compute_missing_questions(task_payload)
    return (len(missing) == 0, missing)


def ensure_task_ready_for_step(task_payload: Dict[str, Any], *, operation: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not str(task_payload.get("task_id") or "").strip():
        errs.append("task_id is required")
    if not str(operation or "").strip():
        errs.append("operation is required")
    return (len(errs) == 0, errs)


def build_web_query(task_payload: Dict[str, Any]) -> str:
    prop = str(task_payload.get("property") or "materials").strip()
    rng = str(task_payload.get("range") or "").strip()
    req = str(task_payload.get("request_text") or "").strip()
    parts = [x for x in [req, prop, rng] if x]
    return " ".join(parts).strip() or "materials molecule dataset"


def parse_duckduckgo_html_results(html_text: str, topk: int) -> List[Dict[str, str]]:
    # lightweight parser for DuckDuckGo html fallback.
    out: List[Dict[str, str]] = []
    pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    for m in pattern.finditer(html_text or ""):
        href = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2) or "").strip()
        if not href:
            continue
        out.append({"title": title or href, "url": href})
        if len(out) >= max(1, topk):
            break
    return out


def run_duckduckgo_search(*, query: str, topk: int = 5, timeout_sec: float = 8.0) -> List[Dict[str, str]]:
    q = str(query or "").strip()
    if not q:
        return []
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    req = urllib.request.Request(url, headers={"User-Agent": "Agent4Mat/0.1"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return parse_duckduckgo_html_results(body, topk=topk)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
