from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from json import JSONDecodeError

from oled_agent.agent.intake import approve_task, run_intake
from oled_agent.agent.planner import DEFAULT_PLANNER_PROVIDER, PlannerValidationError
from oled_agent.agent.request_contract import (
    RequestValidationError,
    load_and_validate_request_json,
    validate_step_request_payload,
    validate_task_v2_payload,
)
from oled_agent.agent.step_runner import run_step, run_step_from_request_payload
from oled_agent.agent.task_v2 import compute_missing_questions, legacy_request_to_task_v2, task_v2_to_request_payload
from oled_agent.agent.session import (
    execute_request,
    execute_request_from_payload,
    plan_request,
    plan_request_from_payload,
)
from oled_agent.diagnostics import run_doctor, run_external_connectivity_debug, run_external_preflight, run_llm_connectivity
from oled_agent.runner import run_pipeline
from oled_agent.smoke import run_smoke


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="oled-agent CLI")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run pipeline from config")
    run_p.add_argument("--config", required=True, help="Pipeline config JSON path")
    run_p.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root (default: current directory)",
    )

    doctor_p = sub.add_parser("doctor", help="Run environment diagnostics")
    doctor_p.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root (default: current directory)",
    )
    doctor_p.add_argument("--json-out", default="", help="Optional path to write doctor JSON report")
    doctor_p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit if any warn/fail)",
    )

    external_p = sub.add_parser("external-preflight", help="Preflight check for external Uni-Mol scorer chain")
    external_p.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root (default: current directory)",
    )

    ext_dbg_p = sub.add_parser(
        "external-connectivity-debug",
        help="Extended external scorer connectivity diagnostics with machine-readable summary",
    )
    ext_dbg_p.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root (default: current directory)",
    )
    ext_dbg_p.add_argument("--json-out", default="", help="Optional path to write debug JSON report")

    llm_conn_p = sub.add_parser(
        "llm-connectivity",
        help="Check LLM planner connectivity for command/backend mode",
    )
    llm_conn_p.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root (default: current directory)",
    )
    llm_conn_p.add_argument("--catalog", default="", help="Optional model catalog path used by command probe")
    llm_conn_p.add_argument("--json-out", default="", help="Optional path to write llm connectivity JSON report")

    smoke_p = sub.add_parser("smoke", help="Run minimal smoke pipeline")
    smoke_p.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root (default: current directory)",
    )
    smoke_p.add_argument("--config", default="", help="Optional smoke config path")

    plan_p = sub.add_parser("agent-plan", help="Build a structured agent plan from user request")
    plan_p.add_argument("--request", required=True, help="User request text")
    plan_p.add_argument("--task-id", required=True, help="Task id")
    plan_p.add_argument("--workspace-root", default=str(Path.cwd()))
    plan_p.add_argument("--predictor-id", default="")
    plan_p.add_argument("--generator-id", default="")
    plan_p.add_argument("--mode", default="fast_screen", choices=["fast_screen", "train_then_design"])
    plan_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    plan_p.add_argument("--catalog", default="")

    plan_json_p = sub.add_parser("agent-plan-json", help="Build plan from structured request JSON")
    plan_json_p.add_argument("--request-json", required=True, help="Request JSON path (request.schema.json)")
    plan_json_p.add_argument("--workspace-root", default=str(Path.cwd()))
    plan_json_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    plan_json_p.add_argument("--catalog", default="")

    run_agent_p = sub.add_parser("agent-run", help="Plan + execute toolchain for a user request")
    run_agent_p.add_argument("--request", required=True, help="User request text")
    run_agent_p.add_argument("--task-id", required=True, help="Task id")
    run_agent_p.add_argument("--workspace-root", default=str(Path.cwd()))
    run_agent_p.add_argument("--predictor-id", default="")
    run_agent_p.add_argument("--generator-id", default="")
    run_agent_p.add_argument("--mode", default="fast_screen", choices=["fast_screen", "train_then_design"])
    run_agent_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    run_agent_p.add_argument("--catalog", default="")
    run_agent_p.add_argument(
        "--require-real-adapters",
        action="store_true",
        help="Fail if fallback/local adapters are used",
    )
    run_agent_p.add_argument(
        "--guardrails-strict",
        action="store_true",
        help="Fail when guardrails strict_status is not pass",
    )

    run_json_p = sub.add_parser("agent-run-json", help="Plan + execute from structured request JSON")
    run_json_p.add_argument("--request-json", required=True, help="Request JSON path (request.schema.json)")
    run_json_p.add_argument("--workspace-root", default=str(Path.cwd()))
    run_json_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    run_json_p.add_argument("--catalog", default="")
    run_json_p.add_argument(
        "--require-real-adapters",
        action="store_true",
        help="Fail if fallback/local adapters are used",
    )
    run_json_p.add_argument(
        "--guardrails-strict",
        action="store_true",
        help="Fail when guardrails strict_status is not pass",
    )

    intake_p = sub.add_parser("agent-intake", help="Build task.v2 draft and missing info questions")
    intake_p.add_argument("--request", required=True, help="User request text")
    intake_p.add_argument("--task-id", required=True, help="Task id")
    intake_p.add_argument("--workspace-root", default=str(Path.cwd()))
    intake_p.add_argument("--disable-web-search", action="store_true")
    intake_p.add_argument("--web-topk", type=int, default=5)

    approve_p = sub.add_parser("agent-approve", help="Approve task.v2 and generate task.json + plan.md")
    approve_p.add_argument("--task-json", required=True, help="Task.v2 JSON path")
    approve_p.add_argument("--workspace-root", default=str(Path.cwd()))
    approve_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    approve_p.add_argument("--catalog", default="")

    run_step_p = sub.add_parser("agent-run-step", help="Run a single operation from task.v2")
    run_step_p.add_argument("--task-json", required=True, help="Task.v2 JSON path")
    run_step_p.add_argument("--operation", required=True)
    run_step_p.add_argument("--workspace-root", default=str(Path.cwd()))
    run_step_p.add_argument("--catalog", default="")
    run_step_p.add_argument("--args-json", default="", help="Optional operation args JSON object string")
    run_step_p.add_argument(
        "--require-real-adapters",
        action="store_true",
        help="Fail if fallback/local adapters are used",
    )
    run_step_p.add_argument(
        "--guardrails-strict",
        action="store_true",
        help="Fail when guardrails strict_status is not pass",
    )

    run_step_json_p = sub.add_parser("agent-run-step-json", help="Run single operation from step request JSON")
    run_step_json_p.add_argument("--step-request-json", required=True)
    run_step_json_p.add_argument("--workspace-root", default=str(Path.cwd()))
    run_step_json_p.add_argument("--catalog", default="")
    run_step_json_p.add_argument(
        "--require-real-adapters",
        action="store_true",
        help="Fail if fallback/local adapters are used",
    )
    run_step_json_p.add_argument(
        "--guardrails-strict",
        action="store_true",
        help="Fail when guardrails strict_status is not pass",
    )

    resume_p = sub.add_parser("agent-resume", help="Resume task from task.json or request.json under runs/agent/<task_id>")
    resume_p.add_argument("--task-id", required=True)
    resume_p.add_argument("--workspace-root", default=str(Path.cwd()))
    resume_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    resume_p.add_argument("--catalog", default="")
    resume_p.add_argument(
        "--require-real-adapters",
        action="store_true",
        help="Fail if fallback/local adapters are used",
    )
    resume_p.add_argument(
        "--guardrails-strict",
        action="store_true",
        help="Fail when guardrails strict_status is not pass",
    )
    resume_p.add_argument("--candidate-data", default="", help="Override task candidate_data before resume")
    resume_p.add_argument("--train-data", default="", help="Override task train_data before resume")
    resume_p.add_argument("--prediction-model", default="", help="Override task prediction_model before resume")
    resume_p.add_argument("--property", default="", help="Override task property before resume")
    resume_p.add_argument("--range", default="", help="Override task range before resume")
    resume_p.add_argument("--n-structures", type=int, default=0, help="Override task n_structures before resume")
    resume_p.add_argument("--predictor-id", default="", help="Override model_preferences.predictor_id before resume")
    resume_p.add_argument("--generator-id", default="", help="Override model_preferences.generator_id before resume")

    return p.parse_args()


def _print_doctor(report: dict) -> None:
    summary = report.get("summary", {})
    print(
        "DOCTOR "
        f"overall={report.get('overall')} "
        f"pass={summary.get('pass', 0)} "
        f"warn={summary.get('warn', 0)} "
        f"fail={summary.get('fail', 0)}"
    )
    for item in report.get("checks", []):
        print(f"[{item['status'].upper()}] {item['name']}: {item['message']}")


def _resolve_and_load_request_json(payload_path: str, workspace_root: Path) -> dict:
    try:
        resolved = Path(payload_path).resolve()
        return load_and_validate_request_json(
            payload_path=resolved,
            workspace_root=workspace_root,
        )
    except RequestValidationError:
        raise
    except (FileNotFoundError, PermissionError, OSError, JSONDecodeError) as exc:
        raise RequestValidationError(str(exc)) from exc


def _resolve_and_load_json(payload_path: str) -> dict:
    p = Path(payload_path).resolve()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RequestValidationError(str(exc)) from exc


def _assert_no_fallback(result_payload: dict) -> None:
    exec_path_raw = str(result_payload.get("execution_path") or "").strip()
    if not exec_path_raw:
        print("[FAIL] require-real-adapters missing execution_path")
        raise SystemExit(3)
    exec_path = Path(exec_path_raw).resolve()
    if not exec_path.exists():
        print(f"[FAIL] require-real-adapters execution_path not found: {exec_path}")
        raise SystemExit(3)
    execution = json.loads(exec_path.read_text(encoding="utf-8"))
    records = execution.get("records", []) if isinstance(execution, dict) else []
    forbidden = {
        "local_deterministic_fallback",
        "stub_generator",
        "dataset_stub_retrieval",
        "train_data_stub_builder",
        "local_cleaning_v1",
    }
    for rec in records:
        if not isinstance(rec, dict):
            continue
        res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        adapter = str(res.get("adapter") or "")
        if adapter in forbidden:
            print(f"[FAIL] require-real-adapters hit fallback/stub adapter: {adapter}")
            raise SystemExit(3)


def _is_true_env(name: str, default: str = "0") -> bool:
    text = str(os.environ.get(name, default) or "").strip().lower()
    return text in ("1", "true", "yes", "on")


def _assert_guardrails(result_payload: dict, *, strict: bool) -> None:
    path_raw = str(result_payload.get("guardrails_report_path") or "").strip()
    if not path_raw:
        path_raw = str(result_payload.get("logging_guardrails_report_path") or "").strip()
    if not path_raw:
        print("[FAIL] guardrails report path is missing in result payload")
        raise SystemExit(4)
    path = Path(path_raw).resolve()
    if not path.exists():
        print(f"[FAIL] guardrails report path not found: {path}")
        raise SystemExit(4)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(f"[FAIL] guardrails report payload is not object: {path}")
        raise SystemExit(4)
    if strict:
        status = str(payload.get("strict_status") or "").strip()
        if status != "pass":
            print(
                "[FAIL] guardrails strict mode blocked execution: "
                f"strict_status={status} strict_blocking_checks={payload.get('strict_blocking_checks', [])}"
            )
            raise SystemExit(4)
        return
    status = str(payload.get("status") or "").strip()
    if status == "fail":
        print(
            "[FAIL] guardrails blocked execution: "
            f"status={status} blocking_checks={payload.get('blocking_checks', [])}"
        )
        raise SystemExit(4)
        if res.get("fallback_error"):
            print(f"[FAIL] require-real-adapters found fallback_error in tool: {rec.get('name')}")
            raise SystemExit(3)

    decision_path_raw = str(result_payload.get("decision_summary_path") or "").strip()
    if not decision_path_raw:
        return
    decision_path = Path(decision_path_raw).resolve()
    if not decision_path.exists():
        print(f"[FAIL] require-real-adapters decision_summary_path not found: {decision_path}")
        raise SystemExit(3)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    for step_key in ("score_step", "inference_step"):
        step = decision.get(step_key)
        if not isinstance(step, dict):
            continue
        used_fallback = bool(step.get("used_fallback", False))
        fallback_code = str(step.get("fallback_code") or "")
        fallback_reason = str(step.get("fallback_reason") or "")
        fallback_error = step.get("fallback_error") if isinstance(step.get("fallback_error"), dict) else {}
        if used_fallback or fallback_code or fallback_reason or fallback_error:
            adapter = str(step.get("adapter") or "")
            print(
                "[FAIL] require-real-adapters decision summary indicates fallback: "
                f"{step_key} adapter={adapter} used_fallback={used_fallback} "
                f"fallback_code={fallback_code}"
            )
            raise SystemExit(3)


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _has_resume_overrides(args: argparse.Namespace) -> bool:
    if str(getattr(args, "candidate_data", "") or "").strip():
        return True
    if str(getattr(args, "train_data", "") or "").strip():
        return True
    if str(getattr(args, "prediction_model", "") or "").strip():
        return True
    if str(getattr(args, "property", "") or "").strip():
        return True
    if str(getattr(args, "range", "") or "").strip():
        return True
    if int(getattr(args, "n_structures", 0) or 0) > 0:
        return True
    if str(getattr(args, "predictor_id", "") or "").strip():
        return True
    if str(getattr(args, "generator_id", "") or "").strip():
        return True
    return False


def _apply_resume_overrides(task_payload: dict, args: argparse.Namespace) -> dict:
    out = dict(task_payload or {})
    cand = str(getattr(args, "candidate_data", "") or "").strip()
    if cand:
        out["candidate_data"] = cand
    train = str(getattr(args, "train_data", "") or "").strip()
    if train:
        out["train_data"] = train
    pred_model = str(getattr(args, "prediction_model", "") or "").strip()
    if pred_model:
        out["prediction_model"] = pred_model
    prop = str(getattr(args, "property", "") or "").strip()
    if prop:
        out["property"] = prop
    rng = str(getattr(args, "range", "") or "").strip()
    if rng:
        out["range"] = rng
    n_structures = int(getattr(args, "n_structures", 0) or 0)
    if n_structures > 0:
        out["n_structures"] = n_structures

    prefs = out.get("model_preferences")
    if not isinstance(prefs, dict):
        prefs = {}
    predictor_id = str(getattr(args, "predictor_id", "") or "").strip()
    generator_id = str(getattr(args, "generator_id", "") or "").strip()
    if predictor_id:
        prefs["predictor_id"] = predictor_id
    if generator_id:
        prefs["generator_id"] = generator_id
    if pred_model and not predictor_id:
        prefs["predictor_id"] = pred_model
    if prefs:
        out["model_preferences"] = prefs

    missing, questions = compute_missing_questions(out)
    out["missing_fields"] = missing
    out["questions"] = questions
    if str(out.get("status") or "").strip() != "approved":
        out["status"] = "need_user_input" if missing else "draft"
    return out


def main() -> None:
    args = parse_args()

    if args.command == "run":
        config_path = Path(args.config).resolve()
        workspace_root = Path(args.workspace_root).resolve()
        manifest = run_pipeline(config_path=config_path, workspace_root=workspace_root)
        print(f"MANIFEST={manifest}")
        return

    if args.command == "doctor":
        workspace_root = Path(args.workspace_root).resolve()
        json_out = Path(args.json_out).resolve() if args.json_out else None
        report = run_doctor(workspace_root=workspace_root, json_out=json_out, strict=args.strict)
        _print_doctor(report)
        if json_out is not None:
            print(f"DOCTOR_JSON={json_out}")
        raise SystemExit(int(report.get("exit_code", 1)))

    if args.command == "external-preflight":
        workspace_root = Path(args.workspace_root).resolve()
        report = run_external_preflight(workspace_root=workspace_root)
        _print_doctor(report)
        raise SystemExit(int(report.get("exit_code", 1)))

    if args.command == "external-connectivity-debug":
        workspace_root = Path(args.workspace_root).resolve()
        json_out = Path(args.json_out).resolve() if args.json_out else None
        report = run_external_connectivity_debug(workspace_root=workspace_root, json_out=json_out)
        _print_doctor(report)
        print(json.dumps(report.get("connectivity", {}), ensure_ascii=False, indent=2))
        if json_out is not None:
            print(f"DEBUG_JSON={json_out}")
        raise SystemExit(int(report.get("exit_code", 1)))

    if args.command == "llm-connectivity":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        json_out = Path(args.json_out).resolve() if args.json_out else None
        report = run_llm_connectivity(workspace_root=workspace_root, catalog_path=catalog, json_out=json_out)
        _print_doctor(report)
        print(json.dumps(report.get("connectivity", {}), ensure_ascii=False, indent=2))
        if json_out is not None:
            print(f"LLM_CONNECTIVITY_JSON={json_out}")
        raise SystemExit(int(report.get("exit_code", 1)))

    if args.command == "smoke":
        workspace_root = Path(args.workspace_root).resolve()
        cfg = Path(args.config).resolve() if args.config else None
        result = run_smoke(workspace_root=workspace_root, config_path=cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") != "success" or not result.get("final_output_exists"):
            raise SystemExit(1)
        return

    if args.command == "agent-plan":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        try:
            plan = plan_request(
                workspace_root=workspace_root,
                user_request=args.request,
                task_id=args.task_id,
                predictor_id=args.predictor_id,
                generator_id=args.generator_id,
                mode=args.mode,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
            )
        except PlannerValidationError as exc:
            print(f"[FAIL] invalid request args: {exc}")
            raise SystemExit(2)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.command == "agent-intake":
        workspace_root = Path(args.workspace_root).resolve()
        result = run_intake(
            workspace_root=workspace_root,
            task_id=args.task_id,
            request_text=args.request,
            enable_web_search=not bool(args.disable_web_search),
            web_topk=max(1, int(args.web_topk)),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if str(result.get("status")) == "need_user_input":
            raise SystemExit(2)
        return

    if args.command == "agent-approve":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        task = _resolve_and_load_json(args.task_json)
        try:
            validate_task_v2_payload(task, workspace_root)
        except RequestValidationError as exc:
            print(f"[FAIL] invalid task json: {exc}")
            raise SystemExit(2)
        result = approve_task(
            workspace_root=workspace_root,
            task_payload=task,
            planner_provider=args.planner_provider,
            catalog_path=catalog,
            plan_fn=plan_request_from_payload,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if str(result.get("status")) != "approved":
            raise SystemExit(2)
        return

    if args.command == "agent-plan-json":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        try:
            payload = _resolve_and_load_request_json(args.request_json, workspace_root)
            plan = plan_request_from_payload(
                workspace_root=workspace_root,
                request_payload=payload,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
            )
        except (RequestValidationError, PlannerValidationError) as exc:
            print(f"[FAIL] invalid request json: {exc}")
            raise SystemExit(2)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.command == "agent-run":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        try:
            result = execute_request(
                workspace_root=workspace_root,
                user_request=args.request,
                task_id=args.task_id,
                predictor_id=args.predictor_id,
                generator_id=args.generator_id,
                mode=args.mode,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
            )
        except PlannerValidationError as exc:
            print(f"[FAIL] invalid request args: {exc}")
            raise SystemExit(2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if bool(getattr(args, "require_real_adapters", False)):
            _assert_no_fallback(result)
        guardrails_strict = bool(getattr(args, "guardrails_strict", False)) or _is_true_env("OLED_AGENT_GUARDRAILS_STRICT")
        if guardrails_strict:
            _assert_guardrails(result, strict=True)
        if result.get("status") != "success":
            raise SystemExit(1)
        return

    if args.command == "agent-run-json":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        try:
            payload = _resolve_and_load_request_json(args.request_json, workspace_root)
            result = execute_request_from_payload(
                workspace_root=workspace_root,
                request_payload=payload,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
            )
        except (RequestValidationError, PlannerValidationError) as exc:
            print(f"[FAIL] invalid request json: {exc}")
            raise SystemExit(2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if bool(getattr(args, "require_real_adapters", False)):
            _assert_no_fallback(result)
        guardrails_strict = bool(getattr(args, "guardrails_strict", False)) or _is_true_env("OLED_AGENT_GUARDRAILS_STRICT")
        if guardrails_strict:
            _assert_guardrails(result, strict=True)
        if result.get("status") != "success":
            raise SystemExit(1)
        return

    if args.command == "agent-run-step":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else (workspace_root / "configs/models/catalog.json").resolve()
        task = _resolve_and_load_json(args.task_json)
        try:
            validate_task_v2_payload(task, workspace_root)
        except RequestValidationError as exc:
            print(f"[FAIL] invalid task json: {exc}")
            raise SystemExit(2)
        args_override = {}
        if str(args.args_json or "").strip():
            try:
                args_override = json.loads(args.args_json)
                if not isinstance(args_override, dict):
                    raise ValueError("args-json must be object")
            except Exception as exc:
                print(f"[FAIL] invalid args-json: {exc}")
                raise SystemExit(2)
        result = run_step(
            workspace_root=workspace_root,
            task_payload=task,
            operation=str(args.operation or ""),
            args_override=args_override,
            catalog_path=catalog,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if bool(getattr(args, "require_real_adapters", False)):
            _assert_no_fallback(result)
        guardrails_strict = bool(getattr(args, "guardrails_strict", False)) or _is_true_env("OLED_AGENT_GUARDRAILS_STRICT")
        if guardrails_strict:
            _assert_guardrails(result, strict=True)
        if result.get("status") != "success":
            raise SystemExit(1)
        return

    if args.command == "agent-run-step-json":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else (workspace_root / "configs/models/catalog.json").resolve()
        payload = _resolve_and_load_json(args.step_request_json)
        try:
            validate_step_request_payload(payload, workspace_root)
        except RequestValidationError as exc:
            print(f"[FAIL] invalid step request json: {exc}")
            raise SystemExit(2)
        result = run_step_from_request_payload(
            workspace_root=workspace_root,
            step_request_payload=payload,
            catalog_path=catalog,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if bool(getattr(args, "require_real_adapters", False)):
            _assert_no_fallback(result)
        guardrails_strict = bool(getattr(args, "guardrails_strict", False)) or _is_true_env("OLED_AGENT_GUARDRAILS_STRICT")
        if guardrails_strict:
            _assert_guardrails(result, strict=True)
        if result.get("status") != "success":
            raise SystemExit(1)
        return

    if args.command == "agent-resume":
        workspace_root = Path(args.workspace_root).resolve()
        catalog = Path(args.catalog).resolve() if args.catalog else None
        run_dir = (workspace_root / "runs" / "agent" / args.task_id).resolve()
        if not run_dir.exists():
            print(f"[FAIL] task run directory not found: {run_dir}")
            raise SystemExit(2)
        task_json = run_dir / "task.json"
        req_json = run_dir / "request_from_task.json"
        draft_json = run_dir / "task.draft.json"

        def _execute_resume(req_payload: dict) -> None:
            result_exec = execute_request_from_payload(
                workspace_root=workspace_root,
                request_payload=req_payload,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
                resume_existing=True,
            )
            print(json.dumps(result_exec, ensure_ascii=False, indent=2))
            if bool(getattr(args, "require_real_adapters", False)):
                _assert_no_fallback(result_exec)
            guardrails_strict = bool(getattr(args, "guardrails_strict", False)) or _is_true_env("OLED_AGENT_GUARDRAILS_STRICT")
            if guardrails_strict:
                _assert_guardrails(result_exec, strict=True)
            if result_exec.get("status") != "success":
                raise SystemExit(1)
            return

        if task_json.exists():
            task = _resolve_and_load_json(str(task_json))
            if _has_resume_overrides(args):
                task = _apply_resume_overrides(task, args)
                _dump_json(task_json, task)
            result = approve_task(
                workspace_root=workspace_root,
                task_payload=task,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
                plan_fn=plan_request_from_payload,
            )
            if result.get("status") != "approved":
                print(json.dumps(result, ensure_ascii=False, indent=2))
                raise SystemExit(2)
            if not req_json.exists():
                _dump_json(req_json, task_v2_to_request_payload(task))
            req_payload = _resolve_and_load_json(str(req_json))
            _execute_resume(req_payload)
            return

        if draft_json.exists():
            task = _resolve_and_load_json(str(draft_json))
            if _has_resume_overrides(args):
                task = _apply_resume_overrides(task, args)
                _dump_json(draft_json, task)
            result = approve_task(
                workspace_root=workspace_root,
                task_payload=task,
                planner_provider=args.planner_provider,
                catalog_path=catalog,
                plan_fn=plan_request_from_payload,
            )
            if result.get("status") != "approved":
                print(json.dumps(result, ensure_ascii=False, indent=2))
                raise SystemExit(2)
            if not req_json.exists():
                req_json = run_dir / "request_from_task.json"
            req_payload = _resolve_and_load_json(str(req_json))
            _execute_resume(req_payload)
            return

        # fallback: resume from request.json of legacy agent-run-json
        legacy_req = run_dir / "request.json"
        if legacy_req.exists():
            req_payload = _resolve_and_load_json(str(legacy_req))
            task_v2 = legacy_request_to_task_v2(req_payload)
            out_task = run_dir / "task.json"
            _dump_json(out_task, task_v2)
            _execute_resume(req_payload)
            return
        print(f"[FAIL] no resumable task.json/task.draft.json/request.json found under: {run_dir}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
