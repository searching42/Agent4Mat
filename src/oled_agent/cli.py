from __future__ import annotations

import argparse
import json
from pathlib import Path
from json import JSONDecodeError

from oled_agent.agent.planner import DEFAULT_PLANNER_PROVIDER, PlannerValidationError
from oled_agent.agent.request_contract import RequestValidationError, load_and_validate_request_json
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

    run_json_p = sub.add_parser("agent-run-json", help="Plan + execute from structured request JSON")
    run_json_p.add_argument("--request-json", required=True, help="Request JSON path (request.schema.json)")
    run_json_p.add_argument("--workspace-root", default=str(Path.cwd()))
    run_json_p.add_argument("--planner-provider", default=DEFAULT_PLANNER_PROVIDER)
    run_json_p.add_argument("--catalog", default="")

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
        if result.get("status") != "success":
            raise SystemExit(1)
        return


if __name__ == "__main__":
    main()
