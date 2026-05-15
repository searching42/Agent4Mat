from __future__ import annotations

import csv
import io
import json
import os
import builtins
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oled_agent.agent.session import execute_request
from oled_agent.agent.request_contract import (
    RequestValidationError,
    _validate_via_jsonschema,
    validate_decision_summary_payload,
    validate_filtering_report_payload,
    validate_model_report_payload,
    validate_plan_payload,
    validate_request_payload,
    validate_step_request_payload,
    validate_task_v2_payload,
    validate_data_report_payload,
    validate_task_state_payload,
)
from oled_agent.agent.task_v2 import legacy_request_to_task_v2
from oled_agent.agent.tool_contracts import build_plan_tool_call_item_schema
from oled_agent.diagnostics import run_external_connectivity_debug, run_external_preflight, run_llm_connectivity
from oled_agent.agent.tools import (
    ExternalScorerError,
    ToolContext,
    ToolError,
    _external_error_payload,
    _local_fallback_scoring,
    _merge_csvs,
    _run_command_with_retry,
    generate_candidates,
    score_candidates,
    train_predictor,
    _try_external_unimol_scoring,
)
import oled_agent.agent.tools as tools_mod


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class RegressionTests(unittest.TestCase):
    def test_clean_dataset_rejects_empty_input_path_instead_of_using_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="clean_empty_input",
                state={},
            )
            with self.assertRaisesRegex(ToolError, "requires input_csv"):
                tools_mod.clean_dataset(ctx, input_csv="")

    def test_search_web_evidence_domain_filter_matches_hostname_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="web_domain_filter",
                state={},
            )
            fake_results = [
                {"title": "ok", "url": "https://sub.nature.com/a"},
                {"title": "bad", "url": "https://notnature.com/x"},
                {"title": "other", "url": "https://example.org/y"},
            ]
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=fake_results):
                out = tools_mod.search_web_evidence(ctx, query="q", topk=5, domains=["nature.com"])
            urls = [x.get("url") for x in out.get("results", [])]
            self.assertEqual(urls, ["https://sub.nature.com/a"])

    def test_search_web_evidence_applies_time_range_to_effective_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="web_time_range",
                state={},
            )
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=[]) as mocked:
                out = tools_mod.search_web_evidence(ctx, query="q", topk=3, time_range="30d")
            self.assertIn("time_range_applied", out)
            self.assertTrue(out["time_range_applied"])
            self.assertEqual(out.get("time_range_kind"), "relative")
            self.assertIn("after:", str(out.get("query_effective", "")))
            self.assertEqual(mocked.call_count, 1)
            called_query = mocked.call_args.kwargs.get("query")
            self.assertIn("after:", str(called_query))
            payload = json.loads(Path(out["web_evidence_json"]).read_text(encoding="utf-8"))
            self.assertIn("time_range_applied", payload)
            self.assertTrue(payload["time_range_applied"])
            self.assertEqual(payload.get("time_range_kind"), "relative")
            self.assertIn("after:", str(payload.get("query_effective", "")))

    def test_search_web_evidence_time_range_invalid_keeps_original_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="web_time_range_invalid",
                state={},
            )
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=[]) as mocked:
                out = tools_mod.search_web_evidence(ctx, query="q", topk=3, time_range="norange")
            self.assertFalse(out["time_range_applied"])
            self.assertEqual(str(out.get("query_effective") or ""), "q")
            called_query = mocked.call_args.kwargs.get("query")
            self.assertEqual(str(called_query or ""), "q")

    def test_search_web_evidence_time_range_hours_note_reports_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="web_time_range_hours",
                state={},
            )
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=[]):
                out = tools_mod.search_web_evidence(ctx, query="q", topk=3, time_range="48h")
            self.assertTrue(out["time_range_applied"])
            self.assertEqual(out.get("time_range_kind"), "relative")
            self.assertIn("rounded from 48h", str(out.get("time_range_note") or ""))

    def test_search_web_evidence_normalizes_no_scheme_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="web_no_scheme",
                state={},
            )
            fake_results = [
                {"title": "ok_noscheme", "url": "nature.com/a"},
                {"title": "ok_double_slash", "url": "//www.nature.com/b"},
                {"title": "ok_with_auth", "url": "https://user:pass@nature.com/c"},
                {"title": "bad_private_v6", "url": "http://[fe80::1]/c"},
            ]
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=fake_results):
                out = tools_mod.search_web_evidence(ctx, query="q", topk=5, domains=["nature.com"])
            urls = [x.get("url") for x in out.get("results", [])]
            self.assertEqual(urls, ["https://nature.com/a", "https://www.nature.com/b", "https://nature.com/c"])
            self.assertTrue(all("@" not in str(u or "") for u in urls))

    def test_search_web_evidence_filters_non_public_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="web_source_quality",
                state={},
            )
            fake_results = [
                {"title": "ok", "url": "https://nature.com/a"},
                {"title": "bad_file", "url": "file:///tmp/a.txt"},
                {"title": "bad_local", "url": "http://127.0.0.1:8000/a"},
                {"title": "bad_private", "url": "http://10.0.0.5/a"},
            ]
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=fake_results):
                out = tools_mod.search_web_evidence(ctx, query="q", topk=5)
            urls = [x.get("url") for x in out.get("results", [])]
            self.assertEqual(urls, ["https://nature.com/a"])
            quality = out.get("source_quality") if isinstance(out.get("source_quality"), dict) else {}
            self.assertEqual(int(quality.get("dropped_non_http", -1)), 1)
            self.assertEqual(int(quality.get("dropped_local_or_private", -1)), 2)

    def test_search_dataset_use_web_search_refreshes_when_state_has_old_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="search_dataset_refresh",
                state={"web_evidence": [{"title": "old", "url": "https://old.example"}]},
            )
            with mock.patch(
                "oled_agent.agent.tools.search_web_evidence",
                return_value={"results": [{"title": "new", "url": "https://new.example"}]},
            ) as mocked:
                out = tools_mod.search_dataset(ctx, preferences=["master_database"], use_web_search=True, web_topk=2)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(out.get("web_evidence"), [{"title": "new", "url": "https://new.example"}])
            self.assertTrue(out.get("web_evidence_refreshed"))

    def test_clean_dataset_uses_soft_mw_when_only_approximation_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            input_csv = td_path / "in.csv"
            _write_csv(
                input_csv,
                fieldnames=["candidate_id", "SMILES"],
                rows=[{"candidate_id": "c1", "SMILES": "C"}],
            )
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="clean_soft_mw",
                state={},
            )
            with mock.patch("oled_agent.agent.tools._estimate_mw", return_value={"value": 1.0, "method": "approx_token_sum"}):
                out = tools_mod.clean_dataset(
                    ctx,
                    input_csv=str(input_csv),
                    constraints={"mw_min": 100.0, "mw_max": 1000.0},
                )
            self.assertEqual(out["status"], "success")
            rep = json.loads(Path(out["cleaning_report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(rep.get("drop_mw_low"), 0)
            self.assertGreaterEqual(rep.get("soft_mw_low", 0), 1)
            self.assertFalse(rep.get("mw_filter_hard_applied"))
            self.assertTrue(rep.get("warnings"))

    def test_clean_dataset_can_force_approx_mw_hard_filter_via_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(json.dumps({"models": []}) + "\n", encoding="utf-8")
            input_csv = td_path / "in_hard.csv"
            _write_csv(
                input_csv,
                fieldnames=["candidate_id", "SMILES"],
                rows=[{"candidate_id": "c1", "SMILES": "C"}],
            )
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="clean_hard_mw",
                state={},
            )
            with mock.patch.dict(os.environ, {"OLED_AGENT_CLEAN_MW_APPROX_HARD_FILTER": "1"}, clear=False):
                with mock.patch("oled_agent.agent.tools._estimate_mw", return_value={"value": 1.0, "method": "approx_token_sum"}):
                    out = tools_mod.clean_dataset(
                        ctx,
                        input_csv=str(input_csv),
                        constraints={"mw_min": 100.0, "mw_max": 1000.0},
                    )
            rep = json.loads(Path(out["cleaning_report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(rep.get("drop_mw_low"), 1)
            self.assertTrue(rep.get("mw_filter_hard_applied"))
    def test_fallback_scoring_uses_real_smiles_when_source_is_uppercase_smiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "generated.csv"
            scored_csv = td_path / "scored.csv"

            _write_csv(
                input_csv,
                fieldnames=["SMILES"],
                rows=[{"SMILES": "c1ccccc1"}],
            )

            result = _local_fallback_scoring(
                input_csv=input_csv,
                scored_csv=scored_csv,
                target_specs=[
                    {
                        "name": "plqy",
                        "objective": "maximize",
                        "target_center": 0.6,
                        "sigma": 0.2,
                    }
                ],
            )
            self.assertEqual(result["adapter"], "local_deterministic_fallback")

            with scored_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["smiles"], "c1ccccc1")
            self.assertFalse(row["smiles"].startswith("DUMMY_SMILES_"))
            self.assertEqual(row["candidate_id"], "cand_000001")
            self.assertTrue(row.get("plqy_pred"))
            self.assertTrue(row.get("plqy_score"))

    def test_merge_csv_raises_when_addon_missing_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            base_csv = td_path / "base.csv"
            addon_csv = td_path / "addon.csv"

            _write_csv(
                base_csv,
                fieldnames=["candidate_id", "SMILES"],
                rows=[
                    {"candidate_id": "cand_000001", "SMILES": "c1ccccc1"},
                    {"candidate_id": "cand_000002", "SMILES": "CCO"},
                ],
            )
            _write_csv(
                addon_csv,
                fieldnames=["plqy_pred"],
                rows=[{"plqy_pred": "0.7"}],
            )

            with self.assertRaisesRegex(ToolError, "candidate_id"):
                _merge_csvs(base_csv, addon_csv, key="candidate_id")

    def test_packaging_metadata_consistent_with_pyproject(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

        # pyproject.toml is the single source of truth.
        self.assertIn('name = "Agent4Mat"', pyproject_text)
        self.assertIn('version = "0.1.0"', pyproject_text)
        self.assertIn('"jsonschema>=4.0"', pyproject_text)

        # setup.py remains compatibility shim and should resolve metadata from pyproject.
        cp_name = subprocess.run(
            [sys.executable, "setup.py", "--name"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        cp_version = subprocess.run(
            [sys.executable, "setup.py", "--version"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(cp_name.returncode, 0, msg=cp_name.stderr)
        self.assertEqual(cp_version.returncode, 0, msg=cp_version.stderr)
        name = cp_name.stdout.strip()
        version = cp_version.stdout.strip()

        self.assertEqual(name, "Agent4Mat")
        self.assertEqual(version, "0.1.0")
        self.assertNotIn("overwritten", cp_name.stderr.lower())
        self.assertNotIn("overwritten", cp_version.stderr.lower())
        self.assertNotIn("ignored", cp_name.stderr.lower())
        self.assertNotIn("ignored", cp_version.stderr.lower())

    def test_external_runner_retries_then_succeeds(self) -> None:
        call_count = {"n": 0}

        def fake_run(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=1)
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

        sleep_calls: list[float] = []

        def fake_sleep(sec: float) -> None:
            sleep_calls.append(sec)

        out = _run_command_with_retry(
            cmd=["python3", "dummy.py"],
            cwd=Path("."),
            timeout_sec=1,
            retries=2,
            backoff_sec=0.5,
            run_fn=fake_run,
            sleep_fn=fake_sleep,
        )
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(sleep_calls, [0.5])

    def test_external_runner_timeout_raises_structured_error(self) -> None:
        def always_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

        with self.assertRaises(ExternalScorerError) as cm:
            _run_command_with_retry(
                cmd=["python3", "dummy.py"],
                cwd=Path("."),
                timeout_sec=1,
                retries=1,
                backoff_sec=0.1,
                run_fn=always_timeout,
                sleep_fn=lambda _: None,
            )

        err = cm.exception
        self.assertEqual(err.code, "external_timeout")
        self.assertTrue(err.retryable)
        self.assertEqual(err.details.get("attempts"), 2)

    def test_external_error_payload_for_known_and_unknown_exceptions(self) -> None:
        known = ExternalScorerError(code="x", message="boom", details={"k": "v"}, retryable=True)
        payload_known = _external_error_payload(known)
        self.assertEqual(payload_known["code"], "x")
        self.assertTrue(payload_known["retryable"])
        self.assertEqual(payload_known["details"], {"k": "v"})

        payload_unknown = _external_error_payload(RuntimeError("oops"))
        self.assertEqual(payload_unknown["code"], "unexpected_external_error")
        self.assertFalse(payload_unknown["retryable"])

    def test_try_external_scorer_disabled_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "in.csv"
            scored_csv = td_path / "scored.csv"
            catalog_path = td_path / "catalog.json"

            _write_csv(
                input_csv,
                fieldnames=["SMILES"],
                rows=[{"SMILES": "c1ccccc1"}],
            )
            catalog_path.write_text('{"models": []}\n', encoding="utf-8")

            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="t1",
            )

            with mock.patch.dict(os.environ, {"OLED_AGENT_USE_EXTERNAL_SCORER": "0"}, clear=False):
                with self.assertRaises(ExternalScorerError) as cm:
                    _try_external_unimol_scoring(
                        ctx=ctx,
                        predictor_id="p1",
                        input_csv=input_csv,
                        target_specs=[{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
                        scored_csv=scored_csv,
                    )
            err = cm.exception
            self.assertEqual(err.code, "external_workspace_missing")

    def test_score_candidates_emits_structured_fallback_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            candidate_csv = td_path / "generated.csv"
            catalog_path = td_path / "catalog.json"

            _write_csv(
                candidate_csv,
                fieldnames=["SMILES"],
                rows=[{"SMILES": "c1ccccc1"}],
            )
            catalog_path.write_text('{"models": []}\n', encoding="utf-8")

            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="t2",
                state={"candidate_csv": str(candidate_csv)},
            )

            result = score_candidates(
                ctx,
                predictor_id="unimol_lambda_plqy_v1",
                targets=["plqy"],
                target_specs=[{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["adapter"], "local_deterministic_fallback")
            self.assertIn("fallback_error", result)
            self.assertEqual(result["fallback_error"]["code"], "external_workspace_missing")

    def test_score_candidates_uses_bundled_unimol_adapter_when_catalog_cmd_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            candidate_csv = td_path / "generated.csv"
            _write_csv(
                candidate_csv,
                fieldnames=["SMILES"],
                rows=[{"SMILES": "c1ccccc1"}],
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "unimol_lambda_plqy_v1",
                                "kind": "predictor",
                                "backend": "unimol_tools",
                                "task_types": ["plqy"],
                                "runtime_profile": "gpu",
                                "params": {"adapters": {"score_candidates_cmd": ""}},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="t_bundled_unimol",
                state={"candidate_csv": str(candidate_csv)},
            )
            with mock.patch.dict(os.environ, {"OLED_AGENT_UNIMOL_SCORE_MODE": "smoke"}, clear=False):
                result = score_candidates(
                    ctx,
                    predictor_id="unimol_lambda_plqy_v1",
                    targets=["plqy"],
                    target_specs=[{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
                )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["adapter"], "unimol_score_adapter_v1")
            self.assertTrue(Path(ctx.state["scored_csv"]).exists())
            scored = Path(ctx.state["scored_csv"]).read_text(encoding="utf-8")
            self.assertIn("plqy_score", scored)

    def test_score_candidates_non_unimol_without_cmd_keeps_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            candidate_csv = td_path / "generated.csv"
            _write_csv(
                candidate_csv,
                fieldnames=["SMILES"],
                rows=[{"SMILES": "c1ccccc1"}],
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "pred_non_unimol_v1",
                                "kind": "predictor",
                                "backend": "sklearn_hist_gbr",
                                "task_types": ["plqy"],
                                "runtime_profile": "cpu",
                                "params": {"adapters": {"score_candidates_cmd": ""}},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="t_non_unimol",
                state={"candidate_csv": str(candidate_csv)},
            )
            result = score_candidates(
                ctx,
                predictor_id="pred_non_unimol_v1",
                targets=["plqy"],
                target_specs=[{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["adapter"], "local_deterministic_fallback")
            self.assertIn("fallback_error", result)
            self.assertEqual(result["fallback_error"]["code"], "external_workspace_missing")

    def test_train_predictor_uses_external_command_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            script = td_path / "train_cmd.py"
            script.write_text(
                (
                    "import json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "print(json.dumps({'status':'success','adapter':'external_train_cmd','predictor_id':payload.get('predictor_id','')}))\n"
                ),
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text('{"models": []}\n', encoding="utf-8")
            ctx = ToolContext(workspace_root=td_path, catalog_path=catalog_path, task_id="t_train")
            with mock.patch.dict(
                os.environ,
                {"OLED_AGENT_TRAIN_CMD": f"{sys.executable} {script}"},
                clear=False,
            ):
                out = train_predictor(
                    ctx,
                    predictor_id="unimol_lambda_plqy_v1",
                    targets=["plqy"],
                    target_specs=[{"name": "plqy", "objective": "maximize"}],
                )
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["adapter"], "external_train_cmd")

    def test_generate_candidates_uses_external_command_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            script = td_path / "generate_cmd.py"
            script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','SMILES'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','SMILES':'c1ccccc1'})\n"
                    "print(json.dumps({'status':'success','adapter':'external_generate_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text('{"models": []}\n', encoding="utf-8")
            ctx = ToolContext(workspace_root=td_path, catalog_path=catalog_path, task_id="t_gen")
            with mock.patch.dict(
                os.environ,
                {"OLED_AGENT_GENERATE_CMD": f"{sys.executable} {script}"},
                clear=False,
            ):
                out = generate_candidates(
                    ctx,
                    generator_id="reinvent4_lambda_em_v2",
                    max_candidates=10,
                    constraints={"mw_max": 700},
                )
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["adapter"], "external_generate_cmd")
            self.assertTrue(Path(ctx.state["candidate_csv"]).exists())

    def test_score_candidates_uses_external_command_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            candidate_csv = td_path / "generated.csv"
            _write_csv(
                candidate_csv,
                fieldnames=["candidate_id", "smiles"],
                rows=[{"candidate_id": "cand_000001", "smiles": "c1ccccc1"}],
            )
            script = td_path / "score_cmd.py"
            script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "inp=payload['input_csv']; out=payload['output_csv']\n"
                    "rows=list(csv.DictReader(open(inp,'r',encoding='utf-8')))\n"
                    "for r in rows:\n"
                    "  r['plqy_pred']='0.66'; r['plqy_score']='0.66'\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)\n"
                    "print(json.dumps({'status':'success','adapter':'external_score_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text('{"models": []}\n', encoding="utf-8")
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="t_score",
                state={"candidate_csv": str(candidate_csv)},
            )
            with mock.patch.dict(
                os.environ,
                {"OLED_AGENT_SCORE_CMD": f"{sys.executable} {script}"},
                clear=False,
            ):
                out = score_candidates(
                    ctx,
                    predictor_id="unimol_lambda_plqy_v1",
                    targets=["plqy"],
                    target_specs=[{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
                )
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["adapter"], "external_score_cmd")
            self.assertTrue(Path(ctx.state["scored_csv"]).exists())

    def test_train_predictor_uses_catalog_model_adapter_when_env_cmd_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            script = td_path / "train_catalog_cmd.py"
            script.write_text(
                (
                    "import json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_train_cmd','predictor_id':payload.get('predictor_id','')}))\n"
                ),
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "unimol_lambda_plqy_v1",
                                "kind": "predictor",
                                "backend": "unimol_tools",
                                "task_types": ["plqy"],
                                "runtime_profile": "gpu",
                                "params": {
                                    "adapters": {
                                        "train_predictor_cmd": f"{sys.executable} {script}",
                                    }
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(workspace_root=td_path, catalog_path=catalog_path, task_id="t_catalog_train")
            with mock.patch.dict(os.environ, {"OLED_AGENT_TRAIN_CMD": ""}, clear=False):
                out = train_predictor(
                    ctx,
                    predictor_id="unimol_lambda_plqy_v1",
                    targets=["plqy"],
                    target_specs=[{"name": "plqy", "objective": "maximize"}],
                )
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["adapter"], "catalog_train_cmd")
            self.assertEqual(out["predictor_id"], "unimol_lambda_plqy_v1")

    def test_generate_candidates_env_cmd_overrides_catalog_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_script = td_path / "generate_catalog_cmd.py"
            catalog_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','smiles'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','smiles':'catalog'})\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_generate_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            env_script = td_path / "generate_env_cmd.py"
            env_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','smiles'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','smiles':'env'})\n"
                    "print(json.dumps({'status':'success','adapter':'external_generate_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "reinvent4_lambda_em_v2",
                                "kind": "generator",
                                "backend": "reinvent4",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "gpu",
                                "params": {
                                    "adapters": {
                                        "generate_candidates_cmd": f"{sys.executable} {catalog_script}",
                                    }
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(workspace_root=td_path, catalog_path=catalog_path, task_id="t_catalog_generate")
            with mock.patch.dict(
                os.environ,
                {"OLED_AGENT_GENERATE_CMD": f"{sys.executable} {env_script}"},
                clear=False,
            ):
                out = generate_candidates(
                    ctx,
                    generator_id="reinvent4_lambda_em_v2",
                    max_candidates=10,
                    constraints={"mw_max": 700},
                )
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["adapter"], "external_generate_cmd")
            self.assertTrue(Path(ctx.state["candidate_csv"]).exists())

    def test_generate_candidates_bundled_reinvent4_adapter_failure_falls_back_to_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "reinvent4_lambda_em_v2",
                                "kind": "generator",
                                "backend": "reinvent4",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "gpu",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(workspace_root=td_path, catalog_path=catalog_path, task_id="t_reinvent4_fallback")
            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_GENERATE_CMD": "",
                    "OLED_AGENT_DISABLE_BUNDLED_REINVENT4_GENERATE_ADAPTER": "0",
                    "OLED_AGENT_REINVENT4_ADAPTER_MODE": "preflight",
                },
                clear=False,
            ):
                out = generate_candidates(
                    ctx,
                    generator_id="reinvent4_lambda_em_v2",
                    max_candidates=7,
                    constraints={"mw_max": 700},
                )
            self.assertEqual(out["status"], "success")
            self.assertIn(out["adapter"], {"stub_generator", "reuse_latest_reinvent_artifact"})
            self.assertIn("fallback_error", out)
            self.assertEqual(out["fallback_error"].get("code"), "reinvent4_generate_cmd_failed")
            self.assertIn("fallback_reason", out)
            self.assertTrue(Path(ctx.state["candidate_csv"]).exists())

    def test_generate_candidates_explicit_env_cmd_failure_raises_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bad_script = td_path / "generate_bad_env.py"
            bad_script.write_text(
                "import sys\nsys.exit(9)\n",
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "reinvent4_lambda_em_v2",
                                "kind": "generator",
                                "backend": "reinvent4",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "gpu",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(workspace_root=td_path, catalog_path=catalog_path, task_id="t_explicit_env_fail")
            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_GENERATE_CMD": f"{sys.executable} {bad_script}",
                },
                clear=False,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    generate_candidates(
                        ctx,
                        generator_id="reinvent4_lambda_em_v2",
                        max_candidates=5,
                        constraints={},
                    )

    def test_score_candidates_uses_catalog_model_adapter_when_env_cmd_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            candidate_csv = td_path / "generated.csv"
            _write_csv(
                candidate_csv,
                fieldnames=["candidate_id", "smiles"],
                rows=[{"candidate_id": "cand_000001", "smiles": "c1ccccc1"}],
            )
            script = td_path / "score_catalog_cmd.py"
            script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "inp=payload['input_csv']; out=payload['output_csv']\n"
                    "rows=list(csv.DictReader(open(inp,'r',encoding='utf-8')))\n"
                    "for r in rows:\n"
                    "  r['plqy_pred']='0.42'; r['plqy_score']='0.42'\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_score_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            catalog_path = td_path / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "unimol_lambda_plqy_v1",
                                "kind": "predictor",
                                "backend": "unimol_tools",
                                "task_types": ["plqy"],
                                "runtime_profile": "gpu",
                                "params": {
                                    "adapters": {
                                        "score_candidates_cmd": f"{sys.executable} {script}",
                                    }
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(
                workspace_root=td_path,
                catalog_path=catalog_path,
                task_id="t_catalog_score",
                state={"candidate_csv": str(candidate_csv)},
            )
            with mock.patch.dict(os.environ, {"OLED_AGENT_SCORE_CMD": ""}, clear=False):
                out = score_candidates(
                    ctx,
                    predictor_id="unimol_lambda_plqy_v1",
                    targets=["plqy"],
                    target_specs=[{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2}],
                )
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["adapter"], "catalog_score_cmd")
            self.assertTrue(Path(ctx.state["scored_csv"]).exists())

    def test_execute_request_writes_decision_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            out = execute_request(
                workspace_root=td_path,
                user_request="设计470nm附近且高PLQY分子",
                task_id="task_decision_summary",
                catalog_path=repo_root / "configs" / "models" / "catalog.json",
            )
            self.assertEqual(out["status"], "success")
            self.assertIn("decision_summary_path", out)

            decision_path = Path(out["decision_summary_path"])
            self.assertTrue(decision_path.exists())

            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], "task_decision_summary")
            self.assertEqual(payload["status"], "success")
            self.assertIn("score_step", payload)
            self.assertIn("used_fallback", payload["score_step"])
            self.assertIn("fallback_code", payload["score_step"])
            self.assertIn("artifacts", payload)
            self.assertIn("task_state_path", out)
            self.assertIn("logging_dir", out)
            self.assertIn("result_dir", out)

            task_state_path = Path(out["task_state_path"])
            self.assertTrue(task_state_path.exists())
            task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
            self.assertEqual(task_state.get("task_id"), "task_decision_summary")
            self.assertIn(task_state.get("current_state"), ("DONE", "FAILED"))
            self.assertIsInstance(task_state.get("history"), list)
            self.assertGreater(len(task_state.get("history", [])), 0)
            allowed_states = {
                "INIT",
                "REQUIREMENT_COLLECTION",
                "VALIDATION",
                "PLAN_GENERATION",
                "USER_CONFIRMATION",
                "DATA_ACQUISITION",
                "PREPROCESSING",
                "ROUTING",
                "TRAINING_OPTIONAL",
                "INFERENCE",
                "FILTERING",
                "SAVING",
                "REPORTING",
                "QA",
                "DONE",
                "FAILED",
            }
            self.assertIn(task_state.get("current_state"), allowed_states)
            self.assertIn(task_state.get("status"), ("success", "failed"))
            for item in task_state.get("history", []):
                self.assertIn(item.get("state"), allowed_states)
                self.assertIn(item.get("status"), ("completed", "success", "failed", "unknown"))

            logging_dir = Path(out["logging_dir"])
            result_dir = Path(out["result_dir"])
            self.assertTrue(logging_dir.exists())
            self.assertTrue(result_dir.exists())
            self.assertTrue((logging_dir / "task.json").exists())
            self.assertTrue((logging_dir / "plan.md").exists())
            self.assertTrue((logging_dir / "execution.log").exists())
            self.assertTrue((logging_dir / "data_report.json").exists())
            self.assertTrue((logging_dir / "model_report.json").exists())
            self.assertTrue((logging_dir / "filtering_report.json").exists())
            self.assertTrue((result_dir / "metadata.json").exists())
            self.assertIn("experiment_trace_path", out)
            trace_path = Path(out["experiment_trace_path"])
            self.assertTrue(trace_path.exists())
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace_payload.get("task_id"), "task_decision_summary")
            self.assertEqual(trace_payload.get("execution_mode"), "full_pipeline")
            self.assertIn("fingerprints", trace_payload)
            self.assertIn("execution_summary", trace_payload)
            self.assertIn("source_artifacts", trace_payload)

    def test_external_preflight_warns_when_workspace_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = run_external_preflight(workspace_root=Path(td))
            self.assertEqual(report["overall"], "warn")
            self.assertEqual(report["exit_code"], 1)
            checks = report.get("checks", [])
            self.assertGreaterEqual(len(checks), 3)
            self.assertEqual(checks[0]["name"], "external:scorer_chain")

    def test_external_preflight_fails_with_partial_remote_runtime_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            scripts_dir = td_path / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            scorer = scripts_dir / "score_unimol_property_candidates.py"
            scorer.write_text(
                "#!/usr/bin/env python3\nimport sys\nif '--help' in sys.argv:\n    print('ok')\n    raise SystemExit(0)\n",
                encoding="utf-8",
            )
            scorer.chmod(0o755)

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_USE_EXTERNAL_SCORER": "1",
                    "UNIMOL_REMOTE_HOST": "example.com",
                    "UNIMOL_REMOTE_PY": "",
                    "UNIMOL_REMOTE_TMP_BASE": "",
                },
                clear=False,
            ):
                report = run_external_preflight(workspace_root=td_path)

            self.assertEqual(report["overall"], "fail")
            checks = {c["name"]: c for c in report.get("checks", [])}
            self.assertIn("external:runtime_config", checks)
            self.assertEqual(checks["external:runtime_config"]["status"], "fail")

    @mock.patch("oled_agent.diagnostics._check_external_remote_tmp_base")
    @mock.patch("oled_agent.diagnostics._check_external_remote_python")
    @mock.patch("oled_agent.diagnostics._check_external_ssh_connectivity")
    @mock.patch("oled_agent.diagnostics._check_command")
    def test_external_preflight_runs_remote_checks_when_configured(
        self,
        mock_check_command: mock.Mock,
        mock_check_ssh: mock.Mock,
        mock_check_remote_py: mock.Mock,
        mock_check_tmp_base: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            scripts_dir = td_path / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            scorer = scripts_dir / "score_unimol_property_candidates.py"
            scorer.write_text(
                "#!/usr/bin/env python3\nimport sys\nif '--help' in sys.argv:\n    print('ok')\n    raise SystemExit(0)\n",
                encoding="utf-8",
            )
            scorer.chmod(0o755)

            def fake_check_command(cmd: str, required: bool = False) -> dict:
                return {"name": f"command:{cmd}", "status": "pass", "message": "ok", "details": {"path": f"/usr/bin/{cmd}"}}

            mock_check_command.side_effect = fake_check_command
            mock_check_ssh.return_value = {"name": "external:ssh_connectivity", "status": "pass", "message": "ok", "details": {}}
            mock_check_remote_py.return_value = {"name": "external:remote_python", "status": "pass", "message": "ok", "details": {}}
            mock_check_tmp_base.return_value = {"name": "external:remote_tmp_base", "status": "pass", "message": "ok", "details": {}}

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_USE_EXTERNAL_SCORER": "1",
                    "UNIMOL_REMOTE_HOST": "u@example.com",
                    "UNIMOL_REMOTE_PY": "/opt/python",
                    "UNIMOL_REMOTE_TMP_BASE": "/tmp/openclaw",
                },
                clear=False,
            ):
                report = run_external_preflight(workspace_root=td_path)

            self.assertEqual(report["overall"], "pass")
            checks = {c["name"]: c for c in report.get("checks", [])}
            self.assertEqual(checks["external:runtime_config"]["status"], "pass")
            self.assertIn("external:ssh_connectivity", checks)
            self.assertIn("external:remote_python", checks)
            self.assertIn("external:remote_tmp_base", checks)

    def test_external_acceptance_script_exports_external_switch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_external_chain_acceptance.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn("export OLED_AGENT_USE_EXTERNAL_SCORER=1", content)

    def test_external_acceptance_with_debug_script_contains_debug_json_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_external_chain_acceptance_with_debug.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn("external-connectivity-debug", content)
        self.assertIn("external_debug.json", content)
        self.assertIn("fallback=", content)
        self.assertNotIn("@Q", content)

    def test_external_acceptance_with_debug_script_uses_debug_default_task_prefix(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_external_chain_acceptance_with_debug.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn('TASK_ID="${1:-accept_external_debug_', content)

    def test_quickstart_chain_script_contains_expected_steps(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "check_quickstart_chain.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn("agent-run-json", content)
        self.assertIn("scripts/adapters/quickstart_catalog.json", content)
        self.assertIn("validate_run_artifacts.py", content)
        self.assertIn("--decision-summary", content)
        self.assertIn("--task-state", content)
        self.assertIn("--data-report", content)
        self.assertIn("--model-report", content)
        self.assertIn("--filtering-report", content)
        self.assertIn("[PASS] quickstart chain completed", content)

    def test_step_mode_guard_script_covers_all_operations_and_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "check_step_mode.py"
        content = script.read_text(encoding="utf-8")
        self.assertIn("run_step_success(", content)
        self.assertIn("run_step_json_success(", content)
        self.assertIn("agent-run-step-json", content)
        self.assertIn('operation="retrieve_candidate_data"', content)
        self.assertIn('operation="clean_dataset"', content)
        self.assertIn('operation="prepare_train_data"', content)
        self.assertIn('operation="generate_candidates"', content)
        self.assertIn('operation="score_candidates"', content)
        self.assertIn('operation="filter_and_rank"', content)
        self.assertIn('operation="make_report"', content)
        self.assertIn('operation="train_predictor"', content)
        self.assertIn("happy(all_operations)", content)
        self.assertIn("step_tool_state.json", content)
        self.assertIn("score_without_candidates_unexpected_success", content)
        self.assertIn("train_nonzero_unexpected_success", content)
        self.assertIn("score_json_without_candidates_unexpected_success", content)
        self.assertIn("train_json_nonzero_unexpected_success", content)
        self.assertIn("OLED_AGENT_TRAIN_CMD", content)

    def test_makefile_release_check_includes_request_template_validation(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
        self.assertIn("request-templates-validate", makefile)
        self.assertIn("step-request-templates-validate", makefile)
        self.assertIn("@$(MAKE) request-templates-validate WORKSPACE_ROOT=\"$(WORKSPACE_ROOT)\"", makefile)
        self.assertIn("@$(MAKE) step-request-templates-validate WORKSPACE_ROOT=\"$(WORKSPACE_ROOT)\"", makefile)

    def test_env_example_uses_tmp_base_name(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env_example = repo_root / ".env.example"
        content = env_example.read_text(encoding="utf-8")
        self.assertIn("UNIMOL_REMOTE_TMP_BASE=", content)
        self.assertNotIn("UNIMOL_REMOTE_BASE=", content)
        self.assertIn("OLED_AGENT_LLM_PLANNER_CMD=", content)
        self.assertIn("OLED_AGENT_LLM_BACKEND=", content)
        self.assertIn("OLED_AGENT_USE_EXTERNAL_SCORER=", content)
        self.assertIn("OLED_AGENT_UNIMOL_TRAIN_MODE=", content)
        self.assertIn("OLED_AGENT_UNIMOL_SCORE_MODE=", content)
        self.assertIn("OLED_AGENT_MINERU_ADAPTER_MODE=", content)
        self.assertIn("OLED_AGENT_REINVENT4_ADAPTER_MODE=", content)
        self.assertIn("OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=", content)
        self.assertIn("OLED_AGENT_MOLSCRIBE_CMD=", content)

    def test_deploy_and_troubleshooting_docs_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        deploy_doc = repo_root / "docs" / "deploy.md"
        troubleshoot_doc = repo_root / "docs" / "troubleshooting.md"
        self.assertTrue(deploy_doc.exists())
        self.assertTrue(troubleshoot_doc.exists())

        deploy = deploy_doc.read_text(encoding="utf-8")
        self.assertIn("make release-check", deploy)
        self.assertIn("make real-adapter-validate", deploy)
        self.assertIn("acceptance-cpu-mock", deploy)
        self.assertIn("acceptance-llm-mock", deploy)
        self.assertIn("acceptance external-adapter (optional)", deploy)

        troubleshoot = troubleshoot_doc.read_text(encoding="utf-8")
        self.assertIn("make llm-smoke", troubleshoot)
        self.assertIn("external-preflight", troubleshoot)
        self.assertIn("external-connectivity-debug", troubleshoot)
        self.assertIn("mineru_not_configured", troubleshoot)
        self.assertIn("molscribe_input_missing", troubleshoot)
        self.assertIn("make request-templates-validate", troubleshoot)

    def test_ci_doc_includes_real_baseline_archive_template(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        ci_doc = repo_root / "docs" / "ci.md"
        self.assertTrue(ci_doc.exists())
        ci = ci_doc.read_text(encoding="utf-8")
        self.assertIn("make real-chain-baseline TASK_ID=<base_task_id>", ci)
        self.assertIn("make real-chain-baseline-archive TASK_ID=<base_task_id>", ci)
        self.assertIn("make real-chain-baseline-archive-tgz TASK_ID=<base_task_id>", ci)
        self.assertIn("make real-chain-release-bundle-check TASK_ID=<base_task_id>", ci)
        self.assertIn("baseline_summary.json", ci)
        self.assertIn("archive_manifest.json", ci)
        self.assertIn("runs/archive/<base_task_id>.tar.gz", ci)
        self.assertIn("strict_acceptance_summary.json", ci)
        self.assertIn("release_evidence.json", ci)

    def test_gitattributes_enforces_lf_for_cross_platform_files(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        gitattributes = repo_root / ".gitattributes"
        self.assertTrue(gitattributes.exists())
        content = gitattributes.read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", content)
        self.assertIn("*.py text eol=lf", content)
        self.assertIn("*.md text eol=lf", content)
        self.assertIn("*.yml text eol=lf", content)

    def test_release_docs_and_version_alignment(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('version = "0.1.0"', pyproject_text)

        changelog = repo_root / "CHANGELOG.md"
        release_doc = repo_root / "docs" / "release_v0.1.0.md"
        self.assertTrue(changelog.exists())
        self.assertTrue(release_doc.exists())

        changelog_text = changelog.read_text(encoding="utf-8")
        release_text = release_doc.read_text(encoding="utf-8")
        self.assertIn("## [0.1.0]", changelog_text)
        self.assertIn("git tag -a v0.1.0", release_text)
        self.assertIn("make release-check", release_text)
        self.assertIn("make real-adapter-validate", release_text)

    def test_release_boundary_doc_mentions_real_chain_baseline_summary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        boundary = (repo_root / "docs" / "release_boundary.md").read_text(encoding="utf-8")
        self.assertIn("make real-chain-baseline TASK_ID=<base_task_id>", boundary)
        self.assertIn("make real-chain-baseline-archive TASK_ID=<base_task_id>", boundary)
        self.assertIn("make real-chain-baseline-archive-tgz TASK_ID=<base_task_id>", boundary)
        self.assertIn("make real-chain-release-bundle-check TASK_ID=<base_task_id>", boundary)
        self.assertIn("baseline_summary.json", boundary)

    def test_readme_includes_task_v2_and_step_mode_paths(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("agent-intake", readme)
        self.assertIn("agent-approve", readme)
        self.assertIn("agent-run-step-json", readme)
        self.assertIn("agent-resume", readme)
        self.assertIn("single-step operation mode", readme)
        self.assertIn("real-chain-release-bundle-check", readme)

    def test_docs_examples_molscribe_requests_are_contract_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        examples_dir = repo_root / "docs" / "examples"
        image_json = examples_dir / "request_molscribe_image.json"
        pdf_json = examples_dir / "request_molscribe_pdf.json"
        examples_readme = examples_dir / "README.md"
        self.assertTrue(image_json.exists(), msg=f"Missing example: {image_json}")
        self.assertTrue(pdf_json.exists(), msg=f"Missing example: {pdf_json}")
        self.assertTrue(examples_readme.exists(), msg=f"Missing example readme: {examples_readme}")

        image_payload = json.loads(image_json.read_text(encoding="utf-8"))
        pdf_payload = json.loads(pdf_json.read_text(encoding="utf-8"))
        validate_request_payload(payload=image_payload, workspace_root=repo_root)
        validate_request_payload(payload=pdf_payload, workspace_root=repo_root)

        image_target = image_payload.get("targets", [{}])[0]
        pdf_target = pdf_payload.get("targets", [{}])[0]
        self.assertEqual(image_target.get("target_value"), 60.0)
        self.assertEqual(pdf_target.get("target_value"), 60.0)

    def test_configs_request_templates_are_contract_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        templates_dir = repo_root / "configs" / "request_templates"
        image_json = templates_dir / "request_molscribe_image.json"
        pdf_json = templates_dir / "request_molscribe_pdf.json"
        templates_readme = templates_dir / "README.md"
        self.assertTrue(image_json.exists(), msg=f"Missing request template: {image_json}")
        self.assertTrue(pdf_json.exists(), msg=f"Missing request template: {pdf_json}")
        self.assertTrue(templates_readme.exists(), msg=f"Missing request template readme: {templates_readme}")

        image_payload = json.loads(image_json.read_text(encoding="utf-8"))
        pdf_payload = json.loads(pdf_json.read_text(encoding="utf-8"))
        validate_request_payload(payload=image_payload, workspace_root=repo_root)
        validate_request_payload(payload=pdf_payload, workspace_root=repo_root)

        image_target = image_payload.get("targets", [{}])[0]
        pdf_target = pdf_payload.get("targets", [{}])[0]
        self.assertEqual(image_target.get("target_value"), 60.0)
        self.assertEqual(pdf_target.get("target_value"), 60.0)

    def test_configs_step_request_templates_are_contract_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        templates_dir = repo_root / "configs" / "request_templates"
        clean_json = templates_dir / "step_request_clean_dataset.json"
        train_json = templates_dir / "step_request_train_predictor.json"
        self.assertTrue(clean_json.exists(), msg=f"Missing step request template: {clean_json}")
        self.assertTrue(train_json.exists(), msg=f"Missing step request template: {train_json}")

        clean_payload = json.loads(clean_json.read_text(encoding="utf-8"))
        train_payload = json.loads(train_json.read_text(encoding="utf-8"))
        validate_step_request_payload(payload=clean_payload, workspace_root=repo_root)
        validate_step_request_payload(payload=train_payload, workspace_root=repo_root)
        self.assertEqual(clean_payload.get("operation"), "clean_dataset")
        self.assertEqual(train_payload.get("operation"), "train_predictor")

    def test_request_templates_validate_script_reports_pass(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/validate_request_examples.py",
                "--workspace-root",
                str(repo_root),
                "--examples-dir",
                "configs/request_templates",
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("failed"), 0)
        self.assertGreaterEqual(int(payload.get("checked", 0)), 2)

    def test_step_request_templates_validate_script_reports_pass(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/validate_step_request_examples.py",
                "--workspace-root",
                str(repo_root),
                "--examples-dir",
                "configs/request_templates",
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("failed"), 0)
        self.assertGreaterEqual(int(payload.get("checked", 0)), 2)

    def test_request_schema_rejects_invalid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "task_id": "task_1",
                "request_text": "design molecule",
                "mode": "invalid_mode",
                "targets": [{"property": "plqy", "objective": "maximize"}],
                "budget": {"max_candidates": 10},
            }
            with self.assertRaises(RequestValidationError):
                validate_request_payload(payload, workspace_root=td_path)

    def test_request_schema_rejects_invalid_target_property(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "task_id": "task_1",
                "request_text": "design molecule",
                "mode": "fast_screen",
                "targets": [{"property": "unknown_prop", "objective": "maximize"}],
                "budget": {"max_candidates": 10},
            }
            with self.assertRaises(RequestValidationError):
                validate_request_payload(payload, workspace_root=td_path)

    def test_request_schema_rejects_extra_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "task_id": "task_1",
                "request_text": "design molecule",
                "mode": "fast_screen",
                "targets": [{"property": "plqy", "objective": "maximize"}],
                "budget": {"max_candidates": 10},
                "unexpected": True,
            }
            with self.assertRaises(RequestValidationError):
                validate_request_payload(payload, workspace_root=td_path)

    def test_task_v2_schema_rejects_invalid_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "version": "2.0",
                "task_id": "task_v2_bad_mode",
                "request_text": "design molecule",
                "execution_mode": "bad_mode",
                "operation": "full_pipeline",
                "property": "plqy",
                "range": "60-100",
                "n_structures": 20,
                "constraints": {},
                "prediction_model": "unimol_lambda_plqy_v1",
            }
            with self.assertRaises(RequestValidationError):
                validate_task_v2_payload(payload, workspace_root=td_path)

    def test_task_v2_schema_rejects_extra_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "version": "2.0",
                "task_id": "task_v2_extra",
                "request_text": "design molecule",
                "execution_mode": "full_pipeline",
                "operation": "full_pipeline",
                "property": "plqy",
                "range": "60-100",
                "n_structures": 20,
                "constraints": {},
                "prediction_model": "unimol_lambda_plqy_v1",
                "unexpected": True,
            }
            with self.assertRaises(RequestValidationError):
                validate_task_v2_payload(payload, workspace_root=td_path)

    def test_task_v2_schema_accepts_null_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "version": "2.0",
                "task_id": "task_v2_nulls",
                "request_text": "design molecule",
                "execution_mode": "single_step",
                "operation": "clean_dataset",
                "property": "plqy",
                "range": "60-100",
                "n_structures": 20,
                "constraints": {"mw_min": 100, "mw_max": 800},
                "train_data": None,
                "candidate_data": None,
                "prediction_model": "unimol_lambda_plqy_v1",
                "model_preferences": {
                    "predictor_id": "unimol_lambda_plqy_v1",
                    "generator_id": "reinvent4_lambda_em_v2",
                },
                "generation_input": {},
                "provenance": {},
                "status": "approved",
                "missing_fields": [],
                "questions": [],
                "compatibility_warnings": [],
            }
            validated = validate_task_v2_payload(payload, workspace_root=td_path)
            self.assertIsNone(validated.get("train_data"))
            self.assertIsNone(validated.get("candidate_data"))

    def test_step_request_schema_rejects_invalid_operation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            step_payload = {
                "task": {
                    "version": "2.0",
                    "task_id": "step_bad_op",
                    "request_text": "design molecule",
                    "execution_mode": "single_step",
                    "operation": "clean_dataset",
                    "property": "plqy",
                    "range": "60-100",
                    "n_structures": 20,
                    "constraints": {},
                    "prediction_model": "unimol_lambda_plqy_v1",
                },
                "operation": "unsupported_op",
            }
            with self.assertRaises(RequestValidationError):
                validate_step_request_payload(step_payload, workspace_root=td_path)

    def test_step_request_schema_rejects_additional_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            step_payload = {
                "task": {
                    "version": "2.0",
                    "task_id": "step_extra_field",
                    "request_text": "design molecule",
                    "execution_mode": "single_step",
                    "operation": "clean_dataset",
                    "property": "plqy",
                    "range": "60-100",
                    "n_structures": 20,
                    "constraints": {},
                    "prediction_model": "unimol_lambda_plqy_v1",
                },
                "operation": "clean_dataset",
                "extra": "x",
            }
            with self.assertRaises(RequestValidationError):
                validate_step_request_payload(step_payload, workspace_root=td_path)

    def test_legacy_request_to_task_v2_compatibility_mapping(self) -> None:
        request_payload = {
            "task_id": "legacy_compat_1",
            "request_text": "design legacy",
            "mode": "fast_screen",
            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
            "constraints": {"candidate_data": "candidates.csv"},
            "budget": {"max_candidates": 77},
            "model_preferences": {
                "predictor_id": "unimol_lambda_plqy_v1",
                "generator_id": "reinvent4_lambda_em_v2",
            },
        }
        task_v2 = legacy_request_to_task_v2(request_payload)
        self.assertEqual(task_v2["version"], "2.0")
        self.assertEqual(task_v2["task_id"], "legacy_compat_1")
        self.assertEqual(task_v2["execution_mode"], "full_pipeline")
        self.assertEqual(task_v2["operation"], "full_pipeline")
        self.assertEqual(task_v2["n_structures"], 77)
        self.assertEqual(task_v2["candidate_data"], "candidates.csv")
        self.assertIn("legacy request payload auto-mapped to task.v2", task_v2["compatibility_warnings"])

    def test_request_minimal_does_not_enforce_decision_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            payload = {
                "task_id": "task_1",
                "request_text": "design molecule",
                "mode": "fast_screen",
                "targets": [{"property": "plqy", "objective": "maximize"}],
                "budget": {"max_candidates": 10},
                "score_step": {
                    "used_fallback": True,
                    "fallback_code": None,
                    "fallback_retryable": None,
                },
            }
            schema = {
                "type": "object",
                "additionalProperties": True,
                "required": ["task_id", "request_text", "mode", "targets", "budget"],
                "properties": {
                    "task_id": {"type": "string"},
                    "request_text": {"type": "string"},
                    "mode": {"type": "string", "enum": ["fast_screen", "train_then_design"]},
                    "targets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["property", "objective"],
                            "properties": {
                                "property": {"type": "string", "enum": ["lambda_em", "plqy", "stability"]},
                                "objective": {"type": "string", "enum": ["maximize", "minimize", "target_window"]},
                            },
                        },
                    },
                    "budget": {"type": "object"},
                },
            }
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "jsonschema":
                    raise ImportError("force minimal")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                _validate_via_jsonschema(payload, schema, contract_kind="request")

    def test_decision_summary_schema_rejects_null_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-03T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                "score_step": {
                    "used_fallback": True,
                    "adapter": "local_deterministic_fallback",
                    "fallback_reason": "oops",
                    "fallback_code": None,
                    "fallback_retryable": True,
                    "fallback_details": {},
                },
            }
            with self.assertRaises(RequestValidationError):
                validate_decision_summary_payload(payload, workspace_root=td_path)

    def test_decision_summary_minimal_rejects_null_fallback_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-03T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                "score_step": {
                    "used_fallback": True,
                    "adapter": "local_deterministic_fallback",
                    "fallback_reason": "oops",
                    "fallback_code": None,
                    "fallback_retryable": None,
                    "fallback_details": {},
                },
            }
            schema = json.loads(
                (Path(__file__).resolve().parents[1] / "schemas" / "decision_summary.schema.json").read_text(encoding="utf-8")
            )
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "jsonschema":
                    raise ImportError("force minimal")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(RequestValidationError):
                    _validate_via_jsonschema(payload, schema, contract_kind="decision_summary")

    def test_data_report_schema_validates_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "dataset_step": {
                    "status": "success",
                    "selected": ["master_database"],
                    "available": ["master_database", "subsidiary_database"],
                },
                "candidate_step": {
                    "status": "success",
                    "adapter": "template_generate_cmd",
                    "rows": 1,
                    "source_csv": "a.csv",
                    "input_csv": "b.csv",
                },
                "artifacts": {"candidate_csv": "cand.csv", "scored_csv": "score.csv"},
            }
            self.assertEqual(validate_data_report_payload(payload, workspace_root=td_path), payload)

    def test_data_report_schema_allows_additional_properties(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "dataset_step": {"status": "success", "selected": [], "available": []},
                "candidate_step": {"status": "success", "rows": 1},
                "artifacts": {"candidate_csv": "cand.csv", "scored_csv": "score.csv"},
            }
            self.assertEqual(validate_data_report_payload(payload, workspace_root=td_path), payload)

    def test_data_report_schema_rejects_non_array_selected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "dataset_step": {"status": "success", "selected": "master_database", "available": []},
                "candidate_step": {"status": "success", "rows": 1},
                "artifacts": {"candidate_csv": "cand.csv", "scored_csv": "score.csv"},
            }
            with self.assertRaises(RequestValidationError):
                validate_data_report_payload(payload, workspace_root=td_path)

    def test_model_report_schema_validates_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                "training_step": {"ran": False, "status": "skipped", "adapter": "", "result": {}},
                "inference_step": {"status": "success", "adapter": "local_deterministic_fallback", "used_fallback": True, "fallback_error": {"code": "x"}, "result": {}},
            }
            self.assertEqual(validate_model_report_payload(payload, workspace_root=td_path), payload)

    def test_model_report_schema_rejects_missing_used_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                "training_step": {"ran": False, "status": "skipped"},
                "inference_step": {"status": "success", "adapter": "local_deterministic_fallback", "fallback_error": {"code": "x"}, "result": {}},
            }
            with self.assertRaises(RequestValidationError):
                validate_model_report_payload(payload, workspace_root=td_path)

    def test_filtering_report_schema_validates_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "filter_step": {"status": "success", "topn": 10, "manifest": "m.json", "final_output": "r.md"},
                "report_step": {"status": "success", "report": "r.md", "latest_run_dir": "runs/1"},
            }
            self.assertEqual(validate_filtering_report_payload(payload, workspace_root=td_path), payload)

    def test_filtering_report_schema_rejects_negative_topn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-09T00:00:00Z",
                "task_id": "task_1",
                "status": "success",
                "filter_step": {"status": "success", "topn": -1},
                "report_step": {"status": "success"},
            }
            with self.assertRaises(RequestValidationError):
                validate_filtering_report_payload(payload, workspace_root=td_path)

    def test_agent_plan_json_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_plan",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [
                            {"property": "lambda_em", "objective": "target_window", "target_min": 460, "target_max": 480},
                            {"property": "plqy", "objective": "maximize", "target_value": 0.6},
                        ],
                        "constraints": {"mw_max": 650},
                        "budget": {"max_candidates": 12},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_v1",
                            "generator_id": "reinvent4_lambda_em_v2",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["design_spec"]["task_id"], "task_json_plan")
            self.assertEqual(payload["design_spec"]["budget"]["max_candidates"], 12)
            self.assertEqual(len(payload["design_spec"]["targets"]), 2)

    def test_agent_plan_json_propagates_generation_input_to_generate_candidates_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            source_img = td_path / "figure1.png"
            source_img.write_bytes(b"\x89PNG\r\n\x1a\n")
            request_json = td_path / "request_plan_generation_input.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_plan_generation_input",
                        "request_text": "从论文图像提取候选分子并设计高PLQY",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "molscribe_generator_real_v1",
                        },
                        "generation_input": {
                            "source_image": str(source_img),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            gen_call = next(c for c in payload["tool_calls"] if c["name"] == "generate_candidates")
            self.assertEqual(gen_call["args"]["source_image"], str(source_img))
            self.assertEqual(gen_call["args"]["generator_id"], "molscribe_generator_real_v1")

    def test_agent_run_json_happy_path_writes_request_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_run.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_run",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_v1",
                            "generator_id": "reinvent4_lambda_em_v2",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["task_id"], "task_json_run")
            request_path = Path(payload["request_path"])
            self.assertTrue(request_path.exists())
            req_saved = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(req_saved["task_id"], "task_json_run")
            self.assertIn("task_state_path", payload)
            self.assertIn("logging_dir", payload)
            self.assertIn("result_dir", payload)
            self.assertTrue(Path(payload["task_state_path"]).exists())
            self.assertTrue(Path(payload["logging_dir"]).exists())
            self.assertTrue(Path(payload["result_dir"]).exists())
            self.assertIn("logging_data_report_path", payload)
            self.assertIn("logging_model_report_path", payload)
            self.assertIn("logging_filtering_report_path", payload)
            self.assertTrue(Path(payload["logging_data_report_path"]).exists())
            self.assertTrue(Path(payload["logging_model_report_path"]).exists())
            self.assertTrue(Path(payload["logging_filtering_report_path"]).exists())

    def test_agent_run_json_molscribe_smoke_uses_generation_input_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            source_img = td_path / "paper_figure.png"
            source_img.write_bytes(b"\x89PNG\r\n\x1a\n")
            request_json = td_path / "request_molscribe_input.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_molscribe_input",
                        "request_text": "从图片中提取分子并筛选高PLQY",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 4},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "molscribe_generator_real_v1",
                        },
                        "generation_input": {
                            "source_image": str(source_img),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_MOLSCRIBE_ADAPTER_MODE": "smoke",
                    "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "success")
            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            by_name = {r.get("name"): r for r in execution.get("records", []) if isinstance(r, dict)}
            gen_rec = by_name["generate_candidates"]
            self.assertEqual(gen_rec["status"], "success")
            self.assertEqual(gen_rec["result"].get("adapter"), "molscribe_generate_adapter_v1")
            self.assertEqual(gen_rec["args"].get("source_image"), str(source_img))

    def test_agent_run_json_molscribe_real_source_pdf_with_pdf_extract_cmd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)

            source_pdf = td_path / "paper.pdf"
            source_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
            marker = td_path / "pdf_extract_ran.marker"

            pdf_extract_script = td_path / "pdf_extract_stub.py"
            pdf_extract_script.write_text(
                (
                    "import argparse, os\n"
                    "ap=argparse.ArgumentParser()\n"
                    "ap.add_argument('--input-pdf', required=True)\n"
                    "ap.add_argument('--output-dir', required=True)\n"
                    "ap.add_argument('--index', required=True)\n"
                    "args=ap.parse_args()\n"
                    "os.makedirs(args.output_dir, exist_ok=True)\n"
                    "img=os.path.join(args.output_dir, f'extract_{args.index}.png')\n"
                    "open(img,'wb').write(b'\\x89PNG\\r\\n\\x1a\\n')\n"
                    "marker=os.environ.get('OLED_AGENT_TEST_PDF_EXTRACT_MARKER','')\n"
                    "if marker:\n"
                    "  open(marker,'w',encoding='utf-8').write('ok')\n"
                ),
                encoding="utf-8",
            )

            molscribe_cmd_script = td_path / "molscribe_cmd_stub.py"
            molscribe_cmd_script.write_text(
                (
                    "import argparse,csv,os\n"
                    "ap=argparse.ArgumentParser()\n"
                    "ap.add_argument('--output-csv', required=True)\n"
                    "ap.add_argument('--input', action='append', default=[])\n"
                    "args=ap.parse_args()\n"
                    "os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)\n"
                    "with open(args.output_csv,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','smiles','source'])\n"
                    "  w.writeheader()\n"
                    "  w.writerow({'candidate_id':'cand_000001','smiles':'c1ccccc1','source':'molscribe_cmd_stub'})\n"
                ),
                encoding="utf-8",
            )

            request_json = td_path / "request_molscribe_pdf.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_molscribe_pdf",
                        "request_text": "从PDF结构图提取分子并筛选高PLQY",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 3},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "molscribe_generator_real_v1",
                        },
                        "generation_input": {
                            "source_pdf": str(source_pdf),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_MOLSCRIBE_ADAPTER_MODE": "real",
                    "OLED_AGENT_MOLSCRIBE_CMD": f"{sys.executable} {molscribe_cmd_script}",
                    "OLED_AGENT_MOLSCRIBE_PDF_EXTRACT_CMD": (
                        f"{sys.executable} {pdf_extract_script} "
                        "--input-pdf {input_pdf} --output-dir {output_dir} --index {index}"
                    ),
                    "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke",
                    "OLED_AGENT_TEST_PDF_EXTRACT_MARKER": str(marker),
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "success")
            self.assertTrue(marker.exists(), msg="pdf extract command was not invoked")
            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            by_name = {r.get("name"): r for r in execution.get("records", []) if isinstance(r, dict)}
            gen_rec = by_name["generate_candidates"]
            self.assertEqual(gen_rec["status"], "success")
            self.assertEqual(gen_rec["result"].get("adapter"), "molscribe_generate_adapter_v1")
            self.assertEqual(gen_rec["args"].get("source_pdf"), str(source_pdf))

    def test_agent_run_json_llm_v1_molscribe_generation_input_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            source_img = td_path / "llm_run_source.png"
            source_img.write_bytes(b"\x89PNG\r\n\x1a\n")
            request_json = td_path / "request_llm_run_molscribe.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_run_molscribe",
                        "request_text": "从图片提取分子并进行高PLQY筛选",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 4},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "molscribe_generator_real_v1",
                        },
                        "generation_input": {"source_image": str(source_img)},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "active",
                    "OLED_AGENT_MOLSCRIBE_ADAPTER_MODE": "smoke",
                    "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "success")
            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            by_name = {r.get("name"): r for r in execution.get("records", []) if isinstance(r, dict)}
            gen_rec = by_name["generate_candidates"]
            self.assertEqual(gen_rec["status"], "success")
            self.assertEqual(gen_rec["args"].get("source_image"), str(source_img))
            self.assertEqual(gen_rec["result"].get("adapter"), "molscribe_generate_adapter_v1")

    def test_agent_intake_returns_need_user_input_when_missing_key_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            task_id = "task_intake_need_info"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-intake",
                    "--workspace-root",
                    str(repo_root),
                    "--task-id",
                    task_id,
                    "--request",
                    "帮我设计分子",
                    "--disable-web-search",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 2, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "need_user_input")
            self.assertIn("missing_fields", payload)
            self.assertIn("questions", payload)
            self.assertTrue(Path(payload["task_draft_path"]).exists())
            self.assertTrue(Path(payload["web_evidence_path"]).exists())
            self.assertTrue(Path(payload["task_state_path"]).exists())
            self.assertEqual(payload.get("current_state"), "NEED_INFO")
            task_state = json.loads(Path(payload["task_state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(task_state.get("current_state"), "NEED_INFO")
            self.assertEqual(task_state.get("status"), "failed")

    def test_agent_resume_from_draft_without_overrides_returns_need_user_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_id = "task_resume_draft_need_info"
            run_dir = td_path / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            draft = {
                "version": "2.0",
                "task_id": task_id,
                "request_text": "设计470nm附近且高PLQY分子",
                "execution_mode": "full_pipeline",
                "operation": "full_pipeline",
                "property": "plqy",
                "range": "458.0-482.0nm",
                "n_structures": 20,
                "constraints": {"mw_min": 150.0, "mw_max": 700.0, "domain_threshold": 0.2, "banned_alerts": []},
                "train_data": None,
                "candidate_data": None,
                "prediction_model": "unimol_lambda_plqy_v1",
                "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                "generation_input": {},
                "provenance": {},
                "status": "need_user_input",
                "missing_fields": ["candidate_data"],
                "questions": ["候选数据来源是什么？本地CSV路径还是数据库关键词？"],
                "compatibility_warnings": [],
            }
            (run_dir / "task.draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-resume",
                    "--workspace-root",
                    str(td_path),
                    "--task-id",
                    task_id,
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 2, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "need_user_input")
            missing = payload.get("missing_fields") if isinstance(payload.get("missing_fields"), list) else []
            self.assertIn("candidate_data", missing)
            self.assertEqual(payload.get("current_state"), "NEED_INFO")
            self.assertTrue(Path(payload["task_state_path"]).exists())

    def test_agent_resume_from_draft_with_candidate_data_override_executes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_id = "task_resume_draft_override"
            run_dir = td_path / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            candidate_csv = td_path / "candidates.csv"
            candidate_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")
            draft = {
                "version": "2.0",
                "task_id": task_id,
                "request_text": "设计470nm附近且高PLQY分子",
                "execution_mode": "full_pipeline",
                "operation": "full_pipeline",
                "property": "plqy",
                "range": "458.0-482.0nm",
                "n_structures": 20,
                "constraints": {"mw_min": 150.0, "mw_max": 700.0, "domain_threshold": 0.2, "banned_alerts": []},
                "train_data": None,
                "candidate_data": None,
                "prediction_model": "unimol_lambda_plqy_v1",
                "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                "generation_input": {},
                "provenance": {},
                "status": "need_user_input",
                "missing_fields": ["candidate_data"],
                "questions": ["候选数据来源是什么？本地CSV路径还是数据库关键词？"],
                "compatibility_warnings": [],
            }
            (run_dir / "task.draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-resume",
                    "--workspace-root",
                    str(td_path),
                    "--task-id",
                    task_id,
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--candidate-data",
                    str(candidate_csv),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "success")
            approved_task = json.loads((run_dir / "task.json").read_text(encoding="utf-8"))
            self.assertEqual(approved_task.get("candidate_data"), str(candidate_csv))
            self.assertTrue((run_dir / "request_from_task.json").exists())
            self.assertTrue((run_dir / "execution.json").exists())

    def test_agent_resume_skips_all_steps_when_task_already_successful(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_id = f"task_resume_already_success_{td_path.name[-8:]}"
            request_json = td_path / "request_resume_success.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_v1",
                            "generator_id": "reinvent4_lambda_em_v2",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            env = {**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"}
            cp_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp_run.returncode, 0, msg=cp_run.stderr + cp_run.stdout)
            run_payload = json.loads(cp_run.stdout)
            exec1 = json.loads(Path(run_payload["execution_path"]).read_text(encoding="utf-8"))
            records1 = exec1.get("records", [])
            self.assertTrue(isinstance(records1, list) and len(records1) > 0)

            cp_resume = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-resume",
                    "--workspace-root",
                    str(repo_root),
                    "--task-id",
                    task_id,
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp_resume.returncode, 0, msg=cp_resume.stderr + cp_resume.stdout)
            resume_payload = json.loads(cp_resume.stdout)
            self.assertTrue(bool(resume_payload.get("resumed")))
            self.assertEqual(int(resume_payload.get("resume_skipped_steps", -1)), len(records1))
            exec2 = json.loads(Path(resume_payload["execution_path"]).read_text(encoding="utf-8"))
            self.assertEqual(exec2.get("records", []), records1)
            self.assertEqual(exec2.get("status"), "success")

    def test_agent_resume_continues_from_first_unfinished_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_id = f"task_resume_partial_{td_path.name[-8:]}"
            counter_path = td_path / "gen_counter.txt"

            generate_script = td_path / "gen_count.py"
            generate_script.write_text(
                (
                    "import csv,json,os,sys\n"
                    "from pathlib import Path\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "counter=Path(os.environ.get('OLED_AGENT_TEST_GEN_COUNTER',''))\n"
                    "if str(counter):\n"
                    "  prev=0\n"
                    "  if counter.exists():\n"
                    "    try:\n"
                    "      prev=int(counter.read_text(encoding='utf-8').strip() or '0')\n"
                    "    except Exception:\n"
                    "      prev=0\n"
                    "  counter.write_text(str(prev+1), encoding='utf-8')\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','smiles'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','smiles':'c1ccccc1'})\n"
                    "print(json.dumps({'status':'success','adapter':'resume_test_generate','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            score_ok_script = td_path / "score_ok.py"
            score_ok_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "inp=payload['input_csv']; out=payload['output_csv']\n"
                    "rows=list(csv.DictReader(open(inp,'r',encoding='utf-8')))\n"
                    "for r in rows:\n"
                    "  r['plqy_pred']='0.81'; r['plqy_score']='0.81'\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)\n"
                    "print(json.dumps({'status':'success','adapter':'resume_test_score','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )

            catalog_json = td_path / "catalog_resume.json"
            catalog_payload = {
                "models": [
                    {
                        "id": "pred_resume",
                        "kind": "predictor",
                        "backend": "mock_predictor",
                        "task_types": ["plqy"],
                        "runtime_profile": "cpu",
                        "params": {"adapters": {"score_candidates_cmd": f"{sys.executable} {score_ok_script}"}},
                    },
                    {
                        "id": "gen_resume",
                        "kind": "generator",
                        "backend": "mock_generator",
                        "task_types": ["molecule_generation"],
                        "runtime_profile": "cpu",
                        "params": {"adapters": {"generate_candidates_cmd": f"{sys.executable} {generate_script}"}},
                    },
                ]
            }
            catalog_json.write_text(json.dumps(catalog_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            request_json = td_path / "request_resume_partial.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
                        "budget": {"max_candidates": 6},
                        "model_preferences": {
                            "predictor_id": "pred_resume",
                            "generator_id": "gen_resume",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(repo_root / "src"),
                "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0",
                "OLED_AGENT_TEST_GEN_COUNTER": str(counter_path),
            }

            cp_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(catalog_json),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp_run.returncode, 0, msg=cp_run.stderr + cp_run.stdout)
            run_payload = json.loads(cp_run.stdout)
            self.assertEqual(counter_path.read_text(encoding="utf-8").strip(), "1")

            execution_path = Path(run_payload["execution_path"])
            execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
            records = execution_payload.get("records", [])
            self.assertTrue(isinstance(records, list) and len(records) >= 3)
            partial_records = records[:-2]
            execution_payload["records"] = partial_records
            execution_payload["status"] = "failed"
            execution_path.write_text(json.dumps(execution_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            cp_resume = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-resume",
                    "--workspace-root",
                    str(repo_root),
                    "--task-id",
                    task_id,
                    "--catalog",
                    str(catalog_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp_resume.returncode, 0, msg=cp_resume.stderr + cp_resume.stdout)
            resume_payload = json.loads(cp_resume.stdout)
            self.assertTrue(bool(resume_payload.get("resumed")))
            self.assertEqual(int(resume_payload.get("resume_skipped_steps", -1)), len(partial_records))
            self.assertEqual(counter_path.read_text(encoding="utf-8").strip(), "1")

            execution = json.loads(Path(resume_payload["execution_path"]).read_text(encoding="utf-8"))
            self.assertEqual(execution.get("status"), "success")
            final_records = execution.get("records", [])
            self.assertEqual(len(final_records), len(records))
            self.assertEqual(final_records[: len(partial_records)], partial_records)
            self.assertTrue(all(isinstance(r, dict) and r.get("status") == "success" for r in final_records))

    def test_agent_run_step_happy_path_clean_dataset_writes_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            input_csv = td_path / "step_input.csv"
            input_csv.write_text(
                "candidate_id,SMILES\n"
                "c1,c1ccccc1\n"
                "c2,c1ccccc1\n",
                encoding="utf-8",
            )
            task_json = td_path / "task_step.json"
            task_json.write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": "task_step_happy",
                        "request_text": "clean candidates",
                        "execution_mode": "single_step",
                        "operation": "clean_dataset",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 10,
                        "constraints": {"mw_min": 100, "mw_max": 900},
                        "prediction_model": "unimol_lambda_plqy_v1",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-step",
                    "--workspace-root",
                    str(repo_root),
                    "--task-json",
                    str(task_json),
                    "--operation",
                    "clean_dataset",
                    "--args-json",
                    json.dumps({"input_csv": str(input_csv)}),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "success")
            for key in (
                "execution_path",
                "tool_state_path",
                "decision_summary_path",
                "task_state_path",
                "experiment_trace_path",
                "logging_data_report_path",
                "logging_model_report_path",
                "logging_filtering_report_path",
                "logging_experiment_trace_path",
                "result_metadata_path",
                "result_experiment_trace_path",
            ):
                self.assertTrue(Path(payload[key]).exists(), msg=f"missing artifact: {key}")
            trace_payload = json.loads(Path(payload["experiment_trace_path"]).read_text(encoding="utf-8"))
            self.assertEqual(trace_payload.get("task_id"), "task_step_happy")
            self.assertEqual(trace_payload.get("execution_mode"), "single_step")

    def test_agent_run_step_failure_path_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_json = td_path / "task_step_fail.json"
            task_json.write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": "task_step_fail",
                        "request_text": "clean candidates",
                        "execution_mode": "single_step",
                        "operation": "clean_dataset",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 10,
                        "constraints": {},
                        "prediction_model": "unimol_lambda_plqy_v1",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-step",
                    "--workspace-root",
                    str(repo_root),
                    "--task-json",
                    str(task_json),
                    "--operation",
                    "clean_dataset",
                    "--args-json",
                    json.dumps({"input_csv": str(td_path / "missing_input.csv")}),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 1, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "failed")
            self.assertIn("not found", str(payload.get("error", "")))

    def test_agent_run_json_require_real_adapters_fails_on_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            generate_script = td_path / "gen_ok.py"
            generate_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','smiles'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','smiles':'c1ccccc1'})\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_generate_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            score_script = td_path / "score_fail.py"
            score_script.write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(2)\n", encoding="utf-8")
            catalog_json = td_path / "catalog.json"
            catalog_json.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "pred_bad_score",
                                "kind": "predictor",
                                "backend": "mock_predictor",
                                "task_types": ["plqy"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "score_candidates_cmd": f"{sys.executable} {score_script}",
                                    }
                                },
                            },
                            {
                                "id": "gen_ok",
                                "kind": "generator",
                                "backend": "mock_generator",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "generate_candidates_cmd": f"{sys.executable} {generate_script}",
                                    }
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            request_json = td_path / "request_real_required.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_require_real_fail",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
                        "budget": {"max_candidates": 6},
                        "model_preferences": {
                            "predictor_id": "pred_bad_score",
                            "generator_id": "gen_ok",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(catalog_json),
                    "--request-json",
                    str(request_json),
                    "--require-real-adapters",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 3, msg=cp.stderr + cp.stdout)
            self.assertIn("require-real-adapters", cp.stdout)

    def test_agent_run_step_require_real_adapters_fails_on_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_id = f"task_step_require_real_{td_path.name[-8:]}"
            input_csv = td_path / "step_input.csv"
            input_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")
            task_json = td_path / "task_step_require_real.json"
            task_json.write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": task_id,
                        "request_text": "score step strict no fallback",
                        "execution_mode": "single_step",
                        "operation": "score_candidates",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 10,
                        "constraints": {},
                        "prediction_model": "unimol_lambda_plqy_real_v1",
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "reinvent4_lambda_em_v2",
                        },
                        "status": "approved",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            env = {**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"}
            cp_retrieve = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-step",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--task-json",
                    str(task_json),
                    "--operation",
                    "retrieve_candidate_data",
                    "--args-json",
                    json.dumps({"candidate_data": str(input_csv)}),
                    "--require-real-adapters",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp_retrieve.returncode, 0, msg=cp_retrieve.stderr + cp_retrieve.stdout)

            cp_score = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-step",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--task-json",
                    str(task_json),
                    "--operation",
                    "score_candidates",
                    "--require-real-adapters",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**env, "OLED_AGENT_SCORE_CMD": f"{sys.executable} -c 'import sys; sys.exit(7)'"},
            )
            self.assertEqual(cp_score.returncode, 3, msg=cp_score.stderr + cp_score.stdout)
            self.assertIn("require-real-adapters", cp_score.stdout)

    def test_agent_run_step_json_require_real_adapters_fails_on_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_id = f"task_step_json_require_real_{td_path.name[-8:]}"
            input_csv = td_path / "step_json_input.csv"
            input_csv.write_text("candidate_id,SMILES\nc1,c1ccccc1\n", encoding="utf-8")
            task_payload = {
                "version": "2.0",
                "task_id": task_id,
                "request_text": "score step-json strict no fallback",
                "execution_mode": "single_step",
                "operation": "score_candidates",
                "property": "plqy",
                "range": "60-100",
                "n_structures": 10,
                "constraints": {},
                "prediction_model": "unimol_lambda_plqy_real_v1",
                "model_preferences": {
                    "predictor_id": "unimol_lambda_plqy_real_v1",
                    "generator_id": "reinvent4_lambda_em_v2",
                },
                "status": "approved",
            }
            retrieve_req = td_path / "step_retrieve.json"
            retrieve_req.write_text(
                json.dumps(
                    {
                        "task": task_payload,
                        "operation": "retrieve_candidate_data",
                        "args": {"candidate_data": str(input_csv)},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            score_req = td_path / "step_score.json"
            score_req.write_text(
                json.dumps(
                    {
                        "task": task_payload,
                        "operation": "score_candidates",
                        "args": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            env = {**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"}
            cp_retrieve = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-step-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--step-request-json",
                    str(retrieve_req),
                    "--require-real-adapters",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp_retrieve.returncode, 0, msg=cp_retrieve.stderr + cp_retrieve.stdout)

            cp_score = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-step-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--step-request-json",
                    str(score_req),
                    "--require-real-adapters",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**env, "OLED_AGENT_SCORE_CMD": f"{sys.executable} -c 'import sys; sys.exit(7)'"},
            )
            self.assertEqual(cp_score.returncode, 3, msg=cp_score.stderr + cp_score.stdout)
            self.assertIn("require-real-adapters", cp_score.stdout)

    def test_agent_run_json_uses_catalog_generate_and_score_adapters_when_env_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)

            generate_script = td_path / "generate_from_catalog.py"
            generate_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','smiles'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','smiles':'c1ccccc1'})\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_generate_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )

            score_script = td_path / "score_from_catalog.py"
            score_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "inp=payload['input_csv']; out=payload['output_csv']\n"
                    "rows=list(csv.DictReader(open(inp,'r',encoding='utf-8')))\n"
                    "for r in rows:\n"
                    "  r['plqy_pred']='0.73'; r['plqy_score']='0.73'\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_score_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )

            catalog_json = td_path / "catalog.json"
            catalog_json.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "pred_catalog_v1",
                                "kind": "predictor",
                                "backend": "mock_predictor",
                                "task_types": ["plqy"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "score_candidates_cmd": f"{sys.executable} {score_script}",
                                    }
                                },
                            },
                            {
                                "id": "gen_catalog_v1",
                                "kind": "generator",
                                "backend": "mock_generator",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "generate_candidates_cmd": f"{sys.executable} {generate_script}",
                                    }
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            request_json = td_path / "request_catalog_adapter_run.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_catalog_adapter",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "pred_catalog_v1",
                            "generator_id": "gen_catalog_v1",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(td_path),
                    "--catalog",
                    str(catalog_json),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_TRAIN_CMD": "",
                    "OLED_AGENT_GENERATE_CMD": "",
                    "OLED_AGENT_SCORE_CMD": "",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "success")
            execution_path = Path(payload["execution_path"])
            self.assertTrue(execution_path.exists())

            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            records = execution.get("records", [])
            by_name = {r.get("name"): r for r in records if isinstance(r, dict)}
            self.assertEqual(by_name["generate_candidates"]["result"]["adapter"], "catalog_generate_cmd")
            self.assertEqual(by_name["score_candidates"]["result"]["adapter"], "catalog_score_cmd")

    def test_agent_run_json_score_catalog_adapter_failure_falls_back_to_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)

            generate_script = td_path / "generate_ok.py"
            generate_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "out=payload['output_csv']\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=['candidate_id','SMILES'])\n"
                    "  w.writeheader(); w.writerow({'candidate_id':'cand_000001','SMILES':'c1ccccc1'})\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_generate_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )
            score_bad_script = td_path / "score_bad.py"
            score_bad_script.write_text(
                "import sys\nsys.exit(3)\n",
                encoding="utf-8",
            )

            catalog_json = td_path / "catalog_bad_score.json"
            catalog_json.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "pred_catalog_bad_score",
                                "kind": "predictor",
                                "backend": "mock_predictor",
                                "task_types": ["plqy"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "score_candidates_cmd": f"{sys.executable} {score_bad_script}",
                                    }
                                },
                            },
                            {
                                "id": "gen_catalog_ok",
                                "kind": "generator",
                                "backend": "mock_generator",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "generate_candidates_cmd": f"{sys.executable} {generate_script}",
                                    }
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            request_json = td_path / "request_bad_score_adapter.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_bad_score_adapter",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "pred_catalog_bad_score",
                            "generator_id": "gen_catalog_ok",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(td_path),
                    "--catalog",
                    str(catalog_json),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_TRAIN_CMD": "",
                    "OLED_AGENT_GENERATE_CMD": "",
                    "OLED_AGENT_SCORE_CMD": "",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "success")

            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            records = execution.get("records", [])
            by_name = {r.get("name"): r for r in records if isinstance(r, dict)}
            score_result = by_name["score_candidates"]["result"]
            self.assertEqual(score_result.get("adapter"), "local_deterministic_fallback")
            self.assertIn("fallback_error", score_result)
            self.assertEqual(score_result["fallback_error"].get("code"), "external_score_cmd_failed")

    def test_agent_run_json_generate_catalog_adapter_failure_fails_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)

            generate_bad_script = td_path / "generate_bad.py"
            generate_bad_script.write_text(
                "import sys\nsys.exit(4)\n",
                encoding="utf-8",
            )
            score_script = td_path / "score_unused.py"
            score_script.write_text(
                (
                    "import csv,json,sys\n"
                    "payload=json.loads(sys.stdin.read())\n"
                    "inp=payload['input_csv']; out=payload['output_csv']\n"
                    "rows=list(csv.DictReader(open(inp,'r',encoding='utf-8')))\n"
                    "for r in rows:\n"
                    "  r['plqy_pred']='0.61'; r['plqy_score']='0.61'\n"
                    "with open(out,'w',encoding='utf-8',newline='') as f:\n"
                    "  w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)\n"
                    "print(json.dumps({'status':'success','adapter':'catalog_score_cmd','output_csv':out}))\n"
                ),
                encoding="utf-8",
            )

            catalog_json = td_path / "catalog_bad_generate.json"
            catalog_json.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "pred_catalog_ok",
                                "kind": "predictor",
                                "backend": "mock_predictor",
                                "task_types": ["plqy"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "score_candidates_cmd": f"{sys.executable} {score_script}",
                                    }
                                },
                            },
                            {
                                "id": "gen_catalog_bad_generate",
                                "kind": "generator",
                                "backend": "mock_generator",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "generate_candidates_cmd": f"{sys.executable} {generate_bad_script}",
                                    }
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            request_json = td_path / "request_bad_generate_adapter.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_bad_generate_adapter",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "pred_catalog_ok",
                            "generator_id": "gen_catalog_bad_generate",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(td_path),
                    "--catalog",
                    str(catalog_json),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_TRAIN_CMD": "",
                    "OLED_AGENT_GENERATE_CMD": "",
                    "OLED_AGENT_SCORE_CMD": "",
                },
            )
            self.assertEqual(cp.returncode, 1, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "failed")

            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            records = execution.get("records", [])
            self.assertGreaterEqual(len(records), 1)
            by_name = {r.get("name"): r for r in records if isinstance(r, dict)}
            self.assertIn("generate_candidates", by_name)
            self.assertEqual(by_name["generate_candidates"].get("status"), "failed")
            self.assertIn("returned non-zero exit status", by_name["generate_candidates"].get("error", ""))
            self.assertNotIn("score_candidates", by_name)

    def test_agent_run_json_with_repo_adapter_templates_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)

            generate_tpl = repo_root / "scripts" / "adapters" / "generate_candidates_adapter_template.py"
            score_tpl = repo_root / "scripts" / "adapters" / "score_candidates_adapter_template.py"
            self.assertTrue(generate_tpl.exists())
            self.assertTrue(score_tpl.exists())

            catalog_json = td_path / "catalog_tpl.json"
            catalog_json.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "pred_tpl_v1",
                                "kind": "predictor",
                                "backend": "mock_predictor",
                                "task_types": ["plqy"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "score_candidates_cmd": f"{sys.executable} {score_tpl}",
                                    }
                                },
                            },
                            {
                                "id": "gen_tpl_v1",
                                "kind": "generator",
                                "backend": "mock_generator",
                                "task_types": ["molecule_generation"],
                                "runtime_profile": "cpu",
                                "params": {
                                    "adapters": {
                                        "generate_candidates_cmd": f"{sys.executable} {generate_tpl}",
                                    }
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            request_json = td_path / "request_tpl_smoke.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_tpl_smoke",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "pred_tpl_v1",
                            "generator_id": "gen_tpl_v1",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(td_path),
                    "--catalog",
                    str(catalog_json),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_TRAIN_CMD": "",
                    "OLED_AGENT_GENERATE_CMD": "",
                    "OLED_AGENT_SCORE_CMD": "",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "success")

            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            records = execution.get("records", [])
            by_name = {r.get("name"): r for r in records if isinstance(r, dict)}
            self.assertEqual(by_name["generate_candidates"]["result"]["adapter"], "template_generate_cmd")
            self.assertEqual(by_name["score_candidates"]["result"]["adapter"], "template_score_cmd")

    def test_agent_run_json_with_quickstart_catalog_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)

            quickstart_catalog = repo_root / "scripts" / "adapters" / "quickstart_catalog.json"
            self.assertTrue(quickstart_catalog.exists())

            request_json = td_path / "request_quickstart_smoke.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_quickstart_smoke",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 5},
                        "model_preferences": {
                            "predictor_id": "pred_tpl_v1",
                            "generator_id": "gen_tpl_v1",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-run-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(quickstart_catalog),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_TRAIN_CMD": "",
                    "OLED_AGENT_GENERATE_CMD": "",
                    "OLED_AGENT_SCORE_CMD": "",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["status"], "success")
            execution = json.loads(Path(payload["execution_path"]).read_text(encoding="utf-8"))
            by_name = {
                r.get("name"): r for r in execution.get("records", []) if isinstance(r, dict)
            }
            self.assertEqual(by_name["generate_candidates"]["result"]["adapter"], "template_generate_cmd")
            self.assertEqual(by_name["score_candidates"]["result"]["adapter"], "template_score_cmd")

    def test_agent_plan_json_rejects_invalid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_invalid_mode.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_bad_mode",
                        "request_text": "design molecule",
                        "mode": "bad_mode",
                        "targets": [{"property": "plqy", "objective": "maximize"}],
                        "budget": {"max_candidates": 10},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 2)
            self.assertIn("[FAIL] invalid request json", cp.stdout)

    def test_request_minimal_generation_input_rejects_extra_field_when_jsonschema_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "task_id": "task_gen_input_minimal",
                "request_text": "design molecule from figure",
                "mode": "fast_screen",
                "targets": [{"property": "plqy", "objective": "maximize"}],
                "budget": {"max_candidates": 8},
                "generation_input": {"unknown_key": "x"},
            }
            real_import = builtins.__import__

            def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "jsonschema":
                    raise ImportError("forced missing jsonschema")
                return real_import(name, globals, locals, fromlist, level)

            with mock.patch("builtins.__import__", side_effect=_fake_import):
                with self.assertRaises(RequestValidationError):
                    validate_request_payload(payload=payload, workspace_root=td_path)

    def test_request_minimal_generation_input_rejects_non_string_array_item_when_jsonschema_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "task_id": "task_gen_input_minimal_bad_array_item",
                "request_text": "design molecule from figure",
                "mode": "fast_screen",
                "targets": [{"property": "plqy", "objective": "maximize"}],
                "budget": {"max_candidates": 8},
                "generation_input": {"source_images": ["ok.png", 42]},
            }
            real_import = builtins.__import__

            def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "jsonschema":
                    raise ImportError("forced missing jsonschema")
                return real_import(name, globals, locals, fromlist, level)

            with mock.patch("builtins.__import__", side_effect=_fake_import):
                with self.assertRaises(RequestValidationError) as cm:
                    validate_request_payload(payload=payload, workspace_root=td_path)
            self.assertIn("$.generation_input.source_images[2]: must be string", str(cm.exception))

    def test_agent_plan_json_rejects_unknown_generator_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_invalid_model.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_bad_model",
                        "request_text": "design molecule",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize"}],
                        "budget": {"max_candidates": 10},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_v1",
                            "generator_id": "not_exists_model",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 2)
            self.assertIn("[FAIL] invalid request json", cp.stdout)

    def test_agent_plan_json_plqy_target_value_ratio_is_normalized_to_percent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_plqy_ratio.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_plqy_ratio",
                        "request_text": "design molecule",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 10},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_v1",
                            "generator_id": "reinvent4_lambda_em_v2",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            plqy_target = next(t for t in payload["design_spec"]["targets"] if t["name"] == "plqy")
            self.assertEqual(plqy_target["target_center"], 60.0)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md.get("plqy_scale"), "percent_0_100")
            self.assertIn("targets[0].target_value", md.get("plqy_scale_converted_fields", []))

    def test_agent_plan_json_plqy_target_value_percent_kept_as_is(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_plqy_percent.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_plqy_percent",
                        "request_text": "design molecule",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 60.0}],
                        "budget": {"max_candidates": 10},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_v1",
                            "generator_id": "reinvent4_lambda_em_v2",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            plqy_target = next(t for t in payload["design_spec"]["targets"] if t["name"] == "plqy")
            self.assertEqual(plqy_target["target_center"], 60.0)
            md = payload["design_spec"]["metadata"]
            self.assertNotIn("plqy_scale_converted_fields", md)

    def test_agent_plan_json_llm_provider_preserves_generation_input_into_generate_candidates_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            source_img = td_path / "llm_source.png"
            source_img.write_bytes(b"\x89PNG\r\n\x1a\n")
            request_json = td_path / "request_llm_generation_input.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_generation_input",
                        "request_text": "从图像提取分子并进行设计",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "molscribe_generator_real_v1",
                        },
                        "generation_input": {"source_image": str(source_img)},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "active",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            gen_call = next(c for c in payload["tool_calls"] if c["name"] == "generate_candidates")
            self.assertEqual(gen_call["args"].get("source_image"), str(source_img))

    def test_agent_plan_default_planner_provider_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_default",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "rule_based_v1")
            self.assertEqual(md["planner_provider_requested"], "rule_based_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "active")

    def test_agent_plan_llm_provider_falls_back_to_rule_based(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "rule_based_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_provider_not_implemented")

    def test_agent_plan_json_llm_provider_falls_back_to_rule_based(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "request_contract_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_provider_not_implemented")

    def test_agent_plan_json_llm_provider_active_with_mock_command(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_active.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_active",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "active",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["summary"], "Mock LLM planner output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "llm_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            self.assertNotIn("planner_provider_reason", md)

    def test_agent_plan_json_llm_provider_command_failure_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_cmd_fail.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_cmd_fail",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "exit_nonzero",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "request_contract_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_command_failed")

    def test_agent_plan_json_llm_provider_invalid_output_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_bad_output.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_bad_output",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "bad_json",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "request_contract_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_json_llm_provider_preserves_generation_input_into_generate_candidates_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            source_img = td_path / "llm_source.png"
            source_img.write_bytes(b"\x89PNG\r\n\x1a\n")
            request_json = td_path / "request_llm_generation_input.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_generation_input",
                        "request_text": "从图像提取分子并进行设计",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                        "model_preferences": {
                            "predictor_id": "unimol_lambda_plqy_real_v1",
                            "generator_id": "molscribe_generator_real_v1",
                        },
                        "generation_input": {"source_image": str(source_img)},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "active",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            gen_call = next(c for c in payload["tool_calls"] if c["name"] == "generate_candidates")
            self.assertEqual(gen_call["args"].get("source_image"), str(source_img))

    def test_agent_plan_json_llm_provider_bad_tools_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_bad_tools.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_bad_tools",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "bad_tools",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "request_contract_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_json_llm_provider_load_model_catalog_alias_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_alias_tools.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_alias_tools",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "alias_load_model_catalog",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "llm_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            list_model_calls = [c for c in payload["tool_calls"] if c["name"] == "list_models"]
            self.assertEqual(len(list_model_calls), 2)
            kinds = sorted(c["args"].get("kind") for c in list_model_calls)
            self.assertEqual(kinds, ["generator", "predictor"])

    def test_agent_plan_json_llm_provider_active_with_extra_tool_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_extra_args.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_extra_args",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "active_with_extra_args",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "llm_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            self.assertNotIn("planner_provider_reason", md)
            list_models_calls = [c for c in payload["tool_calls"] if c["name"] == "list_models"]
            self.assertEqual(len(list_models_calls), 2)
            self.assertEqual(sorted(c["args"].get("kind") for c in list_models_calls), ["generator", "predictor"])
            by_name = {c["name"]: c for c in payload["tool_calls"]}
            self.assertEqual(
                by_name["generate_candidates"]["args"],
                {"generator_id": "reinvent4_lambda_em_v2", "max_candidates": 6, "constraints": {}},
            )
            self.assertEqual(
                by_name["score_candidates"]["args"],
                {"predictor_id": "unimol_lambda_plqy_v1", "targets": ["plqy"]},
            )
            self.assertEqual(by_name["filter_and_rank"]["args"], {"topn": 10})
            self.assertEqual(by_name["make_report"]["args"], {})

    def test_agent_plan_json_llm_provider_bad_model_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            request_json = td_path / "request_llm_provider_bad_model.json"
            request_json.write_text(
                json.dumps(
                    {
                        "task_id": "task_json_llm_provider_bad_model",
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                        "budget": {"max_candidates": 6},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan-json",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--request-json",
                    str(request_json),
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "bad_model",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "request_contract_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_llm_provider_active_with_repo_mock_script(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            mock_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_active_plain",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {mock_script}",
                    "MOCK_LLM_MODE": "active",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["summary"], "Mock LLM planner output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "llm_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            self.assertNotIn("planner_provider_reason", md)

    def test_agent_plan_llm_provider_invalid_output_fallback_plain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_bad_plain",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "bad_json",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "rule_based_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_llm_provider_exit_nonzero_fallback_plain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_exit_nonzero_plain",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "MOCK_LLM_MODE": "exit_nonzero",
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner"], "rule_based_v1")
            self.assertEqual(md["planner_provider_requested"], "llm_v1")
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_command_failed")

    def test_agent_plan_llm_backend_not_configured_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env.pop("OLED_AGENT_LLM_BACKEND", None)
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_backend_none",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_provider_not_implemented")

    def test_agent_plan_llm_backend_failed_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-test"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "1"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_backend_fail",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_backend_failed")

    def test_agent_plan_llm_backend_openai_compat_active_with_http_mock(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            request_capture: dict[str, object] = {}
            from oled_agent.agent.planner import build_plan

            def fake_urlopen(req, timeout=None):
                body = req.data.decode("utf-8")
                request_capture["request"] = json.loads(body)
                request_capture["headers"] = dict(req.headers)
                llm_content = json.dumps(
                    {
                        "summary": "HTTP mock llm planner output",
                        "design_spec": {
                            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                            "constraints": {},
                            "budget": {"max_candidates": 6},
                            "model_choice": {
                                "predictor_id": "unimol_lambda_plqy_v1",
                                "generator_id": "reinvent4_lambda_em_v2",
                            },
                        },
                        "tool_calls": [
                            {"name": "list_models", "args": {"kind": "predictor"}},
                            {"name": "list_models", "args": {"kind": "generator"}},
                            {"name": "search_dataset", "args": {"preferences": ["master_database"]}},
                            {"name": "generate_candidates", "args": {"generator_id": "reinvent4_lambda_em_v2", "max_candidates": 6, "constraints": {}}},
                            {"name": "score_candidates", "args": {"predictor_id": "unimol_lambda_plqy_v1", "targets": ["plqy"]}},
                            {"name": "filter_and_rank", "args": {"topn": 10}},
                            {"name": "make_report", "args": {}},
                        ],
                    },
                    ensure_ascii=False,
                )
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": llm_content}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://mock.local/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_active_http_mock",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            self.assertIn("request", request_capture)
            req_payload = request_capture["request"]
            self.assertIsInstance(req_payload, dict)
            headers = request_capture.get("headers")
            self.assertIsInstance(headers, dict)
            self.assertEqual(headers.get("Authorization"), "Bearer test-key")
            payload = plan.to_dict()
            self.assertEqual(payload["summary"], "HTTP mock llm planner output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")
            self.assertNotIn("planner_provider_reason", md)

    def test_agent_plan_llm_backend_openai_compat_custom_proxy_headers_and_path(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            from oled_agent.agent.planner import build_plan

            capture: dict[str, object] = {}

            def fake_urlopen(req, timeout=None):
                capture["url"] = req.full_url
                capture["headers"] = dict(req.headers)
                llm_content = json.dumps(
                    {
                        "summary": "proxy route output",
                        "design_spec": {
                            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                            "constraints": {},
                            "budget": {"max_candidates": 6},
                            "model_choice": {
                                "predictor_id": "unimol_lambda_plqy_v1",
                                "generator_id": "reinvent4_lambda_em_v2",
                            },
                        },
                        "tool_calls": [
                            {"name": "list_models", "args": {"kind": "predictor"}},
                            {"name": "list_models", "args": {"kind": "generator"}},
                            {"name": "search_dataset", "args": {"preferences": ["master_database"]}},
                            {"name": "generate_candidates", "args": {"generator_id": "reinvent4_lambda_em_v2", "max_candidates": 6, "constraints": {}}},
                            {"name": "score_candidates", "args": {"predictor_id": "unimol_lambda_plqy_v1", "targets": ["plqy"]}},
                            {"name": "filter_and_rank", "args": {"topn": 10}},
                            {"name": "make_report", "args": {}},
                        ],
                    },
                    ensure_ascii=False,
                )
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": llm_content}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "https://proxy.local/api"
            env["OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH"] = "/v1/chat/completions"
            env["OLED_AGENT_LLM_AUTH_HEADER"] = "X-API-Key"
            env["OLED_AGENT_LLM_AUTH_SCHEME"] = ""
            env["OLED_AGENT_LLM_EXTRA_HEADERS_JSON"] = '{"X-Client":"agent4mat","X-Trace":"ci"}'
            env["OLED_AGENT_LLM_DISABLE_RESPONSE_FORMAT"] = "1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_proxy_custom",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            self.assertEqual(capture.get("url"), "https://proxy.local/api/v1/chat/completions")
            headers = capture.get("headers")
            self.assertIsInstance(headers, dict)
            self.assertEqual(headers.get("X-api-key"), "test-key")
            self.assertEqual(headers.get("X-client"), "agent4mat")
            self.assertEqual(headers.get("X-trace"), "ci")
            payload = plan.to_dict()
            self.assertEqual(payload["summary"], "proxy route output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")

    def test_agent_plan_llm_backend_invalid_extra_headers_json_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-test"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            env["OLED_AGENT_LLM_EXTRA_HEADERS_JSON"] = "[1,2,3]"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_backend_invalid_extra_headers",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_backend_failed")

    def test_agent_plan_llm_backend_openai_compat_invalid_content_fallback(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]

            def fake_urlopen(req, timeout=None):
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "NOT_JSON"}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://mock.local/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"

            from oled_agent.agent.planner import build_plan

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_invalid_content",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            payload = plan.to_dict()
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_llm_backend_openai_compat_function_style_tool_calls_normalized(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            from oled_agent.agent.planner import build_plan

            def fake_urlopen(req, timeout=None):
                llm_content = json.dumps(
                    {
                        "summary": "function-style output",
                        "design_spec": {
                            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                            "constraints": {},
                            "budget": {"max_candidates": 6},
                            "model_choice": {
                                "predictor_id": "unimol_lambda_plqy_v1",
                                "generator_id": "reinvent4_lambda_em_v2",
                            },
                        },
                        "tool_calls": [
                            {"type": "function", "function": {"name": "list_models", "arguments": "{\"kind\":\"predictor\"}"}},
                            {"type": "function", "function": {"name": "list_models", "arguments": "{\"kind\":\"generator\"}"}},
                            {"type": "function", "function": {"name": "search_dataset", "arguments": "{\"preferences\":[\"master_database\"]}"}},
                            {"type": "function", "function": {"name": "generate_candidates", "arguments": "{\"generator_id\":\"reinvent4_lambda_em_v2\",\"max_candidates\":6,\"constraints\":{}}"}},
                            {"type": "function", "function": {"name": "score_candidates", "arguments": "{\"predictor_id\":\"unimol_lambda_plqy_v1\",\"targets\":[\"plqy\"]}"}},
                            {"type": "function", "function": {"name": "filter_and_rank", "arguments": "{\"topn\":10}"}},
                            {"type": "function", "function": {"name": "make_report", "arguments": "{}"}},
                        ],
                    },
                    ensure_ascii=False,
                )
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": llm_content}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://mock.local/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_function_style",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            payload = plan.to_dict()
            self.assertEqual(payload["summary"], "function-style output")
            self.assertEqual(payload["tool_calls"][0]["name"], "list_models")
            self.assertEqual(payload["tool_calls"][0]["args"], {"kind": "predictor"})
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")

    def test_agent_plan_llm_backend_bad_tool_calls_classified_as_llm_output_invalid(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            from oled_agent.agent.planner import build_plan

            def fake_urlopen(req, timeout=None):
                llm_content = json.dumps(
                    {
                        "summary": "bad-tool-calls",
                        "design_spec": {
                            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                            "constraints": {},
                            "budget": {"max_candidates": 6},
                            "model_choice": {
                                "predictor_id": "unimol_lambda_plqy_v1",
                                "generator_id": "reinvent4_lambda_em_v2",
                            },
                        },
                        "tool_calls": [{"name": "list_models", "args": {"kind": "predictor"}}, {"args": {"kind": "generator"}}],
                    },
                    ensure_ascii=False,
                )
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": llm_content}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://mock.local/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_bad_tool_calls_reason",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            payload = plan.to_dict()
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_llm_backend_openai_compat_retries_without_response_format(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            from oled_agent.agent.planner import build_plan
            import urllib.error

            call_state = {"n": 0}

            def fake_urlopen(req, timeout=None):
                call_state["n"] += 1
                req_payload = json.loads(req.data.decode("utf-8"))
                if call_state["n"] == 1:
                    self.assertIn("response_format", req_payload)
                    err_body = json.dumps({"error": {"message": "response_format unsupported"}}).encode("utf-8")
                    raise urllib.error.HTTPError(
                        url=req.full_url,
                        code=400,
                        msg="Bad Request",
                        hdrs=None,
                        fp=mock.Mock(read=lambda: err_body),
                    )
                self.assertNotIn("response_format", req_payload)
                llm_content = json.dumps(
                    {
                        "summary": "HTTP retry fallback output",
                        "design_spec": {
                            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                            "constraints": {},
                            "budget": {"max_candidates": 6},
                            "model_choice": {
                                "predictor_id": "unimol_lambda_plqy_v1",
                                "generator_id": "reinvent4_lambda_em_v2",
                            },
                        },
                        "tool_calls": [
                            {"name": "list_models", "args": {"kind": "predictor"}},
                            {"name": "list_models", "args": {"kind": "generator"}},
                            {"name": "search_dataset", "args": {"preferences": ["master_database"]}},
                            {"name": "generate_candidates", "args": {"generator_id": "reinvent4_lambda_em_v2", "max_candidates": 6, "constraints": {}}},
                            {"name": "score_candidates", "args": {"predictor_id": "unimol_lambda_plqy_v1", "targets": ["plqy"]}},
                            {"name": "filter_and_rank", "args": {"topn": 10}},
                            {"name": "make_report", "args": {}},
                        ],
                    },
                    ensure_ascii=False,
                )
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": llm_content}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://mock.local/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_retry_without_response_format",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            self.assertEqual(call_state["n"], 2)
            payload = plan.to_dict()
            self.assertEqual(payload["summary"], "HTTP retry fallback output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")

    def test_agent_plan_llm_backend_openai_compat_retryable_http_code_retries(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            from oled_agent.agent.planner import build_plan
            import urllib.error

            call_state = {"n": 0}

            def fake_urlopen(req, timeout=None):
                call_state["n"] += 1
                req_payload = json.loads(req.data.decode("utf-8"))
                self.assertIn("response_format", req_payload)
                if call_state["n"] == 1:
                    err_body = json.dumps({"error": {"message": "rate limit"}}).encode("utf-8")
                    raise urllib.error.HTTPError(
                        url=req.full_url,
                        code=429,
                        msg="Too Many Requests",
                        hdrs=None,
                        fp=mock.Mock(read=lambda: err_body),
                    )
                llm_content = json.dumps(
                    {
                        "summary": "HTTP retryable code output",
                        "design_spec": {
                            "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                            "constraints": {},
                            "budget": {"max_candidates": 6},
                            "model_choice": {
                                "predictor_id": "unimol_lambda_plqy_v1",
                                "generator_id": "reinvent4_lambda_em_v2",
                            },
                        },
                        "tool_calls": [
                            {"name": "list_models", "args": {"kind": "predictor"}},
                            {"name": "list_models", "args": {"kind": "generator"}},
                            {"name": "search_dataset", "args": {"preferences": ["master_database"]}},
                            {"name": "generate_candidates", "args": {"generator_id": "reinvent4_lambda_em_v2", "max_candidates": 6, "constraints": {}}},
                            {"name": "score_candidates", "args": {"predictor_id": "unimol_lambda_plqy_v1", "targets": ["plqy"]}},
                            {"name": "filter_and_rank", "args": {"topn": 10}},
                            {"name": "make_report", "args": {}},
                        ],
                    },
                    ensure_ascii=False,
                )
                resp_obj = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": llm_content}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-mock"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://mock.local/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "3"
            env["OLED_AGENT_LLM_BACKEND_MAX_RETRIES"] = "1"
            env["OLED_AGENT_LLM_BACKEND_BACKOFF_SEC"] = "0"

            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    plan = build_plan(
                        user_request="设计470nm附近且高PLQY分子",
                        task_id="task_provider_llm_backend_retryable_code",
                        catalog_path=repo_root / "configs" / "models" / "catalog.json",
                        planner_provider="llm_v1",
                    )

            self.assertEqual(call_state["n"], 2)
            payload = plan.to_dict()
            self.assertEqual(payload["summary"], "HTTP retryable code output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")

    def test_agent_plan_llm_backend_invalid_retry_env_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-test"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "1"
            env["OLED_AGENT_LLM_BACKEND_MAX_RETRIES"] = "bad"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_backend_invalid_retry_env",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_backend_failed")

    def test_agent_plan_llm_backend_debug_error_detail_exposed_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env.pop("OLED_AGENT_LLM_PLANNER_CMD", None)
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-test"
            env["OLED_AGENT_LLM_API_KEY"] = "secret-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "1"
            env["OLED_AGENT_LLM_BACKEND_MAX_RETRIES"] = "bad"
            env["OLED_AGENT_LLM_DEBUG_ERROR"] = "1"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_backend_debug_detail",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_backend_failed")
            self.assertIn("planner_provider_error_detail", md)
            self.assertIn("OLED_AGENT_LLM_BACKEND_MAX_RETRIES", md["planner_provider_error_detail"])

    def test_agent_plan_llm_command_has_priority_over_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env["OLED_AGENT_LLM_PLANNER_CMD"] = f"{sys.executable} {llm_script}"
            env["MOCK_LLM_MODE"] = "active"
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-test"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_priority",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload["summary"], "Mock LLM planner output")
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "llm_v1")
            self.assertEqual(md["planner_provider_status"], "active")

    def test_agent_plan_llm_command_priority_not_overridden_by_bad_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")
            env["OLED_AGENT_LLM_PLANNER_CMD"] = f"{sys.executable} {llm_script}"
            env["MOCK_LLM_MODE"] = "bad_json"
            env["OLED_AGENT_LLM_BACKEND"] = "openai_compat"
            env["OLED_AGENT_LLM_MODEL"] = "gpt-test"
            env["OLED_AGENT_LLM_API_KEY"] = "test-key"
            env["OLED_AGENT_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            env["OLED_AGENT_LLM_TIMEOUT_SEC"] = "1"
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--catalog",
                    str(repo_root / "configs" / "models" / "catalog.json"),
                    "--task-id",
                    "task_provider_llm_cmd_priority_bad_backend",
                    "--request",
                    "设计470nm附近且高PLQY分子",
                    "--planner-provider",
                    "llm_v1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            md = payload["design_spec"]["metadata"]
            self.assertEqual(md["planner_provider_effective"], "rule_based_v1")
            self.assertEqual(md["planner_provider_status"], "fallback")
            self.assertEqual(md["planner_provider_reason"], "llm_output_invalid")

    def test_agent_plan_rejects_invalid_planner_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            cp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oled_agent.cli",
                    "agent-plan",
                    "--workspace-root",
                    str(repo_root),
                    "--task-id",
                    "task_provider_bad",
                    "--request",
                    "design molecule",
                    "--planner-provider",
                    "bad_provider",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 2)
            self.assertIn("[FAIL] invalid request args:", cp.stdout)
            self.assertIn("Unknown planner_provider", cp.stdout)

    def test_validate_decision_summary_script_rejects_null_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            decision = td_path / "decision_summary.json"
            decision.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "generated_at": "2026-05-03T00:00:00Z",
                        "task_id": "task_bad",
                        "status": "success",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "score_step": {
                            "used_fallback": True,
                            "adapter": "local_deterministic_fallback",
                            "fallback_reason": "x",
                            "fallback_code": None,
                            "fallback_retryable": True,
                            "fallback_details": {},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [sys.executable, "scripts/validate_decision_summary.py", str(decision)],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 1)
            self.assertIn("[FAIL] decision summary schema invalid", cp.stdout)

    def test_validate_decision_summary_script_accepts_valid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            decision = td_path / "decision_summary.json"
            decision.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "generated_at": "2026-05-03T00:00:00Z",
                        "task_id": "task_ok",
                        "status": "success",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "score_step": {
                            "used_fallback": True,
                            "adapter": "local_deterministic_fallback",
                            "fallback_reason": "x",
                            "fallback_code": "external_command_failed",
                            "fallback_retryable": True,
                            "fallback_details": {},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [sys.executable, "scripts/validate_decision_summary.py", str(decision)],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            self.assertIn("[PASS] decision summary schema valid", cp.stdout)

    def test_validate_task_state_script_rejects_empty_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_state = td_path / "task_state.json"
            task_state.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "generated_at": "2026-05-03T00:00:00Z",
                        "task_id": "task_bad_state",
                        "status": "success",
                        "current_state": "DONE",
                        "history": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [sys.executable, "scripts/validate_task_state.py", str(task_state)],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 1)
            self.assertIn("[FAIL] task state schema invalid", cp.stdout)

    def test_validate_task_state_script_rejects_unknown_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_state = td_path / "task_state.json"
            task_state.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "generated_at": "2026-05-03T00:00:00Z",
                        "task_id": "task_bad_state",
                        "status": "success",
                        "current_state": "ALIEN_STATE",
                        "history": [
                            {"state": "INIT", "status": "completed", "at": "2026-05-03T00:00:00Z"},
                            {"state": "ALIEN_STATE", "status": "completed", "at": "2026-05-03T00:00:01Z"}
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [sys.executable, "scripts/validate_task_state.py", str(task_state)],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 1)
            self.assertIn("[FAIL] task state schema invalid", cp.stdout)

    def test_validate_task_state_script_accepts_valid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            task_state = td_path / "task_state.json"
            task_state.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "generated_at": "2026-05-03T00:00:00Z",
                        "task_id": "task_ok_state",
                        "status": "success",
                        "current_state": "DONE",
                        "history": [
                            {"state": "INIT", "status": "completed", "at": "2026-05-03T00:00:00Z"},
                            {"state": "DONE", "status": "success", "at": "2026-05-03T00:00:01Z"},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            cp = subprocess.run(
                [sys.executable, "scripts/validate_task_state.py", str(task_state)],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            self.assertIn("[PASS] task state schema valid", cp.stdout)

    def test_validate_task_state_payload_accepts_runtime_generated_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            out = execute_request(
                workspace_root=td_path,
                user_request="设计470nm附近且高PLQY分子",
                task_id="task_task_state_contract",
                catalog_path=repo_root / "configs" / "models" / "catalog.json",
            )
            task_state_path = Path(out["task_state_path"])
            payload = json.loads(task_state_path.read_text(encoding="utf-8"))
            validated = validate_task_state_payload(payload=payload, workspace_root=td_path)
            self.assertEqual(validated.get("task_id"), "task_task_state_contract")

    def test_validate_run_artifacts_script_accepts_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            out = execute_request(
                workspace_root=td_path,
                user_request="设计470nm附近且高PLQY分子",
                task_id="task_validate_run_artifacts",
                catalog_path=repo_root / "configs" / "models" / "catalog.json",
            )
            result_json = td_path / "run_result.json"
            result_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_run_artifacts.py",
                    "--workspace-root",
                    str(td_path),
                    "--result-json",
                    str(result_json),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
            self.assertIn("[PASS] decision summary schema valid", cp.stdout)
            self.assertIn("[PASS] task state schema valid", cp.stdout)
            self.assertIn("[PASS] data report schema valid", cp.stdout)
            self.assertIn("[PASS] model report schema valid", cp.stdout)
            self.assertIn("[PASS] filtering report schema valid", cp.stdout)

    def test_validate_run_artifacts_script_rejects_missing_result_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            broken_result_json = td_path / "broken_result.json"
            broken_result_json.write_text(
                json.dumps(
                    {
                        "decision_summary_path": "/tmp/a.json",
                        "task_state_path": "/tmp/b.json",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_run_artifacts.py",
                    "--workspace-root",
                    str(td_path),
                    "--result-json",
                    str(broken_result_json),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 1)
            self.assertIn("[FAIL] invalid artifact inputs:", cp.stdout)

    def test_validate_plan_payload_accepts_valid_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "ok",
                "design_spec": {
                    "task_id": "task_plan_schema_ok",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "list_models", "args": {"kind": "predictor"}},
                    {"name": "make_report", "args": {}},
                ],
            }
            validate_plan_payload(payload, workspace_root=td_path)

    def test_validate_plan_payload_rejects_missing_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "bad",
                "design_spec": {
                    "task_id": "task_plan_schema_bad",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
            }
            with self.assertRaises(RequestValidationError):
                validate_plan_payload(payload, workspace_root=td_path)

    def test_validate_plan_payload_tools_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "ok-tools",
                "design_spec": {
                    "task_id": "task_plan_tools_ok",
                    "request_text": "design molecule",
                    "mode": "train_then_design",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "list_models", "args": {"kind": "predictor"}},
                    {"name": "search_dataset", "args": {"preferences": ["master_database"]}},
                    {"name": "train_predictor", "args": {"predictor_id": "p1", "targets": ["plqy"]}},
                    {"name": "generate_candidates", "args": {"generator_id": "g1", "max_candidates": 8, "constraints": {}}},
                    {"name": "score_candidates", "args": {"predictor_id": "p1", "targets": ["plqy"]}},
                    {"name": "filter_and_rank", "args": {"topn": 5}},
                    {"name": "make_report", "args": {}},
                ],
            }
            validate_plan_payload(payload, workspace_root=td_path)

    def test_validate_plan_payload_rejects_unknown_tool_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "bad-tool-name",
                "design_spec": {
                    "task_id": "task_plan_tools_bad_name",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "unsupported_tool", "args": {}},
                ],
            }
            with self.assertRaises(RequestValidationError):
                validate_plan_payload(payload, workspace_root=td_path)

    def test_validate_plan_payload_rejects_tool_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "bad-tool-missing",
                "design_spec": {
                    "task_id": "task_plan_tools_bad_missing",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "score_candidates", "args": {"predictor_id": "p1"}},
                ],
            }
            with self.assertRaises(RequestValidationError):
                validate_plan_payload(payload, workspace_root=td_path)

    def test_validate_plan_payload_rejects_tool_type_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "bad-tool-type",
                "design_spec": {
                    "task_id": "task_plan_tools_bad_type",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "filter_and_rank", "args": {"topn": "10"}},
                ],
            }
            with self.assertRaises(RequestValidationError):
                validate_plan_payload(payload, workspace_root=td_path)

    def test_validate_plan_payload_rejects_tool_extra_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "bad-tool-extra",
                "design_spec": {
                    "task_id": "task_plan_tools_bad_extra",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "make_report", "args": {"unexpected": 1}},
                ],
            }
            with self.assertRaises(RequestValidationError):
                validate_plan_payload(payload, workspace_root=td_path)

    def test_plan_minimal_tool_args_validation_when_jsonschema_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = {
                "summary": "minimal-path-bad-tool-args",
                "design_spec": {
                    "task_id": "task_plan_tools_minimal",
                    "request_text": "design molecule",
                    "mode": "fast_screen",
                    "targets": [{"name": "plqy", "objective": "maximize"}],
                    "constraints": {},
                    "budget": {"max_candidates": 10},
                    "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                },
                "tool_calls": [
                    {"name": "generate_candidates", "args": {"generator_id": "g1", "max_candidates": "bad"}},
                ],
            }
            schema = json.loads(
                (Path(__file__).resolve().parents[1] / "schemas" / "plan.schema.json").read_text(encoding="utf-8")
            )
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "jsonschema":
                    raise ImportError("force minimal")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(RequestValidationError):
                    _validate_via_jsonschema(payload, schema, contract_kind="plan")

    def test_plan_schema_tool_items_synced_with_shared_contract(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        schema = json.loads((repo_root / "schemas" / "plan.schema.json").read_text(encoding="utf-8"))
        schema_item = schema["properties"]["tool_calls"]["items"]
        expected_item = build_plan_tool_call_item_schema()
        self.assertEqual(schema_item, expected_item)

    def test_external_connectivity_debug_contains_structured_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = run_external_connectivity_debug(workspace_root=Path(td))
            self.assertIn("report_type", report)
            self.assertEqual(report["report_type"], "external_connectivity_debug_v1")
            self.assertIn("connectivity", report)
            c = report["connectivity"]
            self.assertIn("chain_ready", c)
            self.assertIn("blocking_checks", c)
            self.assertIn("check_status", c)
            self.assertIn("external:scorer_chain", c["check_status"])

    def test_external_connectivity_debug_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            json_out = td_path / "ext_debug.json"
            report = run_external_connectivity_debug(workspace_root=td_path, json_out=json_out)
            self.assertTrue(json_out.exists())
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("report_type"), "external_connectivity_debug_v1")
            self.assertEqual(payload.get("overall"), report.get("overall"))

    def test_llm_connectivity_fails_when_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": "",
                    "OLED_AGENT_LLM_BACKEND": "",
                },
                clear=False,
            ):
                report = run_llm_connectivity(workspace_root=td_path)
            self.assertEqual(report["report_type"], "llm_connectivity_v1")
            self.assertEqual(report["source"], "none")
            self.assertEqual(report["overall"], "fail")
            checks = report["connectivity"]["check_status"]
            self.assertEqual(checks.get("llm:source"), "fail")
            self.assertEqual(checks.get("llm:config"), "fail")
            by_name = {c["name"]: c for c in report.get("checks", [])}
            self.assertEqual(by_name["llm:source"]["message"], "LLM source unresolved (none)")
            self.assertEqual(
                by_name["llm:config"]["message"],
                "LLM required config missing (set OLED_AGENT_LLM_PLANNER_CMD or OLED_AGENT_LLM_BACKEND)",
            )

    def test_llm_connectivity_command_probe_passes_with_mock_planner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            llm_script = repo_root / "scripts" / "mock_llm_planner.py"
            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} {llm_script}",
                    "OLED_AGENT_LLM_BACKEND": "",
                    "MOCK_LLM_MODE": "active",
                },
                clear=False,
            ):
                report = run_llm_connectivity(
                    workspace_root=repo_root,
                    catalog_path=repo_root / "configs" / "models" / "catalog.json",
                )
            self.assertEqual(report["source"], "command")
            self.assertEqual(report["overall"], "pass")
            checks = report["connectivity"]["check_status"]
            self.assertEqual(checks.get("llm:source"), "pass")
            self.assertEqual(checks.get("llm:command_probe"), "pass")

    def test_llm_connectivity_backend_probe_http_error(self) -> None:
        class _MockHttpErrorResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            def fake_urlopen(req, timeout=None):
                import urllib.error

                body = json.dumps({"error": {"message": "invalid token"}})
                raise urllib.error.HTTPError(
                    url=req.full_url,
                    code=401,
                    msg="Unauthorized",
                    hdrs=None,
                    fp=_MockHttpErrorResponse(body),
                )

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": "",
                    "OLED_AGENT_LLM_BACKEND": "openai_compat",
                    "OLED_AGENT_LLM_MODEL": "gpt-test",
                    "OLED_AGENT_LLM_API_KEY": "test-key",
                    "OLED_AGENT_LLM_BASE_URL": "http://mock.local/v1",
                    "OLED_AGENT_LLM_TIMEOUT_SEC": "3",
                },
                clear=False,
            ):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    report = run_llm_connectivity(workspace_root=td_path)
            self.assertEqual(report["source"], "backend")
            self.assertEqual(report["overall"], "fail")
            checks = report["connectivity"]["check_status"]
            self.assertEqual(checks.get("llm:backend_config"), "pass")
            self.assertEqual(checks.get("llm:backend_probe"), "fail")
            self.assertIn("llm:backend_probe", report["connectivity"]["blocking_checks"])
            by_name = {c["name"]: c for c in report.get("checks", [])}
            self.assertNotIn("body_tail", by_name["llm:backend_probe"].get("details", {}))

    def test_llm_connectivity_backend_probe_http_error_debug_mode_redacts_body(self) -> None:
        class _MockHttpErrorResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            leaked_key = "test-key"

            def fake_urlopen(req, timeout=None):
                import urllib.error

                body = (
                    "{\"error\":{\"message\":\"invalid token\","
                    f"\"debug\":\"Authorization: Bearer {leaked_key}\"}}"
                )
                raise urllib.error.HTTPError(
                    url=req.full_url,
                    code=401,
                    msg="Unauthorized",
                    hdrs=None,
                    fp=_MockHttpErrorResponse(body),
                )

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": "",
                    "OLED_AGENT_LLM_BACKEND": "openai_compat",
                    "OLED_AGENT_LLM_MODEL": "gpt-test",
                    "OLED_AGENT_LLM_API_KEY": leaked_key,
                    "OLED_AGENT_LLM_BASE_URL": "http://mock.local/v1",
                    "OLED_AGENT_LLM_TIMEOUT_SEC": "3",
                    "OLED_AGENT_LLM_DEBUG_ERROR": "1",
                },
                clear=False,
            ):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    report = run_llm_connectivity(workspace_root=td_path)
            by_name = {c["name"]: c for c in report.get("checks", [])}
            details = by_name["llm:backend_probe"].get("details", {})
            self.assertIn("body_tail", details)
            self.assertNotIn(leaked_key, str(details.get("body_tail", "")))
            self.assertIn("***", str(details.get("body_tail", "")))

    def test_llm_connectivity_backend_probe_socket_timeout_classified_as_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            def fake_urlopen(req, timeout=None):
                raise socket.timeout("timed out")

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": "",
                    "OLED_AGENT_LLM_BACKEND": "openai_compat",
                    "OLED_AGENT_LLM_MODEL": "gpt-test",
                    "OLED_AGENT_LLM_API_KEY": "test-key",
                    "OLED_AGENT_LLM_BASE_URL": "http://mock.local/v1",
                    "OLED_AGENT_LLM_TIMEOUT_SEC": "3",
                },
                clear=False,
            ):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    report = run_llm_connectivity(workspace_root=td_path)
            by_name = {c["name"]: c for c in report.get("checks", [])}
            backend_probe = by_name["llm:backend_probe"]
            self.assertEqual(backend_probe.get("status"), "fail")
            self.assertEqual(backend_probe.get("message"), "LLM backend timeout")

    def test_llm_connectivity_backend_probe_urlerror_timeout_classified_as_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            def fake_urlopen(req, timeout=None):
                import urllib.error

                raise urllib.error.URLError(socket.timeout("timed out"))

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": "",
                    "OLED_AGENT_LLM_BACKEND": "openai_compat",
                    "OLED_AGENT_LLM_MODEL": "gpt-test",
                    "OLED_AGENT_LLM_API_KEY": "test-key",
                    "OLED_AGENT_LLM_BASE_URL": "http://mock.local/v1",
                    "OLED_AGENT_LLM_TIMEOUT_SEC": "3",
                },
                clear=False,
            ):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    report = run_llm_connectivity(workspace_root=td_path)
            by_name = {c["name"]: c for c in report.get("checks", [])}
            backend_probe = by_name["llm:backend_probe"]
            self.assertEqual(backend_probe.get("status"), "fail")
            self.assertEqual(backend_probe.get("message"), "LLM backend timeout")

    def test_llm_connectivity_probe_body_minimal_and_supports_extra_json(self) -> None:
        class _MockResponse:
            def __init__(self, body: str):
                self._body = body.encode("utf-8")

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            seen_payloads: list[dict] = []

            def fake_urlopen(req, timeout=None):
                seen_payloads.append(json.loads(req.data.decode("utf-8")))
                resp_obj = {
                    "id": "chatcmpl-probe",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                }
                return _MockResponse(json.dumps(resp_obj, ensure_ascii=False))

            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": "",
                    "OLED_AGENT_LLM_BACKEND": "openai_compat",
                    "OLED_AGENT_LLM_MODEL": "gpt-test",
                    "OLED_AGENT_LLM_API_KEY": "test-key",
                    "OLED_AGENT_LLM_BASE_URL": "http://mock.local/v1",
                    "OLED_AGENT_LLM_TIMEOUT_SEC": "3",
                    "OLED_AGENT_LLM_CONNECTIVITY_PROBE_EXTRA_BODY_JSON": '{"top_p":0.9}',
                },
                clear=False,
            ):
                with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    report = run_llm_connectivity(workspace_root=td_path)

            self.assertEqual(report.get("overall"), "pass")
            self.assertGreaterEqual(len(seen_payloads), 1)
            sent = seen_payloads[0]
            self.assertIn("model", sent)
            self.assertIn("messages", sent)
            self.assertNotIn("temperature", sent)
            self.assertEqual(sent.get("top_p"), 0.9)

    def test_llm_connectivity_exit_code_is_fail_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            with mock.patch.dict(
                os.environ,
                {
                    "OLED_AGENT_LLM_PLANNER_CMD": f"{sys.executable} /definitely/not/found.py",
                    "OLED_AGENT_LLM_BACKEND": "",
                },
                clear=False,
            ):
                report = run_llm_connectivity(workspace_root=td_path)
            self.assertEqual(report.get("overall"), "fail")
            self.assertEqual(report.get("exit_code"), 1)


if __name__ == "__main__":
    unittest.main()


class SchemaSyncScriptTests(unittest.TestCase):
    def test_sync_plan_tool_schema_check_exit_code_zero_when_in_sync(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cp = subprocess.run(
            [sys.executable, "scripts/sync_plan_tool_schema.py", "--check"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)

    def test_sync_plan_tool_schema_check_exit_code_nonzero_when_drifted(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / "schemas" / "plan.schema.json"
        original = schema_path.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload["properties"]["tool_calls"]["items"] = {"type": "object", "properties": {}}
        schema_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            cp = subprocess.run(
                [sys.executable, "scripts/sync_plan_tool_schema.py", "--check"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        finally:
            schema_path.write_text(original, encoding="utf-8")

    def test_sync_plan_tool_schema_check_json_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cp = subprocess.run(
            [sys.executable, "scripts/sync_plan_tool_schema.py", "--check", "--json"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("action"), "check")
        self.assertTrue(str(payload.get("schema_path", "")).endswith("schemas/plan.schema.json"))
        self.assertTrue(isinstance(payload.get("git_sha"), str))
        self.assertEqual(len(payload.get("git_sha", "")), 40)


class WorkflowPolicyTests(unittest.TestCase):
    @staticmethod
    def _workflow_path(repo_root: Path) -> Path:
        local = repo_root / ".github" / "workflows" / "agent4mat-ci.yml"
        if local.exists():
            return local
        return repo_root.parent / ".github" / "workflows" / "oled-agent-ci.yml"

    @staticmethod
    def _extract_job_block(content: str, job_name: str) -> str:
        marker = f"\n  {job_name}:\n"
        start = content.find(marker)
        if start < 0:
            raise AssertionError(f"job not found: {job_name}")
        start += 1  # keep two-space indentation for regex anchor
        pattern = re.compile(r"^  [a-zA-Z0-9_-]+:\n", flags=re.MULTILINE)
        m = pattern.search(content, pos=start + len(marker))
        end = m.start() if m else len(content)
        return content[start:end]

    def test_oled_agent_ci_external_acceptance_only_manual_trigger(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("external-chain-acceptance:", content)
        self.assertIn("github.event_name == 'workflow_dispatch'", content)
        self.assertIn("github.event.inputs.run_external_acceptance == 'true'", content)
        self.assertNotIn("vars.OLED_AGENT_RUN_EXTERNAL_ACCEPTANCE", content)

    def test_oled_agent_ci_uses_schema_check_json_and_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("scripts/sync_plan_tool_schema.py --check --json", content)
        self.assertIn("plan_tool_schema_check.json", content)
        self.assertIn("name: plan-tool-schema-check", content)
        self.assertIn("Publish schema-check summary", content)
        self.assertIn("GITHUB_STEP_SUMMARY", content)

    def test_oled_agent_ci_has_llm_backend_retry_guard_job(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("llm-backend-retry-guard:", content)
        self.assertIn("Run openai_compat retry guard tests", content)
        self.assertIn(
            "test_agent_plan_llm_backend_openai_compat_retryable_http_code_retries",
            content,
        )
        self.assertIn(
            "test_agent_plan_llm_backend_invalid_retry_env_fallback",
            content,
        )

    def test_oled_agent_ci_has_acceptance_matrix_jobs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("acceptance-cpu-mock:", content)
        self.assertIn("name: acceptance cpu-mock", content)
        self.assertIn("make release-check TASK_ID=ci_accept_cpu_mock", content)
        self.assertIn("acceptance-llm-mock:", content)
        self.assertIn("name: acceptance llm-mock", content)
        self.assertIn("Run acceptance (llm-mock)", content)
        self.assertIn("make llm-smoke", content)
        self.assertIn("name: acceptance external-adapter (optional)", content)

    def test_oled_agent_ci_has_adapter_contract_guard_job(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("adapter-contract-guard:", content)
        self.assertIn("Validate adapter templates contract", content)
        self.assertIn("scripts/adapters/validate_adapter_contract.py", content)
        self.assertIn("--tool train_predictor", content)
        self.assertIn("--tool generate_candidates", content)
        self.assertIn("--tool score_candidates", content)

    def test_oled_agent_ci_has_make_entrypoint_guard_job(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        section = self._extract_job_block(content, "make-entrypoint-guard")
        self.assertIn("make-entrypoint-guard:", content)
        self.assertIn("Run make entrypoints", section)
        self.assertIn("make release-check TASK_ID=ci_release_check", section)
        self.assertIn("make real-adapter-validate", section)
        self.assertNotIn("make adapter-validate", section)
        self.assertNotIn("make quickstart TASK_ID=ci_make_quickstart", section)
        self.assertNotIn("make doctor", section)
        self.assertNotIn("make llm-smoke", section)

    def test_oled_agent_ci_external_acceptance_uses_shell_entrypoint(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("run_external_chain_acceptance_with_debug.sh", content)
        self.assertNotIn("run_external_chain_acceptance.py", content)

    def test_oled_agent_ci_has_manual_real_chain_acceptance_job(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("run_real_chain_acceptance:", content)
        self.assertIn("real-chain-minimal-acceptance:", content)
        self.assertIn("acceptance real-chain-minimal (manual)", content)
        self.assertIn("github.event.inputs.run_real_chain_acceptance == 'true'", content)
        self.assertIn("make real-chain-acceptance TASK_ID=ci_real_chain_manual", content)
        self.assertIn("Collect real-chain release evidence", content)
        self.assertIn("make real-chain-evidence", content)
        self.assertIn("RESULT_JSON=runs/ci/agent_run_real_chain_ci_real_chain_manual.json", content)
        self.assertIn("real-chain-minimal-artifacts", content)
        self.assertIn("real-chain-evidence-artifacts", content)
        self.assertIn("release_evidence.json", content)
        self.assertIn("release_evidence.md", content)

    def test_oled_agent_ci_validates_structured_reports_schema(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("Validate structured artifacts schema", content)
        self.assertIn("scripts/validate_run_artifacts.py", content)
        self.assertIn("--result-json runs/ci/agent_run_ci_smoke.json", content)

    def test_oled_agent_ci_publishes_experiment_summary_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("Generate experiment summary", content)
        self.assertIn("scripts/summarize_experiments.py", content)
        self.assertIn("runs/ci/experiment_summary.json", content)
        self.assertIn("runs/ci/experiment_summary.md", content)
        self.assertIn("Upload experiment-summary artifact", content)
        self.assertIn("name: experiment-summary", content)
        self.assertIn("Publish experiment summary", content)
        self.assertIn("### Experiment Summary", content)

    def test_oled_agent_ci_has_new_intake_step_and_web_guard_jobs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("intake-contract-guard:", content)
        self.assertIn("step-mode-guard:", content)
        self.assertIn("web-evidence-guard:", content)
        self.assertIn("experiment-trace-guard:", content)
        self.assertIn("real-no-fallback-gate:", content)
        self.assertIn("make intake-contract-guard", content)
        self.assertIn("make step-mode-guard", content)
        self.assertIn("make web-evidence-guard", content)
        self.assertIn("make experiment-trace-guard", content)
        self.assertIn("make real-no-fallback-gate", content)


class BuildEntrypointTests(unittest.TestCase):
    def test_makefile_contains_adapter_and_quickstart_targets(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        makefile = repo_root / "Makefile"
        content = makefile.read_text(encoding="utf-8")
        self.assertIn("llm-smoke:", content)
        self.assertIn("release-check:", content)
        self.assertIn("quickstart:", content)
        self.assertIn("adapter-validate:", content)
        self.assertIn("real-adapter-validate:", content)
        self.assertIn("adapter-self-check:", content)
        self.assertIn("doctor:", content)
        self.assertIn("test-regressions:", content)
        self.assertIn("test-adapters:", content)
        self.assertIn("scripts/adapters/check_quickstart_chain.sh", content)
        self.assertIn("scripts/adapters/validate_adapter_contract.py", content)
        self.assertIn("scripts/check_llm_planner_modes.py", content)
        self.assertIn("scripts/validate_step_request_examples.py", content)
        self.assertIn("train_predictor_unimol_adapter.py", content)
        self.assertIn("score_candidates_unimol_adapter.py", content)
        self.assertIn("generate_candidates_mineru_adapter.py", content)
        self.assertIn("generate_candidates_reinvent4_adapter.py", content)
        self.assertIn("generate_candidates_molscribe_adapter.py", content)
        self.assertIn("$(MAKE) adapter-validate", content)
        self.assertIn("$(MAKE) quickstart", content)
        self.assertIn("$(MAKE) llm-smoke", content)
        self.assertIn("$(MAKE) doctor", content)
        self.assertIn("oled_agent.cli doctor", content)

    def test_makefile_contains_release_boundary_and_plan_targets(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        makefile = repo_root / "Makefile"
        content = makefile.read_text(encoding="utf-8")
        self.assertIn("release-boundary:", content)
        self.assertIn("script-map:", content)
        self.assertIn("real-chain-acceptance:", content)
        self.assertIn("real-chain-acceptance-real:", content)
        self.assertIn("real-chain-baseline:", content)
        self.assertIn("real-chain-baseline-archive:", content)
        self.assertIn("real-chain-baseline-archive-tgz:", content)
        self.assertIn("real-chain-release-bundle-check:", content)
        self.assertIn("real-chain-evidence:", content)
        self.assertIn("ui-smoke:", content)
        self.assertIn("scripts/check_release_boundary.py", content)
        self.assertIn("scripts/build_script_migration_map.py", content)
        self.assertIn("scripts/summarize_experiments.py", content)
        self.assertIn("scripts/collect_real_chain_evidence.py", content)
        self.assertIn("scripts/archive_real_chain_baseline.py", content)
        self.assertIn("scripts/check_real_chain_release_bundle.py", content)
        self.assertIn("step-request-templates-validate:", content)
        self.assertIn("scripts/run_real_chain_acceptance_minimal.sh", content)
        self.assertIn("scripts/run_real_chain_acceptance_real.sh", content)
        self.assertIn("scripts/run_real_chain_baseline.sh", content)
        self.assertIn("archive_real_chain_baseline.py --workspace-root \"$(WORKSPACE_ROOT)\" --base-task-id \"$(TASK_ID)\" --tar-gz", content)
        self.assertIn("check_real_chain_release_bundle.py --workspace-root \"$(WORKSPACE_ROOT)\" --base-task-id \"$(TASK_ID)\" --require-tar-gz", content)
        self.assertIn("ui/app.py", content)
        self.assertIn("input-smoke:", content)
        self.assertIn("experiment-summary:", content)
        self.assertIn("scripts/run_molscribe_input_smoke.sh", content)
        self.assertIn("intake-contract-guard:", content)
        self.assertIn("step-mode-guard:", content)
        self.assertIn("web-evidence-guard:", content)
        self.assertIn("experiment-trace-guard:", content)
        self.assertIn("real-no-fallback-gate:", content)
        self.assertIn("scripts/check_experiment_trace.py", content)


class PlanProgressAssetsTests(unittest.TestCase):
    def test_plan_progress_scripts_exist_and_are_executable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected_scripts = [
            repo_root / "scripts" / "check_release_boundary.py",
            repo_root / "scripts" / "check_real_chain_release_bundle.py",
            repo_root / "scripts" / "build_script_migration_map.py",
            repo_root / "scripts" / "collect_real_chain_evidence.py",
            repo_root / "scripts" / "archive_real_chain_baseline.py",
            repo_root / "scripts" / "validate_step_request_examples.py",
            repo_root / "scripts" / "check_experiment_trace.py",
            repo_root / "scripts" / "summarize_experiments.py",
            repo_root / "scripts" / "run_molscribe_input_smoke.sh",
            repo_root / "scripts" / "run_real_chain_acceptance_minimal.sh",
            repo_root / "scripts" / "run_real_chain_acceptance_real.sh",
            repo_root / "scripts" / "run_real_chain_baseline.sh",
        ]
        for script in expected_scripts:
            self.assertTrue(script.exists(), msg=f"missing script: {script}")
            self.assertTrue(os.access(script, os.X_OK), msg=f"script is not executable: {script}")

    def test_summarize_experiments_script_outputs_aggregate_payload(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            exp_a = td_path / "runs" / "agent" / "exp_a" / "artifacts"
            exp_b = td_path / "runs" / "agent" / "exp_b" / "artifacts"
            exp_a.mkdir(parents=True, exist_ok=True)
            exp_b.mkdir(parents=True, exist_ok=True)
            (exp_a / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_a",
                        "run_label": "exp_a-1",
                        "generated_at": "2026-05-14T01:00:00+00:00",
                        "execution_mode": "full_pipeline",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "execution_summary": {"status": "success", "record_count": 3, "failed_count": 0},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (exp_b / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_b",
                        "run_label": "exp_b-1",
                        "generated_at": "2026-05-14T02:00:00+00:00",
                        "execution_mode": "single_step",
                        "model_choice": {"predictor_id": "p2", "generator_id": "g2"},
                        "execution_summary": {"status": "failed", "record_count": 1, "failed_count": 1},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            out_json = td_path / "summary.json"
            out_md = td_path / "summary.md"
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_experiments.py",
                    "--workspace-root",
                    str(td_path),
                    "--limit",
                    "1",
                    "--json-out",
                    str(out_json),
                    "--md-out",
                    str(out_md),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(int(payload.get("count") or 0), 2)
            self.assertEqual(int(payload.get("limit") or 0), 1)
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            by_status = summary.get("by_status") if isinstance(summary.get("by_status"), dict) else {}
            self.assertEqual(int(by_status.get("success") or 0), 1)
            self.assertEqual(int(by_status.get("failed") or 0), 1)
            recent = payload.get("recent") if isinstance(payload.get("recent"), list) else []
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].get("task_id"), "exp_b")
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

    def test_summarize_experiments_script_warns_and_skips_invalid_trace(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            exp_ok = td_path / "runs" / "agent" / "exp_ok" / "artifacts"
            exp_bad = td_path / "runs" / "agent" / "exp_bad" / "artifacts"
            exp_ok.mkdir(parents=True, exist_ok=True)
            exp_bad.mkdir(parents=True, exist_ok=True)
            (exp_ok / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_ok",
                        "run_label": "exp_ok-1",
                        "generated_at": "2026-05-14T03:00:00+00:00",
                        "execution_mode": "full_pipeline",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "execution_summary": {"status": "success", "record_count": 2, "failed_count": 0},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (exp_bad / "experiment_trace.json").write_text("{not-json}\n", encoding="utf-8")
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_experiments.py",
                    "--workspace-root",
                    str(td_path),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
            self.assertIn("[WARN] skip invalid experiment trace:", cp.stderr)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(int(payload.get("count") or 0), 1)
            recent = payload.get("recent") if isinstance(payload.get("recent"), list) else []
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].get("task_id"), "exp_ok")

    def test_summarize_experiments_prioritizes_failed_and_extracts_score_fallback(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            run_a = td_path / "runs" / "agent" / "exp_a"
            run_b = td_path / "runs" / "agent" / "exp_b"
            (run_a / "artifacts").mkdir(parents=True, exist_ok=True)
            (run_b / "artifacts").mkdir(parents=True, exist_ok=True)
            (run_a / "artifacts" / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_a",
                        "run_label": "exp_a-1",
                        "generated_at": "2026-05-14T04:00:00+00:00",
                        "execution_mode": "full_pipeline",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "execution_summary": {
                            "status": "success",
                            "record_count": 3,
                            "failed_count": 0,
                            "failed_steps": [],
                            "adapters": ["a_gen"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_b / "artifacts" / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_b",
                        "run_label": "exp_b-1",
                        "generated_at": "2026-05-14T03:00:00+00:00",
                        "execution_mode": "full_pipeline",
                        "model_choice": {"predictor_id": "p2", "generator_id": "g2"},
                        "execution_summary": {
                            "status": "failed",
                            "record_count": 2,
                            "failed_count": 1,
                            "failed_steps": ["score_candidates"],
                            "adapters": ["a_score"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_b / "decision_summary.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_b",
                        "score_step": {
                            "adapter": "unimol_score_adapter_v1",
                            "used_fallback": True,
                            "fallback_code": "adapter_timeout",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            out_md = td_path / "summary.md"
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_experiments.py",
                    "--workspace-root",
                    str(td_path),
                    "--limit",
                    "2",
                    "--md-out",
                    str(out_md),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            recent_failed_first = payload.get("recent_failed_first") if isinstance(payload.get("recent_failed_first"), list) else []
            self.assertEqual(len(recent_failed_first), 2)
            self.assertEqual(recent_failed_first[0].get("task_id"), "exp_b")
            self.assertEqual(recent_failed_first[0].get("score_adapter"), "unimol_score_adapter_v1")
            self.assertEqual(recent_failed_first[0].get("score_used_fallback"), True)
            self.assertEqual(recent_failed_first[0].get("score_fallback_code"), "adapter_timeout")
            md = out_md.read_text(encoding="utf-8")
            self.assertIn("## Failed Runs (Newest First)", md)
            self.assertIn("task_id=exp_b", md)
            self.assertIn("score_used_fallback=True", md)

    def test_plan_progress_docs_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected_docs = [
            repo_root / "docs" / "release_boundary.md",
            repo_root / "docs" / "script_migration_whitelist.md",
            repo_root / "docs" / "real_chain_minimal_acceptance.md",
            repo_root / "docs" / "real_chain_acceptance_real.md",
            repo_root / "docs" / "real_chain_no_fallback_quickstart.md",
            repo_root / "docs" / "ui_prototype.md",
            repo_root / "docs" / "script_migration_map.json",
        ]
        for doc in expected_docs:
            self.assertTrue(doc.exists(), msg=f"missing doc: {doc}")

    def test_real_chain_real_acceptance_script_forbids_stub_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_real_chain_acceptance_real.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn("stub_unimol_score.py", content)
        self.assertIn("stub_reinvent4_pipeline.sh", content)
        self.assertIn("stub-like value", content)
        self.assertIn('"target_value": 60.0', content)
        self.assertIn("plqy target_center is not percent-scale", content)
        self.assertIn("collect_real_chain_evidence.py", content)
        self.assertIn("--require-real-adapters", content)
        self.assertIn("strict_acceptance_summary.json", content)

    def test_real_chain_baseline_script_runs_three_strict_acceptance_rounds(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_real_chain_baseline.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn('RUN_COUNT="${4:-3}"', content)
        self.assertIn("./scripts/run_real_chain_acceptance_real.sh", content)
        self.assertIn("strict_acceptance_summary.json", content)
        self.assertIn("release_evidence.json", content)
        self.assertIn("baseline_summary.json", content)

    def test_real_chain_acceptance_script_uses_runtime_task_id_substitution(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_real_chain_acceptance_minimal.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn('python3 - "$REQ" "$TASK_ID" <<\'PY\'', content)
        self.assertIn("task_id = sys.argv[2]", content)
        self.assertIn('python3 - "$TASK_ID" <<\'PY\'', content)
        self.assertIn("task_id = sys.argv[1]", content)
        self.assertIn('"target_value": 60.0', content)
        self.assertIn("plqy target_center is not percent-scale", content)

    def test_collect_real_chain_evidence_script_writes_release_evidence(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            run_dir = td_path / "runs" / "agent" / "demo_task"
            run_dir.mkdir(parents=True, exist_ok=True)
            plan_path = run_dir / "plan.json"
            execution_path = run_dir / "execution.json"
            decision_path = run_dir / "decision_summary.json"
            task_state_path = run_dir / "task_state.json"
            result_path = run_dir / "acceptance_result.json"

            plan_path.write_text(
                json.dumps(
                    {
                        "summary": "demo",
                        "design_spec": {
                            "task_id": "demo_task",
                            "request_text": "demo",
                            "mode": "fast_screen",
                            "targets": [{"name": "plqy", "objective": "maximize", "target_center": 60.0, "sigma": 20.0}],
                            "budget": {"max_candidates": 8},
                            "model_choice": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                            "metadata": {"planner": "request_contract_v1"},
                        },
                        "tool_calls": [
                            {"name": "generate_candidates", "args": {"generator_id": "reinvent4_lambda_em_v2"}},
                            {"name": "score_candidates", "args": {"predictor_id": "unimol_lambda_plqy_v1"}},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            execution_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {"name": "generate_candidates", "result": {"adapter": "reinvent4_generate_adapter_v1"}},
                            {"name": "score_candidates", "result": {"adapter": "unimol_score_adapter_v1"}},
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path.write_text(
                json.dumps({"score_step": {"used_fallback": False}}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            task_state_path.write_text(json.dumps({"status": "success"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result_path.write_text(
                json.dumps(
                    {
                        "task_id": "demo_task",
                        "status": "success",
                        "plan_path": str(plan_path),
                        "execution_path": str(execution_path),
                        "decision_summary_path": str(decision_path),
                        "task_state_path": str(task_state_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            script = repo_root / "scripts" / "collect_real_chain_evidence.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--workspace-root",
                    str(repo_root),
                    "--result-json",
                    str(result_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            evidence_json = run_dir / "release_evidence.json"
            evidence_md = run_dir / "release_evidence.md"
            self.assertTrue(evidence_json.exists())
            self.assertTrue(evidence_md.exists())
            evidence = json.loads(evidence_json.read_text(encoding="utf-8"))
            self.assertEqual(evidence["overall"], "pass")
            self.assertTrue(evidence["checks"]["plqy_center_percent_scale"])

    def test_archive_real_chain_baseline_script_writes_archive_manifest(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            base_task_id = "demo_baseline"
            base_dir = td_path / "runs" / "agent" / base_task_id
            base_dir.mkdir(parents=True, exist_ok=True)

            runs = []
            for idx in (1, 2, 3):
                run_task_id = f"{base_task_id}_r{idx}"
                run_dir = td_path / "runs" / "agent" / run_task_id
                run_dir.mkdir(parents=True, exist_ok=True)

                strict_path = run_dir / "strict_acceptance_summary.json"
                result_path = run_dir / "acceptance_result.json"
                release_json = run_dir / "release_evidence.json"
                release_md = run_dir / "release_evidence.md"
                plan_path = run_dir / "plan.json"
                execution_path = run_dir / "execution.json"
                decision_path = run_dir / "decision_summary.json"
                task_state_path = run_dir / "task_state.json"
                tool_state_path = run_dir / "tool_state.json"

                strict_path.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "task_id": run_task_id,
                            "generate_adapter": "reinvent4_generate_adapter_v1",
                            "score_adapter": "unimol_score_adapter_v1",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                plan_path.write_text(json.dumps({"summary": "ok"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                execution_path.write_text(json.dumps({"records": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                decision_path.write_text(json.dumps({"score_step": {"used_fallback": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                task_state_path.write_text(json.dumps({"status": "success"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                tool_state_path.write_text(json.dumps({"status": "ok"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                release_json.write_text(json.dumps({"overall": "pass"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                release_md.write_text("# evidence\n", encoding="utf-8")
                result_path.write_text(
                    json.dumps(
                        {
                            "task_id": run_task_id,
                            "status": "success",
                            "plan_path": str(plan_path),
                            "execution_path": str(execution_path),
                            "decision_summary_path": str(decision_path),
                            "task_state_path": str(task_state_path),
                            "tool_state_path": str(tool_state_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                runs.append(
                    {
                        "task_id": run_task_id,
                        "strict_summary": str(strict_path.relative_to(td_path)),
                        "result_json": str(result_path.relative_to(td_path)),
                        "release_evidence_json": str(release_json.relative_to(td_path)),
                    }
                )

            baseline_summary = base_dir / "baseline_summary.json"
            baseline_summary.write_text(
                json.dumps(
                    {"status": "pass", "base_task_id": base_task_id, "run_count": 3, "runs": runs, "failures": []},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            out_dir = td_path / "runs" / "archive" / base_task_id
            script = repo_root / "scripts" / "archive_real_chain_baseline.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--workspace-root",
                    str(td_path),
                    "--base-task-id",
                    base_task_id,
                    "--out-dir",
                    str(out_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            manifest_json = out_dir / "archive_manifest.json"
            manifest_md = out_dir / "archive_manifest.md"
            self.assertTrue(manifest_json.exists())
            self.assertTrue(manifest_md.exists())
            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("status"), "pass")
            self.assertEqual(int(manifest.get("missing_required_count", -1)), 0)
            copied = manifest.get("copied", [])
            self.assertTrue(isinstance(copied, list) and len(copied) >= 10)
            self.assertTrue((out_dir / "files" / "runs" / "agent" / base_task_id / "baseline_summary.json").exists())

    def test_archive_real_chain_baseline_script_writes_tar_gz_package(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            base_task_id = "demo_baseline_tgz"
            base_dir = td_path / "runs" / "agent" / base_task_id
            base_dir.mkdir(parents=True, exist_ok=True)

            run_task_id = f"{base_task_id}_r1"
            run_dir = td_path / "runs" / "agent" / run_task_id
            run_dir.mkdir(parents=True, exist_ok=True)

            strict_path = run_dir / "strict_acceptance_summary.json"
            result_path = run_dir / "acceptance_result.json"
            release_json = run_dir / "release_evidence.json"
            release_md = run_dir / "release_evidence.md"
            plan_path = run_dir / "plan.json"
            execution_path = run_dir / "execution.json"
            decision_path = run_dir / "decision_summary.json"
            task_state_path = run_dir / "task_state.json"
            tool_state_path = run_dir / "tool_state.json"

            strict_path.write_text(json.dumps({"status": "pass"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            plan_path.write_text(json.dumps({"summary": "ok"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            execution_path.write_text(json.dumps({"records": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            decision_path.write_text(json.dumps({"score_step": {"used_fallback": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            task_state_path.write_text(json.dumps({"status": "success"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tool_state_path.write_text(json.dumps({"status": "ok"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            release_json.write_text(json.dumps({"overall": "pass"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            release_md.write_text("# evidence\n", encoding="utf-8")
            result_path.write_text(
                json.dumps(
                    {
                        "task_id": run_task_id,
                        "status": "success",
                        "plan_path": str(plan_path),
                        "execution_path": str(execution_path),
                        "decision_summary_path": str(decision_path),
                        "task_state_path": str(task_state_path),
                        "tool_state_path": str(tool_state_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            baseline_summary = base_dir / "baseline_summary.json"
            baseline_summary.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "base_task_id": base_task_id,
                        "run_count": 1,
                        "runs": [
                            {
                                "task_id": run_task_id,
                                "strict_summary": str(strict_path.relative_to(td_path)),
                                "result_json": str(result_path.relative_to(td_path)),
                                "release_evidence_json": str(release_json.relative_to(td_path)),
                            }
                        ],
                        "failures": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            out_dir = td_path / "runs" / "archive" / base_task_id
            script = repo_root / "scripts" / "archive_real_chain_baseline.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--workspace-root",
                    str(td_path),
                    "--base-task-id",
                    base_task_id,
                    "--out-dir",
                    str(out_dir),
                    "--tar-gz",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            tar_path = out_dir.with_suffix(".tar.gz")
            self.assertTrue(tar_path.exists(), msg=f"missing tar package: {tar_path}")
            manifest = json.loads((out_dir / "archive_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(str(manifest.get("tar_gz_path") or "")).resolve(), tar_path.resolve())
            with tarfile.open(tar_path, mode="r:gz") as tf:
                names = tf.getnames()
            self.assertIn(f"{base_task_id}/archive_manifest.json", names)
            self.assertIn(f"{base_task_id}/archive_manifest.md", names)

    def test_check_real_chain_release_bundle_script_passes_on_complete_bundle(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            base_task_id = "demo_bundle_check"

            baseline_dir = td_path / "runs" / "agent" / base_task_id
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_summary = baseline_dir / "baseline_summary.json"
            baseline_summary.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "base_task_id": base_task_id,
                        "run_count": 3,
                        "runs": [],
                        "failures": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            archive_dir = td_path / "runs" / "archive" / base_task_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_manifest = archive_dir / "archive_manifest.json"
            archive_manifest.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "base_task_id": base_task_id,
                        "missing_required_count": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (archive_dir / "archive_manifest.md").write_text("# manifest\n", encoding="utf-8")
            tar_path = (td_path / "runs" / "archive" / f"{base_task_id}.tar.gz").resolve()
            with tarfile.open(tar_path, mode="w:gz") as tf:
                tf.add(archive_dir, arcname=archive_dir.name)

            script = repo_root / "scripts" / "check_real_chain_release_bundle.py"
            cp = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--workspace-root",
                    str(td_path),
                    "--base-task-id",
                    base_task_id,
                    "--require-tar-gz",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr + cp.stdout)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("baseline_status"), "pass")
            self.assertEqual(payload.get("archive_status"), "pass")


class ModelCatalogTests(unittest.TestCase):
    def test_default_catalog_contains_real_adapter_commands(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        catalog = json.loads((repo_root / "configs" / "models" / "catalog.json").read_text(encoding="utf-8"))
        entries = {item["id"]: item for item in catalog.get("models", []) if isinstance(item, dict)}
        self.assertEqual(
            entries["unimol_lambda_plqy_v1"]["params"]["adapters"]["score_candidates_cmd"],
            "python3 scripts/adapters/score_candidates_unimol_adapter.py",
        )
        self.assertEqual(
            entries["unimol_lambda_plqy_v1"]["params"]["adapters"]["train_predictor_cmd"],
            "python3 scripts/adapters/train_predictor_unimol_adapter.py",
        )
        self.assertEqual(
            entries["reinvent4_lambda_em_v2"]["params"]["adapters"]["generate_candidates_cmd"],
            "python3 scripts/adapters/generate_candidates_reinvent4_adapter.py",
        )


class OpenclawEnvExportScriptTests(unittest.TestCase):
    def test_export_openclaw_llm_env_exports_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = td_path / "models.json"
            cfg.write_text(
                json.dumps(
                    {
                        "providers": {
                            "p1": {
                                "baseUrl": "https://chat.example.com/v1",
                                "apiKey": "sk-test",
                                "api": "openai-completions",
                                "models": [{"id": "gpt-5.4"}],
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cp = subprocess.run(
                [sys.executable, "scripts/export_openclaw_llm_env.py", "--config", str(cfg), "--provider", "p1"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr)
            out = cp.stdout
            self.assertIn("export OLED_AGENT_LLM_BACKEND=openai_compat", out)
            self.assertIn("export OLED_AGENT_LLM_MODEL=gpt-5.4", out)
            self.assertIn("export OLED_AGENT_LLM_BASE_URL=https://chat.example.com/v1", out)
            self.assertIn("export OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH=/chat/completions", out)
            self.assertIn("export OLED_AGENT_LLM_AUTH_SCHEME=Bearer", out)

    def test_export_openclaw_llm_env_dotenv_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = td_path / "models.json"
            cfg.write_text(
                json.dumps(
                    {
                        "providers": {
                            "p2": {
                                "baseUrl": "https://chat.example.com",
                                "apiKey": "sk-test-2",
                                "models": [{"id": "gpt-5.5"}],
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_openclaw_llm_env.py",
                    "--config",
                    str(cfg),
                    "--provider",
                    "p2",
                    "--format",
                    "dotenv",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr)
            out = cp.stdout
            self.assertIn('OLED_AGENT_LLM_BACKEND="openai_compat"', out)
            self.assertIn('OLED_AGENT_LLM_MODEL="gpt-5.5"', out)
            self.assertIn('OLED_AGENT_LLM_BASE_URL="https://chat.example.com"', out)
            self.assertIn('OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH="/v1/chat/completions"', out)


class AdapterContractValidatorTests(unittest.TestCase):
    def test_validate_adapter_contract_score_template_passes_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "score_candidates_adapter_template.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "score_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "score_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "template_score_cmd")

    def test_validate_adapter_contract_generate_template_passes_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "generate_candidates_adapter_template.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "generate_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "generate_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "template_generate_cmd")

    def test_validate_adapter_contract_fails_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(__file__).resolve().parents[1]
            td_path = Path(td)
            bad = td_path / "bad_adapter.py"
            bad.write_text("import sys\nsys.exit(7)\n", encoding="utf-8")
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/adapters/validate_adapter_contract.py",
                    "--tool",
                    "train_predictor",
                    "--cmd",
                    f"{sys.executable} {bad}",
                    "--workspace-root",
                    str(repo_root),
                    "--json",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 2, msg=cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "fail")
            self.assertEqual(payload.get("tool"), "train_predictor")
            self.assertEqual(payload.get("error", {}).get("code"), "adapter_nonzero_exit")

    def test_validate_adapter_contract_real_unimol_score_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "score_candidates_unimol_adapter.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "score_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke"},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "score_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "unimol_score_adapter_v1")

    def test_validate_adapter_contract_real_unimol_score_real_mode_with_stub_scorer(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "score_candidates_unimol_adapter.py"
        stub_scorer = repo_root / "scripts" / "adapters" / "stub_unimol_score.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "score_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "OLED_AGENT_UNIMOL_SCORE_MODE": "real",
                "OLED_AGENT_UNIMOL_SCORE_SCRIPT": str(stub_scorer),
                "UNIMOL_REMOTE_HOST": "stub_host",
                "UNIMOL_REMOTE_PY": "stub_py",
                "UNIMOL_REMOTE_TMP_BASE": "/tmp",
            },
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "score_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "unimol_score_adapter_v1")

    def test_validate_adapter_contract_real_unimol_score_prefers_property_model_dir_env(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "score_candidates_unimol_adapter.py"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            scorer = td_path / "stub_score_expect_plqy_model.py"
            expected_model_dir = "/remote/model/plqy_v2"
            scorer.write_text(
                (
                    "import argparse, csv, json\n"
                    "from pathlib import Path\n"
                    "ap = argparse.ArgumentParser()\n"
                    "ap.add_argument('input_csv')\n"
                    "ap.add_argument('output_csv')\n"
                    "ap.add_argument('--model-dir', default='')\n"
                    "ap.add_argument('--property-name', default='plqy')\n"
                    "ap.add_argument('--objective-type', default='maximize')\n"
                    "ap.add_argument('--target-center', default='0.6')\n"
                    "ap.add_argument('--sigma', default='0.2')\n"
                    "args = ap.parse_args()\n"
                    f"assert args.model_dir == {expected_model_dir!r}, f'bad model-dir: {{args.model_dir}}'\n"
                    "rows = list(csv.DictReader(open(args.input_csv, 'r', encoding='utf-8')))\n"
                    "for r in rows:\n"
                    "  r[f\"{args.property_name}_pred\"] = '0.6600'\n"
                    "  r[f\"{args.property_name}_score\"] = '0.660000'\n"
                    "Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)\n"
                    "with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:\n"
                    "  w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n"
                    "  w.writeheader(); w.writerows(rows)\n"
                    "print('ok')\n"
                ),
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/adapters/validate_adapter_contract.py",
                    "--tool",
                    "score_candidates",
                    "--cmd",
                    f"{sys.executable} {script}",
                    "--workspace-root",
                    str(repo_root),
                    "--json",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "OLED_AGENT_UNIMOL_SCORE_MODE": "real",
                    "OLED_AGENT_UNIMOL_SCORE_SCRIPT": str(scorer),
                    "UNIMOL_REMOTE_HOST": "stub_host",
                    "UNIMOL_REMOTE_PY": "stub_py",
                    "UNIMOL_REMOTE_TMP_BASE": "/tmp",
                    "UNIMOL_REMOTE_MODEL_DIR": "/remote/model/default_lambda",
                    "UNIMOL_REMOTE_MODEL_DIR_PLQY": expected_model_dir,
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("tool"), "score_candidates")

    def test_validate_adapter_contract_real_mineru_generate_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "generate_candidates_mineru_adapter.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "generate_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "OLED_AGENT_MINERU_ADAPTER_MODE": "smoke"},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "generate_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "mineru_generate_adapter_v1")

    def test_validate_adapter_contract_real_reinvent4_generate_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "generate_candidates_reinvent4_adapter.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "generate_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "OLED_AGENT_REINVENT4_ADAPTER_MODE": "smoke"},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "generate_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "reinvent4_generate_adapter_v1")

    def test_validate_adapter_contract_real_reinvent4_generate_real_mode_with_stub_pipeline(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "generate_candidates_reinvent4_adapter.py"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source_csv = td_path / "sampling.csv"
            source_csv.write_text(
                "SMILES\nc1ccccc1\nCCO\n",
                encoding="utf-8",
            )
            rankready_csv = td_path / "rankready.csv"
            stub_pipeline = td_path / "stub_reinvent4_pipeline.sh"
            stub_pipeline.write_text(
                (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "SOURCE_CSV=\"$1\"\n"
                    "if [ ! -f \"$SOURCE_CSV\" ]; then\n"
                    "  echo \"missing source csv: $SOURCE_CSV\" >&2\n"
                    "  exit 9\n"
                    "fi\n"
                    "OUT=\"${OLED_AGENT_REINVENT4_RANKREADY_CSV:?}\"\n"
                    "mkdir -p \"$(dirname \"$OUT\")\"\n"
                    "cat > \"$OUT\" <<'CSV'\n"
                    "SMILES\n"
                    "c1ccccc1\n"
                    "CCO\n"
                    "CSV\n"
                ),
                encoding="utf-8",
            )
            cp = subprocess.run(
                [
                    sys.executable,
                    "scripts/adapters/validate_adapter_contract.py",
                    "--tool",
                    "generate_candidates",
                    "--cmd",
                    f"{sys.executable} {script}",
                    "--workspace-root",
                    str(repo_root),
                    "--json",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "OLED_AGENT_REINVENT4_ADAPTER_MODE": "real",
                    "OLED_AGENT_REINVENT4_SOURCE_CSV": str(source_csv),
                    "OLED_AGENT_REINVENT4_PIPELINE_SCRIPT": str(stub_pipeline),
                    "OLED_AGENT_REINVENT4_RANKREADY_CSV": str(rankready_csv),
                },
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("tool"), "generate_candidates")
            preview = payload.get("result_preview", {})
            self.assertEqual(preview.get("adapter"), "reinvent4_generate_adapter_v1")

    def test_validate_adapter_contract_real_molscribe_generate_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "generate_candidates_molscribe_adapter.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "generate_candidates",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "OLED_AGENT_MOLSCRIBE_ADAPTER_MODE": "smoke"},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "generate_candidates")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "molscribe_generate_adapter_v1")

    def test_validate_adapter_contract_real_unimol_train_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "train_predictor_unimol_adapter.py"
        cp = subprocess.run(
            [
                sys.executable,
                "scripts/adapters/validate_adapter_contract.py",
                "--tool",
                "train_predictor",
                "--cmd",
                f"{sys.executable} {script}",
                "--workspace-root",
                str(repo_root),
                "--json",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "OLED_AGENT_UNIMOL_TRAIN_MODE": "smoke",
                "UNIMOL_REMOTE_HOST": "stub_host",
                "UNIMOL_REMOTE_PY": "python3",
                "UNIMOL_REMOTE_TMP_BASE": "/tmp",
            },
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "pass")
        self.assertEqual(payload.get("tool"), "train_predictor")
        preview = payload.get("result_preview", {})
        self.assertEqual(preview.get("adapter"), "unimol_train_adapter_v1")

    def test_agent_run_json_with_real_adapter_catalog_reinvent4_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        request_json = repo_root / "runs" / "test_request_real_adapter_reinvent4_smoke.json"
        request_json.parent.mkdir(parents=True, exist_ok=True)
        request_json.write_text(
            json.dumps(
                {
                    "task_id": "task_real_adapter_reinvent4_smoke",
                    "request_text": "设计470nm附近且高PLQY分子",
                    "mode": "fast_screen",
                    "targets": [{"property": "plqy", "objective": "maximize", "target_value": 0.6}],
                    "budget": {"max_candidates": 5},
                    "model_preferences": {
                        "predictor_id": "unimol_lambda_plqy_real_v1",
                        "generator_id": "reinvent4_generator_real_v1",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "oled_agent.cli",
                "agent-run-json",
                "--workspace-root",
                str(repo_root),
                "--catalog",
                str(repo_root / "scripts" / "adapters" / "real_adapters_catalog.json"),
                "--request-json",
                str(request_json),
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(repo_root / "src"),
                "OLED_AGENT_REINVENT4_ADAPTER_MODE": "smoke",
                "OLED_AGENT_UNIMOL_SCORE_MODE": "smoke",
            },
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        payload = json.loads(cp.stdout)
        self.assertEqual(payload.get("status"), "success")
        execution_path = Path(str(payload.get("execution_path") or ""))
        self.assertTrue(execution_path.exists())
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        records = execution.get("records", [])
        by_name = {r.get("name"): r for r in records}
        gen = by_name.get("generate_candidates", {}).get("result", {})
        score = by_name.get("score_candidates", {}).get("result", {})
        self.assertEqual(gen.get("adapter"), "reinvent4_generate_adapter_v1")
        self.assertEqual(score.get("adapter"), "unimol_score_adapter_v1")

    def test_check_quickstart_chain_script_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "adapters" / "check_quickstart_chain.sh"
        cp = subprocess.run(
            [str(script), "test_quickstart_chain_script"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src"), "OLED_AGENT_ENABLE_WEB_EVIDENCE": "0"},
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stdout + cp.stderr)
        self.assertIn("[PASS] quickstart chain completed", cp.stdout)
        self.assertIn("generate_adapter=template_generate_cmd", cp.stdout)
        self.assertIn("score_adapter=template_score_cmd", cp.stdout)


class UiPrototypeTests(unittest.TestCase):
    def _load_ui_module(self):
        try:
            from ui import app as ui_app_mod  # type: ignore
        except ModuleNotFoundError as exc:
            self.skipTest(f"ui dependency missing: {exc}")
        return ui_app_mod

    def test_ui_health_endpoint(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "pass")
        self.assertTrue(str(payload.get("repo_root") or ""))

    def test_ui_projects_create_history_and_upload_ref(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                create_resp = client.post(
                    "/api/projects",
                    json={
                        "project_id": "ui_proj_a",
                        "title": "project A",
                        "options": {
                            "planner_provider": "rule_based_v1",
                            "catalog_path": "configs/models/catalog.json",
                            "web_search_enabled": True,
                            "web_topk": 6,
                        },
                    },
                )
                self.assertEqual(create_resp.status_code, 200)
                created = create_resp.get_json()
                self.assertEqual(created.get("status"), "pass")
                project = created.get("project") if isinstance(created.get("project"), dict) else {}
                self.assertEqual(project.get("project_id"), "ui_proj_a")
                self.assertEqual(project.get("title"), "project A")

                upload_resp = client.post(
                    "/api/projects/ui_proj_a/upload-ref",
                    json={"path": "/tmp/demo_candidates.csv", "label": "manual", "kind": "path_ref"},
                )
                self.assertEqual(upload_resp.status_code, 200)
                uploaded = upload_resp.get_json()
                self.assertEqual(uploaded.get("status"), "pass")
                attachment = uploaded.get("attachment") if isinstance(uploaded.get("attachment"), dict) else {}
                self.assertEqual(attachment.get("path"), "/tmp/demo_candidates.csv")

                hist_resp = client.get("/api/projects/ui_proj_a/history?limit=50")
                self.assertEqual(hist_resp.status_code, 200)
                hist = hist_resp.get_json()
                self.assertEqual(hist.get("status"), "pass")
                messages = hist.get("messages") if isinstance(hist.get("messages"), list) else []
                self.assertTrue(any(str(m.get("kind") or "") == "attachment" for m in messages if isinstance(m, dict)))
                attachments = hist.get("attachments") if isinstance(hist.get("attachments"), list) else []
                self.assertEqual(len(attachments), 1)

    def test_ui_project_memory_roundtrip_persists(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                create_resp = client.post(
                    "/api/projects",
                    json={
                        "project_id": "ui_proj_memory",
                        "title": "memory project",
                        "options": {"memory_enabled": True, "planner_provider": "rule_based_v1"},
                        "memory_notes": "固定目标: 470nm附近\n禁用 alert: azo",
                    },
                )
                self.assertEqual(create_resp.status_code, 200)
                created = create_resp.get_json()
                self.assertEqual(created.get("status"), "pass")
                proj = created.get("project") if isinstance(created.get("project"), dict) else {}
                opts = proj.get("options") if isinstance(proj.get("options"), dict) else {}
                self.assertEqual(bool(opts.get("memory_enabled")), True)
                self.assertIn("470nm", str(proj.get("memory_notes") or ""))
                self.assertTrue(str(proj.get("memory_updated_at") or ""))

                hist_resp = client.get("/api/projects/ui_proj_memory/history?limit=20")
                self.assertEqual(hist_resp.status_code, 200)
                hist = hist_resp.get_json()
                hist_proj = hist.get("project") if isinstance(hist.get("project"), dict) else {}
                self.assertIn("禁用 alert", str(hist_proj.get("memory_notes") or ""))

    def test_ui_chat_send_injects_project_memory_into_intake_request_when_enabled(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={
                        "project_id": "ui_chat_mem_on",
                        "title": "memory on",
                        "options": {"memory_enabled": True},
                        "memory_notes": "优先考虑470nm +/- 10nm，避免已知高风险骨架。",
                    },
                )
                intake_result = {
                    "task_id": "ui_chat_mem_on_20260515_000001",
                    "status": "need_user_input",
                    "task_draft_path": str(root / "runs" / "agent" / "ui_chat_mem_on_20260515_000001" / "task.draft.json"),
                    "missing_fields": ["candidate_data"],
                    "questions": ["候选数据来源是什么？"],
                }
                fake_cp = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-intake"],
                    returncode=2,
                    stdout=json.dumps(intake_result, ensure_ascii=False),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
                    resp = client.post(
                        "/api/chat/send",
                        json={
                            "project_id": "ui_chat_mem_on",
                            "message": "设计470nm附近且高PLQY分子",
                            "options": {"planner_provider": "rule_based_v1", "memory_enabled": True},
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "need_user_input")
                cmd = mocked.call_args.args[0]
                self.assertIn("--request", cmd)
                request_text = str(cmd[cmd.index("--request") + 1])
                self.assertIn("Project memory context:", request_text)
                self.assertIn("470nm", request_text)
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                self.assertTrue(
                    any(isinstance(m, dict) and str(m.get("kind") or "") == "memory_context" for m in messages)
                )

    def test_ui_chat_send_does_not_inject_memory_when_disabled(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={
                        "project_id": "ui_chat_mem_off",
                        "title": "memory off",
                        "options": {"memory_enabled": False},
                        "memory_notes": "这个文本不应被注入。",
                    },
                )
                intake_result = {
                    "task_id": "ui_chat_mem_off_20260515_000001",
                    "status": "need_user_input",
                    "task_draft_path": str(root / "runs" / "agent" / "ui_chat_mem_off_20260515_000001" / "task.draft.json"),
                    "missing_fields": ["candidate_data"],
                    "questions": ["候选数据来源是什么？"],
                }
                fake_cp = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-intake"],
                    returncode=2,
                    stdout=json.dumps(intake_result, ensure_ascii=False),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
                    resp = client.post(
                        "/api/chat/send",
                        json={
                            "project_id": "ui_chat_mem_off",
                            "message": "设计470nm附近且高PLQY分子",
                            "options": {"planner_provider": "rule_based_v1", "memory_enabled": False},
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "need_user_input")
                cmd = mocked.call_args.args[0]
                request_text = str(cmd[cmd.index("--request") + 1])
                self.assertNotIn("Project memory context:", request_text)
                self.assertEqual(request_text, "设计470nm附近且高PLQY分子")

    def test_ui_project_history_roundtrip_preserves_project_identity(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                create_resp = client.post(
                    "/api/projects",
                    json={"project_id": "ui_proj_switch", "title": "switch target"},
                )
                self.assertEqual(create_resp.status_code, 200)
                created = create_resp.get_json()
                self.assertEqual(created.get("status"), "pass")

                hist_resp = client.get("/api/projects/ui_proj_switch/history?limit=20")
                self.assertEqual(hist_resp.status_code, 200)
                hist = hist_resp.get_json()
                self.assertEqual(hist.get("status"), "pass")
                proj = hist.get("project") if isinstance(hist.get("project"), dict) else {}
                self.assertEqual(proj.get("project_id"), "ui_proj_switch")
                self.assertEqual(proj.get("title"), "switch target")

    def test_ui_projects_summary_includes_runtime_health_from_current_task(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                create = client.post(
                    "/api/projects",
                    json={"project_id": "ui_proj_health", "title": "health"},
                )
                self.assertEqual(create.status_code, 200)
                run_task_id = "ui_proj_health_t1"
                run_dir = root / "runs" / "agent" / run_task_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "execution.json").write_text(
                    json.dumps(
                        {
                            "task_id": run_task_id,
                            "status": "failed",
                            "records": [
                                {"name": "search_dataset", "status": "success"},
                                {"name": "score_candidates", "status": "failed", "error": "adapter timeout after 300s"},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                project_path = root / "runs" / "ui_sessions" / "projects" / "ui_proj_health.json"
                project_payload = json.loads(project_path.read_text(encoding="utf-8"))
                project_payload["current_task_id"] = run_task_id
                project_path.write_text(json.dumps(project_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                list_resp = client.get("/api/projects?limit=20")
                self.assertEqual(list_resp.status_code, 200)
                payload = list_resp.get_json()
                self.assertEqual(payload.get("status"), "pass")
                projects = payload.get("projects") if isinstance(payload.get("projects"), list) else []
                row = next((x for x in projects if isinstance(x, dict) and x.get("project_id") == "ui_proj_health"), {})
                health = row.get("runtime_health") if isinstance(row.get("runtime_health"), dict) else {}
                self.assertEqual(health.get("status"), "failed")
                self.assertEqual(int(health.get("record_count") or 0), 2)
                self.assertEqual(int(health.get("success_steps") or 0), 1)
                self.assertEqual(int(health.get("failed_steps") or 0), 1)
                self.assertAlmostEqual(float(health.get("success_ratio") or 0.0), 0.5, places=6)
                self.assertEqual(health.get("latest_failed_step"), "score_candidates")
                self.assertIn("timeout", str(health.get("latest_failed_error") or ""))
                self.assertEqual(int(health.get("recent_duration_ms") or 0), 0)

    def test_ui_html_contains_project_runtime_health_field(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("project_runtime_health", html)
        self.assertIn("formatRuntimeHealth(", html)

    def test_ui_html_contains_prompt_history_and_shortcut_hints(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("prompt_history_box", html)
        self.assertIn("Ctrl/Cmd+Enter 发送", html)
        self.assertIn("prompt-chip", html)

    def test_ui_html_contains_task_compare_controls(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("compare_other_task_id", html)
        self.assertIn("compareTasks()", html)
        self.assertIn("compareSelectedArtifact()", html)

    def test_ui_html_contains_workspace_hud_and_web_search_action(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("workspace-hud", html)
        self.assertIn("hud_project_id", html)
        self.assertIn("current_task_id_hud", html)
        self.assertIn("sendWebSearchHint()", html)
        self.assertIn("downloadTaskBundle()", html)
        self.assertIn("Download Task Bundle", html)
        self.assertIn("memory_enabled", html)
        self.assertIn("memory_notes", html)
        self.assertIn("updateMemoryStatus()", html)
        self.assertIn("project_read_only", html)
        self.assertIn("updateProjectLockStatus()", html)
        self.assertIn("snapshot_note", html)
        self.assertIn("createProjectSnapshot()", html)
        self.assertIn("loadProjectSnapshots()", html)

    def test_ui_html_contains_workspace_url_controls(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("openWorkspaceWindow()", html)
        self.assertIn("copyWorkspaceLink()", html)
        self.assertIn("readProjectIdFromUrl()", html)
        self.assertIn("syncWorkspaceUrl(", html)
        self.assertIn("const hist = await loadHistory();", html)
        self.assertIn("if (!ok) {", html)
        self.assertIn("await saveProject();", html)
        self.assertIn("project_session_list", html)
        self.assertIn("renderProjectSessionBoard(", html)
        self.assertIn("openProjectWorkspace(", html)
        self.assertIn("resumeProjectTask(", html)
        self.assertIn("latest_failed_step=", html)
        self.assertIn("failed_error=", html)
        self.assertIn("Retry Failed", html)
        self.assertIn("retryProjectFailedStep(", html)
        self.assertIn("Timeline", html)
        self.assertIn("showProjectTimeline(", html)
        self.assertIn("Copy Task ID", html)
        self.assertIn("copyProjectTaskId(", html)
        self.assertIn("session_filter_text", html)
        self.assertIn("session_filter_health", html)
        self.assertIn("session_sort_mode", html)
        self.assertIn("applySessionBoardControls()", html)
        self.assertIn("quickFilterFailedOnly()", html)
        self.assertIn("quickFilterByHealth('failed')", html)
        self.assertIn("quickFilterByHealth('success')", html)
        self.assertIn("quickFilterByHealth('none')", html)
        self.assertIn("quickSortPriority()", html)
        self.assertIn("openTopPrioritySession()", html)
        self.assertIn("openNextFailedSession()", html)
        self.assertIn("togglePinnedOnly()", html)
        self.assertIn("toggleSessionBoardGroupedView()", html)
        self.assertIn("batchShowProjectSummary()", html)
        self.assertIn("batchValidateProjectTask()", html)
        self.assertIn("batchRetryFailedProjectStep()", html)
        self.assertIn("exportSessionBoardBatchResult()", html)
        self.assertIn("exportSessionBoardBatchResultPersisted()", html)
        self.assertIn("persistBatchPayload(", html)
        self.assertIn("loadBatchHistory()", html)
        self.assertIn("replayLatestBatchAction()", html)
        self.assertIn("replayFailedLatestBatchAction()", html)
        self.assertIn("loadFailedReplayQueueById()", html)
        self.assertIn("replayFailedQueueNow()", html)
        self.assertIn("readBatchHistoryControls()", html)
        self.assertIn("resetBatchHistoryOffsetAndReload()", html)
        self.assertIn("prevBatchHistoryPage()", html)
        self.assertIn("nextBatchHistoryPage()", html)
        self.assertIn("renderBatchHistoryList(", html)
        self.assertIn("viewBatchExportById()", html)
        self.assertIn("replayBatchExportById()", html)
        self.assertIn("replayFailedBatchExportById()", html)
        self.assertIn("deleteBatchExportById()", html)
        self.assertIn("compareBatchExportsById()", html)
        self.assertIn("downloadBatchExportById('json')", html)
        self.assertIn("downloadBatchExportById('csv')", html)
        self.assertIn("readBatchCompareExportId()", html)
        self.assertIn("readBatchReplayOptions()", html)
        self.assertIn("renderBatchHistoryMetrics(", html)
        self.assertIn("applyReplayPreset('safe')", html)
        self.assertIn("applyReplayPreset('fast')", html)
        self.assertIn("applyReplayPreset('dryrun')", html)
        self.assertIn("saveReplayDefaultsToProject()", html)
        self.assertIn("applyBatchReplayOptions(", html)
        self.assertIn("batch_replay_defaults", html)
        self.assertIn("batch_export_id", html)
        self.assertIn("batch_export_compare_id", html)
        self.assertIn("batch_replay_dry_run", html)
        self.assertIn("batch_replay_failed_only", html)
        self.assertIn("batch_replay_retry_max", html)
        self.assertIn("batch_replay_retry_backoff_ms", html)
        self.assertIn("batch_replay_max_concurrency", html)
        self.assertIn("batch_history_action_filter", html)
        self.assertIn("batch_history_status_filter", html)
        self.assertIn("batch_history_page_size", html)
        self.assertIn("batch_history_offset", html)
        self.assertIn("session_batch_limit", html)
        self.assertIn("project_batch_history_summary", html)
        self.assertIn("project_batch_history_metrics", html)
        self.assertIn("project_failed_queue_summary", html)
        self.assertIn("project_failed_queue", html)
        self.assertIn("project_batch_history", html)
        self.assertIn("project_batch_history_list", html)
        self.assertIn("clearSessionBoardControls()", html)
        self.assertIn("session_auto_refresh", html)
        self.assertIn("session_refresh_seconds", html)
        self.assertIn("onSessionAutoRefreshChanged()", html)
        self.assertIn("SESSION_BOARD_KEY", html)
        self.assertIn("loadSessionBoardState()", html)
        self.assertIn("saveSessionBoardState(", html)
        self.assertIn("ensureSessionAutoRefresh()", html)
        self.assertIn("project_board_summary", html)
        self.assertIn("pinnedProjectIds", html)
        self.assertIn("pinnedOnly", html)
        self.assertIn("groupedView", html)
        self.assertIn("batchLimit", html)
        self.assertIn("toggleProjectPin(", html)
        self.assertIn("computeSessionBoardRows(", html)
        self.assertIn("recentSessionBatchRows(", html)
        self.assertIn("storeLatestBatchPayload(", html)
        self.assertIn("readLatestBatchPayload(", html)
        self.assertIn("project-session-section", html)
        self.assertIn("project-session-status", html)
        self.assertIn("Pin", html)
        self.assertIn("pinned=", html)
        self.assertIn("mode=", html)
        self.assertIn("batch_limit=", html)
        self.assertIn("Failed Count (", html)
        self.assertIn("Success Count (", html)
        self.assertIn("None Count (", html)
        self.assertIn("batch_retry_failed", html)
        self.assertIn("batch_export", html)
        self.assertIn("/batch-export", html)
        self.assertIn("/batch-exports", html)
        self.assertIn("/batch-exports/compare", html)
        self.assertIn("/batch-exports/replay-latest", html)
        self.assertIn("/batch-exports/${encodeURIComponent(eid)}/failed-queue", html)
        self.assertIn("?${qs.toString()}", html)
        self.assertIn("/batch-exports/${encodeURIComponent(eid)}", html)
        self.assertIn("/batch-exports/${encodeURIComponent(eid)}/download", html)
        self.assertIn("/batch-exports/${encodeURIComponent(eid)}/replay", html)
        self.assertIn("options: readBatchReplayOptions()", html)
        self.assertIn("readBatchReplayOptions(true)", html)
        self.assertIn("Summary", html)
        self.assertIn("showProjectSummary(", html)
        self.assertIn("Validate", html)
        self.assertIn("validateProjectTask(", html)
        self.assertIn("recent_duration=", html)
        self.assertIn("success_ratio=", html)
        self.assertIn("records=", html)
        self.assertIn("project-session-progress-bar", html)
        self.assertIn("clone_project_id", html)
        self.assertIn("cloneProject()", html)
        self.assertIn("cloneAndOpenProject()", html)
        self.assertIn("snapshotLockProject()", html)
        self.assertIn("/api/projects/${encodeURIComponent(sourceProjectId)}/clone", html)
        self.assertIn("/api/projects/${encodeURIComponent(pid)}/snapshots", html)
        self.assertIn("/api/projects/${encodeURIComponent(pid)}/snapshots/${encodeURIComponent(sid)}/restore", html)

    def test_ui_upload_ref_accepts_multipart_file(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_upload_case", "title": "upload"})
                data = {
                    "label": "browser_upload",
                    "file": (io.BytesIO(b"smiles,plqy\nc1ccccc1,0.6\n"), "candidates.csv"),
                }
                resp = client.post("/api/projects/ui_upload_case/upload-ref", data=data, content_type="multipart/form-data")
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "pass")
                att = payload.get("attachment") if isinstance(payload.get("attachment"), dict) else {}
                path = Path(str(att.get("path") or ""))
                self.assertTrue(path.exists())
                self.assertIn("runs/ui_sessions/uploads/ui_upload_case", str(path))

    def test_ui_project_export_import_roundtrip(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={"project_id": "ui_proj_export_src", "title": "src", "options": {"planner_provider": "rule_based_v1"}},
                )
                client.post(
                    "/api/projects/ui_proj_export_src/upload-ref",
                    json={"path": "/tmp/src.csv", "label": "src_path", "kind": "path_ref"},
                )
                exp_resp = client.get("/api/projects/ui_proj_export_src/export")
                self.assertEqual(exp_resp.status_code, 200)
                exp_payload = exp_resp.get_json()
                self.assertEqual(exp_payload.get("status"), "pass")
                project_blob = exp_payload.get("project") if isinstance(exp_payload.get("project"), dict) else {}
                self.assertEqual(project_blob.get("project_id"), "ui_proj_export_src")
                src_options = project_blob.get("options") if isinstance(project_blob.get("options"), dict) else {}
                replay_defaults = src_options.get("batch_replay_defaults") if isinstance(src_options.get("batch_replay_defaults"), dict) else {}
                self.assertIn("dry_run", replay_defaults)
                self.assertIn("failed_only", replay_defaults)

                import_resp = client.post(
                    "/api/projects/import",
                    json={"project": project_blob, "project_id": "ui_proj_export_dst", "override": False},
                )
                self.assertEqual(import_resp.status_code, 200)
                imported = import_resp.get_json()
                self.assertEqual(imported.get("status"), "pass")
                proj_summary = imported.get("project") if isinstance(imported.get("project"), dict) else {}
                self.assertEqual(proj_summary.get("project_id"), "ui_proj_export_dst")

                hist_resp = client.get("/api/projects/ui_proj_export_dst/history")
                self.assertEqual(hist_resp.status_code, 200)
                hist_payload = hist_resp.get_json()
                self.assertEqual(hist_payload.get("status"), "pass")
                attachments = hist_payload.get("attachments") if isinstance(hist_payload.get("attachments"), list) else []
                self.assertGreaterEqual(len(attachments), 1)

                conflict_resp = client.post(
                    "/api/projects/import",
                    json={"project": project_blob, "project_id": "ui_proj_export_dst", "override": False},
                )
                self.assertEqual(conflict_resp.status_code, 409)
                conflict_payload = conflict_resp.get_json()
                self.assertEqual(conflict_payload.get("error"), "project_exists")

    def test_ui_project_clone_default_clears_runtime_and_keeps_context(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={
                        "project_id": "ui_clone_src",
                        "title": "clone src",
                        "options": {"memory_enabled": True},
                        "memory_notes": "memory from source",
                    },
                )
                source = ui_app_mod._load_project_state("ui_clone_src")
                self.assertIsInstance(source, dict)
                source["current_task_id"] = "clone_src_task"
                source["task_draft_path"] = "/tmp/a.json"
                source["task_json_path"] = "/tmp/b.json"
                source["request_path"] = "/tmp/c.json"
                source["last_runtime"] = {"status": "success"}
                source["pending_input"] = {"stage": "intake", "missing_fields": ["candidate_data"], "questions": []}
                source["attachments"] = [
                    {
                        "id": "att1",
                        "kind": "path_ref",
                        "label": "dataset",
                        "name": "src.csv",
                        "path": "/tmp/src.csv",
                        "created_at": "2026-05-15T12:00:00+08:00",
                    }
                ]
                ui_app_mod._append_message(source, role="user", content="source message", kind="chat")
                ui_app_mod._save_project_state(source)

                clone_resp = client.post(
                    "/api/projects/ui_clone_src/clone",
                    json={"target_project_id": "ui_clone_dst"},
                )
                self.assertEqual(clone_resp.status_code, 200)
                payload = clone_resp.get_json()
                self.assertEqual(payload.get("status"), "pass")
                proj = payload.get("project") if isinstance(payload.get("project"), dict) else {}
                self.assertEqual(proj.get("project_id"), "ui_clone_dst")

                hist_resp = client.get("/api/projects/ui_clone_dst/history?limit=300")
                self.assertEqual(hist_resp.status_code, 200)
                hist = hist_resp.get_json()
                hist_proj = hist.get("project") if isinstance(hist.get("project"), dict) else {}
                self.assertEqual(hist_proj.get("current_task_id"), "")
                self.assertEqual(hist_proj.get("task_draft_path"), "")
                self.assertEqual(hist_proj.get("task_json_path"), "")
                self.assertEqual(hist_proj.get("request_path"), "")
                self.assertEqual(hist_proj.get("pending_input"), {})
                self.assertIn("memory from source", str(hist_proj.get("memory_notes") or ""))
                attachments = hist.get("attachments") if isinstance(hist.get("attachments"), list) else []
                self.assertEqual(len(attachments), 1)
                messages = hist.get("messages") if isinstance(hist.get("messages"), list) else []
                contents = [str(m.get("content") or "") for m in messages if isinstance(m, dict)]
                self.assertTrue(any("source message" in c for c in contents))
                self.assertTrue(any("Project cloned from ui_clone_src" in c for c in contents))

    def test_ui_project_clone_options_can_drop_messages_and_attachments_and_keep_runtime(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_clone_opt_src", "title": "clone options src"})
                source = ui_app_mod._load_project_state("ui_clone_opt_src")
                self.assertIsInstance(source, dict)
                source["current_task_id"] = "clone_opt_task"
                source["last_runtime"] = {"status": "failed"}
                source["attachments"] = [
                    {"id": "att1", "kind": "path_ref", "label": "dataset", "name": "src.csv", "path": "/tmp/src.csv", "created_at": "2026-05-15T12:00:00+08:00"}
                ]
                ui_app_mod._append_message(source, role="user", content="old message", kind="chat")
                ui_app_mod._save_project_state(source)

                clone_resp = client.post(
                    "/api/projects/ui_clone_opt_src/clone",
                    json={
                        "target_project_id": "ui_clone_opt_dst",
                        "options": {
                            "copy_messages": False,
                            "copy_attachments": False,
                            "carry_runtime": True,
                        },
                    },
                )
                self.assertEqual(clone_resp.status_code, 200)

                hist_resp = client.get("/api/projects/ui_clone_opt_dst/history?limit=300")
                self.assertEqual(hist_resp.status_code, 200)
                hist = hist_resp.get_json()
                hist_proj = hist.get("project") if isinstance(hist.get("project"), dict) else {}
                self.assertEqual(hist_proj.get("current_task_id"), "clone_opt_task")
                attachments = hist.get("attachments") if isinstance(hist.get("attachments"), list) else []
                self.assertEqual(len(attachments), 0)
                messages = hist.get("messages") if isinstance(hist.get("messages"), list) else []
                self.assertEqual(len(messages), 1)
                clone_msg = messages[0] if isinstance(messages[0], dict) else {}
                self.assertEqual(str(clone_msg.get("kind") or ""), "project_clone")

    def test_ui_project_clone_rejects_conflict_without_override(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_clone_conf_src", "title": "src"})
                client.post("/api/projects", json={"project_id": "ui_clone_conf_dst", "title": "dst"})
                clone_resp = client.post(
                    "/api/projects/ui_clone_conf_src/clone",
                    json={"target_project_id": "ui_clone_conf_dst", "override": False},
                )
                self.assertEqual(clone_resp.status_code, 409)
                payload = clone_resp.get_json()
                self.assertEqual(payload.get("status"), "fail")
                self.assertEqual(payload.get("error"), "project_exists")

    def test_ui_project_clone_can_set_read_only_target_and_block_chat_send(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_clone_ro_src", "title": "src"})
                clone_resp = client.post(
                    "/api/projects/ui_clone_ro_src/clone",
                    json={
                        "target_project_id": "ui_clone_ro_dst",
                        "target_options": {"project_read_only": True},
                    },
                )
                self.assertEqual(clone_resp.status_code, 200)
                clone_payload = clone_resp.get_json()
                self.assertEqual(clone_payload.get("status"), "pass")
                proj = clone_payload.get("project") if isinstance(clone_payload.get("project"), dict) else {}
                opts = proj.get("options") if isinstance(proj.get("options"), dict) else {}
                self.assertEqual(bool(opts.get("project_read_only")), True)

                send_resp = client.post(
                    "/api/chat/send",
                    json={"project_id": "ui_clone_ro_dst", "message": "设计470nm附近且高PLQY分子"},
                )
                self.assertEqual(send_resp.status_code, 409)
                send_payload = send_resp.get_json()
                self.assertEqual(send_payload.get("status"), "fail")
                self.assertEqual(send_payload.get("error"), "project_read_only")

    def test_ui_upload_ref_rejects_read_only_project(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={"project_id": "ui_upload_ro", "title": "ro", "options": {"project_read_only": True}},
                )
                resp = client.post(
                    "/api/projects/ui_upload_ro/upload-ref",
                    json={"path": "/tmp/demo.csv", "label": "demo", "kind": "path_ref"},
                )
                self.assertEqual(resp.status_code, 409)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "fail")
                self.assertEqual(payload.get("error"), "project_read_only")

    def test_ui_project_snapshot_create_list_and_restore(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={"project_id": "ui_snap_proj", "title": "snapshot source", "options": {"memory_enabled": True}},
                )
                project = ui_app_mod._load_project_state("ui_snap_proj")
                self.assertIsInstance(project, dict)
                project["current_task_id"] = "snap_task_1"
                ui_app_mod._append_message(project, role="user", kind="chat", content="before snapshot")
                ui_app_mod._save_project_state(project)

                create_resp = client.post(
                    "/api/projects/ui_snap_proj/snapshots",
                    json={"note": "before edit"},
                )
                self.assertEqual(create_resp.status_code, 200)
                created = create_resp.get_json()
                self.assertEqual(created.get("status"), "pass")
                snap = created.get("snapshot") if isinstance(created.get("snapshot"), dict) else {}
                sid = str(snap.get("snapshot_id") or "")
                self.assertTrue(sid)

                client.post(
                    "/api/projects",
                    json={"project_id": "ui_snap_proj", "title": "after edit", "memory_notes": "changed memory"},
                )
                updated = ui_app_mod._load_project_state("ui_snap_proj")
                self.assertIsInstance(updated, dict)
                ui_app_mod._append_message(updated, role="user", kind="chat", content="after snapshot mutation")
                ui_app_mod._save_project_state(updated)

                restore_resp = client.post(
                    f"/api/projects/ui_snap_proj/snapshots/{sid}/restore",
                    json={"auto_snapshot_before": True, "restore_note": "rollback test"},
                )
                self.assertEqual(restore_resp.status_code, 200)
                restored_payload = restore_resp.get_json()
                self.assertEqual(restored_payload.get("status"), "pass")
                proj_summary = restored_payload.get("project") if isinstance(restored_payload.get("project"), dict) else {}
                self.assertEqual(proj_summary.get("title"), "snapshot source")
                auto_before = restored_payload.get("auto_snapshot_before") if isinstance(restored_payload.get("auto_snapshot_before"), dict) else {}
                self.assertTrue(str(auto_before.get("snapshot_id") or ""))

                hist_resp = client.get("/api/projects/ui_snap_proj/history?limit=200")
                self.assertEqual(hist_resp.status_code, 200)
                hist = hist_resp.get_json()
                messages = hist.get("messages") if isinstance(hist.get("messages"), list) else []
                self.assertTrue(any("restored from snapshot" in str(m.get("content") or "").lower() for m in messages if isinstance(m, dict)))

                list_resp = client.get("/api/projects/ui_snap_proj/snapshots?limit=20&offset=0")
                self.assertEqual(list_resp.status_code, 200)
                listed = list_resp.get_json()
                self.assertEqual(listed.get("status"), "pass")
                snaps = listed.get("snapshots") if isinstance(listed.get("snapshots"), list) else []
                ids = [str(x.get("snapshot_id") or "") for x in snaps if isinstance(x, dict)]
                self.assertIn(sid, ids)
                self.assertGreaterEqual(len(ids), 2)

    def test_ui_project_snapshot_restore_rejects_read_only_project(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post(
                    "/api/projects",
                    json={"project_id": "ui_snap_ro", "title": "ro", "options": {"project_read_only": True}},
                )
                create_resp = client.post(
                    "/api/projects/ui_snap_ro/snapshots",
                    json={"note": "ro snapshot"},
                )
                self.assertEqual(create_resp.status_code, 200)
                payload = create_resp.get_json()
                snap = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
                sid = str(snap.get("snapshot_id") or "")
                self.assertTrue(sid)
                restore_resp = client.post(
                    f"/api/projects/ui_snap_ro/snapshots/{sid}/restore",
                    json={"auto_snapshot_before": False},
                )
                self.assertEqual(restore_resp.status_code, 409)
                restore_payload = restore_resp.get_json()
                self.assertEqual(restore_payload.get("status"), "fail")
                self.assertEqual(restore_payload.get("error"), "project_read_only")

    def test_ui_batch_export_list_and_replay_latest(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_proj_batch", "title": "batch"})

                task_id = "ui_proj_batch_t1"
                run_dir = root / "runs" / "agent" / task_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "execution.json").write_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "status": "success",
                            "records": [
                                {"name": "search_dataset", "status": "success"},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (run_dir / "decision_summary.json").write_text(
                    json.dumps({"task_id": task_id, "status": "success", "inference_step": {"status": "success"}}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                (run_dir / "task_state.json").write_text(
                    json.dumps({"task_id": task_id, "current_stage": "DONE", "status": "success"}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                save_resp = client.post(
                    "/api/projects/ui_proj_batch/batch-export",
                    json={
                        "payload": {
                            "status": "pass",
                            "action": "batch_summary",
                            "limit": 2,
                            "count": 2,
                            "rows": [
                                {"task_id": task_id, "project_id": "ui_proj_batch"},
                                {"task_id": "ui_proj_batch_missing", "project_id": "ui_proj_batch"},
                            ],
                            "results": [
                                {
                                    "task_id": task_id,
                                    "project_id": "ui_proj_batch",
                                    "http_status": 200,
                                    "data": {"status": "pass"},
                                },
                                {
                                    "task_id": "ui_proj_batch_missing",
                                    "project_id": "ui_proj_batch",
                                    "http_status": 404,
                                    "data": {"status": "missing"},
                                },
                            ],
                            "created_at": "2026-05-15T10:00:00+08:00",
                        }
                    },
                )
                self.assertEqual(save_resp.status_code, 200)
                saved = save_resp.get_json()
                self.assertEqual(saved.get("status"), "pass")
                entry = saved.get("batch_export") if isinstance(saved.get("batch_export"), dict) else {}
                export_path = Path(str(entry.get("path") or ""))
                self.assertTrue(export_path.exists())
                self.assertIn("runs/ui_sessions/exports/ui_proj_batch", str(export_path))

                save_resp_2 = client.post(
                    "/api/projects/ui_proj_batch/batch-export",
                    json={
                        "payload": {
                            "status": "fail",
                            "action": "batch_validate",
                            "limit": 1,
                            "count": 1,
                            "rows": [{"task_id": task_id, "project_id": "ui_proj_batch"}],
                            "results": [{"task_id": task_id, "project_id": "ui_proj_batch"}],
                            "created_at": "2026-05-15T09:00:00+08:00",
                        }
                    },
                )
                self.assertEqual(save_resp_2.status_code, 200)

                save_resp_3 = client.post(
                    "/api/projects/ui_proj_batch/batch-export",
                    json={
                        "payload": {
                            "status": "pass",
                            "action": "batch_summary",
                            "limit": 1,
                            "count": 1,
                            "rows": [{"task_id": task_id, "project_id": "ui_proj_batch"}],
                            "results": [{"task_id": task_id, "project_id": "ui_proj_batch"}],
                            "created_at": "2026-05-15T08:30:00+08:00",
                        }
                    },
                )
                self.assertEqual(save_resp_3.status_code, 200)

                filtered_page_1 = client.get("/api/projects/ui_proj_batch/batch-exports?limit=1&offset=0&action=batch_summary&status=pass")
                self.assertEqual(filtered_page_1.status_code, 200)
                fp1_payload = filtered_page_1.get_json()
                self.assertEqual(fp1_payload.get("status"), "pass")
                self.assertEqual(int(fp1_payload.get("total_count") or 0), 2)
                self.assertEqual(bool(fp1_payload.get("has_more")), True)
                fp1_exports = fp1_payload.get("exports") if isinstance(fp1_payload.get("exports"), list) else []
                self.assertEqual(len(fp1_exports), 1)
                self.assertEqual(str(fp1_exports[0].get("action") or ""), "batch_summary")
                self.assertEqual(str(fp1_exports[0].get("status") or ""), "pass")

                filtered_page_2 = client.get("/api/projects/ui_proj_batch/batch-exports?limit=1&offset=1&action=batch_summary&status=pass")
                self.assertEqual(filtered_page_2.status_code, 200)
                fp2_payload = filtered_page_2.get_json()
                self.assertEqual(fp2_payload.get("status"), "pass")
                self.assertEqual(int(fp2_payload.get("total_count") or 0), 2)
                self.assertEqual(bool(fp2_payload.get("has_more")), False)
                fp2_exports = fp2_payload.get("exports") if isinstance(fp2_payload.get("exports"), list) else []
                self.assertEqual(len(fp2_exports), 1)

                partial_filter_resp = client.get("/api/projects/ui_proj_batch/batch-exports?status=partial")
                self.assertEqual(partial_filter_resp.status_code, 200)

                list_resp = client.get("/api/projects/ui_proj_batch/batch-exports?limit=10")
                self.assertEqual(list_resp.status_code, 200)
                listed = list_resp.get_json()
                self.assertEqual(listed.get("status"), "pass")
                exports = listed.get("exports") if isinstance(listed.get("exports"), list) else []
                self.assertGreaterEqual(len(exports), 1)
                export_id = str(exports[0].get("export_id") or "")
                self.assertTrue(export_id)
                other_export_id = str(exports[1].get("export_id") or "") if len(exports) > 1 else ""
                self.assertTrue(other_export_id)
                self.assertNotEqual(export_id, other_export_id)

                compare_resp = client.get(
                    f"/api/projects/ui_proj_batch/batch-exports/compare?primary_export_id={export_id}&other_export_id={other_export_id}"
                )
                self.assertEqual(compare_resp.status_code, 200)
                compare_payload = compare_resp.get_json()
                self.assertEqual(compare_payload.get("status"), "pass")
                self.assertEqual(compare_payload.get("primary_export_id"), export_id)
                self.assertEqual(compare_payload.get("other_export_id"), other_export_id)
                self.assertIn("diff", compare_payload)
                compare_lines = compare_payload.get("compare_lines") if isinstance(compare_payload.get("compare_lines"), list) else []
                self.assertGreaterEqual(len(compare_lines), 1)

                compare_same_resp = client.get(
                    f"/api/projects/ui_proj_batch/batch-exports/compare?primary_export_id={export_id}&other_export_id={export_id}"
                )
                self.assertEqual(compare_same_resp.status_code, 400)

                download_json_resp = client.get(f"/api/projects/ui_proj_batch/batch-exports/{export_id}/download?format=json")
                self.assertEqual(download_json_resp.status_code, 200)
                self.assertIn("attachment;", str(download_json_resp.headers.get("Content-Disposition") or ""))
                self.assertIn("application/json", str(download_json_resp.content_type or ""))
                downloaded_payload = json.loads(download_json_resp.get_data(as_text=True))
                self.assertEqual(str(downloaded_payload.get("export_id") or ""), export_id)

                download_csv_resp = client.get(f"/api/projects/ui_proj_batch/batch-exports/{export_id}/download?format=csv")
                self.assertEqual(download_csv_resp.status_code, 200)
                self.assertIn("attachment;", str(download_csv_resp.headers.get("Content-Disposition") or ""))
                self.assertIn("text/csv", str(download_csv_resp.content_type or ""))
                csv_body = download_csv_resp.get_data(as_text=True)
                self.assertIn("section,index,export_id,project_id,action,status", csv_body)
                self.assertIn(export_id, csv_body)

                invalid_format_resp = client.get(f"/api/projects/ui_proj_batch/batch-exports/{export_id}/download?format=txt")
                self.assertEqual(invalid_format_resp.status_code, 400)

                detail_resp = client.get(f"/api/projects/ui_proj_batch/batch-exports/{export_id}")
                self.assertEqual(detail_resp.status_code, 200)
                detail_payload = detail_resp.get_json()
                self.assertEqual(detail_payload.get("status"), "pass")
                detail_entry = detail_payload.get("batch_export") if isinstance(detail_payload.get("batch_export"), dict) else {}
                self.assertEqual(str(detail_entry.get("export_id") or ""), export_id)

                failed_queue_resp = client.get(f"/api/projects/ui_proj_batch/batch-exports/{export_id}/failed-queue")
                self.assertEqual(failed_queue_resp.status_code, 200)
                failed_queue_payload = failed_queue_resp.get_json()
                self.assertEqual(failed_queue_payload.get("status"), "pass")
                queue = failed_queue_payload.get("queue") if isinstance(failed_queue_payload.get("queue"), dict) else {}
                self.assertEqual(int(queue.get("count") or 0), 1)
                queue_rows = queue.get("rows") if isinstance(queue.get("rows"), list) else []
                self.assertEqual(len(queue_rows), 1)
                self.assertEqual(str(queue_rows[0].get("task_id") or ""), "ui_proj_batch_missing")
                reason_rows = queue.get("failure_reasons") if isinstance(queue.get("failure_reasons"), list) else []
                self.assertGreaterEqual(len(reason_rows), 1)

                replay_by_id_resp = client.post(
                    f"/api/projects/ui_proj_batch/batch-exports/{export_id}/replay",
                    json={"options": {"dry_run": True, "failed_only": True, "retry_max": 2, "retry_backoff_ms": 10, "max_concurrency": 3}},
                )
                self.assertEqual(replay_by_id_resp.status_code, 200)
                replay_by_id_payload = replay_by_id_resp.get_json()
                self.assertEqual(replay_by_id_payload.get("status"), "pass")
                self.assertEqual(replay_by_id_payload.get("replay_status"), "pass")
                replay_by_id_entry = replay_by_id_payload.get("batch_export") if isinstance(replay_by_id_payload.get("batch_export"), dict) else {}
                self.assertEqual(replay_by_id_payload.get("source_export_id"), export_id)
                replay_by_id_path = Path(str(replay_by_id_entry.get("path") or ""))
                self.assertTrue(replay_by_id_path.exists())
                replay_options = replay_by_id_entry.get("replay_options") if isinstance(replay_by_id_entry.get("replay_options"), dict) else {}
                self.assertEqual(bool(replay_options.get("dry_run")), True)
                self.assertEqual(bool(replay_options.get("failed_only")), True)
                replay_metrics = replay_by_id_entry.get("replay_metrics") if isinstance(replay_by_id_entry.get("replay_metrics"), dict) else {}
                self.assertGreaterEqual(int(replay_metrics.get("dry_run_count") or 0), 1)
                self.assertEqual(int(replay_metrics.get("max_concurrency_requested") or 0), 3)
                self.assertEqual(int(replay_metrics.get("failed_only") or 0), 1)
                self.assertGreaterEqual(int(replay_metrics.get("failed_source_count") or 0), 1)
                self.assertEqual(int(replay_metrics.get("base_rows_count") or 0), 2)
                self.assertEqual(int(replay_metrics.get("effective_rows_count") or 0), 1)

                replay_resp = client.post(
                    "/api/projects/ui_proj_batch/batch-exports/replay-latest",
                    json={"options": {"dry_run": False, "retry_max": 1, "retry_backoff_ms": 0, "max_concurrency": 2}},
                )
                self.assertEqual(replay_resp.status_code, 200)
                replayed = replay_resp.get_json()
                self.assertEqual(replayed.get("status"), "pass")
                replay_entry = replayed.get("batch_export") if isinstance(replayed.get("batch_export"), dict) else {}
                replay_path = Path(str(replay_entry.get("path") or ""))
                self.assertTrue(replay_path.exists())
                self.assertEqual(replayed.get("action"), "batch_summary")
                self.assertIn(str(replayed.get("replay_status") or ""), {"pass", "partial"})
                replayed_metrics = replay_entry.get("replay_metrics") if isinstance(replay_entry.get("replay_metrics"), dict) else {}
                self.assertGreaterEqual(int(replayed_metrics.get("elapsed_ms") or 0), 0)

                delete_resp = client.delete(f"/api/projects/ui_proj_batch/batch-exports/{export_id}")
                self.assertEqual(delete_resp.status_code, 200)
                delete_payload = delete_resp.get_json()
                self.assertEqual(delete_payload.get("status"), "pass")

                detail_after_delete = client.get(f"/api/projects/ui_proj_batch/batch-exports/{export_id}")
                self.assertEqual(detail_after_delete.status_code, 404)

    def test_ui_chat_send_need_user_input_path(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_chat_need_input", "title": "need input"})
                intake_result = {
                    "task_id": "ui_chat_need_input_20260514_000001",
                    "status": "need_user_input",
                    "task_draft_path": str(root / "runs" / "agent" / "ui_chat_need_input_20260514_000001" / "task.draft.json"),
                    "missing_fields": ["candidate_data"],
                    "questions": ["候选数据来源是什么？本地CSV路径还是数据库关键词？"],
                }
                fake_cp = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-intake"],
                    returncode=2,
                    stdout=json.dumps(intake_result, ensure_ascii=False),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", return_value=fake_cp):
                    resp = client.post(
                        "/api/chat/send",
                        json={
                            "project_id": "ui_chat_need_input",
                            "message": "设计470nm附近且高PLQY分子",
                            "options": {"planner_provider": "rule_based_v1", "catalog_path": "configs/models/catalog.json"},
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "need_user_input")
                pending = payload.get("pending_input") if isinstance(payload.get("pending_input"), dict) else {}
                self.assertEqual(pending.get("stage"), "intake")
                missing_fields = pending.get("missing_fields") if isinstance(pending.get("missing_fields"), list) else []
                self.assertIn("candidate_data", missing_fields)
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                assistant_text = "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
                self.assertIn("candidate_data", assistant_text)

    def test_ui_chat_pending_submit_resume_success(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_chat_pending_resume", "title": "pending"})
                project = ui_app_mod._load_project_state("ui_chat_pending_resume")
                self.assertIsInstance(project, dict)
                project["current_task_id"] = "ui_chat_pending_task"
                project["pending_input"] = {
                    "stage": "intake",
                    "missing_fields": ["candidate_data"],
                    "questions": ["候选数据来源是什么？"],
                    "task_draft_path": str(root / "runs" / "agent" / "ui_chat_pending_task" / "task.draft.json"),
                }
                ui_app_mod._save_project_state(project)

                fake_resume = {
                    "task_id": "ui_chat_pending_task",
                    "status": "success",
                    "run_label": "ui_chat_pending_task-20260515-120000",
                    "result_dir": str(root / "result" / "ui_chat_pending_task-20260515-120000"),
                }
                fake_cp = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-resume"],
                    returncode=0,
                    stdout=json.dumps(fake_resume, ensure_ascii=False),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
                    resp = client.post(
                        "/api/chat/pending-submit",
                        json={
                            "project_id": "ui_chat_pending_resume",
                            "patch": {"candidate_data": "/tmp/candidates.csv"},
                            "options": {"planner_provider": "rule_based_v1", "catalog_path": "configs/models/catalog.json"},
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "pass")
                events = payload.get("events") if isinstance(payload.get("events"), list) else []
                self.assertTrue(any(isinstance(e, dict) and e.get("stage") == "resume" and e.get("status") == "success" for e in events))
                project_out = payload.get("project") if isinstance(payload.get("project"), dict) else {}
                self.assertEqual(project_out.get("current_task_id"), "ui_chat_pending_task")
                self.assertEqual(project_out.get("pending_input"), {})
                cmd = mocked.call_args.args[0]
                self.assertIn("agent-resume", cmd)
                self.assertIn("--candidate-data", cmd)
                self.assertIn("/tmp/candidates.csv", cmd)

    def test_ui_chat_pending_submit_resume_need_user_input(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_chat_pending_need_input", "title": "pending"})
                project = ui_app_mod._load_project_state("ui_chat_pending_need_input")
                self.assertIsInstance(project, dict)
                project["current_task_id"] = "ui_chat_pending_need_task"
                project["pending_input"] = {
                    "stage": "approve",
                    "missing_fields": ["candidate_data"],
                    "questions": ["候选数据来源是什么？"],
                    "task_draft_path": str(root / "runs" / "agent" / "ui_chat_pending_need_task" / "task.draft.json"),
                }
                ui_app_mod._save_project_state(project)

                fake_resume = {
                    "task_id": "ui_chat_pending_need_task",
                    "status": "need_user_input",
                    "missing_fields": ["candidate_data"],
                    "questions": ["候选数据来源是什么？本地CSV路径还是数据库关键词？"],
                    "task_draft_path": str(root / "runs" / "agent" / "ui_chat_pending_need_task" / "task.draft.json"),
                }
                fake_cp = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-resume"],
                    returncode=2,
                    stdout=json.dumps(fake_resume, ensure_ascii=False),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", return_value=fake_cp):
                    resp = client.post(
                        "/api/chat/pending-submit",
                        json={
                            "project_id": "ui_chat_pending_need_input",
                            "patch": {"property": "plqy"},
                            "options": {"planner_provider": "rule_based_v1", "catalog_path": "configs/models/catalog.json"},
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "need_user_input")
                pending = payload.get("pending_input") if isinstance(payload.get("pending_input"), dict) else {}
                self.assertEqual(pending.get("stage"), "resume")
                missing = pending.get("missing_fields") if isinstance(pending.get("missing_fields"), list) else []
                self.assertIn("candidate_data", missing)

    def test_ui_chat_send_happy_path_runs_pipeline(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_chat_run_20260514_000002"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            draft_path = run_dir / "task.draft.json"
            request_from_task = run_dir / "request_from_task.json"
            task_json_path = run_dir / "task.json"
            draft_payload = {
                "version": "2.0",
                "task_id": task_id,
                "request_text": "设计470nm附近且高PLQY分子",
                "execution_mode": "full_pipeline",
                "operation": "full_pipeline",
                "property": "plqy",
                "range": "458.0-482.0nm",
                "n_structures": 120,
                "constraints": {"mw_min": 150.0, "mw_max": 700.0, "domain_threshold": 0.2, "banned_alerts": []},
                "train_data": None,
                "candidate_data": "/tmp/candidates.csv",
                "prediction_model": "unimol_lambda_plqy_v1",
                "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                "generation_input": {},
                "provenance": {"intake_source": "agent-intake", "web_evidence": [], "web_evidence_json": ""},
                "status": "draft",
                "missing_fields": [],
                "questions": [],
                "compatibility_warnings": [],
            }
            draft_path.write_text(json.dumps(draft_payload, ensure_ascii=False) + "\n", encoding="utf-8")
            request_from_task.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "request_text": "设计470nm附近且高PLQY分子",
                        "mode": "fast_screen",
                        "targets": [{"property": "plqy", "objective": "maximize"}],
                        "constraints": {"candidate_data": "/tmp/candidates.csv"},
                        "model_choice": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                        "budget": {"max_candidates": 120},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            task_json_path.write_text(json.dumps(draft_payload, ensure_ascii=False) + "\n", encoding="utf-8")
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_chat_run", "title": "run"})
                cp_intake = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-intake"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "task_id": task_id,
                            "status": "draft",
                            "task_draft_path": str(draft_path),
                            "web_evidence_path": str(run_dir / "web_evidence.json"),
                            "missing_fields": [],
                            "questions": [],
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )
                cp_approve = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-approve"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "task_id": task_id,
                            "status": "approved",
                            "task_path": str(task_json_path),
                            "request_path": str(request_from_task),
                            "plan_path": str(run_dir / "plan.json"),
                            "plan_md_path": str(run_dir / "plan.md"),
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )
                cp_run = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-run-json"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "task_id": task_id,
                            "status": "success",
                            "run_label": f"{task_id}-20260514-010101",
                            "result_dir": str(root / "result" / f"{task_id}-20260514-010101"),
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", side_effect=[cp_intake, cp_approve, cp_run]) as mocked:
                    resp = client.post(
                        "/api/chat/send",
                        json={
                            "project_id": "ui_chat_run",
                            "message": "设计470nm附近且高PLQY分子",
                            "options": {
                                "planner_provider": "rule_based_v1",
                                "catalog_path": "configs/models/catalog.json",
                                "web_search_enabled": True,
                                "web_topk": 4,
                            },
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "pass")
                run_result = payload.get("run_result") if isinstance(payload.get("run_result"), dict) else {}
                self.assertEqual(run_result.get("status"), "success")
                events = payload.get("events") if isinstance(payload.get("events"), list) else []
                self.assertTrue(any((isinstance(e, dict) and str(e.get("stage")) == "run") for e in events))
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                assistant_text = "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
                self.assertIn("任务执行完成", assistant_text)
                event_msgs = [
                    m for m in messages
                    if isinstance(m, dict)
                    and str(m.get("kind") or "") == "event_trace"
                    and isinstance(m.get("meta"), dict)
                ]
                self.assertTrue(len(event_msgs) >= 1)
                meta_events = event_msgs[-1].get("meta", {}).get("events") if isinstance(event_msgs[-1].get("meta"), dict) else []
                self.assertTrue(any(isinstance(e, dict) and str(e.get("stage") or "") == "run" for e in (meta_events if isinstance(meta_events, list) else [])))
                self.assertEqual(mocked.call_count, 3)

    def test_ui_chat_send_step_command_runs_step_json(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_chat_step", "title": "step"})
                task_payload = {
                    "version": "2.0",
                    "task_id": "ui_chat_step_task",
                    "request_text": "单步执行",
                    "execution_mode": "single_step",
                    "operation": "clean_dataset",
                    "property": "plqy",
                    "range": "60-100",
                    "n_structures": 100,
                    "constraints": {"mw_min": 150.0, "mw_max": 700.0, "domain_threshold": 0.2, "banned_alerts": []},
                    "train_data": None,
                    "candidate_data": "/tmp/candidates.csv",
                    "prediction_model": "unimol_lambda_plqy_v1",
                    "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                    "generation_input": {},
                    "provenance": {},
                    "status": "draft",
                    "missing_fields": [],
                    "questions": [],
                    "compatibility_warnings": [],
                }
                msg_payload = {
                    "operation": "clean_dataset",
                    "args": {"input_csv": "/tmp/candidates.csv", "dedupe_by_smiles": True},
                    "task": task_payload,
                }
                cp_step = subprocess.CompletedProcess(
                    args=["python3", "-m", "oled_agent.cli", "agent-run-step-json"],
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "task_id": "ui_chat_step_task",
                            "status": "success",
                            "operation": "clean_dataset",
                            "run_label": "ui_chat_step_task-20260514-111111",
                            "execution_path": str(root / "runs" / "agent" / "ui_chat_step_task" / "execution.json"),
                            "task_path": str(root / "runs" / "agent" / "ui_chat_step_task" / "task.json"),
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )
                with mock.patch("ui.app.subprocess.run", return_value=cp_step) as mocked:
                    resp = client.post(
                        "/api/chat/send",
                        json={
                            "project_id": "ui_chat_step",
                            "message": json.dumps(msg_payload, ensure_ascii=False),
                            "options": {"catalog_path": "configs/models/catalog.json"},
                        },
                    )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "pass")
                events = payload.get("events") if isinstance(payload.get("events"), list) else []
                self.assertTrue(any((isinstance(e, dict) and e.get("stage") == "step") for e in events))
                step_result = payload.get("step_result") if isinstance(payload.get("step_result"), dict) else {}
                self.assertEqual(step_result.get("operation"), "clean_dataset")
                self.assertEqual(step_result.get("status"), "success")
                cmd = mocked.call_args.args[0]
                self.assertIn("agent-run-step-json", cmd)
                self.assertIn("--step-request-json", cmd)

    def test_ui_chat_send_step_command_without_task_returns_need_input(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                client.post("/api/projects", json={"project_id": "ui_chat_step_need_input", "title": "step need input"})
                resp = client.post(
                    "/api/chat/send",
                    json={
                        "project_id": "ui_chat_step_need_input",
                        "message": "/step clean_dataset {\"input_csv\":\"/tmp/a.csv\"}",
                    },
                )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload.get("status"), "need_user_input")
                pending = payload.get("pending_input") if isinstance(payload.get("pending_input"), dict) else {}
                self.assertEqual(pending.get("stage"), "step")
                missing_fields = pending.get("missing_fields") if isinstance(pending.get("missing_fields"), list) else []
                self.assertIn("task_context", missing_fields)

    def test_ui_tasks_endpoint_lists_recent_runs(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "runs" / "agent"
            run_a = base / "ui_task_a"
            run_b = base / "ui_task_b"
            run_skip = base / "bad..id"
            for run in [run_a, run_b, run_skip]:
                run.mkdir(parents=True, exist_ok=True)
            (run_a / "execution.json").write_text(
                json.dumps({"status": "success", "records": [{"status": "success"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (run_a / "task_state.json").write_text(json.dumps({"status": "SUCCESS"}, ensure_ascii=False) + "\n", encoding="utf-8")
            (run_b / "execution.json").write_text(
                json.dumps({"status": "failed", "records": [{"status": "failed"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (run_b / "task_state.json").write_text(json.dumps({"status": "FAILED"}, ensure_ascii=False) + "\n", encoding="utf-8")
            os.utime(run_a, (1700000000, 1700000000))
            os.utime(run_b, (1800000000, 1800000000))
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get("/api/tasks?limit=1")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(int(payload.get("count") or 0), 1)
            tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].get("task_id"), "ui_task_b")
            self.assertEqual(tasks[0].get("execution_status"), "failed")

    def test_ui_tasks_endpoint_rejects_invalid_prefix(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/tasks?prefix=bad/../x")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid prefix")

    def test_ui_experiments_endpoint_filters_records(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs = root / "runs" / "agent"
            a = runs / "exp_a" / "artifacts"
            b = runs / "exp_b" / "artifacts"
            a.mkdir(parents=True, exist_ok=True)
            b.mkdir(parents=True, exist_ok=True)
            (a / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_a",
                        "run_label": "exp_a-20260514-010101",
                        "generated_at": "2026-05-14T01:01:01+00:00",
                        "execution_mode": "full_pipeline",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "execution_summary": {"status": "success", "record_count": 4, "failed_count": 0, "adapters": ["a1"]},
                        "source_artifacts": {"candidate_csv": {"exists": True}, "scored_csv": {"exists": True}},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (b / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "task_id": "exp_b",
                        "run_label": "exp_b-20260514-010102",
                        "generated_at": "2026-05-14T01:01:02+00:00",
                        "execution_mode": "single_step",
                        "model_choice": {"predictor_id": "p2", "generator_id": "g2"},
                        "execution_summary": {"status": "failed", "record_count": 1, "failed_count": 1, "adapters": ["a2"]},
                        "source_artifacts": {"candidate_csv": {"exists": False}, "scored_csv": {"exists": False}},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get("/api/experiments?predictor_id=p1&execution_mode=full_pipeline&status=success")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            exps = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
            self.assertEqual(len(exps), 1)
            self.assertEqual(exps[0].get("task_id"), "exp_a")
            self.assertEqual(exps[0].get("predictor_id"), "p1")

    def test_ui_experiments_endpoint_rejects_invalid_filters(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/experiments?status=weird")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid status")

        resp2 = client.get("/api/experiments?execution_mode=abc")
        self.assertEqual(resp2.status_code, 400)
        payload2 = resp2.get_json()
        self.assertEqual(payload2.get("status"), "fail")
        self.assertEqual(payload2.get("error"), "invalid execution_mode")

    def test_ui_timeline_groups_endpoint_recent_tasks(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs = root / "runs" / "agent"
            a = runs / "tg_a"
            b = runs / "tg_b"
            a.mkdir(parents=True, exist_ok=True)
            b.mkdir(parents=True, exist_ok=True)
            (a / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": "tg_a",
                        "status": "success",
                        "records": [
                            {"name": "generate_candidates", "status": "success", "args": {}, "result": {}, "error": ""},
                            {"name": "score_candidates", "status": "failed", "args": {"input_csv": "/tmp/a.csv"}, "result": {}, "error": "boom"},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (b / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": "tg_b",
                        "status": "success",
                        "records": [
                            {"name": "search_dataset", "status": "success", "args": {}, "result": {}, "error": ""},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.utime(a, (1700000000, 1700000000))
            os.utime(b, (1800000000, 1800000000))
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get("/api/timeline-groups?scope=recent_tasks&limit=2")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("scope"), "recent_tasks")
            self.assertEqual(int(payload.get("task_count") or 0), 2)
            failed_items = payload.get("failed_items") if isinstance(payload.get("failed_items"), list) else []
            self.assertTrue(any(isinstance(it, dict) and str(it.get("name") or "").endswith(":score_candidates") for it in failed_items))

    def test_ui_timeline_groups_endpoint_rejects_invalid_scope(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/timeline-groups?scope=bad_scope")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid scope")

    def test_ui_html_contains_retry_tool_name_and_step_template_controls(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("retry_failed_tool_name", html)
        self.assertIn("loadSuggestedRetryArgs()", html)
        self.assertIn("Load Suggested Retry Args", html)
        self.assertIn("applyStepArgsTemplate(", html)
        self.assertIn("Load Args Template", html)

    def test_ui_run_step_endpoint_shells_out_to_agent_run_step_json(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        fake_step_result = {"status": "success", "task_id": "ui_step", "operation": "search_dataset"}
        fake_cp = subprocess.CompletedProcess(
            args=["python3", "-m", "oled_agent.cli", "agent-run-step-json"],
            returncode=0,
            stdout=json.dumps(fake_step_result, ensure_ascii=False),
            stderr="",
        )
        payload_text = json.dumps(
            {
                "task": {
                    "task_id": "ui_step",
                    "request_text": "single step",
                    "domain": "oled_molecule_design",
                    "targets": [{"name": "plqy", "objective": "maximize", "target_center": 0.6, "sigma": 0.2, "weight": 1.0}],
                    "constraints": {"mw_min": 150, "mw_max": 700, "domain_threshold": 0.2, "banned_alerts": []},
                    "model_choice": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                    "budget": {"max_candidates": 10},
                    "dataset_preferences": ["master_database"],
                },
                "operation": "search_dataset",
                "args": {"preferences": ["master_database"]},
            },
            ensure_ascii=False,
        )
        with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
            resp = client.post(
                "/api/run-step",
                json={"payload_text": payload_text, "catalog_path": "configs/models/catalog.json"},
            )
        self.assertEqual(resp.status_code, 200)
        out = resp.get_json()
        self.assertEqual(out.get("status"), "pass")
        result = out.get("result") if isinstance(out.get("result"), dict) else {}
        self.assertEqual(result.get("status"), "success")
        cmd = mocked.call_args.args[0]
        self.assertIn("agent-run-step-json", cmd)
        self.assertIn("--step-request-json", cmd)

    def test_ui_approve_endpoint_shells_out_to_agent_approve(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        fake_result = {"status": "approved", "task_id": "ui_intake_demo"}
        fake_cp = subprocess.CompletedProcess(
            args=["python3", "-m", "oled_agent.cli", "agent-approve"],
            returncode=0,
            stdout=json.dumps(fake_result, ensure_ascii=False),
            stderr="",
        )
        with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
            resp = client.post(
                "/api/approve",
                json={
                    "task_json_path": "runs/agent/ui_intake_demo/task.draft.json",
                    "planner_provider": "rule_based_v1",
                    "catalog_path": "configs/models/catalog.json",
                },
            )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "pass")
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        self.assertEqual(result.get("status"), "approved")
        cmd = mocked.call_args.args[0]
        self.assertIn("agent-approve", cmd)
        self.assertIn("--task-json", cmd)

    def test_ui_resume_endpoint_shells_out_to_agent_resume(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        fake_result = {"status": "success", "task_id": "ui_resume_demo"}
        fake_cp = subprocess.CompletedProcess(
            args=["python3", "-m", "oled_agent.cli", "agent-resume"],
            returncode=0,
            stdout=json.dumps(fake_result, ensure_ascii=False),
            stderr="",
        )
        with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
            resp = client.post(
                "/api/resume",
                json={
                    "task_id": "ui_resume_demo",
                    "planner_provider": "rule_based_v1",
                    "catalog_path": "configs/models/catalog.json",
                    "candidate_data": "/tmp/candidates.csv",
                    "predictor_id": "unimol_lambda_plqy_v1",
                },
            )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "pass")
        cmd = mocked.call_args.args[0]
        self.assertIn("agent-resume", cmd)
        self.assertIn("ui_resume_demo", cmd)
        self.assertIn("--candidate-data", cmd)
        self.assertIn("/tmp/candidates.csv", cmd)
        self.assertIn("--predictor-id", cmd)

    def test_ui_retry_failed_step_endpoint_runs_agent_step_retry(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_retry_failed_step_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "records": [
                            {"name": "search_dataset", "status": "success", "args": {"preferences": ["db_a"]}},
                            {"name": "search_dataset", "status": "failed", "args": {"preferences": ["db_a"]}},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "task.json").write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": task_id,
                        "request_text": "retry failed step",
                        "execution_mode": "full_pipeline",
                        "operation": "full_pipeline",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 50,
                        "constraints": {"mw_min": 150, "mw_max": 700, "domain_threshold": 0.2, "banned_alerts": []},
                        "train_data": None,
                        "candidate_data": "/tmp/candidates.csv",
                        "prediction_model": "unimol_lambda_plqy_v1",
                        "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                        "generation_input": {},
                        "provenance": {},
                        "status": "approved",
                        "missing_fields": [],
                        "questions": [],
                        "compatibility_warnings": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            fake_step_result = {
                "status": "success",
                "task_id": task_id,
                "operation": "retrieve_candidate_data",
                "run_label": f"{task_id}-20260514-010101",
            }
            fake_cp = subprocess.CompletedProcess(
                args=["python3", "-m", "oled_agent.cli", "agent-run-step-json"],
                returncode=0,
                stdout=json.dumps(fake_step_result, ensure_ascii=False),
                stderr="",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                with mock.patch("ui.app.subprocess.run", return_value=fake_cp) as mocked:
                    resp = client.post(
                        f"/api/task/{task_id}/retry-failed-step",
                        json={"catalog_path": "configs/models/catalog.json"},
                    )
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("retry_operation"), "retrieve_candidate_data")
            self.assertEqual(payload.get("failed_tool_name"), "search_dataset")
            cmd = mocked.call_args.args[0]
            self.assertIn("agent-run-step-json", cmd)
            self.assertIn("--step-request-json", cmd)

    def test_ui_retry_failed_step_endpoint_handles_no_failed_step(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_retry_no_failed_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "success",
                        "records": [
                            {"name": "search_dataset", "status": "success", "args": {}},
                            {"name": "score_candidates", "status": "success", "args": {}},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.post(f"/api/task/{task_id}/retry-failed-step", json={})
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "fail")
            self.assertEqual(payload.get("error"), "no_failed_step")

    def test_ui_retry_failed_step_endpoint_dry_run_preview_does_not_execute(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_retry_dry_run_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "records": [
                            {"name": "clean_dataset", "status": "failed", "args": {"input_csv": "/tmp/a.csv"}},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "task.json").write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": task_id,
                        "request_text": "retry dry run",
                        "execution_mode": "full_pipeline",
                        "operation": "full_pipeline",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 20,
                        "constraints": {"mw_min": 150, "mw_max": 700, "domain_threshold": 0.2, "banned_alerts": []},
                        "train_data": None,
                        "candidate_data": "/tmp/candidates.csv",
                        "prediction_model": "unimol_lambda_plqy_v1",
                        "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                        "generation_input": {},
                        "provenance": {},
                        "status": "approved",
                        "missing_fields": [],
                        "questions": [],
                        "compatibility_warnings": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                with mock.patch("ui.app._run_agent_step_json", side_effect=AssertionError("should not execute on dry_run")):
                    resp = client.post(
                        f"/api/task/{task_id}/retry-failed-step",
                        json={"dry_run": True},
                    )
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("mode"), "dry_run")
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(payload.get("retry_operation"), "clean_dataset")

    def test_ui_retry_failed_step_endpoint_uses_override_args(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_retry_override_args_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "records": [
                            {"name": "score_candidates", "status": "failed", "args": {"input_csv": "/tmp/old.csv"}},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "task.json").write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": task_id,
                        "request_text": "retry override args",
                        "execution_mode": "full_pipeline",
                        "operation": "full_pipeline",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 20,
                        "constraints": {"mw_min": 150, "mw_max": 700, "domain_threshold": 0.2, "banned_alerts": []},
                        "train_data": None,
                        "candidate_data": "/tmp/candidates.csv",
                        "prediction_model": "unimol_lambda_plqy_v1",
                        "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                        "generation_input": {},
                        "provenance": {},
                        "status": "approved",
                        "missing_fields": [],
                        "questions": [],
                        "compatibility_warnings": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            override_args = {"input_csv": "/tmp/new.csv", "force": True}
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                with mock.patch(
                    "ui.app._run_agent_step_json",
                    return_value={"status": "pass", "result": {"status": "success", "task_id": task_id}},
                ) as mocked_step:
                    resp = client.post(
                        f"/api/task/{task_id}/retry-failed-step",
                        json={"args": override_args, "catalog_path": "configs/models/catalog.json"},
                    )
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("retry_args"), override_args)
            called_payload = mocked_step.call_args.kwargs.get("payload") if mocked_step.call_args and mocked_step.call_args.kwargs else {}
            self.assertEqual(called_payload.get("args"), override_args)

    def test_ui_retry_failed_step_endpoint_respects_failed_tool_name_filter(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_retry_target_failed_name"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "records": [
                            {"name": "train_predictor", "status": "failed", "args": {"predictor_id": "p1"}},
                            {"name": "score_candidates", "status": "failed", "args": {"input_csv": "/tmp/scored.csv"}},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "task.json").write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": task_id,
                        "request_text": "retry targeted failed name",
                        "execution_mode": "full_pipeline",
                        "operation": "full_pipeline",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 20,
                        "constraints": {"mw_min": 150, "mw_max": 700, "domain_threshold": 0.2, "banned_alerts": []},
                        "train_data": None,
                        "candidate_data": "/tmp/candidates.csv",
                        "prediction_model": "unimol_lambda_plqy_v1",
                        "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                        "generation_input": {},
                        "provenance": {},
                        "status": "approved",
                        "missing_fields": [],
                        "questions": [],
                        "compatibility_warnings": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                with mock.patch(
                    "ui.app._run_agent_step_json",
                    return_value={"status": "pass", "result": {"status": "success", "task_id": task_id}},
                ) as mocked_step:
                    resp = client.post(
                        f"/api/task/{task_id}/retry-failed-step",
                        json={"failed_tool_name": "train_predictor", "catalog_path": "configs/models/catalog.json"},
                    )
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("failed_tool_name"), "train_predictor")
            self.assertEqual(payload.get("retry_operation"), "train_predictor")
            called_payload = mocked_step.call_args.kwargs.get("payload") if mocked_step.call_args and mocked_step.call_args.kwargs else {}
            self.assertEqual(called_payload.get("operation"), "train_predictor")
            self.assertEqual(called_payload.get("args"), {"predictor_id": "p1"})

    def test_ui_retry_failed_step_endpoint_rejects_non_object_args(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_retry_bad_args_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "records": [
                            {"name": "clean_dataset", "status": "failed", "args": {"input_csv": "/tmp/a.csv"}},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "task.json").write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "task_id": task_id,
                        "request_text": "retry bad args",
                        "execution_mode": "full_pipeline",
                        "operation": "full_pipeline",
                        "property": "plqy",
                        "range": "60-100",
                        "n_structures": 20,
                        "constraints": {"mw_min": 150, "mw_max": 700, "domain_threshold": 0.2, "banned_alerts": []},
                        "train_data": None,
                        "candidate_data": "/tmp/candidates.csv",
                        "prediction_model": "unimol_lambda_plqy_v1",
                        "model_preferences": {"predictor_id": "unimol_lambda_plqy_v1", "generator_id": "reinvent4_lambda_em_v2"},
                        "generation_input": {},
                        "provenance": {},
                        "status": "approved",
                        "missing_fields": [],
                        "questions": [],
                        "compatibility_warnings": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.post(
                    f"/api/task/{task_id}/retry-failed-step",
                    json={"args": [1, 2, 3]},
                )
            self.assertEqual(resp.status_code, 400)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "fail")
            self.assertEqual(payload.get("error"), "args_must_be_object")

    def test_ui_approve_rejects_missing_task_json_path(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.post("/api/approve", json={})
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "missing task_json_path")

    def test_ui_task_summary_reads_artifacts(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_summary_case"
            run_dir = root / "runs" / "agent" / task_id
            (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            (run_dir / "execution.json").write_text(
                json.dumps({"status": "success", "records": [{"name": "search_dataset"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (run_dir / "task_state.json").write_text(
                json.dumps({"status": "RUNNING", "current_stage": "MODEL_TRAINING"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (run_dir / "decision_summary.json").write_text(
                json.dumps(
                    {"task_id": task_id, "selected_models": {}, "inference_step": {"used_fallback": False}},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "artifacts" / "web_evidence.json").write_text(
                json.dumps({"results": [{"title": "n", "url": "https://nature.com/a"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (run_dir / "artifacts" / "experiment_trace.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "task_id": task_id,
                        "run_label": "ui_summary_case-20260514-000001",
                        "execution_mode": "full_pipeline",
                        "model_choice": {"predictor_id": "p1", "generator_id": "g1"},
                        "execution_summary": {"status": "success", "record_count": 1},
                        "source_artifacts": {"candidate_csv": {"exists": False}},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_id}/summary")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("task_id"), task_id)
            execution_summary = payload.get("execution_summary") if isinstance(payload.get("execution_summary"), dict) else {}
            self.assertEqual(execution_summary.get("record_count"), 1)
            preview = payload.get("web_evidence_preview") if isinstance(payload.get("web_evidence_preview"), list) else []
            self.assertEqual(len(preview), 1)
            trace_preview = payload.get("experiment_trace_preview") if isinstance(payload.get("experiment_trace_preview"), dict) else {}
            self.assertEqual(trace_preview.get("execution_mode"), "full_pipeline")
            self.assertEqual(trace_preview.get("run_label"), "ui_summary_case-20260514-000001")

    def test_ui_task_summary_rejects_invalid_task_id(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/task/bad..id/summary")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid task_id")

    def test_ui_task_bundle_download_includes_run_and_output_dirs(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_bundle_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            result_dir = root / "result" / f"{task_id}-20260515-010101"
            logging_dir = root / "logging" / f"{task_id}-20260515-010101"
            rank_dir = root / "runs" / f"agent_rank_{task_id}_20260515T010101.000000+0000"
            result_dir.mkdir(parents=True, exist_ok=True)
            logging_dir.mkdir(parents=True, exist_ok=True)
            rank_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "execution.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": "success",
                        "records": [
                            {
                                "name": "make_report",
                                "status": "success",
                                "result": {
                                    "latest_run_dir": str(rank_dir),
                                    "report": str(rank_dir / "06_report.md"),
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "decision_summary.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "artifacts": {
                            "final_output": str(rank_dir / "06_report.md"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "task_state.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
            (run_dir / "plan.json").write_text(json.dumps({"summary": "ok"}, ensure_ascii=False) + "\n", encoding="utf-8")
            (run_dir / "tool_state.json").write_text(json.dumps({"ok": True}, ensure_ascii=False) + "\n", encoding="utf-8")
            (result_dir / "metadata.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
            (logging_dir / "task.json").write_text(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n", encoding="utf-8")
            (rank_dir / "06_report.md").write_text("# report\n", encoding="utf-8")

            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_id}/bundle")

            self.assertEqual(resp.status_code, 200)
            self.assertIn("application/gzip", str(resp.content_type or ""))
            disp = str(resp.headers.get("Content-Disposition") or "")
            self.assertIn("attachment;", disp)
            self.assertIn(f"agent4mat-task-{task_id}", disp)

            buf = io.BytesIO(resp.get_data())
            with tarfile.open(fileobj=buf, mode="r:gz") as tf:
                names = tf.getnames()
                self.assertTrue(any(name.endswith("manifest.json") for name in names))
                self.assertTrue(any(name.endswith(f"runs/agent/{task_id}/execution.json") for name in names))
                self.assertTrue(any(name.endswith(f"result/{result_dir.name}/metadata.json") for name in names))
                self.assertTrue(any(name.endswith(f"logging/{logging_dir.name}/task.json") for name in names))
                self.assertTrue(any(name.endswith(f"runs/{rank_dir.name}/06_report.md") for name in names))
                manifest_name = next((name for name in names if name.endswith("manifest.json")), "")
                self.assertTrue(manifest_name)
                manifest_file = tf.extractfile(manifest_name)
                self.assertIsNotNone(manifest_file)
                manifest = json.loads((manifest_file.read() if manifest_file is not None else b"{}").decode("utf-8"))
                self.assertEqual(manifest.get("task_id"), task_id)
                self.assertGreaterEqual(int(manifest.get("file_count") or 0), 5)

    def test_ui_task_bundle_returns_missing_for_unknown_task(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get("/api/task/ui_bundle_missing/bundle")
        self.assertEqual(resp.status_code, 404)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "missing")
        self.assertEqual(payload.get("task_id"), "ui_bundle_missing")

    def test_ui_task_artifact_preview_reads_json(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_artifact_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "plan.json").write_text(
                json.dumps({"summary": "demo", "tool_calls": [{"name": "list_models", "args": {}}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_id}/artifact/plan")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("artifact"), "plan")
            preview = payload.get("json_preview") if isinstance(payload.get("json_preview"), dict) else {}
            self.assertEqual(preview.get("summary"), "demo")

    def test_ui_task_artifact_rejects_invalid_artifact_name(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/task/ui_task_demo/artifact/not_exists")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid artifact_name")

    def test_ui_task_timeline_reads_execution_records(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_timeline_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            execution_payload = {
                "task_id": task_id,
                "status": "success",
                "started_at": "2026-05-14T00:00:00+00:00",
                "ended_at": "2026-05-14T00:00:05+00:00",
                "records": [
                    {
                        "name": "search_dataset",
                        "status": "success",
                        "started_at": "2026-05-14T00:00:00+00:00",
                        "ended_at": "2026-05-14T00:00:01+00:00",
                        "result": {"status": "success"},
                        "error": "",
                    },
                    {
                        "name": "score_candidates",
                        "status": "success",
                        "started_at": "2026-05-14T00:00:01+00:00",
                        "ended_at": "2026-05-14T00:00:03+00:00",
                        "result": {"status": "success", "adapter": "unimol_score_adapter_v1"},
                        "error": "",
                    },
                ],
            }
            (run_dir / "execution.json").write_text(json.dumps(execution_payload, ensure_ascii=False) + "\n", encoding="utf-8")
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_id}/timeline")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            self.assertEqual(summary.get("total_steps"), 2)
            self.assertEqual(summary.get("failed_steps"), 0)
            events = payload.get("events") if isinstance(payload.get("events"), list) else []
            self.assertEqual(len(events), 2)
            self.assertEqual(events[1].get("adapter"), "unimol_score_adapter_v1")
            self.assertEqual(events[0].get("duration_ms"), 1000)
            lines = payload.get("timeline_lines") if isinstance(payload.get("timeline_lines"), list) else []
            self.assertEqual(len(lines), 2)
            self.assertIn("[PASS]", lines[0])

    def test_ui_task_timeline_filters_and_sorts(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_timeline_filter_case"
            run_dir = root / "runs" / "agent" / task_id
            run_dir.mkdir(parents=True, exist_ok=True)
            execution_payload = {
                "task_id": task_id,
                "status": "success",
                "started_at": "2026-05-14T00:00:00+00:00",
                "ended_at": "2026-05-14T00:00:10+00:00",
                "records": [
                    {
                        "name": "search_dataset",
                        "status": "success",
                        "started_at": "2026-05-14T00:00:00+00:00",
                        "ended_at": "2026-05-14T00:00:03+00:00",
                        "result": {"status": "success"},
                        "error": "",
                    },
                    {
                        "name": "score_candidates",
                        "status": "failed",
                        "started_at": "2026-05-14T00:00:03+00:00",
                        "ended_at": "2026-05-14T00:00:09+00:00",
                        "result": {"status": "failed", "adapter": "x"},
                        "error": "boom",
                    },
                ],
            }
            (run_dir / "execution.json").write_text(json.dumps(execution_payload, ensure_ascii=False) + "\n", encoding="utf-8")
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_id}/timeline?tool=score&status_filter=failed&sort=duration_desc")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            self.assertEqual(summary.get("total_steps_before_filter"), 2)
            self.assertEqual(summary.get("total_steps"), 1)
            self.assertEqual(summary.get("failed_steps"), 1)
            events = payload.get("events") if isinstance(payload.get("events"), list) else []
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].get("name"), "score_candidates")
            self.assertTrue(bool(events[0].get("is_failed")))
            self.assertEqual(events[0].get("highlight"), "fail")

    def test_ui_task_timeline_rejects_invalid_task_id(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/task/bad..id/timeline")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid task_id")

    def test_ui_task_timeline_rejects_invalid_filters(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/task/ui_task_demo/timeline?status_filter=weird")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid status_filter")
        resp2 = client.get("/api/task/ui_task_demo/timeline?sort=bad")
        self.assertEqual(resp2.status_code, 400)
        payload2 = resp2.get_json()
        self.assertEqual(payload2.get("status"), "fail")
        self.assertEqual(payload2.get("error"), "invalid sort")

    def test_ui_task_compare_returns_diff(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_a = "ui_compare_a"
            task_b = "ui_compare_b"
            run_a = root / "runs" / "agent" / task_a
            run_b = root / "runs" / "agent" / task_b
            (run_a / "artifacts").mkdir(parents=True, exist_ok=True)
            (run_b / "artifacts").mkdir(parents=True, exist_ok=True)
            (run_a / "execution.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "started_at": "2026-05-14T00:00:00+00:00",
                        "ended_at": "2026-05-14T00:00:06+00:00",
                        "records": [
                            {
                                "name": "generate_candidates",
                                "status": "success",
                                "result": {"adapter": "reinvent4_generate_adapter_v1"},
                            },
                            {
                                "name": "score_candidates",
                                "status": "failed",
                                "result": {"adapter": "unimol_score_adapter_v1"},
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_b / "execution.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "started_at": "2026-05-14T00:00:00+00:00",
                        "ended_at": "2026-05-14T00:00:03+00:00",
                        "records": [
                            {
                                "name": "generate_candidates",
                                "status": "success",
                                "result": {"adapter": "template_generate_cmd"},
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_a / "artifacts" / "web_evidence.json").write_text(
                json.dumps({"results": [{"url": "https://nature.com/a"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (run_b / "artifacts" / "web_evidence.json").write_text(
                json.dumps({"results": []}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_a}/compare?other_task_id={task_b}")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            diff = payload.get("diff") if isinstance(payload.get("diff"), dict) else {}
            self.assertEqual(diff.get("record_count_delta"), 1)
            self.assertEqual(diff.get("failed_step_count_delta"), 1)
            self.assertEqual(diff.get("web_evidence_count_delta"), 1)
            self.assertEqual(diff.get("total_duration_ms_delta"), 3000)
            self.assertEqual(diff.get("adapters_only_in_primary"), ["reinvent4_generate_adapter_v1", "unimol_score_adapter_v1"])
            self.assertEqual(diff.get("adapters_only_in_other"), ["template_generate_cmd"])
            lines = payload.get("compare_lines") if isinstance(payload.get("compare_lines"), list) else []
            self.assertTrue(any("record_count" in str(line) for line in lines))

    def test_ui_task_compare_rejects_invalid_inputs(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/task/ui_task_demo/compare")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "missing other_task_id")

        resp2 = client.get("/api/task/ui_task_demo/compare?other_task_id=bad..id")
        self.assertEqual(resp2.status_code, 400)
        payload2 = resp2.get_json()
        self.assertEqual(payload2.get("status"), "fail")
        self.assertEqual(payload2.get("error"), "invalid other_task_id")

        resp3 = client.get("/api/task/ui_task_demo/compare?other_task_id=ui_task_demo")
        self.assertEqual(resp3.status_code, 400)
        payload3 = resp3.get_json()
        self.assertEqual(payload3.get("status"), "fail")
        self.assertEqual(payload3.get("error"), "other_task_id must differ from task_id")

    def test_ui_task_artifact_diff_returns_changed_paths(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_a = "ui_diff_a"
            task_b = "ui_diff_b"
            run_a = root / "runs" / "agent" / task_a
            run_b = root / "runs" / "agent" / task_b
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)
            (run_a / "decision_summary.json").write_text(
                json.dumps(
                    {
                        "task_id": task_a,
                        "score_step": {"used_fallback": False, "adapter": "unimol_score_adapter_v1"},
                        "selected_models": {"predictor_id": "unimol_lambda_plqy_v1"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_b / "decision_summary.json").write_text(
                json.dumps(
                    {
                        "task_id": task_b,
                        "score_step": {"used_fallback": True, "adapter": "template_score_cmd"},
                        "selected_models": {"predictor_id": "template_predictor_v1"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_a}/artifact-diff?other_task_id={task_b}&artifact=decision_summary")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("artifact"), "decision_summary")
            diff = payload.get("diff") if isinstance(payload.get("diff"), dict) else {}
            self.assertGreater(int(diff.get("changed_count") or 0), 0)
            changed = diff.get("changed") if isinstance(diff.get("changed"), list) else []
            changed_paths = [str(item.get("path") or "") for item in changed if isinstance(item, dict)]
            self.assertIn("score_step.used_fallback", changed_paths)

    def test_ui_task_artifact_diff_rejects_invalid_artifact(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.get("/api/task/ui_demo/artifact-diff?other_task_id=ui_other&artifact=not_exists")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid artifact")

    def test_ui_task_validate_handles_missing_run_dir(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get("/api/task/ui_missing_case/validate")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "missing")
            self.assertEqual(payload.get("error"), "run_dir_missing")

    def test_ui_task_validate_fails_when_required_files_missing(self) -> None:
        ui_app_mod = self._load_ui_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_id = "ui_validate_case"
            (root / "runs" / "agent" / task_id).mkdir(parents=True, exist_ok=True)
            with mock.patch.object(ui_app_mod, "REPO_ROOT", root):
                client = ui_app_mod.app.test_client()
                resp = client.get(f"/api/task/{task_id}/validate")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload.get("status"), "fail")
            blocking = payload.get("blocking_checks") if isinstance(payload.get("blocking_checks"), list) else []
            self.assertIn("plan", blocking)
            self.assertIn("execution_records", blocking)

    def test_ui_resume_rejects_invalid_task_id(self) -> None:
        ui_app_mod = self._load_ui_module()
        client = ui_app_mod.app.test_client()
        resp = client.post("/api/resume", json={"task_id": "bad..id"})
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload.get("status"), "fail")
        self.assertEqual(payload.get("error"), "invalid task_id")
