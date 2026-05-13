from __future__ import annotations

import csv
import json
import os
import builtins
import socket
import subprocess
import sys
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

    def test_search_web_evidence_returns_time_range_applied_false(self) -> None:
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
            with mock.patch("oled_agent.agent.tools.run_duckduckgo_search", return_value=[]):
                out = tools_mod.search_web_evidence(ctx, query="q", topk=3, time_range="30d")
            self.assertIn("time_range_applied", out)
            self.assertFalse(out["time_range_applied"])
            payload = json.loads(Path(out["web_evidence_json"]).read_text(encoding="utf-8"))
            self.assertIn("time_range_applied", payload)
            self.assertFalse(payload["time_range_applied"])

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

    def test_makefile_release_check_includes_request_template_validation(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
        self.assertIn("request-templates-validate", makefile)
        self.assertIn("@$(MAKE) request-templates-validate WORKSPACE_ROOT=\"$(WORKSPACE_ROOT)\"", makefile)

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
                "logging_data_report_path",
                "logging_model_report_path",
                "logging_filtering_report_path",
                "result_metadata_path",
            ):
                self.assertTrue(Path(payload[key]).exists(), msg=f"missing artifact: {key}")

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
        section_start = content.index("make-entrypoint-guard:")
        section_end = content.index("  acceptance-cpu-mock:", section_start)
        section = content[section_start:section_end]
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

    def test_oled_agent_ci_has_new_intake_step_and_web_guard_jobs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = self._workflow_path(repo_root)
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("intake-contract-guard:", content)
        self.assertIn("step-mode-guard:", content)
        self.assertIn("web-evidence-guard:", content)
        self.assertIn("real-no-fallback-gate:", content)
        self.assertIn("make intake-contract-guard", content)
        self.assertIn("make step-mode-guard", content)
        self.assertIn("make web-evidence-guard", content)
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
        self.assertIn("real-chain-evidence:", content)
        self.assertIn("ui-smoke:", content)
        self.assertIn("scripts/check_release_boundary.py", content)
        self.assertIn("scripts/build_script_migration_map.py", content)
        self.assertIn("scripts/collect_real_chain_evidence.py", content)
        self.assertIn("scripts/run_real_chain_acceptance_minimal.sh", content)
        self.assertIn("scripts/run_real_chain_acceptance_real.sh", content)
        self.assertIn("ui/app.py", content)
        self.assertIn("input-smoke:", content)
        self.assertIn("scripts/run_molscribe_input_smoke.sh", content)
        self.assertIn("intake-contract-guard:", content)
        self.assertIn("step-mode-guard:", content)
        self.assertIn("web-evidence-guard:", content)
        self.assertIn("real-no-fallback-gate:", content)


class PlanProgressAssetsTests(unittest.TestCase):
    def test_plan_progress_scripts_exist_and_are_executable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected_scripts = [
            repo_root / "scripts" / "check_release_boundary.py",
            repo_root / "scripts" / "build_script_migration_map.py",
            repo_root / "scripts" / "collect_real_chain_evidence.py",
            repo_root / "scripts" / "run_molscribe_input_smoke.sh",
            repo_root / "scripts" / "run_real_chain_acceptance_minimal.sh",
            repo_root / "scripts" / "run_real_chain_acceptance_real.sh",
        ]
        for script in expected_scripts:
            self.assertTrue(script.exists(), msg=f"missing script: {script}")
            self.assertTrue(os.access(script, os.X_OK), msg=f"script is not executable: {script}")

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
