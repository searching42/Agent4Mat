PYTHON ?= python3
PYTHONPATH_ENV := PYTHONPATH=src
WORKSPACE_ROOT ?= .
TASK_ID ?= make_quickstart
RESULT_JSON ?= runs/agent/$(TASK_ID)/acceptance_result.json

.PHONY: help quickstart adapter-validate real-adapter-validate adapter-self-check test-regressions test-adapters
.PHONY: doctor llm-smoke llm-connectivity release-check release-boundary script-map request-templates-validate step-request-templates-validate input-smoke
.PHONY: intake-contract-guard step-mode-guard web-evidence-guard real-no-fallback-gate
.PHONY: real-chain-acceptance real-chain-acceptance-real real-chain-baseline real-chain-baseline-archive real-chain-baseline-archive-tgz real-chain-evidence ui-smoke

help:
	@echo "Available targets:"
	@echo "  make llm-smoke           - verify LLM integration path with mock planner"
	@echo "  make llm-connectivity    - run LLM connectivity diagnostic (command/backend)"
	@echo "  make release-check       - run adapter-validate + quickstart + llm-smoke + doctor"
	@echo "  make release-boundary    - check repo hygiene for release boundary"
	@echo "  make script-map          - generate workspace script migration map"
	@echo "  make request-templates-validate - validate request templates against request schema"
	@echo "  make step-request-templates-validate - validate step request templates against step_request schema"
	@echo "  make input-smoke         - run MolScribe image/pdf input smoke acceptance"
	@echo "  make intake-contract-guard - validate task.v2 + step_request contracts"
	@echo "  make step-mode-guard     - smoke check agent-run-step and agent-run-step-json"
	@echo "  make web-evidence-guard  - smoke check intake web evidence artifact"
	@echo "  make real-no-fallback-gate - run require-real-adapters acceptance smoke"
	@echo "  make real-chain-acceptance - run minimal real-chain acceptance with local stubs"
	@echo "  make real-chain-acceptance-real - run non-stub real-chain acceptance (requires real env)"
	@echo "  make real-chain-baseline   - run strict real-chain acceptance repeatedly (default x3)"
	@echo "  make real-chain-baseline-archive - archive baseline artifacts into one bundle"
	@echo "  make real-chain-baseline-archive-tgz - archive baseline artifacts and create tar.gz package"
	@echo "  make real-chain-evidence   - collect release evidence from acceptance_result.json"
	@echo "  make ui-smoke            - run lightweight UI smoke check"
	@echo "  make quickstart          - run quickstart chain self-check"
	@echo "  make adapter-validate    - validate adapter templates contract"
	@echo "  make real-adapter-validate - validate real adapter shells (preflight/smoke)"
	@echo "  make adapter-self-check  - alias of quickstart"
	@echo "  make doctor              - run environment diagnostics"
	@echo "  make test-regressions    - run full regression tests"
	@echo "  make test-adapters       - run adapter-focused regression subset"

quickstart:
	@./scripts/adapters/check_quickstart_chain.sh "$(TASK_ID)"

adapter-self-check: quickstart

doctor:
	@$(PYTHONPATH_ENV) $(PYTHON) -m oled_agent.cli doctor --workspace-root "$(WORKSPACE_ROOT)"

adapter-validate:
	@$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool train_predictor --cmd "$(PYTHON) scripts/adapters/train_predictor_adapter_template.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "$(PYTHON) scripts/adapters/generate_candidates_adapter_template.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool score_candidates --cmd "$(PYTHON) scripts/adapters/score_candidates_adapter_template.py" --workspace-root "$(WORKSPACE_ROOT)" --json

real-adapter-validate:
	@OLED_AGENT_UNIMOL_TRAIN_MODE=smoke UNIMOL_REMOTE_HOST=stub_host UNIMOL_REMOTE_PY=python3 UNIMOL_REMOTE_TMP_BASE=/tmp \
		$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool train_predictor --cmd "$(PYTHON) scripts/adapters/train_predictor_unimol_adapter.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@OLED_AGENT_MINERU_ADAPTER_MODE=smoke \
		$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "$(PYTHON) scripts/adapters/generate_candidates_mineru_adapter.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@OLED_AGENT_REINVENT4_ADAPTER_MODE=smoke \
		$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "$(PYTHON) scripts/adapters/generate_candidates_reinvent4_adapter.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@OLED_AGENT_REINVENT4_ADAPTER_MODE=real OLED_AGENT_REINVENT4_SOURCE_CSV="$(CURDIR)/configs/pipelines/demo_input.csv" \
		OLED_AGENT_REINVENT4_PIPELINE_SCRIPT="$(CURDIR)/scripts/adapters/stub_reinvent4_pipeline.sh" \
		OLED_AGENT_REINVENT4_RANKREADY_CSV="$(CURDIR)/runs/contract/reinvent4_real_stub_rankready.csv" \
		$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "$(PYTHON) scripts/adapters/generate_candidates_reinvent4_adapter.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@OLED_AGENT_MOLSCRIBE_ADAPTER_MODE=smoke \
		$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool generate_candidates --cmd "$(PYTHON) scripts/adapters/generate_candidates_molscribe_adapter.py" --workspace-root "$(WORKSPACE_ROOT)" --json
	@OLED_AGENT_UNIMOL_SCORE_MODE=real OLED_AGENT_UNIMOL_SCORE_SCRIPT="$(CURDIR)/scripts/adapters/stub_unimol_score.py" \
		UNIMOL_REMOTE_HOST=stub_host UNIMOL_REMOTE_PY=stub_py UNIMOL_REMOTE_TMP_BASE=/tmp \
		$(PYTHON) scripts/adapters/validate_adapter_contract.py --tool score_candidates --cmd "$(PYTHON) scripts/adapters/score_candidates_unimol_adapter.py" --workspace-root "$(WORKSPACE_ROOT)" --json

test-regressions:
	@$(PYTHONPATH_ENV) $(PYTHON) -m unittest -v tests.test_regressions

test-adapters:
	@$(PYTHONPATH_ENV) $(PYTHON) -m unittest -v \
		tests.test_regressions.AdapterContractValidatorTests \
		tests.test_regressions.RegressionTests.test_agent_run_json_with_quickstart_catalog_smoke \
		tests.test_regressions.RegressionTests.test_agent_run_json_with_repo_adapter_templates_smoke

llm-smoke:
	@$(PYTHONPATH_ENV) MOCK_LLM_MODE=active \
		$(PYTHON) scripts/check_llm_planner_modes.py

llm-connectivity:
	@$(PYTHONPATH_ENV) $(PYTHON) -m oled_agent.cli llm-connectivity --workspace-root "$(WORKSPACE_ROOT)" --catalog "$(WORKSPACE_ROOT)/configs/models/catalog.json"

release-boundary:
	@$(PYTHON) scripts/check_release_boundary.py --workspace-root "$(WORKSPACE_ROOT)" --json

script-map:
	@$(PYTHON) scripts/build_script_migration_map.py --workspace-scripts-root "$(WORKSPACE_ROOT)/../scripts" --out "$(WORKSPACE_ROOT)/docs/script_migration_map.json"

request-templates-validate:
	@$(PYTHONPATH_ENV) $(PYTHON) scripts/validate_request_examples.py --workspace-root "$(WORKSPACE_ROOT)" --examples-dir "configs/request_templates"

step-request-templates-validate:
	@$(PYTHONPATH_ENV) $(PYTHON) scripts/validate_step_request_examples.py --workspace-root "$(WORKSPACE_ROOT)" --examples-dir "configs/request_templates"

input-smoke:
	@./scripts/run_molscribe_input_smoke.sh "input_smoke"

intake-contract-guard:
	@$(PYTHONPATH_ENV) $(PYTHON) scripts/check_intake_contracts.py

step-mode-guard:
	@$(PYTHONPATH_ENV) $(PYTHON) scripts/check_step_mode.py

web-evidence-guard:
	@$(PYTHONPATH_ENV) $(PYTHON) scripts/check_web_evidence.py

real-no-fallback-gate:
	@$(PYTHONPATH_ENV) $(PYTHON) scripts/check_real_no_fallback.py

real-chain-acceptance:
	@./scripts/run_real_chain_acceptance_minimal.sh "$(WORKSPACE_ROOT)" "$(TASK_ID)"

real-chain-acceptance-real:
	@./scripts/run_real_chain_acceptance_real.sh "$(TASK_ID)" "设计470nm附近且高PLQY分子" "scripts/adapters/real_adapters_catalog.json" "runs/agent/$(TASK_ID)/external_debug.json"

real-chain-baseline:
	@./scripts/run_real_chain_baseline.sh "$(TASK_ID)" "设计470nm附近且高PLQY分子" "scripts/adapters/real_adapters_catalog.json" "3"

real-chain-baseline-archive:
	@$(PYTHON) scripts/archive_real_chain_baseline.py --workspace-root "$(WORKSPACE_ROOT)" --base-task-id "$(TASK_ID)"

real-chain-baseline-archive-tgz:
	@$(PYTHON) scripts/archive_real_chain_baseline.py --workspace-root "$(WORKSPACE_ROOT)" --base-task-id "$(TASK_ID)" --tar-gz

real-chain-evidence:
	@$(PYTHON) scripts/collect_real_chain_evidence.py --workspace-root "$(WORKSPACE_ROOT)" --result-json "$(RESULT_JSON)"

ui-smoke:
	@PYTHONPYCACHEPREFIX="$${TMPDIR:-/tmp}/agent4mat_pycache" $(PYTHON) -m py_compile ui/app.py

release-check:
	@$(MAKE) adapter-validate WORKSPACE_ROOT="$(WORKSPACE_ROOT)"
	@$(MAKE) request-templates-validate WORKSPACE_ROOT="$(WORKSPACE_ROOT)"
	@$(MAKE) step-request-templates-validate WORKSPACE_ROOT="$(WORKSPACE_ROOT)"
	@$(MAKE) quickstart TASK_ID="$(TASK_ID)"
	@$(MAKE) llm-smoke
	@$(MAKE) doctor WORKSPACE_ROOT="$(WORKSPACE_ROOT)"
