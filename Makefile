PYTHON ?= python3
PYTHONPATH_ENV := PYTHONPATH=src
WORKSPACE_ROOT ?= .
TASK_ID ?= make_quickstart

.PHONY: help quickstart adapter-validate real-adapter-validate adapter-self-check test-regressions test-adapters
.PHONY: doctor llm-smoke release-check

help:
	@echo "Available targets:"
	@echo "  make llm-smoke           - verify LLM integration path with mock planner"
	@echo "  make release-check       - run adapter-validate + quickstart + llm-smoke + doctor"
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
	@OLED_AGENT_UNIMOL_SCORE_MODE=smoke UNIMOL_REMOTE_HOST=stub_host UNIMOL_REMOTE_PY=stub_py UNIMOL_REMOTE_TMP_BASE=/tmp \
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

release-check:
	@$(MAKE) adapter-validate WORKSPACE_ROOT="$(WORKSPACE_ROOT)"
	@$(MAKE) quickstart TASK_ID="$(TASK_ID)"
	@$(MAKE) llm-smoke
	@$(MAKE) doctor WORKSPACE_ROOT="$(WORKSPACE_ROOT)"
