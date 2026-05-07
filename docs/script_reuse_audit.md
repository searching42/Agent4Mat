# workspace/scripts 可复用审计（for oled-agent）

更新时间：2026-04-29

目标：评估 `workspace/scripts/*.py` 在新仓库中的迁移价值。

评估维度：
- `A` 可直接迁移为库函数（轻微改路径/参数）
- `B` 可复用核心逻辑，但需要拆分与重构
- `C` 仅适合保留为历史/分析脚本，不建议进入主流水线

---

## 1) 数据摄取与治理（Data ingest/curation）

### A 级（优先迁移）
- `init_schema.py`
- `create_reference_stubs.py`
- `generate_review_actions.py`
- `prioritize_curation_lists.py`
- `export_training_ready_subset_v1.py`
- `export_unimol_lambda_em_dataset_v1.py`
- `export_unimol_multitask_molecule_dataset_v1.py`

迁移建议：
- 去掉硬编码 `ROOT=/home/node/.openclaw/workspace`，统一使用 `workspace_root` 参数。
- 从 CLI 脚本改成 `def run(ctx, input, params)` 风格 stage。
- 输出路径统一写入 run 目录，再由 runner 复制/发布到 reports。

### B 级（可复用但需重构）
- `batch_extract_unimol_embeddings.py`（依赖远端执行环境）
- `prepare_unimol_demo_dataset.py`

---

## 2) 评分与多目标筛选（Scoring/ranking）

### A 级（优先迁移）
- `compose_multi_objective_scores.py`
- `filter_multi_objective_prefilter.py`
- `filter_multi_objective_candidates.py`
- `generate_explained_shortlist.py`
- `extract_explained_shortlist_observation.py`

迁移建议：
- 这些脚本的核心逻辑已较独立，直接提取为 `stages/` 模块即可。
- 将配置读取统一交给 `PipelineConfig`，避免 stage 内再读额外 config 文件。

### B 级（需适配远端依赖）
- `score_unimol_property_candidates.py`
- `score_reinvent4_lambda_em_candidates_unimol.py`
- `run_single_property_task.py`
- `run_multi_objective_task.py`

重构重点：
- 当前强绑定 `ssh/scp/REMOTE_HOST`，建议抽象成 `adapters/unimol.py` 与 `adapters/reinvent4.py`。
- adapter 需要支持本地模式与远端模式（配置切换）。

---

## 3) 编排与策略执行（Orchestration/policy）

### A 级（核心可迁移）
- `intake_schema.py`
- `bootstrap_baseline_config.py`
- `check_prelaunch_live_trigger_gate.py`
- `check_tail_retention_offline.py`
- `check_b1_viability_replay.py`
- `check_targeted_reopen_eligibility.py`
- `evaluate_targeted_hold_decision.py`
- `classify_targeted_hold_replay_family.py`

### B 级（框架参考价值高）
- `run_current_default_shortlist_policy.py`
- `run_generalized_workflow.py`
- `run_b1_type1_discount_branch.py`
- `run_max1_type1_pilot.py`
- `run_conditional_second_stage_branch_v2.py`

重构重点：
- 当前大量通过 subprocess 串脚本，建议改成 runner 直接调 Python 函数。
- 各分支（baseline/B1/max1/second-stage）改成 pipeline graph 或 stage list。

---

## 4) 验收、回归与守护（QA/guardrails）

### A 级（建议保留并迁移）
- `check_generalized_workflow_acceptance.py`
- `check_generalized_workflow_regression_suite.py`
- `check_prelaunch_intake_regression.py`
- `check_legacy_intake_usage.py`
- `check_markdown_path_leaks.py`
- `check_run_manifest_consistency.py`

迁移建议：
- 整理为 `tests/regression/`（pytest 或 plain python CLI 均可）。
- CI 中至少保留：intake gate + manifest consistency + one acceptance replay。

---

## 5) 分析型脚本（历史复盘/机制分析）

### C 级（不进主流水线，留在 analyses）
- `analyze_*`
- `replay_*`
- `generate_*_report.py`（特定轮次报告）
- `rdkit_priority4_nn_analysis.py`

处理建议：
- 迁移到 `analysis/` 或 `notebooks/`，只作为研究支持，不作为 release 必需能力。

---

## 6) 立即可执行的迁移优先级

P0（两周内）：
1. `compose_multi_objective_scores.py`
2. `filter_multi_objective_prefilter.py`
3. `filter_multi_objective_candidates.py`
4. `generate_explained_shortlist.py`
5. `intake_schema.py`
6. `check_prelaunch_live_trigger_gate.py`

P1（随后）：
1. `run_current_default_shortlist_policy.py`（拆为纯函数）
2. `check_tail_retention_offline.py`
3. `check_b1_viability_replay.py`
4. `check_targeted_reopen_eligibility.py`

P2（平台化）：
1. 远端 adapter：`score_unimol_property_candidates.py` / `run_multi_objective_task.py`
2. 统一 generalized workflow runner（替代 subprocess 链）

---

## 7) 关键风险与对应动作

风险：脚本普遍有路径约定和命令行副作用。  
动作：先提纯“纯输入->纯输出”函数，再包 CLI。

风险：远端 Uni-Mol/REINVENT4 环境强耦合。  
动作：adapter 化 + 环境探测 + mock mode。

风险：历史策略语义复杂（hold/reopen gates）。  
动作：先迁移 gate 检查器，再迁移调度器。
