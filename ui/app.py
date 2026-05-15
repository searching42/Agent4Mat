from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request
from oled_agent.agent.task_v2 import compute_missing_questions, legacy_request_to_task_v2
from oled_agent.agent.request_contract import validate_decision_summary_payload, validate_task_state_payload


app = Flask(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "scripts/adapters/real_adapters_catalog.json"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PROJECTS_DIR_REL = Path("runs/ui_sessions/projects")
UPLOADS_DIR_REL = Path("runs/ui_sessions/uploads")
MAX_PROJECT_HISTORY = 400
STEP_OPERATIONS = (
    "retrieve_candidate_data",
    "clean_dataset",
    "prepare_train_data",
    "train_predictor",
    "generate_candidates",
    "score_candidates",
    "filter_and_rank",
    "make_report",
)
ARTIFACT_NAME_TO_FILE = {
    "plan": "plan.json",
    "execution": "execution.json",
    "tool_state": "tool_state.json",
    "decision_summary": "decision_summary.json",
    "task_state": "task_state.json",
    "web_evidence": "artifacts/web_evidence.json",
    "experiment_trace": "artifacts/experiment_trace.json",
}


HTML = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Agent4Mat Chat Console</title>
    <style>
      :root {
        --bg: #f3f5f9;
        --card: #ffffff;
        --line: #d6deea;
        --txt: #1b2433;
        --muted: #6b7483;
        --brand: #0b5ed7;
        --brand-soft: #dbe9ff;
        --ok: #0f766e;
        --warn: #b45309;
        --fail: #b42318;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background: radial-gradient(circle at 20% 0%, #ecf3ff, var(--bg) 48%);
        color: var(--txt);
      }
      .layout {
        display: grid;
        grid-template-columns: 280px minmax(520px, 1fr) 360px;
        gap: 12px;
        min-height: 100vh;
        padding: 12px;
        align-items: start;
      }
      .panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.04);
      }
      .panel.left-drawer,
      .panel.right-drawer {
        position: sticky;
        top: 12px;
        max-height: calc(100vh - 24px);
        overflow: auto;
      }
      .panel.chat-workspace {
        min-height: calc(100vh - 24px);
        display: grid;
        grid-template-rows: auto 1fr auto;
        gap: 10px;
      }
      h2, h3 { margin: 0 0 8px 0; }
      h2 { font-size: 1.0rem; }
      h3 { font-size: 0.92rem; color: var(--muted); }
      .muted { color: var(--muted); font-size: 0.84rem; }
      label {
        display: block;
        margin-top: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #3a4252;
      }
      input, textarea, select, button {
        font: inherit;
      }
      input, textarea, select {
        width: 100%;
        margin-top: 5px;
        padding: 8px 9px;
        border: 1px solid #cfd7e5;
        border-radius: 9px;
        background: white;
      }
      textarea { resize: vertical; }
      button {
        margin-top: 8px;
        padding: 8px 11px;
        border-radius: 9px;
        border: 1px solid #bed1f8;
        background: var(--brand-soft);
        color: #114293;
        cursor: pointer;
      }
      button.primary {
        background: var(--brand);
        color: white;
        border-color: var(--brand);
      }
      .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
      .project-meta {
        margin-top: 8px;
        padding: 8px;
        background: #f7f9fd;
        border: 1px solid #e2e7f1;
        border-radius: 9px;
        font-size: 0.82rem;
      }
      .chat-wrap { display: grid; grid-template-rows: 1fr auto; gap: 10px; min-height: 82vh; }
      .workspace-hud {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
        padding: 10px 12px;
        border: 1px solid #d8e2ef;
        border-radius: 12px;
        background: linear-gradient(180deg, #ffffff, #f7faff);
      }
      .hud-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6a7280;
        margin-bottom: 4px;
      }
      .hud-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        font-size: 0.82rem;
        color: #334155;
      }
      .hud-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 5px 8px;
        border-radius: 999px;
        background: #eef4ff;
        border: 1px solid #d0dcfa;
        color: #244b8f;
        font-size: 0.76rem;
      }
      .hud-actions {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }
      .hud-actions button {
        margin-top: 0;
      }
      .chat-log {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fbfcff;
        padding: 10px;
        overflow: auto;
      }
      .msg {
        max-width: 88%;
        margin-bottom: 10px;
        padding: 9px 10px;
        border-radius: 10px;
        line-height: 1.45;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .msg.user {
        margin-left: auto;
        background: #dbe9ff;
        border: 1px solid #bad1ff;
      }
      .msg.assistant {
        margin-right: auto;
        background: #eef2f8;
        border: 1px solid #d7e0ee;
      }
      .msg.system {
        margin-right: auto;
        background: #fff8ea;
        border: 1px solid #f3deb5;
      }
      .msg .meta {
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.72rem;
      }
      .timeline {
        margin-top: 8px;
        border-top: 1px dashed #ccd7ea;
        padding-top: 7px;
        font-size: 0.75rem;
        color: #3f4d62;
      }
      .timeline-item {
        margin: 2px 0;
      }
      .timeline-groups {
        margin-top: 10px;
        border: 1px solid #d6dfef;
        border-radius: 9px;
        padding: 8px;
        background: #f7faff;
      }
      .tg-head {
        font-size: 0.78rem;
        color: #39465c;
        margin-bottom: 6px;
      }
      .tg-cols {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
      }
      .tg-col {
        border: 1px solid #dce5f3;
        border-radius: 8px;
        background: #fff;
        min-height: 66px;
        padding: 6px;
      }
      .tg-col h4 {
        margin: 0 0 5px 0;
        font-size: 0.74rem;
        color: #485772;
      }
      .tg-col ul {
        margin: 0;
        padding-left: 14px;
        font-size: 0.72rem;
      }
      .tg-col li {
        margin: 2px 0;
      }
      .chat-input {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px;
        background: #fff;
      }
      .chat-input textarea { min-height: 84px; }
      .tool-box {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 9px;
        margin-top: 10px;
        background: #fbfcff;
      }
      details.drawer {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0;
        margin-top: 10px;
        background: #fbfcff;
      }
      details.drawer > summary {
        list-style: none;
        cursor: pointer;
        padding: 9px 11px;
        font-weight: 700;
        color: #334155;
      }
      details.drawer > summary::-webkit-details-marker {
        display: none;
      }
      details.drawer[open] > summary {
        border-bottom: 1px solid #e4eaf4;
      }
      .drawer-body {
        padding: 9px 11px 11px 11px;
      }
      .pending-fields {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
      }
      .pending-q {
        margin: 6px 0 0 16px;
        padding: 0;
        color: #3b4455;
        font-size: 0.84rem;
      }
      .prompt-history {
        margin-top: 8px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .prompt-history .empty {
        color: var(--muted);
        font-size: 0.76rem;
      }
      .prompt-chip {
        margin-top: 0;
        padding: 6px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: #eef4ff;
        border: 1px solid #c8d9fb;
        color: #244b8f;
        cursor: pointer;
      }
      pre {
        margin: 0;
        background: #0f1729;
        color: #d3ddf3;
        border-radius: 10px;
        padding: 10px;
        overflow: auto;
        max-height: 380px;
        font-size: 0.78rem;
      }
      .runtime {
        border: 1px solid #d8e2ef;
        background: #f7faff;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 10px;
        font-size: 0.82rem;
      }
      .progress-wrap {
        width: 100%;
        height: 10px;
        border: 1px solid #ced8e9;
        background: #ecf2fb;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
      }
      .progress-bar {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
      }
      .state-pass { color: var(--ok); }
      .state-fail { color: var(--fail); }
      .state-warn { color: var(--warn); }
      @media (max-width: 1200px) {
        .layout { grid-template-columns: 1fr; }
        .chat-wrap { min-height: 65vh; }
        .tg-cols { grid-template-columns: 1fr; }
        .panel.left-drawer,
        .panel.right-drawer {
          position: static;
          max-height: unset;
          overflow: visible;
        }
        .workspace-hud {
          flex-direction: column;
          align-items: flex-start;
        }
      }
    </style>
  </head>
  <body>
    <div class=\"layout\">
      <section class=\"panel left-drawer\">
        <h2>Projects</h2>
        <h3>Independent chat memory per project</h3>
        <label>Project picker</label>
        <select id=\"project_picker\" onchange=\"switchProjectFromPicker()\">
          <option value=\"\">(select)</option>
        </select>
        <button onclick=\"refreshProjects()\">Refresh Projects</button>

        <label>Project ID</label>
        <input id=\"project_id\" value=\"demo_chat_project\" />
        <label>Project title</label>
        <input id=\"project_title\" value=\"OLED chat test\" />

        <label>Planner provider</label>
        <select id=\"planner\">
          <option value=\"rule_based_v1\">rule_based_v1</option>
          <option value=\"llm_v1\">llm_v1</option>
        </select>
        <label>Catalog path</label>
        <input id=\"catalog\" value=\"scripts/adapters/real_adapters_catalog.json\" />

        <label><input id=\"web_enabled\" type=\"checkbox\" checked /> Enable web evidence</label>
        <label>Web topk</label>
        <input id=\"web_topk\" value=\"5\" />

        <div class=\"btn-row\">
          <button class=\"primary\" onclick=\"saveProject()\">Save/Load Project</button>
          <button onclick=\"sendChat(true)\">Start New Task</button>
        </div>
        <div class=\"btn-row\">
          <button onclick=\"openWorkspaceWindow()\">Open in New Window</button>
          <button onclick=\"copyWorkspaceLink()\">Copy Workspace Link</button>
        </div>
        <div class=\"muted\">当前项目会同步到 URL 的 <code>?project_id=...</code>，便于独立窗口和分享。</div>

        <div class=\"project-meta\" id=\"project_meta\">
          <div>task_id: <span id=\"current_task_id\">-</span></div>
          <div>runtime_health: <span id=\"project_runtime_health\">-</span></div>
          <div>updated_at: <span id=\"project_updated_at\">-</span></div>
          <div>session_file: <span id=\"project_file\">-</span></div>
        </div>

        <div class=\"tool-box\">
          <h3>Project Import/Export</h3>
          <div class=\"btn-row\">
            <button onclick=\"exportProject()\">Export Project JSON</button>
            <button onclick=\"importProject(false)\">Import JSON</button>
            <button onclick=\"importProject(true)\">Import JSON (override)</button>
          </div>
          <label>Import JSON payload</label>
          <textarea id=\"project_import_json\" rows=\"6\" placeholder='{"project": {...}}'></textarea>
        </div>
      </section>

      <section class=\"panel chat-workspace chat-wrap\">
        <div class=\"workspace-hud\">
          <div>
            <div class=\"hud-label\">Current Workspace</div>
            <div class=\"hud-row\">
              <span class=\"hud-chip\">project <span id=\"hud_project_id\">-</span></span>
              <span class=\"hud-chip\">task <span id=\"current_task_id_hud\">-</span></span>
              <span class=\"hud-chip\">health <span id=\"project_runtime_health_hud\">-</span></span>
            </div>
          </div>
          <div class=\"hud-actions\">
            <button class=\"primary\" onclick=\"sendChat(true)\">Start New Task</button>
            <button onclick=\"loadHistory()\">Reload History</button>
            <button onclick=\"loadRunRuntime()\">Refresh Runtime</button>
          </div>
        </div>
        <div class=\"chat-log\" id=\"chat_log\"></div>
        <div class=\"chat-input\">
          <label>Chat with agent</label>
          <textarea id=\"message_input\" placeholder=\"例如：设计470nm附近且高PLQY分子；补充字段：{&quot;candidate_data&quot;:&quot;/abs/path/data.csv&quot;}；或单步：/step clean_dataset {&quot;input_csv&quot;:&quot;/abs/path/data.csv&quot;}\"></textarea>
          <div class=\"muted\">Step mode: 支持 `/step <operation> {args_json}` 或直接发送 `{\"operation\":\"...\",\"args\":{...}}`。</div>
          <div class=\"muted\">快捷键: Ctrl/Cmd+Enter 发送，Shift+Enter 换行。</div>
          <div class=\"prompt-history\" id=\"prompt_history_box\"></div>
          <div class=\"btn-row\">
            <button class=\"primary\" onclick=\"sendChat(false)\">Send</button>
            <button onclick=\"sendWebSearchHint()\">Web Search</button>
          </div>

          <div class=\"tool-box\" id=\"pending_input_box\" style=\"display:none;\">
            <h3>Need Input</h3>
            <div class=\"muted\" id=\"pending_stage_text\">stage: -</div>
            <ul class=\"pending-q\" id=\"pending_questions\"></ul>
            <div class=\"pending-fields\" id=\"pending_fields\"></div>
            <div class=\"btn-row\">
              <button class=\"primary\" onclick=\"sendPendingForm(false)\">Send Form Patch</button>
              <button onclick=\"sendPendingForm(true)\">Send + Run</button>
              <button onclick=\"clearPendingInput()\">Hide</button>
            </div>
          </div>

          <details class=\"drawer\" open>
            <summary>File Input Entry</summary>
            <div class=\"drawer-body\">
              <label>Local file path (recommended)</label>
              <input id=\"attachment_path\" placeholder=\"/absolute/path/to/file.csv\" />
              <div class=\"btn-row\">
                <button onclick=\"attachPath()\">Attach Path</button>
                <button onclick=\"setCandidateDataFromPath()\">Use As candidate_data</button>
              </div>
              <label>Upload file copy (optional)</label>
              <input id=\"attachment_file\" type=\"file\" />
              <button onclick=\"uploadFileRef()\">Upload File To Session</button>
              <div class=\"muted\">上传文件将保存到 runs/ui_sessions/uploads/&lt;project_id&gt;/，并记录到项目会话。</div>
            </div>
          </details>

          <details class=\"drawer\">
            <summary>Single Step Runner</summary>
            <div class=\"drawer-body\">
              <label>Operation</label>
              <select id=\"step_operation\" onchange=\"applyStepArgsTemplate(false)\">
                <option value=\"retrieve_candidate_data\">retrieve_candidate_data</option>
                <option value=\"clean_dataset\">clean_dataset</option>
                <option value=\"prepare_train_data\">prepare_train_data</option>
                <option value=\"train_predictor\">train_predictor</option>
                <option value=\"generate_candidates\">generate_candidates</option>
                <option value=\"score_candidates\">score_candidates</option>
                <option value=\"filter_and_rank\">filter_and_rank</option>
                <option value=\"make_report\">make_report</option>
              </select>
              <label>Args JSON</label>
              <textarea id=\"step_args_json\" rows=\"4\">{}</textarea>
              <div class=\"btn-row\">
                <button onclick=\"applyStepArgsTemplate(true)\">Load Args Template</button>
                <button onclick=\"runStepPanel()\">Run Step From Panel</button>
              </div>
            </div>
          </details>
        </div>
      </section>

      <section class=\"panel right-drawer\">
        <h2>Outputs</h2>
        <h3>Runtime + artifacts</h3>
        <div class=\"runtime\" id=\"runtime_box\">runtime: (waiting)</div>
        <div class=\"muted\" id=\"runtime_stage_text\">stage: -</div>
        <div class=\"progress-wrap\"><div class=\"progress-bar\" id=\"runtime_progress_bar\"></div></div>
        <div class=\"muted\" id=\"runtime_progress_text\">progress: -</div>
        <label>Failed Tool Name (optional)</label>
        <input id=\"retry_failed_tool_name\" placeholder=\"e.g. score_candidates (empty = latest failed step)\" />
        <label>Retry Args JSON (optional override)</label>
        <textarea id=\"retry_failed_args_json\" rows=\"3\">{}</textarea>
        <div class=\"btn-row\">
          <button onclick=\"loadSuggestedRetryArgs()\">Load Suggested Retry Args</button>
          <button onclick=\"previewRetryFailedStep()\">Preview Failed-Step Retry</button>
          <button onclick=\"retryFailedStep()\">Retry Latest Failed Step</button>
          <button onclick=\"retryCurrentTask()\">Retry Current Task (resume)</button>
        </div>
        <label>Recent Events</label>
        <pre id=\"event_out\">(no events)</pre>

        <details class=\"drawer\" open>
          <summary>Run Timeline Groups</summary>
          <div class=\"drawer-body timeline-groups\" id=\"timeline_groups_box\">
            <div class=\"tg-head\" id=\"timeline_groups_head\">Run Timeline Groups (current task)</div>
            <div class=\"btn-row\">
              <label style=\"margin-top:0; font-weight:600;\">Scope</label>
              <select id=\"timeline_scope\" style=\"max-width: 180px;\">
                <option value=\"current_task\">Current Task</option>
                <option value=\"recent_tasks\">Recent Tasks</option>
              </select>
              <label style=\"margin-top:0; font-weight:600;\">Recent N</label>
              <input id=\"timeline_recent_limit\" value=\"5\" style=\"max-width: 70px;\" />
              <button onclick=\"loadTimelineGroupsByScope()\">Apply</button>
            </div>
            <div class=\"tg-cols\">
              <div class=\"tg-col\">
                <h4>Running</h4>
                <ul id=\"tg_running\"><li>(empty)</li></ul>
              </div>
              <div class=\"tg-col\">
                <h4>Completed</h4>
                <ul id=\"tg_completed\"><li>(empty)</li></ul>
              </div>
              <div class=\"tg-col\">
                <h4>Failed</h4>
                <ul id=\"tg_failed\"><li>(empty)</li></ul>
              </div>
            </div>
          </div>
        </details>

        <details class=\"drawer\" open>
          <summary>Artifacts & Validation</summary>
          <div class=\"drawer-body\">
            <label>Artifact</label>
            <select id=\"artifact_name\">
              <option value=\"plan\">plan</option>
              <option value=\"execution\">execution</option>
              <option value=\"decision_summary\">decision_summary</option>
              <option value=\"task_state\">task_state</option>
              <option value=\"tool_state\">tool_state</option>
              <option value=\"web_evidence\">web_evidence</option>
              <option value=\"experiment_trace\">experiment_trace</option>
            </select>
            <div class=\"btn-row\">
              <button onclick=\"previewArtifact()\">Preview Artifact</button>
              <button onclick=\"showTimeline()\">Show Timeline</button>
              <button onclick=\"validateTask()\">Validate Task</button>
            </div>
          </div>
        </details>

        <details class=\"drawer\">
          <summary>Task Compare</summary>
          <div class=\"drawer-body\">
            <label>Other Task ID</label>
            <input id=\"compare_other_task_id\" placeholder=\"e.g. acc_local_20260514_095552\" />
            <div class=\"btn-row\">
              <button onclick=\"compareTasks()\">Compare Tasks</button>
              <button onclick=\"compareSelectedArtifact()\">Compare Selected Artifact Diff</button>
            </div>
            <div class=\"muted\">使用当前 task 与另一个 task 做 summary / artifact diff 对比。</div>
          </div>
        </details>
        <pre id=\"out\">(waiting)</pre>
      </div>
    </div>
    <script>
      const state = {
        project: null,
        pendingInput: null,
        promptHistory: [],
      };

      const PROMPT_HISTORY_LIMIT = 8;

      const pendingFieldMeta = {
        property: {label: 'property', placeholder: 'plqy / lambda_em / stability'},
        range: {label: 'range', placeholder: '470+-12nm or 60-100'},
        n_structures: {label: 'n_structures', placeholder: 'e.g. 500', type: 'number'},
        prediction_model: {label: 'prediction_model', placeholder: 'e.g. unimol_lambda_plqy_v1'},
        candidate_data: {label: 'candidate_data', placeholder: '/abs/path/to/candidates.csv'},
        train_data: {label: 'train_data', placeholder: '/abs/path/to/train.csv'},
      };

      const stepArgsTemplates = {
        retrieve_candidate_data: {
          candidate_data: "/abs/path/to/candidate_source.csv"
        },
        clean_dataset: {
          input_csv: "/abs/path/to/candidates.csv",
          constraints: {
            mw_min: 150,
            mw_max: 700,
            domain_threshold: 0.2,
            banned_alerts: []
          }
        },
        prepare_train_data: {
          train_data: "/abs/path/to/train.csv"
        },
        train_predictor: {
          predictor_id: "unimol_lambda_plqy_v1",
          targets: ["plqy"]
        },
        generate_candidates: {
          generator_id: "reinvent4_lambda_em_v2",
          max_candidates: 300,
          constraints: {
            mw_min: 150,
            mw_max: 700,
            domain_threshold: 0.2,
            banned_alerts: []
          },
          input_csv: "/abs/path/to/candidates.csv"
        },
        score_candidates: {
          predictor_id: "unimol_lambda_plqy_v1",
          targets: ["plqy"],
          input_csv: "/abs/path/to/generated.csv"
        },
        filter_and_rank: {
          topn: 10,
          target_specs: [
            {"property_name": "lambda_em", "weight": 0.65},
            {"property_name": "plqy", "weight": 0.25}
          ]
        },
        make_report: {}
      };

      function nowIso() {
        return new Date().toISOString();
      }

      function renderJsonOut(payload) {
        document.getElementById('out').textContent = JSON.stringify(payload, null, 2);
      }

      function currentProjectKey() {
        return String(selectedProjectId() || 'demo_chat_project').trim() || 'demo_chat_project';
      }

      function messageDraftKey(projectId) {
        return `agent4mat.ui.message_draft.${String(projectId || '').trim() || 'demo_chat_project'}`;
      }

      function promptHistoryKey(projectId) {
        return `agent4mat.ui.prompt_history.${String(projectId || '').trim() || 'demo_chat_project'}`;
      }

      function loadPromptHistory(projectId) {
        try {
          const raw = localStorage.getItem(promptHistoryKey(projectId));
          if (!raw) return [];
          const parsed = JSON.parse(raw);
          if (!Array.isArray(parsed)) return [];
          return parsed.filter((item) => typeof item === 'string' && item.trim()).slice(0, PROMPT_HISTORY_LIMIT);
        } catch (e) {
          return [];
        }
      }

      function savePromptHistory(projectId, items) {
        try {
          localStorage.setItem(promptHistoryKey(projectId), JSON.stringify(items.slice(0, PROMPT_HISTORY_LIMIT)));
        } catch (e) {
          // ignore storage failures
        }
      }

      function renderPromptHistory(projectId) {
        const box = document.getElementById('prompt_history_box');
        if (!box) return;
        const items = loadPromptHistory(projectId);
        state.promptHistory = items;
        box.innerHTML = '';
        if (items.length < 1) {
          const empty = document.createElement('div');
          empty.className = 'empty';
          empty.textContent = 'Recent prompts: (empty)';
          box.appendChild(empty);
          return;
        }
        for (const prompt of items) {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'prompt-chip';
          const short = prompt.length > 42 ? `${prompt.slice(0, 42)}...` : prompt;
          btn.textContent = short;
          btn.title = prompt;
          btn.onclick = () => {
            setMessageInput(prompt, {persist: true});
            document.getElementById('message_input').focus();
          };
          box.appendChild(btn);
        }
      }

      function capturePromptHistory(projectId, message) {
        const text = String(message || '').trim();
        if (!text) return;
        const existing = loadPromptHistory(projectId);
        const deduped = [text, ...existing.filter((item) => String(item || '').trim() !== text)];
        savePromptHistory(projectId, deduped);
        renderPromptHistory(projectId);
      }

      function restoreMessageDraft(projectId) {
        const key = messageDraftKey(projectId);
        try {
          const saved = localStorage.getItem(key);
          if (saved !== null) {
            document.getElementById('message_input').value = saved;
          }
        } catch (e) {
          // ignore storage failures
        }
      }

      function persistMessageDraft() {
        const pid = currentProjectKey();
        const text = String(document.getElementById('message_input').value || '');
        try {
          if (text.trim()) {
            localStorage.setItem(messageDraftKey(pid), text);
          } else {
            localStorage.removeItem(messageDraftKey(pid));
          }
        } catch (e) {
          // ignore storage failures
        }
      }

      function setMessageInput(text, opts) {
        const value = String(text || '');
        document.getElementById('message_input').value = value;
        if (!opts || opts.persist !== false) {
          persistMessageDraft();
        }
      }

      function clearPendingInput() {
        state.pendingInput = null;
        document.getElementById('pending_input_box').style.display = 'none';
        document.getElementById('pending_stage_text').textContent = 'stage: -';
        document.getElementById('pending_questions').innerHTML = '';
        document.getElementById('pending_fields').innerHTML = '';
      }

      function pendingFieldDefault(field) {
        if (field === 'candidate_data') {
          const p = (document.getElementById('attachment_path').value || '').trim();
          if (p) return p;
        }
        return '';
      }

      function renderPendingInput(pending) {
        if (!pending || typeof pending !== 'object') {
          clearPendingInput();
          return;
        }
        state.pendingInput = pending;
        const box = document.getElementById('pending_input_box');
        box.style.display = 'block';

        const stage = String(pending.stage || '');
        document.getElementById('pending_stage_text').textContent = `stage: ${stage || '-'}`;

        const qList = document.getElementById('pending_questions');
        qList.innerHTML = '';
        const questions = Array.isArray(pending.questions) ? pending.questions : [];
        for (const q of questions) {
          const li = document.createElement('li');
          li.textContent = String(q || '');
          qList.appendChild(li);
        }

        const fieldsWrap = document.getElementById('pending_fields');
        fieldsWrap.innerHTML = '';
        const missing = Array.isArray(pending.missing_fields) ? pending.missing_fields : [];
        for (const field of missing) {
          const f = String(field || '').trim();
          if (!f) continue;
          const meta = pendingFieldMeta[f] || {label: f, placeholder: ''};
          const row = document.createElement('div');
          const label = document.createElement('label');
          label.textContent = `${meta.label}`;
          const input = document.createElement('input');
          input.id = `pending_field_${f}`;
          input.type = meta.type || 'text';
          input.placeholder = meta.placeholder || '';
          input.value = pendingFieldDefault(f);
          row.appendChild(label);
          row.appendChild(input);
          fieldsWrap.appendChild(row);
        }
      }

      function collectPendingPatch() {
        const pending = state.pendingInput;
        if (!pending || typeof pending !== 'object') return {};
        const out = {};
        const missing = Array.isArray(pending.missing_fields) ? pending.missing_fields : [];
        for (const field of missing) {
          const f = String(field || '').trim();
          if (!f) continue;
          const ele = document.getElementById(`pending_field_${f}`);
          if (!ele) continue;
          const raw = String(ele.value || '').trim();
          if (!raw) continue;
          if (f === 'n_structures') {
            const n = Number(raw);
            if (Number.isFinite(n) && n > 0) out[f] = Math.floor(n);
            continue;
          }
          out[f] = raw;
        }
        return out;
      }

      function renderEvents(events) {
        const arr = Array.isArray(events) ? events : [];
        if (arr.length < 1) {
          document.getElementById('event_out').textContent = '(no events)';
          return;
        }
        const lines = [];
        for (const e of arr) {
          if (!e || typeof e !== 'object') continue;
          const stage = String(e.stage || 'stage');
          const status = String(e.status || 'unknown');
          const op = String(e.operation || '');
          const reason = String(e.reason || '');
          let line = `${stage}: ${status}`;
          if (op) line += ` | op=${op}`;
          if (reason) line += ` | reason=${reason}`;
          lines.push(line);
        }
        document.getElementById('event_out').textContent = lines.length > 0 ? lines.join('\n') : '(no events)';
      }

      function renderRuntimeProgress(summary) {
        const bar = document.getElementById('runtime_progress_bar');
        const text = document.getElementById('runtime_progress_text');
        const total = Number(summary && summary.total_steps ? summary.total_steps : 0);
        const success = Number(summary && summary.success_steps ? summary.success_steps : 0);
        const failed = Number(summary && summary.failed_steps ? summary.failed_steps : 0);
        if (!Number.isFinite(total) || total <= 0) {
          bar.style.width = '0%';
          text.textContent = 'progress: -';
          return;
        }
        const ratio = Math.max(0, Math.min(1, success / total));
        bar.style.width = `${(ratio * 100).toFixed(1)}%`;
        text.textContent = `progress: ${success}/${total} success, failed=${failed}`;
      }

      function renderRuntimeStage(summaryPayload, timelinePayload) {
        const ele = document.getElementById('runtime_stage_text');
        const taskState = (summaryPayload && summaryPayload.task_state && typeof summaryPayload.task_state === 'object')
          ? summaryPayload.task_state
          : {};
        const stage = String(taskState.current_stage || taskState.currentState || '-');
        const status = String(taskState.status || '-');
        const events = Array.isArray(timelinePayload && timelinePayload.events) ? timelinePayload.events : [];
        const failed = events.find((e) => e && typeof e === 'object' && Boolean(e.is_failed));
        let txt = `stage: ${stage} | task_state: ${status}`;
        if (failed) {
          const name = String(failed.name || '');
          txt += ` | latest_failed_step: ${name || '-'}`;
        }
        ele.textContent = txt;
      }

      function groupItemText(item) {
        if (!item || typeof item !== 'object') return 'step';
        const name = String(item.name || 'step');
        const status = String(item.status || '-');
        const dur = (typeof item.duration_ms === 'number') ? `${item.duration_ms}ms` : 'n/a';
        return `${name} | status=${status} | dur=${dur}`;
      }

      async function retrySpecificFailedItem(item) {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const parsed = parseRetryArgsOptional();
        if (!parsed.ok) {
          renderJsonOut({status: 'fail', error: parsed.error});
          return;
        }
        const body = {
          catalog_path: document.getElementById('catalog').value,
          failed_tool_name: String(item && item.name ? item.name : ''),
        };
        if (parsed.args && Object.keys(parsed.args).length > 0) {
          body.args = parsed.args;
        } else if (item && item.args && typeof item.args === 'object' && !Array.isArray(item.args)) {
          body.args = item.args;
        }
        const r = await apiPost(`/api/task/${encodeURIComponent(tid)}/retry-failed-step`, body);
        renderJsonOut(r.data);
        const status = String((r.data && r.data.status) || 'unknown');
        const op = String((r.data && r.data.retry_operation) || '');
        renderEvents([{stage: 'retry_failed_item', status: status, operation: op || undefined}]);
        await loadRunRuntime();
      }

      function selectedRetryFailedToolName() {
        const raw = document.getElementById('retry_failed_tool_name');
        return String(raw && raw.value ? raw.value : '').trim();
      }

      async function loadSuggestedRetryArgs() {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const body = {
          catalog_path: document.getElementById('catalog').value,
          dry_run: true,
        };
        const failedToolName = selectedRetryFailedToolName();
        if (failedToolName) {
          body.failed_tool_name = failedToolName;
        }
        const r = await apiPost(`/api/task/${encodeURIComponent(tid)}/retry-failed-step`, body);
        renderJsonOut(r.data);
        const args = (r.data && r.data.retry_args && typeof r.data.retry_args === 'object' && !Array.isArray(r.data.retry_args))
          ? r.data.retry_args
          : null;
        if (args) {
          document.getElementById('retry_failed_args_json').value = JSON.stringify(args, null, 2);
        }
        const failedName = String((r.data && r.data.failed_tool_name) || '');
        if (failedName) {
          document.getElementById('retry_failed_tool_name').value = failedName;
        }
        const status = String((r.data && r.data.status) || 'unknown');
        const op = String((r.data && r.data.retry_operation) || '');
        renderEvents([{stage: 'load_retry_args', status: status, operation: op || undefined}]);
      }

      function setListItems(targetId, items) {
        const ul = document.getElementById(targetId);
        ul.innerHTML = '';
        const arr = Array.isArray(items) ? items : [];
        if (arr.length < 1) {
          const li = document.createElement('li');
          li.textContent = '(empty)';
          ul.appendChild(li);
          return;
        }
        for (const it of arr) {
          const li = document.createElement('li');
          li.textContent = groupItemText(it);
          li.style.cursor = 'pointer';
          li.title = 'Click to inspect details';
          li.onclick = () => {
            const detail = {
              stage: 'timeline_item',
              name: it && it.name ? it.name : '',
              status: it && it.status ? it.status : '',
              duration_ms: it && typeof it.duration_ms === 'number' ? it.duration_ms : null,
              started_at: it && it.started_at ? it.started_at : '',
              ended_at: it && it.ended_at ? it.ended_at : '',
              adapter: it && it.adapter ? it.adapter : '',
              error: it && it.error ? it.error : '',
              result_summary: it && it.result_summary ? it.result_summary : {},
              args: it && it.args ? it.args : {},
            };
            renderJsonOut({status: 'pass', item: detail});
            if (it && it.is_failed && it.name) {
              document.getElementById('retry_failed_tool_name').value = String(it.name);
              const args = (it.args && typeof it.args === 'object' && !Array.isArray(it.args)) ? it.args : {};
              document.getElementById('retry_failed_args_json').value = JSON.stringify(args, null, 2);
            }
          };
          ul.appendChild(li);
          if (it && it.is_failed) {
            const btn = document.createElement('button');
            btn.textContent = 'Retry';
            btn.style.marginLeft = '6px';
            btn.onclick = (evt) => {
              evt.stopPropagation();
              retrySpecificFailedItem(it);
            };
            li.appendChild(btn);
          }
        }
      }

      function renderTimelineGroups(timelinePayload) {
        const head = document.getElementById('timeline_groups_head');
        const summary = (timelinePayload && typeof timelinePayload === 'object' && timelinePayload.summary && typeof timelinePayload.summary === 'object')
          ? timelinePayload.summary
          : {};
        const events = Array.isArray(timelinePayload && timelinePayload.events) ? timelinePayload.events : [];
        const running = [];
        const completed = [];
        const failed = [];
        for (const ev of events) {
          if (!ev || typeof ev !== 'object') continue;
          const status = String(ev.status || '');
          const startedAt = String(ev.started_at || '');
          if (String(status).toLowerCase() === 'running') {
            running.push(ev);
          } else if (Boolean(ev.is_failed)) {
            failed.push(ev);
          } else {
            completed.push(ev);
          }
          if (!status && startedAt && !ev.ended_at) {
            running.push(ev);
          }
        }
        const total = Number(summary.total_steps || events.length || 0);
        const succ = Number(summary.success_steps || 0);
        const fail = Number(summary.failed_steps || failed.length || 0);
        head.textContent = `Run Timeline Groups (total=${total}, success=${succ}, failed=${fail})`;
        setListItems('tg_running', running);
        setListItems('tg_completed', completed);
        setListItems('tg_failed', failed);
      }

      function renderTimelineGroupsAggregate(payload) {
        const head = document.getElementById('timeline_groups_head');
        const running = Array.isArray(payload && payload.running_items) ? payload.running_items : [];
        const completed = Array.isArray(payload && payload.completed_items) ? payload.completed_items : [];
        const failed = Array.isArray(payload && payload.failed_items) ? payload.failed_items : [];
        const total = Number(payload && payload.total_steps ? payload.total_steps : (running.length + completed.length + failed.length));
        const scope = String(payload && payload.scope ? payload.scope : 'recent_tasks');
        const tasksN = Number(payload && payload.task_count ? payload.task_count : 0);
        head.textContent = `Run Timeline Groups (${scope}, tasks=${tasksN}, total=${total}, success=${completed.length}, failed=${failed.length})`;
        setListItems('tg_running', running);
        setListItems('tg_completed', completed);
        setListItems('tg_failed', failed);
      }

      async function loadTimelineGroupsByScope() {
        const scope = String(document.getElementById('timeline_scope').value || 'current_task');
        if (scope === 'current_task') {
          await loadRunRuntime();
          return;
        }
        const limitRaw = Number(document.getElementById('timeline_recent_limit').value || 5);
        const limit = Number.isFinite(limitRaw) ? Math.max(1, Math.min(20, Math.floor(limitRaw))) : 5;
        const r = await apiGet(`/api/timeline-groups?scope=recent_tasks&limit=${encodeURIComponent(String(limit))}`);
        renderJsonOut(r.data);
        renderTimelineGroupsAggregate(r.data);
      }

      function taskId() {
        if (state.project && state.project.current_task_id) return state.project.current_task_id;
        const span = document.getElementById('current_task_id');
        return (span && span.textContent) ? span.textContent.trim() : '';
      }

      function refreshWorkspaceHud() {
        const projectId = selectedProjectId();
        const task = taskId();
        const health = (state.project && state.project.runtime_health && typeof state.project.runtime_health === 'object')
          ? state.project.runtime_health
          : {};
        const pidEle = document.getElementById('hud_project_id');
        const tidEle = document.getElementById('current_task_id_hud');
        const hEle = document.getElementById('project_runtime_health_hud');
        if (pidEle) pidEle.textContent = projectId || '-';
        if (tidEle) tidEle.textContent = task || '-';
        if (hEle) hEle.textContent = formatRuntimeHealth(health);
      }

      function renderProjectOptions(project) {
        if (!project || typeof project !== 'object') return;
        const opts = (project.options && typeof project.options === 'object') ? project.options : {};
        const title = String(project.title || '');
        if (title) document.getElementById('project_title').value = title;
        if (opts.planner_provider) document.getElementById('planner').value = String(opts.planner_provider);
        if (opts.catalog_path) document.getElementById('catalog').value = String(opts.catalog_path);
        if (Object.prototype.hasOwnProperty.call(opts, 'web_search_enabled')) {
          document.getElementById('web_enabled').checked = Boolean(opts.web_search_enabled);
        }
        if (Object.prototype.hasOwnProperty.call(opts, 'web_topk')) {
          document.getElementById('web_topk').value = String(opts.web_topk);
        }
      }

      function renderProjectMeta(project) {
        if (!project) return;
        document.getElementById('current_task_id').textContent = project.current_task_id || '-';
        document.getElementById('project_runtime_health').textContent = formatRuntimeHealth(project.runtime_health);
        document.getElementById('project_updated_at').textContent = project.updated_at || '-';
        document.getElementById('project_file').textContent = project.project_path || '-';
        refreshWorkspaceHud();
      }

      function formatRuntimeHealth(health) {
        if (!health || typeof health !== 'object') return '-';
        const status = String(health.status || 'none');
        const taskId = String(health.task_id || '');
        const failed = Number(health.failed_steps || 0);
        const success = Number(health.success_steps || 0);
        const latest = String(health.latest_failed_step || '');
        if (status === 'none') {
          return String(health.reason || 'none');
        }
        let txt = `${status} ${success}✓/${failed}✗`;
        if (taskId) txt += ` @${taskId}`;
        if (latest) txt += ` ${latest}`;
        return txt;
      }

      function msgClass(role) {
        if (role === 'assistant') return 'assistant';
        if (role === 'user') return 'user';
        return 'system';
      }

      function renderChat(messages) {
        const log = document.getElementById('chat_log');
        log.innerHTML = '';
        for (const m of messages || []) {
          const role = String(m.role || 'system');
          const row = document.createElement('div');
          row.className = `msg ${msgClass(role)}`;
          const content = document.createElement('div');
          content.textContent = String(m.content || '');
          row.appendChild(content);
          const meta = document.createElement('div');
          const ts = String(m.created_at || '');
          const kind = String(m.kind || 'text');
          meta.className = 'meta';
          meta.textContent = `${role} • ${kind} • ${ts}`;
          row.appendChild(meta);

          const metaObj = (m && typeof m === 'object' && m.meta && typeof m.meta === 'object') ? m.meta : {};
          const timelineItems = [];
          if (metaObj.events && Array.isArray(metaObj.events)) {
            for (const ev of metaObj.events) {
              if (!ev || typeof ev !== 'object') continue;
              const stage = String(ev.stage || '');
              const status = String(ev.status || '');
              const op = String(ev.operation || '');
              let line = `${stage || 'stage'}: ${status || 'unknown'}`;
              if (op) line += ` | op=${op}`;
              timelineItems.push(line);
            }
          }
          if (timelineItems.length > 0) {
            const wrap = document.createElement('div');
            wrap.className = 'timeline';
            for (const line of timelineItems) {
              const item = document.createElement('div');
              item.className = 'timeline-item';
              item.textContent = line;
              wrap.appendChild(item);
            }
            row.appendChild(wrap);
          }
          log.appendChild(row);
        }
        log.scrollTop = log.scrollHeight;
      }

      async function apiGet(url) {
        const resp = await fetch(url);
        const data = await resp.json();
        return {status: resp.status, data};
      }

      async function apiPost(url, payload) {
        const resp = await fetch(url, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        return {status: resp.status, data};
      }

      function selectedProjectId() {
        const v = (document.getElementById('project_id').value || '').trim();
        return v || 'demo_chat_project';
      }

      function isSafeProjectId(projectId) {
        return /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(String(projectId || '').trim());
      }

      function readProjectIdFromUrl() {
        try {
          const raw = new URLSearchParams(window.location.search || '').get('project_id');
          const pid = String(raw || '').trim();
          return isSafeProjectId(pid) ? pid : '';
        } catch (e) {
          return '';
        }
      }

      function workspaceUrlForProject(projectId) {
        const url = new URL(window.location.href);
        const pid = String(projectId || '').trim();
        if (pid) {
          url.searchParams.set('project_id', pid);
        } else {
          url.searchParams.delete('project_id');
        }
        return url.toString();
      }

      function syncProjectPickerValue(projectId) {
        const picker = document.getElementById('project_picker');
        if (!picker) return;
        const pid = String(projectId || '').trim();
        const hasOption = Array.from(picker.options || []).some((opt) => String(opt.value || '') === pid);
        picker.value = hasOption ? pid : '';
      }

      function syncWorkspaceUrl(projectId, opts) {
        const pid = String(projectId || '').trim();
        const next = workspaceUrlForProject(pid);
        try {
          if (opts && opts.push) {
            window.history.pushState({project_id: pid}, '', next);
          } else {
            window.history.replaceState({project_id: pid}, '', next);
          }
        } catch (e) {
          // ignore history updates when the browser blocks them
        }
      }

      function applyProjectStateToUi(project, opts) {
        if (!project || typeof project !== 'object') return;
        const pid = String(project.project_id || selectedProjectId() || '').trim() || 'demo_chat_project';
        document.getElementById('project_id').value = pid;
        syncProjectPickerValue(pid);
        if (!opts || opts.updateUrl !== false) {
          syncWorkspaceUrl(pid, opts);
        }
        renderProjectOptions(project);
        renderProjectMeta(project);
        renderPendingInput(project.pending_input || null);
        restoreMessageDraft(pid);
        renderPromptHistory(pid);
        refreshWorkspaceHud();
      }

      function bindWorkspaceUrlNavigation() {
        window.addEventListener('popstate', () => {
          const pid = readProjectIdFromUrl();
          if (!pid) return;
          document.getElementById('project_id').value = pid;
          syncProjectPickerValue(pid);
          void loadHistory();
          void loadRunRuntime();
        });
      }

      function openWorkspaceWindow() {
        const url = workspaceUrlForProject(selectedProjectId());
        window.open(url, '_blank', 'noopener,noreferrer');
      }

      async function copyWorkspaceLink() {
        const url = workspaceUrlForProject(selectedProjectId());
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(url);
            renderJsonOut({status: 'pass', copied: url});
            return;
          }
        } catch (e) {
          // fall through to the failure payload below
        }
        renderJsonOut({status: 'fail', error: 'clipboard_unavailable', url: url});
      }

      function collectOptions() {
        const planner = document.getElementById('planner').value;
        const catalog = document.getElementById('catalog').value;
        const webEnabled = document.getElementById('web_enabled').checked;
        const webTopk = Number(document.getElementById('web_topk').value || 5);
        return {
          planner_provider: planner,
          catalog_path: catalog,
          web_search_enabled: Boolean(webEnabled),
          web_topk: Number.isFinite(webTopk) ? webTopk : 5,
        };
      }

      async function refreshProjects() {
        const r = await apiGet('/api/projects?limit=120');
        renderJsonOut(r.data);
        const picker = document.getElementById('project_picker');
        while (picker.options.length > 1) picker.remove(1);
        const projects = Array.isArray(r.data.projects) ? r.data.projects : [];
        for (const p of projects) {
          const pid = String(p.project_id || '');
          if (!pid) continue;
          const label = `${pid} [${String(p.current_task_id || '-')}] · ${formatRuntimeHealth(p.runtime_health)}`;
          const opt = document.createElement('option');
          opt.value = pid;
          opt.textContent = label;
          picker.appendChild(opt);
        }
        syncProjectPickerValue(selectedProjectId());
      }

      async function switchProjectFromPicker() {
        const picker = document.getElementById('project_picker');
        const pid = String(picker.value || '').trim();
        if (!pid) return;
        document.getElementById('project_id').value = pid;
        await saveProject();
      }

      async function saveProject() {
        const projectId = selectedProjectId();
        const title = (document.getElementById('project_title').value || '').trim();
        const r = await apiPost('/api/projects', {
          project_id: projectId,
          title: title,
          options: collectOptions(),
        });
        renderJsonOut(r.data);
        const project = r.data && r.data.project ? r.data.project : null;
        if (project) {
          state.project = project;
          applyProjectStateToUi(project);
        }
        await refreshProjects();
        await loadHistory();
      }

      async function exportProject() {
        const pid = selectedProjectId();
        const r = await apiGet(`/api/projects/${encodeURIComponent(pid)}/export`);
        renderJsonOut(r.data);
        if (r.data && r.data.project) {
          document.getElementById('project_import_json').value = JSON.stringify({project: r.data.project}, null, 2);
        }
      }

      async function importProject(override) {
        const text = String(document.getElementById('project_import_json').value || '').trim();
        if (!text) {
          renderJsonOut({status: 'fail', error: 'empty import json'});
          return;
        }
        let payload = null;
        try {
          payload = JSON.parse(text);
        } catch (e) {
          renderJsonOut({status: 'fail', error: `invalid import json: ${String(e)}`});
          return;
        }
        const r = await apiPost('/api/projects/import', {
          project: (payload && typeof payload === 'object' && payload.project && typeof payload.project === 'object')
            ? payload.project
            : payload,
          project_id: selectedProjectId(),
          override: Boolean(override),
        });
        renderJsonOut(r.data);
        if (r.data && r.data.project) {
          state.project = r.data.project;
          applyProjectStateToUi(r.data.project);
        }
        await refreshProjects();
        await loadHistory();
      }

      async function loadHistory() {
        const pid = selectedProjectId();
        const r = await apiGet(`/api/projects/${encodeURIComponent(pid)}/history?limit=300`);
        renderJsonOut(r.data);
        if (r.data && r.data.project) {
          state.project = r.data.project;
          applyProjectStateToUi(r.data.project);
        }
        const messages = Array.isArray(r.data.messages) ? r.data.messages : [];
        renderChat(messages);
        restoreMessageDraft(pid);
        renderPromptHistory(pid);
      }

      async function sendChat(newTask) {
        const pid = selectedProjectId();
        const message = (document.getElementById('message_input').value || '').trim();
        if (!message && !newTask) {
          renderJsonOut({status: 'fail', error: 'empty message'});
          return;
        }
        const r = await apiPost('/api/chat/send', {
          project_id: pid,
          message: message,
          options: collectOptions(),
          new_task: Boolean(newTask),
        });
        if (message) {
          capturePromptHistory(pid, message);
        }
        renderJsonOut(r.data);
        renderEvents(r.data && r.data.events ? r.data.events : []);
        const pending = (r.data && r.data.pending_input)
          ? r.data.pending_input
          : ((r.data && r.data.project && r.data.project.pending_input) ? r.data.project.pending_input : null);
        renderPendingInput(pending);
        if (r.data && r.data.project) {
          state.project = r.data.project;
          applyProjectStateToUi(r.data.project);
        }
        const msgs = Array.isArray(r.data.messages) ? r.data.messages : [];
        if (msgs.length > 0) {
          renderChat(msgs);
        } else {
          await loadHistory();
        }
        setMessageInput('', {persist: false});
        await loadRunRuntime();
      }

      async function sendPendingForm(sendNow) {
        const patch = collectPendingPatch();
        if (!patch || Object.keys(patch).length < 1) {
          renderJsonOut({status: 'fail', error: 'pending form has no values'});
          return;
        }
        setMessageInput(JSON.stringify(patch, null, 2));
        if (sendNow) {
          await sendChat(false);
        }
      }

      async function previewArtifact() {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const artifact = document.getElementById('artifact_name').value;
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/artifact/${encodeURIComponent(artifact)}?max_chars=20000`);
        renderJsonOut(r.data);
      }

      async function showTimeline() {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/timeline?sort=duration_desc`);
        renderJsonOut(r.data);
      }

      async function validateTask() {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/validate`);
        renderJsonOut(r.data);
      }

      async function compareTasks() {
        const tid = taskId();
        const other = String(document.getElementById('compare_other_task_id').value || '').trim();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        if (!other) {
          renderJsonOut({status: 'fail', error: 'missing other_task_id'});
          return;
        }
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/compare?other_task_id=${encodeURIComponent(other)}`);
        renderJsonOut(r.data);
      }

      async function compareSelectedArtifact() {
        const tid = taskId();
        const other = String(document.getElementById('compare_other_task_id').value || '').trim();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        if (!other) {
          renderJsonOut({status: 'fail', error: 'missing other_task_id'});
          return;
        }
        const artifact = String(document.getElementById('artifact_name').value || 'decision_summary').trim();
        const r = await apiGet(
          `/api/task/${encodeURIComponent(tid)}/artifact-diff?other_task_id=${encodeURIComponent(other)}&artifact=${encodeURIComponent(artifact)}`
        );
        renderJsonOut(r.data);
      }

      function sendWebSearchHint() {
        const topk = String(document.getElementById('web_topk').value || '5').trim();
        const msg = [
          "请先做web search证据收集，再进入后续设计流程。",
          `建议参数: {"web_search_enabled": true, "web_topk": ${topk || "5"}}`,
          "请输出来源链接和时间范围。"
        ].join("\n");
        setMessageInput(msg);
      }

      async function attachPath() {
        const pid = selectedProjectId();
        const p = (document.getElementById('attachment_path').value || '').trim();
        if (!p) {
          renderJsonOut({status: 'fail', error: 'empty attachment_path'});
          return;
        }
        const r = await apiPost(`/api/projects/${encodeURIComponent(pid)}/upload-ref`, {
          path: p,
          label: 'manual_path',
          kind: 'path',
        });
        renderJsonOut(r.data);
        await loadHistory();
      }

      async function setCandidateDataFromPath() {
        const p = (document.getElementById('attachment_path').value || '').trim();
        if (!p) {
          renderJsonOut({status: 'fail', error: 'empty attachment_path'});
          return;
        }
        setMessageInput(JSON.stringify({candidate_data: p}, null, 2));
      }

      async function uploadFileRef() {
        const pid = selectedProjectId();
        const fileInput = document.getElementById('attachment_file');
        if (!fileInput.files || fileInput.files.length < 1) {
          renderJsonOut({status: 'fail', error: 'no file selected'});
          return;
        }
        const form = new FormData();
        form.append('file', fileInput.files[0]);
        form.append('label', 'browser_upload');
        const resp = await fetch(`/api/projects/${encodeURIComponent(pid)}/upload-ref`, {
          method: 'POST',
          body: form,
        });
        const data = await resp.json();
        renderJsonOut(data);
        if (data && data.attachment && data.attachment.path) {
          document.getElementById('attachment_path').value = String(data.attachment.path);
        }
        await loadHistory();
      }

      async function runStepPanel() {
        const op = String(document.getElementById('step_operation').value || '').trim();
        const argsText = String(document.getElementById('step_args_json').value || '').trim();
        let args = {};
        if (argsText) {
          try {
            const parsed = JSON.parse(argsText);
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
              args = parsed;
            } else {
              renderJsonOut({status: 'fail', error: 'step args must be JSON object'});
              return;
            }
          } catch (e) {
            renderJsonOut({status: 'fail', error: `invalid step args json: ${String(e)}`});
            return;
          }
        }
        setMessageInput(JSON.stringify({operation: op, args: args}, null, 2));
        await sendChat(false);
      }

      function bindComposerShortcuts() {
        const input = document.getElementById('message_input');
        if (!input) return;
        input.addEventListener('input', () => {
          persistMessageDraft();
        });
        input.addEventListener('keydown', (evt) => {
          if ((evt.ctrlKey || evt.metaKey) && evt.key === 'Enter') {
            evt.preventDefault();
            sendChat(false);
          }
        });
      }

      function applyStepArgsTemplate(forceOverwrite) {
        const op = String(document.getElementById('step_operation').value || '').trim();
        const area = document.getElementById('step_args_json');
        if (!area) return;
        const current = String(area.value || '').trim();
        if (!forceOverwrite && current && current !== '{}') {
          return;
        }
        const tpl = stepArgsTemplates[op] || {};
        area.value = JSON.stringify(tpl, null, 2);
      }

      async function retryCurrentTask() {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const r = await apiPost('/api/resume', {
          task_id: tid,
          planner_provider: document.getElementById('planner').value,
          catalog_path: document.getElementById('catalog').value,
        });
        renderJsonOut(r.data);
        const status = String((r.data && r.data.status) || 'unknown');
        renderEvents([{stage: 'resume', status: status}]);
        await loadRunRuntime();
      }

      async function retryFailedStep() {
        await retryFailedStepInternal(false);
      }

      async function previewRetryFailedStep() {
        await retryFailedStepInternal(true);
      }

      function parseRetryArgsOptional() {
        const txt = String(document.getElementById('retry_failed_args_json').value || '').trim();
        if (!txt) return {ok: true, args: null};
        try {
          const payload = JSON.parse(txt);
          if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
            return {ok: true, args: payload};
          }
          return {ok: false, error: 'retry args must be JSON object'};
        } catch (e) {
          return {ok: false, error: `invalid retry args json: ${String(e)}`};
        }
      }

      async function retryFailedStepInternal(dryRun) {
        const tid = taskId();
        if (!tid || tid === '-') {
          renderJsonOut({status: 'fail', error: 'no current_task_id'});
          return;
        }
        const parsed = parseRetryArgsOptional();
        if (!parsed.ok) {
          renderJsonOut({status: 'fail', error: parsed.error});
          return;
        }
        const body = {
          catalog_path: document.getElementById('catalog').value,
          dry_run: Boolean(dryRun),
        };
        const failedToolName = selectedRetryFailedToolName();
        if (failedToolName) {
          body.failed_tool_name = failedToolName;
        }
        if (parsed.args && Object.keys(parsed.args).length > 0) {
          body.args = parsed.args;
        }
        const r = await apiPost(`/api/task/${encodeURIComponent(tid)}/retry-failed-step`, {
          ...body,
        });
        renderJsonOut(r.data);
        const status = String((r.data && r.data.status) || 'unknown');
        const op = String((r.data && r.data.retry_operation) || '');
        const stage = dryRun ? 'preview_retry_failed_step' : 'retry_failed_step';
        renderEvents([{stage: stage, status: status, operation: op || undefined}]);
        if (!dryRun) {
          await loadRunRuntime();
        }
      }

      async function loadRunRuntime() {
        const tid = taskId();
        if (!tid || tid === '-') {
          document.getElementById('runtime_box').textContent = 'runtime: no active task';
          document.getElementById('runtime_stage_text').textContent = 'stage: -';
          renderRuntimeProgress(null);
          renderTimelineGroups(null);
          return;
        }
        const [summaryResp, timelineResp] = await Promise.all([
          apiGet(`/api/task/${encodeURIComponent(tid)}/summary`),
          apiGet(`/api/task/${encodeURIComponent(tid)}/timeline`),
        ]);
        const s = summaryResp.data || {};
        const tl = timelineResp.data || {};
        const lines = [];
        lines.push(`task_id: ${tid}`);
        lines.push(`summary_status: ${String(s.status || '-')}`);
        const exec = (s.execution_summary && typeof s.execution_summary === 'object') ? s.execution_summary : {};
        lines.push(`execution_status: ${String(exec.status || '-')}`);
        lines.push(`record_count: ${String(exec.record_count || 0)}`);
        const totalMs = tl.total_duration_ms;
        if (typeof totalMs === 'number') {
          lines.push(`duration_sec: ${(totalMs / 1000).toFixed(2)}`);
        }
        const text = lines.join(' | ');
        document.getElementById('runtime_box').textContent = text;
        renderRuntimeStage(s, tl);
        renderRuntimeProgress(tl.summary || null);
        renderTimelineGroups(tl);
      }

      async function boot() {
        applyStepArgsTemplate(true);
        bindComposerShortcuts();
        bindWorkspaceUrlNavigation();
        const urlProjectId = readProjectIdFromUrl();
        if (urlProjectId) {
          document.getElementById('project_id').value = urlProjectId;
        }
        await refreshProjects();
        await saveProject();
        renderPromptHistory(currentProjectKey());
        await loadRunRuntime();
        refreshWorkspaceHud();
      }

      boot();
    </script>
  </body>
</html>
"""


def _resolve_catalog(catalog_path: str) -> Path:
    p = Path(str(catalog_path or DEFAULT_CATALOG))
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _run_cli_with_json_payload(
    *,
    cli_base_args: List[str],
    payload: Dict[str, Any],
    payload_filename: str,
    payload_arg_name: str,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        payload_path = td_path / payload_filename
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        cmd = [
            os.environ.get("PYTHON", "python3"),
            "-m",
            "oled_agent.cli",
            *cli_base_args,
            payload_arg_name,
            str(payload_path),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        stdout_text = str(cp.stdout or "").strip()
        parsed: Any = None
        if stdout_text:
            try:
                parsed = json.loads(stdout_text)
            except json.JSONDecodeError:
                parsed = None
        if cp.returncode != 0:
            return {
                "status": "fail",
                "returncode": cp.returncode,
                "command": cmd,
                "stdout": cp.stdout,
                "stderr": cp.stderr,
                "result": parsed,
            }
        return {
            "status": "pass",
            "returncode": cp.returncode,
            "command": cmd,
            "result": parsed if parsed is not None else {"raw_stdout": cp.stdout},
            "stderr": cp.stderr,
        }


def _run_cli_command(*, cli_args: List[str], ok_returncodes: Optional[List[int]] = None) -> Dict[str, Any]:
    cmd = [
        os.environ.get("PYTHON", "python3"),
        "-m",
        "oled_agent.cli",
        *cli_args,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    raw = str(cp.stdout or "").strip()
    parsed: Any = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    allowed = set(ok_returncodes or [0])
    return {
        "status": "pass" if cp.returncode in allowed else "fail",
        "returncode": cp.returncode,
        "command": cmd,
        "result": parsed if parsed is not None else {"raw_stdout": cp.stdout},
        "stderr": cp.stderr,
    }


def _run_agent_run_json(*, payload: Dict[str, Any], planner_provider: str, catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_with_json_payload(
        cli_base_args=[
            "agent-run-json",
            "--workspace-root",
            str(REPO_ROOT),
            "--catalog",
            str(catalog),
            "--planner-provider",
            str(planner_provider or "rule_based_v1"),
        ],
        payload=payload,
        payload_filename="request.json",
        payload_arg_name="--request-json",
    )


def _run_agent_step_json(*, payload: Dict[str, Any], catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_with_json_payload(
        cli_base_args=[
            "agent-run-step-json",
            "--workspace-root",
            str(REPO_ROOT),
            "--catalog",
            str(catalog),
        ],
        payload=payload,
        payload_filename="step_request.json",
        payload_arg_name="--step-request-json",
    )


def _run_agent_intake(*, task_id: str, request_text: str, web_topk: int, enable_web_search: bool = True) -> Dict[str, Any]:
    cli_args = [
        "agent-intake",
        "--workspace-root",
        str(REPO_ROOT),
        "--task-id",
        task_id,
        "--request",
        request_text,
        "--web-topk",
        str(max(1, int(web_topk))),
    ]
    if not enable_web_search:
        cli_args.append("--disable-web-search")
    return _run_cli_command(
        cli_args=cli_args,
        ok_returncodes=[0, 2],
    )


def _run_agent_approve(*, task_json_path: Path, planner_provider: str, catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_command(
        cli_args=[
            "agent-approve",
            "--workspace-root",
            str(REPO_ROOT),
            "--task-json",
            str(task_json_path),
            "--planner-provider",
            str(planner_provider or "rule_based_v1"),
            "--catalog",
            str(catalog),
        ],
        ok_returncodes=[0, 2],
    )


def _run_agent_resume(*, task_id: str, planner_provider: str, catalog_path: str) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    return _run_cli_command(
        cli_args=[
            "agent-resume",
            "--workspace-root",
            str(REPO_ROOT),
            "--task-id",
            task_id,
            "--planner-provider",
            str(planner_provider or "rule_based_v1"),
            "--catalog",
            str(catalog),
        ],
        ok_returncodes=[0],
    )


def _task_artifact_path(task_id: str, filename: str) -> Path:
    return (REPO_ROOT / "runs" / "agent" / task_id / filename).resolve()


def _task_artifact_paths(task_id: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for name, rel in ARTIFACT_NAME_TO_FILE.items():
        out[name] = _task_artifact_path(task_id, rel)
    return out


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tool_name_to_retry_operation(tool_name: str) -> Optional[str]:
    name = str(tool_name or "").strip()
    mapping = {
        "search_dataset": "retrieve_candidate_data",
        "retrieve_candidate_data": "retrieve_candidate_data",
        "clean_dataset": "clean_dataset",
        "prepare_train_data": "prepare_train_data",
        "train_predictor": "train_predictor",
        "generate_candidates": "generate_candidates",
        "score_candidates": "score_candidates",
        "filter_and_rank": "filter_and_rank",
        "make_report": "make_report",
    }
    op = mapping.get(name)
    if op not in STEP_OPERATIONS:
        return None
    return op


def _load_task_payload_for_retry(task_id: str) -> Optional[Dict[str, Any]]:
    run_dir = (REPO_ROOT / "runs" / "agent" / task_id).resolve()
    task_json = run_dir / "task.json"
    draft_json = run_dir / "task.draft.json"
    req_task_json = run_dir / "request_from_task.json"
    legacy_req_json = run_dir / "request.json"
    for p in (task_json, draft_json):
        payload = _load_json_if_exists(p)
        if isinstance(payload, dict):
            return payload
    req_payload = _load_json_if_exists(req_task_json)
    if isinstance(req_payload, dict):
        try:
            return legacy_request_to_task_v2(req_payload)
        except Exception:
            return None
    legacy_req = _load_json_if_exists(legacy_req_json)
    if isinstance(legacy_req, dict):
        try:
            return legacy_request_to_task_v2(legacy_req)
        except Exception:
            return None
    return None


def _build_retry_args(
    *,
    operation: str,
    task_payload: Dict[str, Any],
    tool_state: Dict[str, Any],
    failed_record_args: Dict[str, Any],
) -> Dict[str, Any]:
    # Prefer the original failed args for deterministic replay.
    if isinstance(failed_record_args, dict) and failed_record_args:
        return dict(failed_record_args)

    candidate_data = str(task_payload.get("candidate_data") or "").strip()
    train_data = str(task_payload.get("train_data") or "").strip()
    n_structures = int(task_payload.get("n_structures") or 10)
    if operation == "retrieve_candidate_data":
        return {"candidate_data": candidate_data}
    if operation == "clean_dataset":
        input_csv = str(tool_state.get("candidate_csv") or candidate_data).strip()
        return {"input_csv": input_csv} if input_csv else {}
    if operation == "prepare_train_data":
        return {"train_data": train_data} if train_data else {}
    if operation == "generate_candidates":
        args: Dict[str, Any] = {"max_candidates": max(1, n_structures)}
        if candidate_data:
            args["input_csv"] = candidate_data
        return args
    if operation == "score_candidates":
        input_csv = str(tool_state.get("generated_csv") or tool_state.get("candidate_csv") or "").strip()
        return {"input_csv": input_csv} if input_csv else {}
    if operation == "filter_and_rank":
        return {"topn": min(10, max(1, n_structures))}
    return {}


def _latest_failed_record(execution_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    records = execution_payload.get("records")
    if not isinstance(records, list):
        return None
    for rec in reversed(records):
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "") != "success":
            return rec
    return None


def _latest_failed_record_by_name(execution_payload: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    name = str(tool_name or "").strip()
    if not name:
        return _latest_failed_record(execution_payload)
    records = execution_payload.get("records")
    if not isinstance(records, list):
        return None
    for rec in reversed(records):
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "") == "success":
            continue
        if str(rec.get("name") or "").strip() == name:
            return rec
    return None


def _task_updated_epoch_ms(run_dir: Path) -> int:
    latest = run_dir.stat().st_mtime if run_dir.exists() else 0.0
    try:
        for p in run_dir.rglob("*"):
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            if mt > latest:
                latest = mt
    except Exception:
        pass
    return int(latest * 1000)


def _task_list_item(task_id: str, run_dir: Path) -> Dict[str, Any]:
    execution = _load_json_if_exists(run_dir / "execution.json")
    task_state = _load_json_if_exists(run_dir / "task_state.json")
    records = execution.get("records", []) if isinstance(execution, dict) and isinstance(execution.get("records"), list) else []
    failed_n = 0
    for rec in records:
        if isinstance(rec, dict) and str(rec.get("status") or "") != "success":
            failed_n += 1
    updated_ms = _task_updated_epoch_ms(run_dir)
    updated_at = datetime.fromtimestamp(updated_ms / 1000.0).isoformat(timespec="seconds")
    return {
        "task_id": task_id,
        "run_dir": str(run_dir),
        "updated_epoch_ms": updated_ms,
        "updated_at": updated_at,
        "execution_status": str(execution.get("status") or "") if isinstance(execution, dict) else "",
        "record_count": len(records),
        "failed_step_count": failed_n,
        "task_state_status": str(task_state.get("status") or "") if isinstance(task_state, dict) else "",
    }


def _experiment_row_from_trace(trace: Dict[str, Any], trace_path: Path) -> Dict[str, Any]:
    model_choice = trace.get("model_choice") if isinstance(trace.get("model_choice"), dict) else {}
    execution_summary = trace.get("execution_summary") if isinstance(trace.get("execution_summary"), dict) else {}
    source_artifacts = trace.get("source_artifacts") if isinstance(trace.get("source_artifacts"), dict) else {}
    candidate = source_artifacts.get("candidate_csv") if isinstance(source_artifacts.get("candidate_csv"), dict) else {}
    scored = source_artifacts.get("scored_csv") if isinstance(source_artifacts.get("scored_csv"), dict) else {}
    return {
        "task_id": str(trace.get("task_id") or ""),
        "run_label": str(trace.get("run_label") or ""),
        "generated_at": str(trace.get("generated_at") or ""),
        "execution_mode": str(trace.get("execution_mode") or ""),
        "status": str(execution_summary.get("status") or ""),
        "record_count": int(execution_summary.get("record_count") or 0),
        "failed_count": int(execution_summary.get("failed_count") or 0),
        "adapters": execution_summary.get("adapters", []) if isinstance(execution_summary.get("adapters"), list) else [],
        "predictor_id": str(model_choice.get("predictor_id") or ""),
        "generator_id": str(model_choice.get("generator_id") or ""),
        "candidate_csv_exists": bool(candidate.get("exists")),
        "scored_csv_exists": bool(scored.get("exists")),
        "trace_path": str(trace_path),
    }


def _safe_filter_token(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _preview_payload(payload: Any, *, artifact_name: str) -> Any:
    if artifact_name == "execution" and isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return {
                "task_id": payload.get("task_id"),
                "status": payload.get("status"),
                "started_at": payload.get("started_at"),
                "ended_at": payload.get("ended_at"),
                "record_count": len(records),
                "records_head": records[:8],
            }
    if artifact_name == "web_evidence" and isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            lite = dict(payload)
            lite["results"] = results[:8]
            lite["result_count"] = len(results)
            return lite
    if artifact_name == "experiment_trace" and isinstance(payload, dict):
        return {
            "schema_version": payload.get("schema_version", ""),
            "task_id": payload.get("task_id", ""),
            "run_label": payload.get("run_label", ""),
            "execution_mode": payload.get("execution_mode", ""),
            "model_choice": payload.get("model_choice", {}),
            "execution_summary": payload.get("execution_summary", {}),
            "source_artifacts": payload.get("source_artifacts", {}),
        }
    return payload


def _artifact_preview(*, artifact_name: str, path: Path, max_chars: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "artifact": artifact_name,
            "path": str(path),
            "exists": False,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    text_preview = text if not truncated else text[:max_chars]
    payload = None
    parse_error = ""
    try:
        payload = json.loads(text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "pass",
        "artifact": artifact_name,
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "truncated": truncated,
        "text_preview": text_preview,
        "json_preview": _preview_payload(payload, artifact_name=artifact_name) if payload is not None else None,
        "json_parse_error": parse_error,
    }


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _duration_ms(started_at: Any, ended_at: Any) -> Optional[int]:
    started = _parse_iso_datetime(started_at)
    ended = _parse_iso_datetime(ended_at)
    if started is None or ended is None:
        return None
    return int((ended - started).total_seconds() * 1000)


def _timeline_result_summary(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ["status", "adapter", "count", "rows", "output_csv", "final_output", "report_path"]:
        if key in result:
            out[key] = result.get(key)
    if "models" in result and isinstance(result.get("models"), list):
        out["models_count"] = len(result.get("models", []))
    if "results" in result and isinstance(result.get("results"), list):
        out["results_count"] = len(result.get("results", []))
    if "topn" in result:
        out["topn"] = result.get("topn")
    return out


def _filter_timeline_events(*, events: List[Dict[str, Any]], tool_filter: str, status_filter: str) -> List[Dict[str, Any]]:
    out = list(events)
    tf = str(tool_filter or "").strip().lower()
    sf = str(status_filter or "all").strip().lower()
    if tf:
        out = [e for e in out if tf in str(e.get("name") or "").lower()]
    if sf == "failed":
        out = [e for e in out if bool(e.get("is_failed"))]
    elif sf == "success":
        out = [e for e in out if not bool(e.get("is_failed"))]
    return out


def _sort_timeline_events(*, events: List[Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
    key = str(sort_key or "original").strip().lower()
    out = list(events)
    if key == "duration_desc":
        out.sort(key=lambda e: int(e.get("duration_ms") or -1), reverse=True)
        return out
    if key == "duration_asc":
        out.sort(key=lambda e: int(e.get("duration_ms") or 10**15))
        return out
    if key == "name_asc":
        out.sort(key=lambda e: str(e.get("name") or ""))
        return out
    return out


def _timeline_line(event: Dict[str, Any]) -> str:
    idx = int(event.get("index") or 0)
    name = str(event.get("name") or "")
    status = str(event.get("status") or "")
    dur = event.get("duration_ms")
    dur_text = f"{dur}ms" if isinstance(dur, int) and dur >= 0 else "n/a"
    adapter = str(event.get("adapter") or "")
    marker = "[FAIL]" if bool(event.get("is_failed")) else "[PASS]"
    if adapter:
        return f"{idx:02d} {marker} {name} status={status} duration={dur_text} adapter={adapter}"
    return f"{idx:02d} {marker} {name} status={status} duration={dur_text}"


def _events_from_execution(execution: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = execution.get("records", []) if isinstance(execution.get("records"), list) else []
    events: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue
        result = rec.get("result")
        event: Dict[str, Any] = {
            "index": idx,
            "name": str(rec.get("name") or ""),
            "status": str(rec.get("status") or ""),
            "started_at": rec.get("started_at"),
            "ended_at": rec.get("ended_at"),
            "duration_ms": _duration_ms(rec.get("started_at"), rec.get("ended_at")),
            "error": str(rec.get("error") or ""),
            "result_summary": _timeline_result_summary(result),
            "is_failed": str(rec.get("status") or "") != "success",
            "args": rec.get("args") if isinstance(rec.get("args"), dict) else {},
        }
        if isinstance(result, dict) and result.get("adapter"):
            event["adapter"] = result.get("adapter")
        event["highlight"] = "fail" if bool(event.get("is_failed")) else "normal"
        events.append(event)
    return events


def _timeline_groups_recent_tasks(*, limit: int) -> Dict[str, Any]:
    runs_root = (REPO_ROOT / "runs" / "agent").resolve()
    if not runs_root.exists():
        return {
            "status": "pass",
            "scope": "recent_tasks",
            "task_count": 0,
            "total_steps": 0,
            "running_items": [],
            "completed_items": [],
            "failed_items": [],
            "tasks": [],
        }

    rows: List[Dict[str, Any]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        tid = str(child.name or "").strip()
        if not _is_safe_task_id(tid):
            continue
        updated_ms = _task_updated_epoch_ms(child)
        rows.append({"task_id": tid, "run_dir": child, "updated_ms": updated_ms})
    rows.sort(key=lambda item: int(item.get("updated_ms") or 0), reverse=True)
    selected = rows[: max(1, min(limit, 50))]

    running_items: List[Dict[str, Any]] = []
    completed_items: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []
    task_ids: List[str] = []
    for item in selected:
        tid = str(item.get("task_id") or "")
        run_dir = item.get("run_dir")
        if not tid or not isinstance(run_dir, Path):
            continue
        task_ids.append(tid)
        execution = _load_json_if_exists(run_dir / "execution.json")
        if not isinstance(execution, dict):
            continue
        events = _events_from_execution(execution)
        for ev in events:
            enriched = dict(ev)
            enriched["task_id"] = tid
            name = str(enriched.get("name") or "")
            status = str(enriched.get("status") or "")
            if name:
                enriched["name"] = f"{tid}:{name}"
            if str(status).lower() == "running":
                running_items.append(enriched)
            elif bool(enriched.get("is_failed")):
                failed_items.append(enriched)
            else:
                completed_items.append(enriched)

    return {
        "status": "pass",
        "scope": "recent_tasks",
        "task_count": len(task_ids),
        "total_steps": len(running_items) + len(completed_items) + len(failed_items),
        "running_items": running_items,
        "completed_items": completed_items,
        "failed_items": failed_items,
        "tasks": task_ids,
    }


def _task_compare_summary(task_id: str) -> Dict[str, Any]:
    run_dir = (REPO_ROOT / "runs" / "agent" / task_id).resolve()
    by_name = _task_artifact_paths(task_id)
    artifact_exists = {name: path.exists() for name, path in by_name.items()}
    artifact_missing = [name for name, ok in artifact_exists.items() if not ok]

    execution = _load_json_if_exists(by_name["execution"])
    records = execution.get("records", []) if isinstance(execution, dict) and isinstance(execution.get("records"), list) else []
    execution_status = str(execution.get("status") or "") if isinstance(execution, dict) else ""
    total_duration_ms = _duration_ms(execution.get("started_at"), execution.get("ended_at")) if isinstance(execution, dict) else None

    failed_steps: List[str] = []
    adapters: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "") != "success":
            failed_steps.append(str(rec.get("name") or ""))
        result = rec.get("result")
        if isinstance(result, dict):
            adapter = str(result.get("adapter") or "").strip()
            if adapter:
                adapters.add(adapter)

    web_evidence = _load_json_if_exists(by_name["web_evidence"])
    web_results = web_evidence.get("results", []) if isinstance(web_evidence, dict) and isinstance(web_evidence.get("results"), list) else []

    return {
        "task_id": task_id,
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "artifacts_exists": artifact_exists,
        "artifacts_missing": artifact_missing,
        "execution_status": execution_status,
        "record_count": len(records),
        "failed_step_count": len(failed_steps),
        "failed_steps": failed_steps,
        "adapters": sorted(adapters),
        "total_duration_ms": total_duration_ms,
        "web_evidence_count": len(web_results),
    }


def _task_compare_diff(primary: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    primary_adapters = set(primary.get("adapters", [])) if isinstance(primary.get("adapters"), list) else set()
    other_adapters = set(other.get("adapters", [])) if isinstance(other.get("adapters"), list) else set()
    primary_failed = set(primary.get("failed_steps", [])) if isinstance(primary.get("failed_steps"), list) else set()
    other_failed = set(other.get("failed_steps", [])) if isinstance(other.get("failed_steps"), list) else set()

    primary_rc = int(primary.get("record_count") or 0)
    other_rc = int(other.get("record_count") or 0)
    primary_fail = int(primary.get("failed_step_count") or 0)
    other_fail = int(other.get("failed_step_count") or 0)
    primary_web = int(primary.get("web_evidence_count") or 0)
    other_web = int(other.get("web_evidence_count") or 0)
    primary_dur = primary.get("total_duration_ms")
    other_dur = other.get("total_duration_ms")

    duration_delta: Optional[int] = None
    if isinstance(primary_dur, int) and isinstance(other_dur, int):
        duration_delta = primary_dur - other_dur

    return {
        "execution_status_changed": str(primary.get("execution_status") or "") != str(other.get("execution_status") or ""),
        "record_count_delta": primary_rc - other_rc,
        "failed_step_count_delta": primary_fail - other_fail,
        "web_evidence_count_delta": primary_web - other_web,
        "total_duration_ms_delta": duration_delta,
        "adapters_only_in_primary": sorted(primary_adapters - other_adapters),
        "adapters_only_in_other": sorted(other_adapters - primary_adapters),
        "failed_steps_only_in_primary": sorted(primary_failed - other_failed),
        "failed_steps_only_in_other": sorted(other_failed - primary_failed),
    }


def _task_compare_lines(primary: Dict[str, Any], other: Dict[str, Any], diff: Dict[str, Any]) -> List[str]:
    p_tid = str(primary.get("task_id") or "")
    o_tid = str(other.get("task_id") or "")
    out = [
        f"record_count {p_tid}={int(primary.get('record_count') or 0)} vs {o_tid}={int(other.get('record_count') or 0)} delta={int(diff.get('record_count_delta') or 0)}",
        f"failed_steps {p_tid}={int(primary.get('failed_step_count') or 0)} vs {o_tid}={int(other.get('failed_step_count') or 0)} delta={int(diff.get('failed_step_count_delta') or 0)}",
        f"web_evidence {p_tid}={int(primary.get('web_evidence_count') or 0)} vs {o_tid}={int(other.get('web_evidence_count') or 0)} delta={int(diff.get('web_evidence_count_delta') or 0)}",
    ]
    if isinstance(diff.get("total_duration_ms_delta"), int):
        out.append(f"duration_ms delta={int(diff.get('total_duration_ms_delta') or 0)}")
    if bool(diff.get("execution_status_changed")):
        out.append(
            f"execution_status changed: {p_tid}={str(primary.get('execution_status') or '')} vs {o_tid}={str(other.get('execution_status') or '')}"
        )
    if isinstance(diff.get("adapters_only_in_primary"), list) and diff.get("adapters_only_in_primary"):
        out.append(f"adapters only in {p_tid}: {', '.join(diff.get('adapters_only_in_primary', []))}")
    if isinstance(diff.get("adapters_only_in_other"), list) and diff.get("adapters_only_in_other"):
        out.append(f"adapters only in {o_tid}: {', '.join(diff.get('adapters_only_in_other', []))}")
    return out


def _normalize_diff_leaf(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 240:
            return value[:240] + "...(truncated)"
        return value
    if isinstance(value, dict):
        return f"<dict:{len(value)}>"
    if isinstance(value, list):
        return f"<list:{len(value)}>"
    return str(value)


def _flatten_json_paths(
    payload: Any,
    *,
    out: Dict[str, Any],
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 60,
    max_nodes: int = 1800,
) -> None:
    if len(out) >= max_nodes:
        return
    if depth >= max_depth:
        key = prefix or "$"
        out[key] = "<max_depth>"
        return
    if isinstance(payload, dict):
        if not payload:
            out[prefix or "$"] = "<empty_dict>"
            return
        keys = sorted(payload.keys(), key=lambda x: str(x))
        for idx, key in enumerate(keys):
            if idx >= max_items:
                out[(prefix + "." if prefix else "") + "__truncated_keys__"] = len(keys) - max_items
                return
            k = str(key)
            next_prefix = f"{prefix}.{k}" if prefix else k
            _flatten_json_paths(
                payload.get(key),
                out=out,
                prefix=next_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_nodes=max_nodes,
            )
            if len(out) >= max_nodes:
                return
        return
    if isinstance(payload, list):
        if not payload:
            out[prefix or "$"] = "<empty_list>"
            return
        limit = min(len(payload), max_items)
        for idx in range(limit):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_json_paths(
                payload[idx],
                out=out,
                prefix=next_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_nodes=max_nodes,
            )
            if len(out) >= max_nodes:
                return
        if len(payload) > limit:
            out[(prefix or "$") + ".__truncated_items__"] = len(payload) - limit
        return
    out[prefix or "$"] = _normalize_diff_leaf(payload)


def _artifact_diff_payload(primary_payload: Any, other_payload: Any) -> Dict[str, Any]:
    primary_flat: Dict[str, Any] = {}
    other_flat: Dict[str, Any] = {}
    _flatten_json_paths(primary_payload, out=primary_flat)
    _flatten_json_paths(other_payload, out=other_flat)

    primary_keys = set(primary_flat.keys())
    other_keys = set(other_flat.keys())
    only_primary = sorted(primary_keys - other_keys)
    only_other = sorted(other_keys - primary_keys)
    common = sorted(primary_keys & other_keys)
    changed: List[Dict[str, Any]] = []
    for key in common:
        if primary_flat.get(key) != other_flat.get(key):
            changed.append({"path": key, "primary": primary_flat.get(key), "other": other_flat.get(key)})

    return {
        "only_in_primary_count": len(only_primary),
        "only_in_other_count": len(only_other),
        "changed_count": len(changed),
        "only_in_primary": only_primary[:200],
        "only_in_other": only_other[:200],
        "changed": changed[:300],
        "primary_paths_total": len(primary_flat),
        "other_paths_total": len(other_flat),
    }


def _is_safe_task_id(task_id: str) -> bool:
    tid = str(task_id or "").strip()
    if not tid:
        return False
    if not TASK_ID_PATTERN.fullmatch(tid):
        return False
    if ".." in tid or "/" in tid or "\\" in tid:
        return False
    return True


def _is_safe_project_id(project_id: str) -> bool:
    return _is_safe_task_id(project_id)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ui_projects_root() -> Path:
    p = (REPO_ROOT / PROJECTS_DIR_REL).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ui_uploads_root(project_id: str) -> Path:
    p = (REPO_ROOT / UPLOADS_DIR_REL / project_id).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _project_file_path(project_id: str) -> Path:
    return (_ui_projects_root() / f"{project_id}.json").resolve()


def _resolve_optional_path(raw_path: Any) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _normalize_project_options(raw: Any) -> Dict[str, Any]:
    options = raw if isinstance(raw, dict) else {}
    planner = str(options.get("planner_provider") or "rule_based_v1").strip() or "rule_based_v1"
    catalog = str(options.get("catalog_path") or DEFAULT_CATALOG).strip() or DEFAULT_CATALOG
    web_enabled = bool(options.get("web_search_enabled", True))
    web_topk = _as_int(options.get("web_topk"), 5)
    web_topk = max(1, min(web_topk, 20))
    return {
        "planner_provider": planner,
        "catalog_path": catalog,
        "web_search_enabled": web_enabled,
        "web_topk": web_topk,
    }


def _new_project_state(project_id: str, *, title: str = "", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "title": str(title or project_id).strip() or project_id,
        "created_at": now,
        "updated_at": now,
        "options": _normalize_project_options(options or {}),
        "current_task_id": "",
        "task_draft_path": "",
        "task_json_path": "",
        "request_path": "",
        "last_runtime": {},
        "pending_input": {},
        "attachments": [],
        "messages": [],
    }


def _project_summary(project: Dict[str, Any]) -> Dict[str, Any]:
    pid = str(project.get("project_id") or "")
    messages = project.get("messages")
    attachments = project.get("attachments")
    return {
        "project_id": pid,
        "title": str(project.get("title") or ""),
        "created_at": str(project.get("created_at") or ""),
        "updated_at": str(project.get("updated_at") or ""),
        "options": project.get("options") if isinstance(project.get("options"), dict) else {},
        "current_task_id": str(project.get("current_task_id") or ""),
        "task_draft_path": str(project.get("task_draft_path") or ""),
        "task_json_path": str(project.get("task_json_path") or ""),
        "request_path": str(project.get("request_path") or ""),
        "last_runtime": project.get("last_runtime") if isinstance(project.get("last_runtime"), dict) else {},
        "pending_input": project.get("pending_input") if isinstance(project.get("pending_input"), dict) else {},
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "attachment_count": len(attachments) if isinstance(attachments, list) else 0,
        "project_path": str(_project_file_path(pid)) if pid else "",
        "runtime_health": _project_runtime_health(project),
    }


def _project_runtime_health(project: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(project.get("current_task_id") or "").strip()
    if not _is_safe_task_id(task_id):
        return {
            "status": "none",
            "reason": "no_current_task",
            "record_count": 0,
            "success_steps": 0,
            "failed_steps": 0,
            "latest_failed_step": "",
        }
    run_dir = (REPO_ROOT / "runs" / "agent" / task_id).resolve()
    execution = _load_json_if_exists(run_dir / "execution.json")
    if not isinstance(execution, dict):
        return {
            "status": "none",
            "reason": "missing_execution",
            "task_id": task_id,
            "record_count": 0,
            "success_steps": 0,
            "failed_steps": 0,
            "latest_failed_step": "",
        }
    records = execution.get("records") if isinstance(execution.get("records"), list) else []
    success_steps = 0
    failed_steps = 0
    latest_failed_step = ""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "") == "success":
            success_steps += 1
        else:
            failed_steps += 1
            latest_failed_step = str(rec.get("name") or latest_failed_step)
    return {
        "status": str(execution.get("status") or "unknown"),
        "task_id": task_id,
        "record_count": len(records),
        "success_steps": success_steps,
        "failed_steps": failed_steps,
        "latest_failed_step": latest_failed_step,
    }


def _load_project_state(project_id: str) -> Optional[Dict[str, Any]]:
    p = _project_file_path(project_id)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload["project_id"] = str(payload.get("project_id") or project_id)
    payload["options"] = _normalize_project_options(payload.get("options"))
    if not isinstance(payload.get("attachments"), list):
        payload["attachments"] = []
    if not isinstance(payload.get("messages"), list):
        payload["messages"] = []
    if not isinstance(payload.get("pending_input"), dict):
        payload["pending_input"] = {}
    return payload


def _save_project_state(project: Dict[str, Any]) -> Dict[str, Any]:
    project = dict(project)
    project_id = str(project.get("project_id") or "").strip()
    if not _is_safe_project_id(project_id):
        raise ValueError("invalid project_id")
    if not str(project.get("created_at") or "").strip():
        project["created_at"] = _now_iso()
    project["updated_at"] = _now_iso()
    project["options"] = _normalize_project_options(project.get("options"))
    messages = project.get("messages")
    if not isinstance(messages, list):
        messages = []
    if len(messages) > MAX_PROJECT_HISTORY:
        messages = messages[-MAX_PROJECT_HISTORY:]
    project["messages"] = messages
    if not isinstance(project.get("pending_input"), dict):
        project["pending_input"] = {}
    attachments = project.get("attachments")
    if not isinstance(attachments, list):
        attachments = []
    if len(attachments) > 120:
        attachments = attachments[-120:]
    project["attachments"] = attachments
    p = _project_file_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return project


def _normalize_import_project(raw: Dict[str, Any], *, project_id: str) -> Dict[str, Any]:
    base = _new_project_state(project_id, title=str(raw.get("title") or project_id), options=raw.get("options") if isinstance(raw.get("options"), dict) else {})
    if str(raw.get("created_at") or "").strip():
        base["created_at"] = str(raw.get("created_at"))
    for key in ("current_task_id", "task_draft_path", "task_json_path", "request_path"):
        base[key] = str(raw.get(key) or "")
    if isinstance(raw.get("last_runtime"), dict):
        base["last_runtime"] = dict(raw.get("last_runtime") or {})
    if isinstance(raw.get("pending_input"), dict):
        base["pending_input"] = dict(raw.get("pending_input") or {})
    if isinstance(raw.get("attachments"), list):
        cleaned_attachments: List[Dict[str, Any]] = []
        for item in raw.get("attachments") or []:
            if not isinstance(item, dict):
                continue
            cleaned_attachments.append(
                {
                    "id": str(item.get("id") or str(uuid.uuid4())),
                    "kind": str(item.get("kind") or "path_ref"),
                    "label": str(item.get("label") or ""),
                    "name": str(item.get("name") or ""),
                    "path": str(item.get("path") or ""),
                    "created_at": str(item.get("created_at") or _now_iso()),
                }
            )
        base["attachments"] = cleaned_attachments[-120:]
    if isinstance(raw.get("messages"), list):
        cleaned_messages: List[Dict[str, Any]] = []
        for item in raw.get("messages") or []:
            if not isinstance(item, dict):
                continue
            cleaned_messages.append(
                {
                    "id": str(item.get("id") or str(uuid.uuid4())),
                    "role": str(item.get("role") or "system"),
                    "kind": str(item.get("kind") or "text"),
                    "content": str(item.get("content") or ""),
                    "created_at": str(item.get("created_at") or _now_iso()),
                    "meta": item.get("meta") if isinstance(item.get("meta"), dict) else {},
                }
            )
        base["messages"] = cleaned_messages[-MAX_PROJECT_HISTORY:]
    return base


def _append_message(
    project: Dict[str, Any],
    *,
    role: str,
    content: str,
    kind: str = "text",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    messages = project.get("messages")
    if not isinstance(messages, list):
        messages = []
        project["messages"] = messages
    messages.append(
        {
            "id": str(uuid.uuid4()),
            "role": str(role or "system"),
            "kind": str(kind or "text"),
            "content": str(content or "").strip(),
            "created_at": _now_iso(),
            "meta": meta if isinstance(meta, dict) else {},
        }
    )
    if len(messages) > MAX_PROJECT_HISTORY:
        project["messages"] = messages[-MAX_PROJECT_HISTORY:]


def _recent_messages(project: Dict[str, Any], *, limit: int = 160) -> List[Dict[str, Any]]:
    messages = project.get("messages")
    if not isinstance(messages, list):
        return []
    cap = max(1, min(int(limit), MAX_PROJECT_HISTORY))
    out: List[Dict[str, Any]] = []
    for item in messages[-cap:]:
        if isinstance(item, dict):
            out.append(item)
    return out


def _create_task_id(project_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{project_id}_{stamp}"
    if len(base) <= 128 and _is_safe_task_id(base):
        return base
    short = f"{project_id[:48]}_{stamp}"
    if _is_safe_task_id(short):
        return short
    return f"task_{stamp}"


def _parse_message_patch(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload

    # Allow plain "candidate_data=/abs/path.csv" or bare csv path.
    m = re.search(r"(?:candidate_data|候选数据)\s*[:=]\s*([^\s]+)", raw, flags=re.IGNORECASE)
    if m:
        return {"candidate_data": str(m.group(1)).strip()}
    raw_l = raw.lower()
    if ".csv" in raw_l and (raw.startswith("/") or raw.startswith("./") or raw.startswith("../")):
        return {"candidate_data": raw}
    return {}


def _parse_step_intent(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None

    # JSON inline style:
    # {"operation":"clean_dataset","args":{"input_csv":"/abs/path.csv"}}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and str(payload.get("operation") or "").strip() in STEP_OPERATIONS:
        op = str(payload.get("operation") or "").strip()
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        task = payload.get("task") if isinstance(payload.get("task"), dict) else None
        return {"operation": op, "args": args, "task": task}

    # Slash command style:
    # /step clean_dataset {"input_csv":"/abs/path.csv"}
    # /step {"operation":"clean_dataset","args":{"input_csv":"..."}}
    if not raw.startswith("/step"):
        return None
    rest = raw[len("/step") :].strip()
    if not rest:
        return {"operation": "", "args": {}, "task": None, "error": "missing operation"}
    if rest.startswith("{"):
        try:
            obj = json.loads(rest)
        except json.JSONDecodeError as exc:
            return {"operation": "", "args": {}, "task": None, "error": f"invalid json after /step: {exc}"}
        if not isinstance(obj, dict):
            return {"operation": "", "args": {}, "task": None, "error": "step json must be object"}
        op = str(obj.get("operation") or "").strip()
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        task = obj.get("task") if isinstance(obj.get("task"), dict) else None
        return {"operation": op, "args": args, "task": task}

    parts = rest.split(" ", 1)
    op = str(parts[0] or "").strip()
    args: Dict[str, Any] = {}
    if len(parts) > 1 and str(parts[1] or "").strip():
        try:
            parsed_args = json.loads(parts[1])
            if isinstance(parsed_args, dict):
                args = parsed_args
            else:
                return {"operation": op, "args": {}, "task": None, "error": "step args must be json object"}
        except json.JSONDecodeError as exc:
            return {"operation": op, "args": {}, "task": None, "error": f"invalid step args json: {exc}"}
    return {"operation": op, "args": args, "task": None}


def _load_project_task_payload(project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("task_json_path", "task_draft_path"):
        p = _resolve_optional_path(project.get(key))
        if p is None or not p.exists():
            continue
        payload = _load_json_path(p)
        if isinstance(payload, dict):
            return payload
    return None


def _merge_task_draft(draft: Dict[str, Any], patch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    updated: List[str] = []
    out = dict(draft)
    for key in (
        "property",
        "range",
        "n_structures",
        "train_data",
        "candidate_data",
        "prediction_model",
        "execution_mode",
        "operation",
        "request_text",
    ):
        if key not in patch:
            continue
        value = patch.get(key)
        if key == "n_structures":
            try:
                value_i = int(value)
            except Exception:
                continue
            if value_i < 1:
                continue
            out[key] = value_i
        else:
            out[key] = value
        updated.append(key)

    if isinstance(patch.get("constraints"), dict):
        constraints = out.get("constraints") if isinstance(out.get("constraints"), dict) else {}
        constraints = dict(constraints)
        constraints.update(patch.get("constraints") or {})
        out["constraints"] = constraints
        updated.append("constraints")

    model_keys = ("model_preferences", "model_choice")
    for mk in model_keys:
        if isinstance(patch.get(mk), dict):
            model = out.get("model_preferences") if isinstance(out.get("model_preferences"), dict) else {}
            model = dict(model)
            model.update(patch.get(mk) or {})
            out["model_preferences"] = model
            updated.append("model_preferences")
            break

    missing, questions = compute_missing_questions(out)
    out["missing_fields"] = missing
    out["questions"] = questions
    out["status"] = "need_user_input" if missing else "draft"
    return out, updated


def _load_json_path(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _assistant_need_input_text(missing_fields: Any, questions: Any) -> str:
    missing = [str(x) for x in (missing_fields if isinstance(missing_fields, list) else []) if str(x).strip()]
    qs = [str(x) for x in (questions if isinstance(questions, list) else []) if str(x).strip()]
    lines = ["还需要补充信息后才能继续执行。"]
    if missing:
        lines.append(f"missing_fields: {', '.join(missing)}")
    for idx, q in enumerate(qs, start=1):
        lines.append(f"{idx}. {q}")
    lines.append('可直接回复 JSON，例如: {"candidate_data": "/abs/path/candidates.csv"}')
    return "\n".join(lines)


def _assistant_cli_fail_text(stage: str, payload: Dict[str, Any]) -> str:
    stderr = str(payload.get("stderr") or "").strip()
    rc = payload.get("returncode")
    msg = f"{stage} 执行失败，returncode={rc}。"
    if stderr:
        msg += f"\nstderr: {stderr[:800]}"
    return msg


def _pending_input_payload(*, stage: str, missing_fields: Any, questions: Any, task_draft_path: Any = "") -> Dict[str, Any]:
    missing = [str(x) for x in (missing_fields if isinstance(missing_fields, list) else []) if str(x).strip()]
    qs = [str(x) for x in (questions if isinstance(questions, list) else []) if str(x).strip()]
    return {
        "stage": str(stage or ""),
        "missing_fields": missing,
        "questions": qs,
        "task_draft_path": str(task_draft_path or ""),
    }


def _chat_run_single_step(*, project: Dict[str, Any], step_intent: Dict[str, Any], message: str) -> Dict[str, Any]:
    options = _normalize_project_options(project.get("options"))
    catalog = str(options.get("catalog_path") or DEFAULT_CATALOG)
    operation = str(step_intent.get("operation") or "").strip()
    if operation not in STEP_OPERATIONS:
        project["pending_input"] = {}
        _append_message(
            project,
            role="assistant",
            kind="assistant",
            content=f"无效 step operation: {operation or '(empty)'}。可选: {', '.join(STEP_OPERATIONS)}",
        )
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "step", "status": "fail", "reason": "invalid_operation"}],
        }

    if str(step_intent.get("error") or "").strip():
        project["pending_input"] = {}
        _append_message(
            project,
            role="assistant",
            kind="assistant",
            content=f"/step 解析失败: {step_intent.get('error')}",
        )
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "step", "status": "fail", "reason": "parse_error"}],
        }

    task_payload = step_intent.get("task") if isinstance(step_intent.get("task"), dict) else None
    if not isinstance(task_payload, dict):
        task_payload = _load_project_task_payload(project)
    if not isinstance(task_payload, dict):
        pending = _pending_input_payload(
            stage="step",
            missing_fields=["task_context"],
            questions=["请先提供任务目标触发 intake，或在 step JSON 中附带完整 task 对象。"],
        )
        project["pending_input"] = pending
        _append_message(
            project,
            role="assistant",
            kind="assistant",
            content=(
                "当前项目没有可用 task 草案/已批准任务。"
                "\n请先发送一个目标请求触发 intake，或在 /step JSON 里附带完整 task 字段。"
            ),
        )
        project = _save_project_state(project)
        return {
            "status": "need_user_input",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "step", "status": "need_user_input"}],
            "pending_input": pending,
        }

    task = dict(task_payload)
    task["execution_mode"] = "single_step"
    task["operation"] = operation
    args = step_intent.get("args") if isinstance(step_intent.get("args"), dict) else {}

    step_request = {"task": task, "operation": operation, "args": args}
    started_at = datetime.now()
    step_result = _run_agent_step_json(payload=step_request, catalog_path=catalog)
    elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
    if step_result.get("status") != "pass":
        project["pending_input"] = {}
        _append_message(
            project,
            role="assistant",
            kind="assistant",
            content=_assistant_cli_fail_text("agent-run-step-json", step_result),
        )
        project["last_runtime"] = {
            "status": "failed",
            "duration_ms": elapsed_ms,
            "operation": operation,
            "updated_at": _now_iso(),
        }
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "step", "status": "fail", "operation": operation}],
            "step_result": step_result.get("result"),
        }

    sr = step_result.get("result") if isinstance(step_result.get("result"), dict) else {}
    status_text = str(sr.get("status") or "unknown")
    project["pending_input"] = {}
    project["current_task_id"] = str(sr.get("task_id") or project.get("current_task_id") or "")
    task_path = _resolve_optional_path(sr.get("task_path"))
    if task_path is not None:
        project["task_json_path"] = str(task_path)
    _append_message(
        project,
        role="assistant",
        kind="assistant",
        content=(
            f"单步执行完成: operation={operation}, status={status_text}"
            f"\nrun_label={sr.get('run_label', '')}"
            f"\nexecution_path={sr.get('execution_path', '')}"
        ),
        meta={"step_result": sr, "source_message": message},
    )
    project["last_runtime"] = {
        "status": status_text,
        "duration_ms": elapsed_ms,
        "operation": operation,
        "run_label": str(sr.get("run_label") or ""),
        "updated_at": _now_iso(),
    }
    project = _save_project_state(project)
    return {
        "status": "pass",
        "project": _project_summary(project),
        "messages": _recent_messages(project),
        "events": [{"stage": "step", "status": status_text, "operation": operation}],
        "step_result": sr,
    }


def _chat_run_pipeline(*, project: Dict[str, Any], message: str, new_task: bool) -> Dict[str, Any]:
    options = _normalize_project_options(project.get("options"))
    planner = str(options.get("planner_provider") or "rule_based_v1")
    catalog = str(options.get("catalog_path") or DEFAULT_CATALOG)
    web_enabled = bool(options.get("web_search_enabled", True))
    web_topk = int(options.get("web_topk") or 5)

    if new_task:
        project["current_task_id"] = ""
        project["task_draft_path"] = ""
        project["task_json_path"] = ""
        project["request_path"] = ""
        project["last_runtime"] = {}
        project["pending_input"] = {}

    if message:
        _append_message(project, role="user", content=message, kind="chat")

    step_intent = _parse_step_intent(message)
    if isinstance(step_intent, dict):
        return _chat_run_single_step(project=project, step_intent=step_intent, message=message)

    task_id = str(project.get("current_task_id") or "").strip()
    if not task_id:
        task_id = _create_task_id(str(project.get("project_id") or "task"))
        project["current_task_id"] = task_id

    draft_path = _resolve_optional_path(project.get("task_draft_path"))
    patch = _parse_message_patch(message)

    # Stage 1: intake (if no draft yet)
    if draft_path is None or not draft_path.exists():
        if not str(message or "").strip():
            project["pending_input"] = {}
            _append_message(project, role="assistant", content="请先输入任务目标，然后我会自动做 intake。", kind="assistant")
            project = _save_project_state(project)
            return {"status": "pass", "project": _project_summary(project), "messages": _recent_messages(project), "events": []}
        intake = _run_agent_intake(task_id=task_id, request_text=message, web_topk=web_topk, enable_web_search=web_enabled)
        intake_result = intake.get("result") if isinstance(intake.get("result"), dict) else {}
        draft_path = _resolve_optional_path(intake_result.get("task_draft_path"))
        if draft_path is not None:
            project["task_draft_path"] = str(draft_path)
        project["current_task_id"] = str(intake_result.get("task_id") or task_id)

        if intake.get("status") != "pass":
            project["pending_input"] = {}
            _append_message(project, role="assistant", content=_assistant_cli_fail_text("agent-intake", intake), kind="assistant")
            project = _save_project_state(project)
            return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "intake", "status": "fail"}]}

        if str(intake_result.get("status") or "") == "need_user_input":
            pending = _pending_input_payload(
                stage="intake",
                missing_fields=intake_result.get("missing_fields"),
                questions=intake_result.get("questions"),
                task_draft_path=intake_result.get("task_draft_path"),
            )
            project["pending_input"] = pending
            _append_message(
                project,
                role="assistant",
                content=_assistant_need_input_text(intake_result.get("missing_fields"), intake_result.get("questions")),
                kind="assistant",
            )
            project = _save_project_state(project)
            return {
                "status": "need_user_input",
                "project": _project_summary(project),
                "messages": _recent_messages(project),
                "events": [{"stage": "intake", "status": "need_user_input"}],
                "pending_input": pending,
            }

    if draft_path is None or not draft_path.exists():
        project["pending_input"] = {}
        _append_message(project, role="assistant", content="intake 未生成可用 task.draft.json。", kind="assistant")
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "intake", "status": "fail"}]}

    draft = _load_json_path(draft_path)
    if not isinstance(draft, dict):
        project["pending_input"] = {}
        _append_message(project, role="assistant", content=f"draft 读取失败: {draft_path}", kind="assistant")
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "draft_read", "status": "fail"}]}

    if patch:
        draft, updated_fields = _merge_task_draft(draft, patch)
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if updated_fields:
            _append_message(
                project,
                role="system",
                content=f"已更新 task 草案字段: {', '.join(updated_fields)}",
                kind="task_patch",
                meta={"updated_fields": updated_fields},
            )

    started_at = datetime.now()
    approve = _run_agent_approve(task_json_path=draft_path, planner_provider=planner, catalog_path=catalog)
    approve_result = approve.get("result") if isinstance(approve.get("result"), dict) else {}
    if approve.get("status") != "pass":
        project["pending_input"] = {}
        _append_message(project, role="assistant", content=_assistant_cli_fail_text("agent-approve", approve), kind="assistant")
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "approve", "status": "fail"}]}

    approve_status = str(approve_result.get("status") or "")
    if approve_status == "need_user_input":
        pending = _pending_input_payload(
            stage="approve",
            missing_fields=approve_result.get("missing_fields"),
            questions=approve_result.get("questions"),
            task_draft_path=str(draft_path),
        )
        project["pending_input"] = pending
        _append_message(
            project,
            role="assistant",
            content=_assistant_need_input_text(approve_result.get("missing_fields"), approve_result.get("questions")),
            kind="assistant",
        )
        project = _save_project_state(project)
        return {
            "status": "need_user_input",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "approve", "status": "need_user_input"}],
            "pending_input": pending,
        }
    if approve_status != "approved":
        project["pending_input"] = {}
        _append_message(project, role="assistant", content=f"agent-approve 返回未知状态: {approve_status}", kind="assistant")
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "approve", "status": "fail"}]}

    request_path = _resolve_optional_path(approve_result.get("request_path"))
    task_json_path = _resolve_optional_path(approve_result.get("task_path"))
    if request_path is not None:
        project["request_path"] = str(request_path)
    if task_json_path is not None:
        project["task_json_path"] = str(task_json_path)
    project["current_task_id"] = str(approve_result.get("task_id") or project.get("current_task_id") or "")

    if request_path is None or not request_path.exists():
        project["pending_input"] = {}
        _append_message(project, role="assistant", content="approved 后未找到 request_path，无法执行 agent-run-json。", kind="assistant")
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "approve", "status": "fail"}]}

    request_payload = _load_json_path(request_path)
    if not isinstance(request_payload, dict):
        project["pending_input"] = {}
        _append_message(project, role="assistant", content=f"request_from_task.json 解析失败: {request_path}", kind="assistant")
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "request_load", "status": "fail"}]}

    run_result = _run_agent_run_json(payload=request_payload, planner_provider=planner, catalog_path=catalog)
    elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)

    if run_result.get("status") != "pass":
        project["pending_input"] = {}
        _append_message(project, role="assistant", content=_assistant_cli_fail_text("agent-run-json", run_result), kind="assistant")
        project["last_runtime"] = {"status": "failed", "duration_ms": elapsed_ms, "updated_at": _now_iso()}
        project = _save_project_state(project)
        return {"status": "fail", "project": _project_summary(project), "messages": _recent_messages(project), "events": [{"stage": "run", "status": "fail"}]}

    rr = run_result.get("result") if isinstance(run_result.get("result"), dict) else {}
    run_label = str(rr.get("run_label") or "")
    result_dir = str(rr.get("result_dir") or "")
    status_text = str(rr.get("status") or "unknown")
    project["pending_input"] = {}
    _append_message(
        project,
        role="assistant",
        content=f"任务执行完成: status={status_text}\nrun_label={run_label}\nresult_dir={result_dir}",
        kind="assistant",
        meta={"run_result": rr},
    )
    project["last_runtime"] = {
        "status": status_text,
        "duration_ms": elapsed_ms,
        "run_label": run_label,
        "result_dir": result_dir,
        "updated_at": _now_iso(),
    }
    project = _save_project_state(project)
    return {
        "status": "pass",
        "project": _project_summary(project),
        "messages": _recent_messages(project),
        "events": [{"stage": "run", "status": status_text}],
        "run_result": rr,
    }


@app.get("/")
def index() -> str:
    return render_template_string(HTML)


@app.get("/api/health")
def api_health():
    return jsonify({"status": "pass", "repo_root": str(REPO_ROOT)})


@app.get("/api/projects")
def api_projects():
    limit = _as_int(request.args.get("limit"), 80)
    limit = max(1, min(limit, 300))
    root = _ui_projects_root()
    rows: List[Dict[str, Any]] = []
    for p in root.glob("*.json"):
        project_id = str(p.stem or "").strip()
        if not _is_safe_project_id(project_id):
            continue
        project = _load_project_state(project_id)
        if not isinstance(project, dict):
            continue
        rows.append(_project_summary(project))
    rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    limited = rows[:limit]
    return jsonify(
        {
            "status": "pass",
            "projects_root": str(root),
            "count": len(limited),
            "count_before_limit": len(rows),
            "limit": limit,
            "projects": limited,
        }
    )


@app.post("/api/projects")
def api_projects_upsert():
    body = request.get_json(silent=True) or {}
    project_id = str(body.get("project_id") or "").strip()
    title = str(body.get("title") or "").strip()
    options = body.get("options")
    if not project_id:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(project_id):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400

    project = _load_project_state(project_id)
    if not isinstance(project, dict):
        project = _new_project_state(project_id, title=title, options=options if isinstance(options, dict) else {})
    else:
        if title:
            project["title"] = title
        if isinstance(options, dict):
            merged = dict(project.get("options") or {})
            merged.update(options)
            project["options"] = merged
    project = _save_project_state(project)
    return jsonify({"status": "pass", "project": _project_summary(project), "messages": _recent_messages(project)})


@app.get("/api/projects/<project_id>/history")
def api_project_history(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    limit = _as_int(request.args.get("limit"), 180)
    limit = max(1, min(limit, MAX_PROJECT_HISTORY))
    project = _load_project_state(pid)
    if not isinstance(project, dict):
        return jsonify({"status": "missing", "error": "project_not_found", "project_id": pid}), 404
    return jsonify(
        {
            "status": "pass",
            "project": _project_summary(project),
            "messages": _recent_messages(project, limit=limit),
            "attachments": project.get("attachments") if isinstance(project.get("attachments"), list) else [],
        }
    )


@app.get("/api/projects/<project_id>/export")
def api_project_export(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    project = _load_project_state(pid)
    if not isinstance(project, dict):
        return jsonify({"status": "missing", "error": "project_not_found", "project_id": pid}), 404
    return jsonify({"status": "pass", "project": project, "project_summary": _project_summary(project)})


@app.post("/api/projects/import")
def api_project_import():
    body = request.get_json(silent=True) or {}
    raw_project = body.get("project")
    if not isinstance(raw_project, dict):
        return jsonify({"status": "fail", "error": "missing project object"}), 400
    target_id = str(body.get("project_id") or raw_project.get("project_id") or "").strip()
    if not target_id:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(target_id):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    override = bool(body.get("override"))
    existing = _load_project_state(target_id)
    if isinstance(existing, dict) and not override:
        return jsonify({"status": "fail", "error": "project_exists", "project_id": target_id}), 409
    normalized = _normalize_import_project(raw_project, project_id=target_id)
    normalized["project_id"] = target_id
    saved = _save_project_state(normalized)
    return jsonify({"status": "pass", "project": _project_summary(saved), "messages": _recent_messages(saved)})


@app.post("/api/projects/<project_id>/upload-ref")
def api_project_upload_ref(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400

    project = _load_project_state(pid)
    if not isinstance(project, dict):
        project = _new_project_state(pid, title=pid, options={})

    attachment: Dict[str, Any] = {}
    file_obj = request.files.get("file")
    if file_obj is not None and str(file_obj.filename or "").strip():
        base_name = Path(str(file_obj.filename or "")).name
        if not base_name:
            base_name = "upload.bin"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._") or "upload.bin"
        out_dir = _ui_uploads_root(pid)
        out_path = (out_dir / f"{stamp}_{safe_name}").resolve()
        file_obj.save(str(out_path))
        attachment = {
            "id": str(uuid.uuid4()),
            "kind": "uploaded_file",
            "label": str(request.form.get("label") or "upload").strip() or "upload",
            "name": base_name,
            "path": str(out_path),
            "created_at": _now_iso(),
        }
    else:
        body = request.get_json(silent=True) or {}
        path_text = str(body.get("path") or "").strip()
        if not path_text:
            return jsonify({"status": "fail", "error": "missing path or file"}), 400
        attachment = {
            "id": str(uuid.uuid4()),
            "kind": str(body.get("kind") or "path_ref").strip() or "path_ref",
            "label": str(body.get("label") or "path_ref").strip() or "path_ref",
            "name": Path(path_text).name,
            "path": path_text,
            "created_at": _now_iso(),
        }

    attachments = project.get("attachments")
    if not isinstance(attachments, list):
        attachments = []
    attachments.append(attachment)
    project["attachments"] = attachments[-120:]
    _append_message(
        project,
        role="system",
        content=f"附件已记录: {attachment.get('path')}",
        kind="attachment",
        meta={"attachment": attachment},
    )
    project = _save_project_state(project)
    return jsonify(
        {
            "status": "pass",
            "project": _project_summary(project),
            "attachment": attachment,
            "messages": _recent_messages(project),
        }
    )


@app.post("/api/chat/send")
def api_chat_send():
    body = request.get_json(silent=True) or {}
    project_id = str(body.get("project_id") or "").strip()
    message = str(body.get("message") or "").strip()
    new_task = bool(body.get("new_task"))
    options = body.get("options")

    if not project_id:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(project_id):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not message and not new_task:
        return jsonify({"status": "fail", "error": "missing message"}), 400

    project = _load_project_state(project_id)
    if not isinstance(project, dict):
        project = _new_project_state(project_id, title=project_id, options={})
    if isinstance(options, dict):
        merged_options = dict(project.get("options") or {})
        merged_options.update(options)
        project["options"] = merged_options

    out = _chat_run_pipeline(project=project, message=message, new_task=new_task)
    events_for_meta = out.get("events") if isinstance(out.get("events"), list) else []
    if events_for_meta:
        _append_message(
            project,
            role="system",
            kind="event_trace",
            content="Execution timeline updated.",
            meta={"events": events_for_meta},
        )
        project = _save_project_state(project)
        out["project"] = _project_summary(project)
        out["messages"] = _recent_messages(project)
    return jsonify(out)


@app.get("/api/tasks")
def api_tasks():
    limit = _as_int(request.args.get("limit"), 50)
    limit = max(1, min(limit, 200))
    prefix = str(request.args.get("prefix") or "").strip()
    if prefix and not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", prefix):
        return jsonify({"status": "fail", "error": "invalid prefix"}), 400
    runs_root = (REPO_ROOT / "runs" / "agent").resolve()
    if not runs_root.exists():
        return jsonify({"status": "pass", "tasks": [], "count": 0, "runs_root": str(runs_root)})

    items: List[Dict[str, Any]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        tid = str(child.name or "").strip()
        if not _is_safe_task_id(tid):
            continue
        if prefix and not tid.startswith(prefix):
            continue
        items.append(_task_list_item(tid, child))
    items.sort(key=lambda it: int(it.get("updated_epoch_ms") or 0), reverse=True)
    limited = items[:limit]
    return jsonify(
        {
            "status": "pass",
            "runs_root": str(runs_root),
            "count": len(limited),
            "count_before_limit": len(items),
            "limit": limit,
            "prefix": prefix,
            "tasks": limited,
        }
    )


@app.get("/api/experiments")
def api_experiments():
    limit = _as_int(request.args.get("limit"), 80)
    limit = max(1, min(limit, 500))
    prefix = str(request.args.get("prefix") or "").strip()
    predictor_id = str(request.args.get("predictor_id") or "").strip()
    generator_id = str(request.args.get("generator_id") or "").strip()
    status = str(request.args.get("status") or "").strip()
    execution_mode = str(request.args.get("execution_mode") or "").strip()
    for token in (prefix, predictor_id, generator_id):
        if token and not _safe_filter_token(token):
            return jsonify({"status": "fail", "error": "invalid filter token"}), 400
    if status and status not in {"success", "failed"}:
        return jsonify({"status": "fail", "error": "invalid status"}), 400
    if execution_mode and execution_mode not in {"full_pipeline", "single_step"}:
        return jsonify({"status": "fail", "error": "invalid execution_mode"}), 400

    runs_root = (REPO_ROOT / "runs" / "agent").resolve()
    if not runs_root.exists():
        return jsonify({"status": "pass", "experiments": [], "count": 0, "runs_root": str(runs_root)})

    rows: List[Dict[str, Any]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        task_id = str(child.name or "").strip()
        if not _is_safe_task_id(task_id):
            continue
        if prefix and not task_id.startswith(prefix):
            continue
        trace_path = child / "artifacts" / "experiment_trace.json"
        if not trace_path.exists():
            continue
        trace = _load_json_if_exists(trace_path)
        if not isinstance(trace, dict):
            continue
        row = _experiment_row_from_trace(trace, trace_path)
        if predictor_id and row.get("predictor_id") != predictor_id:
            continue
        if generator_id and row.get("generator_id") != generator_id:
            continue
        if status and row.get("status") != status:
            continue
        if execution_mode and row.get("execution_mode") != execution_mode:
            continue
        rows.append(row)
    rows.sort(key=lambda item: str(item.get("generated_at") or ""), reverse=True)
    limited = rows[:limit]
    return jsonify(
        {
            "status": "pass",
            "runs_root": str(runs_root),
            "count": len(limited),
            "count_before_limit": len(rows),
            "limit": limit,
            "filters": {
                "prefix": prefix,
                "predictor_id": predictor_id,
                "generator_id": generator_id,
                "status": status,
                "execution_mode": execution_mode,
            },
            "experiments": limited,
        }
    )


@app.get("/api/timeline-groups")
def api_timeline_groups():
    scope = str(request.args.get("scope") or "recent_tasks").strip()
    if scope not in {"recent_tasks"}:
        return jsonify({"status": "fail", "error": "invalid scope"}), 400
    limit = _as_int(request.args.get("limit"), 5)
    limit = max(1, min(limit, 50))
    out = _timeline_groups_recent_tasks(limit=limit)
    out["limit"] = limit
    return jsonify(out)


@app.post("/api/run")
def api_run():
    body = request.get_json(silent=True) or {}
    payload_text = str(body.get("payload_text") or "")
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return jsonify({"status": "fail", "error": f"invalid request json: {exc}"}), 400
    return jsonify(_run_agent_run_json(payload=payload, planner_provider=planner, catalog_path=catalog))


@app.post("/api/run-step")
def api_run_step():
    body = request.get_json(silent=True) or {}
    payload_text = str(body.get("payload_text") or "")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return jsonify({"status": "fail", "error": f"invalid step request json: {exc}"}), 400
    if not isinstance(payload, dict):
        return jsonify({"status": "fail", "error": "step request must be JSON object"}), 400
    return jsonify(_run_agent_step_json(payload=payload, catalog_path=catalog))


@app.post("/api/intake")
def api_intake():
    body = request.get_json(silent=True) or {}
    task_id = str(body.get("task_id") or "").strip()
    request_text = str(body.get("request_text") or "").strip()
    web_topk = int(body.get("web_topk") or 5)
    web_enabled = bool(body.get("web_search_enabled", True))
    if not task_id:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(task_id):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    if not request_text:
        return jsonify({"status": "fail", "error": "missing request_text"}), 400
    return jsonify(_run_agent_intake(task_id=task_id, request_text=request_text, web_topk=web_topk, enable_web_search=web_enabled))


@app.post("/api/approve")
def api_approve():
    body = request.get_json(silent=True) or {}
    task_json_path = str(body.get("task_json_path") or "").strip()
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    if not task_json_path:
        return jsonify({"status": "fail", "error": "missing task_json_path"}), 400
    task_path = Path(task_json_path)
    if not task_path.is_absolute():
        task_path = (REPO_ROOT / task_path).resolve()
    else:
        task_path = task_path.resolve()
    return jsonify(_run_agent_approve(task_json_path=task_path, planner_provider=planner, catalog_path=catalog))


@app.post("/api/resume")
def api_resume():
    body = request.get_json(silent=True) or {}
    task_id = str(body.get("task_id") or "").strip()
    planner = str(body.get("planner_provider") or "rule_based_v1")
    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    if not task_id:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(task_id):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    return jsonify(_run_agent_resume(task_id=task_id, planner_provider=planner, catalog_path=catalog))


@app.post("/api/task/<task_id>/retry-failed-step")
def api_task_retry_failed_step(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400

    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    if not run_dir.exists():
        return jsonify({"status": "missing", "task_id": tid, "error": "run_dir_missing"}), 404

    body = request.get_json(silent=True) or {}
    target_failed_tool_name = str(body.get("failed_tool_name") or "").strip()

    execution = _load_json_if_exists(run_dir / "execution.json")
    if not isinstance(execution, dict):
        return jsonify({"status": "fail", "task_id": tid, "error": "missing_or_invalid_execution"}), 200
    failed_rec = _latest_failed_record_by_name(execution, target_failed_tool_name)
    if not isinstance(failed_rec, dict):
        return jsonify({"status": "fail", "task_id": tid, "error": "no_failed_step"}), 200

    failed_tool_name = str(failed_rec.get("name") or "").strip()
    operation = _tool_name_to_retry_operation(failed_tool_name)
    if not operation:
        return jsonify(
            {
                "status": "fail",
                "task_id": tid,
                "error": "unsupported_failed_step_for_retry",
                "failed_tool_name": failed_tool_name,
            }
        ), 200

    task_payload = _load_task_payload_for_retry(tid)
    if not isinstance(task_payload, dict):
        return jsonify({"status": "fail", "task_id": tid, "error": "missing_task_payload_for_retry"}), 200

    catalog = str(body.get("catalog_path") or DEFAULT_CATALOG)
    dry_run = bool(body.get("dry_run"))
    override_args = body.get("args")
    if override_args is not None and not isinstance(override_args, dict):
        return jsonify({"status": "fail", "task_id": tid, "error": "args_must_be_object"}), 400
    tool_state = _load_json_if_exists(run_dir / "tool_state.json")
    if not isinstance(tool_state, dict):
        tool_state = {}
    failed_args = failed_rec.get("args") if isinstance(failed_rec.get("args"), dict) else {}
    retry_args = _build_retry_args(
        operation=operation,
        task_payload=task_payload,
        tool_state=tool_state,
        failed_record_args=failed_args,
    )
    if isinstance(override_args, dict):
        retry_args = dict(override_args)
    step_request = {
        "task": task_payload,
        "operation": operation,
        "args": retry_args,
    }
    out: Dict[str, Any]
    if dry_run:
        out = {"status": "pass", "mode": "dry_run"}
    else:
        out = _run_agent_step_json(payload=step_request, catalog_path=catalog)
    response: Dict[str, Any] = {
        "task_id": tid,
        "failed_tool_name": failed_tool_name,
        "retry_operation": operation,
        "retry_args": retry_args,
        "dry_run": dry_run,
        **out,
    }
    return jsonify(response)


@app.get("/api/task/<task_id>/summary")
def api_task_summary(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    by_name = _task_artifact_paths(tid)
    artifacts = {
        "plan_path": by_name["plan"],
        "execution_path": by_name["execution"],
        "tool_state_path": by_name["tool_state"],
        "decision_summary_path": by_name["decision_summary"],
        "task_state_path": by_name["task_state"],
        "web_evidence_path": by_name["web_evidence"],
        "experiment_trace_path": by_name["experiment_trace"],
    }
    files = {k: {"path": str(v), "exists": v.exists()} for k, v in artifacts.items()}
    execution = _load_json_if_exists(artifacts["execution_path"])
    task_state = _load_json_if_exists(artifacts["task_state_path"])
    decision = _load_json_if_exists(artifacts["decision_summary_path"])
    web_evidence = _load_json_if_exists(artifacts["web_evidence_path"])
    experiment_trace = _load_json_if_exists(artifacts["experiment_trace_path"])
    return jsonify(
        {
            "status": "pass" if run_dir.exists() else "missing",
            "task_id": tid,
            "run_dir": str(run_dir),
            "run_dir_exists": run_dir.exists(),
            "artifacts": files,
            "execution_summary": {
                "record_count": len(execution.get("records", [])) if isinstance(execution, dict) else 0,
                "status": execution.get("status") if isinstance(execution, dict) else None,
            },
            "task_state": task_state if isinstance(task_state, dict) else {},
            "decision_summary": decision if isinstance(decision, dict) else {},
            "web_evidence_preview": (
                web_evidence.get("results", [])[:5]
                if isinstance(web_evidence, dict) and isinstance(web_evidence.get("results"), list)
                else []
            ),
            "experiment_trace_preview": (
                _preview_payload(experiment_trace, artifact_name="experiment_trace")
                if isinstance(experiment_trace, dict)
                else {}
            ),
        }
    )


@app.get("/api/task/<task_id>/artifact/<artifact_name>")
def api_task_artifact(task_id: str, artifact_name: str):
    tid = str(task_id or "").strip()
    name = str(artifact_name or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    if name not in ARTIFACT_NAME_TO_FILE:
        return jsonify({"status": "fail", "error": "invalid artifact_name"}), 400
    max_chars = max(2000, min(_as_int(request.args.get("max_chars"), 12000), 200000))
    paths = _task_artifact_paths(tid)
    payload = _artifact_preview(artifact_name=name, path=paths[name], max_chars=max_chars)
    return jsonify(payload)


@app.get("/api/task/<task_id>/timeline")
def api_task_timeline(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    tool_filter = str(request.args.get("tool") or "").strip()
    status_filter = str(request.args.get("status_filter") or "all").strip().lower()
    sort = str(request.args.get("sort") or "original").strip().lower()
    if status_filter not in {"all", "failed", "success"}:
        return jsonify({"status": "fail", "error": "invalid status_filter"}), 400
    if sort not in {"original", "duration_desc", "duration_asc", "name_asc"}:
        return jsonify({"status": "fail", "error": "invalid sort"}), 400
    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    if not run_dir.exists():
        return jsonify({"status": "missing", "task_id": tid, "error": "run_dir_missing"}), 200
    execution_path = _task_artifact_path(tid, "execution.json")
    if not execution_path.exists():
        return jsonify({"status": "fail", "task_id": tid, "error": "missing_execution_json"}), 200
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"status": "fail", "task_id": tid, "error": f"invalid_execution_json: {type(exc).__name__}: {exc}"}), 200

    records = execution.get("records", []) if isinstance(execution, dict) else []
    events: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue
        result = rec.get("result")
        event: Dict[str, Any] = {
            "index": idx,
            "name": str(rec.get("name") or ""),
            "args": rec.get("args") if isinstance(rec.get("args"), dict) else {},
            "status": str(rec.get("status") or ""),
            "started_at": rec.get("started_at"),
            "ended_at": rec.get("ended_at"),
            "duration_ms": _duration_ms(rec.get("started_at"), rec.get("ended_at")),
            "error": str(rec.get("error") or ""),
            "result_summary": _timeline_result_summary(result),
            "is_failed": str(rec.get("status") or "") != "success",
        }
        if isinstance(result, dict) and result.get("adapter"):
            event["adapter"] = result.get("adapter")
        event["highlight"] = "fail" if bool(event.get("is_failed")) else "normal"
        events.append(event)

    filtered = _filter_timeline_events(events=events, tool_filter=tool_filter, status_filter=status_filter)
    sorted_events = _sort_timeline_events(events=filtered, sort_key=sort)
    timeline_lines = [_timeline_line(e) for e in sorted_events]

    total_ms = _duration_ms(execution.get("started_at"), execution.get("ended_at")) if isinstance(execution, dict) else None
    success_n = sum(1 for e in sorted_events if not bool(e.get("is_failed")))
    fail_n = sum(1 for e in sorted_events if bool(e.get("is_failed")))
    return jsonify(
        {
            "status": "pass",
            "task_id": tid,
            "run_dir": str(run_dir),
            "execution_status": execution.get("status") if isinstance(execution, dict) else "",
            "started_at": execution.get("started_at") if isinstance(execution, dict) else "",
            "ended_at": execution.get("ended_at") if isinstance(execution, dict) else "",
            "total_duration_ms": total_ms,
            "summary": {
                "total_steps_before_filter": len(events),
                "total_steps": len(sorted_events),
                "success_steps": success_n,
                "failed_steps": fail_n,
                "tool_filter": tool_filter,
                "status_filter": status_filter,
                "sort": sort,
            },
            "events": sorted_events,
            "timeline_lines": timeline_lines,
        }
    )


@app.get("/api/task/<task_id>/compare")
def api_task_compare(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    other = str(request.args.get("other_task_id") or "").strip()
    if not other:
        return jsonify({"status": "fail", "error": "missing other_task_id"}), 400
    if not _is_safe_task_id(other):
        return jsonify({"status": "fail", "error": "invalid other_task_id"}), 400
    if other == tid:
        return jsonify({"status": "fail", "error": "other_task_id must differ from task_id"}), 400

    primary = _task_compare_summary(tid)
    other_summary = _task_compare_summary(other)
    diff = _task_compare_diff(primary, other_summary)
    warnings: List[str] = []
    if not bool(primary.get("run_dir_exists")):
        warnings.append("primary_run_dir_missing")
    if not bool(other_summary.get("run_dir_exists")):
        warnings.append("other_run_dir_missing")
    return jsonify(
        {
            "status": "pass" if not warnings else "partial",
            "task_id": tid,
            "other_task_id": other,
            "warnings": warnings,
            "primary": primary,
            "other": other_summary,
            "diff": diff,
            "compare_lines": _task_compare_lines(primary, other_summary, diff),
        }
    )


@app.get("/api/task/<task_id>/artifact-diff")
def api_task_artifact_diff(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    other = str(request.args.get("other_task_id") or "").strip()
    if not other:
        return jsonify({"status": "fail", "error": "missing other_task_id"}), 400
    if not _is_safe_task_id(other):
        return jsonify({"status": "fail", "error": "invalid other_task_id"}), 400
    if other == tid:
        return jsonify({"status": "fail", "error": "other_task_id must differ from task_id"}), 400
    artifact = str(request.args.get("artifact") or "decision_summary").strip()
    if artifact not in ARTIFACT_NAME_TO_FILE:
        return jsonify({"status": "fail", "error": "invalid artifact"}), 400

    primary_path = _task_artifact_paths(tid).get(artifact)
    other_path = _task_artifact_paths(other).get(artifact)
    if primary_path is None or other_path is None:
        return jsonify({"status": "fail", "error": "internal_artifact_resolution_error"}), 500
    if not primary_path.exists() or not other_path.exists():
        return jsonify(
            {
                "status": "missing",
                "task_id": tid,
                "other_task_id": other,
                "artifact": artifact,
                "primary_exists": primary_path.exists(),
                "other_exists": other_path.exists(),
                "primary_path": str(primary_path),
                "other_path": str(other_path),
            }
        ), 200

    try:
        primary_payload = json.loads(primary_path.read_text(encoding="utf-8"))
        other_payload = json.loads(other_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"status": "fail", "error": f"invalid_json: {type(exc).__name__}: {exc}"}), 200

    diff = _artifact_diff_payload(primary_payload, other_payload)
    return jsonify(
        {
            "status": "pass",
            "task_id": tid,
            "other_task_id": other,
            "artifact": artifact,
            "primary_path": str(primary_path),
            "other_path": str(other_path),
            "diff": diff,
        }
    )


@app.get("/api/task/<task_id>/validate")
def api_task_validate(task_id: str):
    tid = str(task_id or "").strip()
    if not tid:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(tid):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    run_dir = (REPO_ROOT / "runs" / "agent" / tid).resolve()
    if not run_dir.exists():
        return jsonify({"status": "missing", "task_id": tid, "error": "run_dir_missing"}), 200

    checks: List[Dict[str, str]] = []
    loaded: Dict[str, Any] = {}
    required = ["plan", "execution", "tool_state", "decision_summary", "task_state"]
    by_name = _task_artifact_paths(tid)

    for name in required:
        path = by_name[name]
        if not path.exists():
            checks.append({"name": name, "status": "fail", "message": f"missing file: {path}"})
            continue
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            checks.append({"name": name, "status": "pass", "message": "json parse ok"})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "message": f"json parse failed: {type(exc).__name__}: {exc}"})

    execution = loaded.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("records"), list) and len(execution.get("records", [])) > 0:
        checks.append({"name": "execution_records", "status": "pass", "message": "records list is non-empty"})
    else:
        checks.append({"name": "execution_records", "status": "fail", "message": "records list missing or empty"})

    decision = loaded.get("decision_summary")
    if isinstance(decision, dict):
        try:
            validate_decision_summary_payload(decision, REPO_ROOT)
            checks.append({"name": "decision_summary_schema", "status": "pass", "message": "schema valid"})
        except Exception as exc:
            checks.append({"name": "decision_summary_schema", "status": "fail", "message": str(exc)})

    task_state = loaded.get("task_state")
    if isinstance(task_state, dict):
        try:
            validate_task_state_payload(task_state, REPO_ROOT)
            checks.append({"name": "task_state_schema", "status": "pass", "message": "schema valid"})
        except Exception as exc:
            checks.append({"name": "task_state_schema", "status": "fail", "message": str(exc)})

    pass_n = sum(1 for c in checks if c.get("status") == "pass")
    fail_n = sum(1 for c in checks if c.get("status") == "fail")
    overall = "pass" if fail_n == 0 else "fail"
    return jsonify(
        {
            "status": overall,
            "task_id": tid,
            "run_dir": str(run_dir),
            "summary": {"pass": pass_n, "fail": fail_n},
            "checks": checks,
            "blocking_checks": [c.get("name") for c in checks if c.get("status") == "fail"],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)
