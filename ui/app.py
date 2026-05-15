from __future__ import annotations

import json
import csv
import io
import os
import re
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, Response, jsonify, render_template_string, request
from oled_agent.agent.task_v2 import compute_missing_questions, legacy_request_to_task_v2
from oled_agent.agent.request_contract import validate_decision_summary_payload, validate_task_state_payload


app = Flask(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "scripts/adapters/real_adapters_catalog.json"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PROJECTS_DIR_REL = Path("runs/ui_sessions/projects")
UPLOADS_DIR_REL = Path("runs/ui_sessions/uploads")
BATCH_EXPORTS_DIR_REL = Path("runs/ui_sessions/exports")
MAX_PROJECT_HISTORY = 400
MAX_MEMORY_NOTES_CHARS = 8000
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
      .project-board {
        margin-top: 10px;
        border: 1px solid #dbe4f2;
        border-radius: 10px;
        background: #f9fbff;
        padding: 8px;
      }
      .project-board h4 {
        margin: 0 0 8px 0;
        font-size: 0.78rem;
        color: #4b5a73;
      }
      .project-board-controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-bottom: 8px;
      }
      .project-board-controls input,
      .project-board-controls select {
        margin-top: 0;
        padding: 6px 7px;
        font-size: 0.74rem;
      }
      .project-board-controls button {
        margin-top: 0;
        padding: 6px 8px;
        font-size: 0.72rem;
      }
      .project-board-quick {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 8px;
      }
      .project-board-quick button {
        margin-top: 0;
        padding: 5px 8px;
        font-size: 0.71rem;
      }
      .project-board-quick label {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-top: 0;
        font-size: 0.72rem;
        color: #3f4d64;
      }
      .project-board-quick select {
        margin-top: 0;
        padding: 5px 7px;
        font-size: 0.72rem;
        width: auto;
      }
      .project-board-quick .project-batch-limit {
        width: 72px;
        margin-top: 0;
        padding: 5px 7px;
        font-size: 0.72rem;
      }
      .project-board-summary {
        margin: 6px 0 8px 0;
        padding: 6px 7px;
        border: 1px solid #d8e1f1;
        border-radius: 8px;
        background: #ffffff;
        color: #334155;
        font-size: 0.72rem;
      }
      .project-batch-history-controls {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 6px;
        margin: 0 0 6px 0;
      }
      .project-batch-history-controls select,
      .project-batch-history-controls input {
        margin-top: 0;
        padding: 5px 7px;
        font-size: 0.72rem;
      }
      .project-batch-history-list {
        display: grid;
        grid-template-columns: 1fr;
        gap: 6px;
        margin: 0 0 8px 0;
      }
      .project-batch-history-item {
        border: 1px solid #d6e2f3;
        border-radius: 8px;
        padding: 6px;
        background: #ffffff;
        font-size: 0.72rem;
        color: #304058;
      }
      .project-batch-history-item button {
        margin-top: 4px;
        padding: 4px 7px;
        font-size: 0.7rem;
      }
      .project-session-section {
        border: 1px dashed #d4dfef;
        border-radius: 8px;
        padding: 6px;
        background: #ffffff;
      }
      .project-session-section-head {
        font-size: 0.72rem;
        color: #4b5a73;
        margin-bottom: 6px;
      }
      .project-session-list {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
      }
      .project-session-item {
        border: 1px solid #d9e3f4;
        border-radius: 8px;
        background: #fff;
        padding: 7px;
        font-size: 0.76rem;
      }
      .project-session-item.active {
        border-color: #9bbcff;
        box-shadow: inset 0 0 0 1px #cddfff;
      }
      .project-session-item.pinned {
        border-color: #ffd27a;
        box-shadow: inset 0 0 0 1px #ffe6b1;
      }
      .project-session-head {
        display: flex;
        justify-content: space-between;
        gap: 6px;
        margin-bottom: 4px;
      }
      .project-session-title {
        font-weight: 700;
        color: #1f3559;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .project-session-id {
        color: #6a7280;
        font-size: 0.7rem;
      }
      .project-session-meta {
        color: #4b5568;
        line-height: 1.45;
      }
      .project-session-status {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-top: 4px;
        padding: 2px 7px;
        border-radius: 999px;
        border: 1px solid #d5deec;
        color: #3e4b61;
        font-size: 0.7rem;
        background: #f2f6fc;
      }
      .project-session-status.fail {
        color: #8a2f2f;
        border-color: #eab9b9;
        background: #fff0f0;
      }
      .project-session-status.pass {
        color: #116b5c;
        border-color: #bfe8df;
        background: #ecfaf6;
      }
      .project-session-failed {
        margin-top: 4px;
        color: #8a2f2f;
        font-size: 0.72rem;
      }
      .project-session-error {
        margin-top: 3px;
        color: #7a3f00;
        font-size: 0.72rem;
      }
      .project-session-runtime {
        margin-top: 4px;
        color: #36506f;
        font-size: 0.72rem;
      }
      .project-session-progress {
        margin-top: 5px;
        width: 100%;
        height: 7px;
        border-radius: 999px;
        border: 1px solid #d3deef;
        background: #eef3fb;
        overflow: hidden;
      }
      .project-session-progress-bar {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #4c8bf5, #2563eb);
      }
      .project-session-actions {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
      }
      .project-session-actions button {
        margin-top: 6px;
        padding: 4px 8px;
        font-size: 0.72rem;
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
        <div class=\"project-board\">
          <h4>Workspace Sessions</h4>
          <div class=\"project-board-controls\">
            <input id=\"session_filter_text\" placeholder=\"filter: project/task\" />
            <select id=\"session_filter_health\">
              <option value=\"all\">health: all</option>
              <option value=\"failed\">health: failed</option>
              <option value=\"success\">health: success</option>
              <option value=\"none\">health: none</option>
            </select>
            <select id=\"session_sort_mode\">
              <option value=\"updated_desc\">sort: updated desc</option>
              <option value=\"failed_desc\">sort: failed desc</option>
              <option value=\"success_ratio_asc\">sort: success ratio asc</option>
              <option value=\"priority_desc\">sort: priority desc</option>
            </select>
            <button type=\"button\" onclick=\"applySessionBoardControls()\">Apply</button>
          </div>
          <div class=\"project-board-quick\">
            <button type=\"button\" onclick=\"quickFilterFailedOnly()\">Failed Only</button>
            <button type=\"button\" onclick=\"quickFilterByHealth('failed')\">Failed Count</button>
            <button type=\"button\" onclick=\"quickFilterByHealth('success')\">Success Count</button>
            <button type=\"button\" onclick=\"quickFilterByHealth('none')\">None Count</button>
            <button type=\"button\" onclick=\"quickSortPriority()\">Priority First</button>
            <button type=\"button\" onclick=\"openTopPrioritySession()\">Open Top Priority</button>
            <button type=\"button\" onclick=\"openNextFailedSession()\">Open Next Failed</button>
            <button type=\"button\" onclick=\"togglePinnedOnly()\">Pinned Only</button>
            <button type=\"button\" onclick=\"toggleSessionBoardGroupedView()\">Status Groups</button>
            <button type=\"button\" onclick=\"batchShowProjectSummary()\">Batch Summary</button>
            <button type=\"button\" onclick=\"batchValidateProjectTask()\">Batch Validate</button>
            <button type=\"button\" onclick=\"batchRetryFailedProjectStep()\">Batch Retry Failed</button>
            <button type=\"button\" onclick=\"exportSessionBoardBatchResult()\">Batch Export JSON</button>
            <label>Batch Limit</label>
            <input id=\"session_batch_limit\" class=\"project-batch-limit\" type=\"number\" min=\"1\" max=\"20\" value=\"5\" />
            <button type=\"button\" onclick=\"clearSessionBoardControls()\">Reset</button>
            <label><input id=\"session_auto_refresh\" type=\"checkbox\" onchange=\"onSessionAutoRefreshChanged()\" /> Auto Refresh</label>
            <select id=\"session_refresh_seconds\" onchange=\"onSessionAutoRefreshChanged()\">
              <option value=\"10\">10s</option>
              <option value=\"20\">20s</option>
              <option value=\"30\">30s</option>
              <option value=\"60\">60s</option>
            </select>
          </div>
          <div class=\"project-board-summary\" id=\"project_board_summary\">summary: -</div>
          <div class=\"btn-row\" style=\"margin: 0 0 6px 0;\">
            <button type=\"button\" onclick=\"loadBatchHistory()\">Load Batch History</button>
            <button type=\"button\" onclick=\"replayLatestBatchAction()\">Replay Latest Batch</button>
            <button type=\"button\" onclick=\"replayFailedLatestBatchAction()\">Replay Failed Latest</button>
            <button type=\"button\" onclick=\"viewBatchExportById()\">View Export By ID</button>
            <button type=\"button\" onclick=\"replayBatchExportById()\">Replay Export By ID</button>
            <button type=\"button\" onclick=\"replayFailedBatchExportById()\">Replay Failed By ID</button>
            <button type=\"button\" onclick=\"deleteBatchExportById()\">Delete Export By ID</button>
            <button type=\"button\" onclick=\"compareBatchExportsById()\">Compare Export IDs</button>
            <button type=\"button\" onclick=\"loadFailedReplayQueueById()\">Load Failed Queue By ID</button>
            <button type=\"button\" onclick=\"replayFailedQueueNow()\">Replay Failed Queue</button>
            <button type=\"button\" onclick=\"downloadBatchExportById('json')\">Download Export JSON</button>
            <button type=\"button\" onclick=\"downloadBatchExportById('csv')\">Download Export CSV</button>
          </div>
          <input id=\"batch_export_id\" placeholder=\"batch export id\" />
          <input id=\"batch_export_compare_id\" placeholder=\"compare export id\" />
          <div class=\"project-batch-history-controls\">
            <label><input id=\"batch_replay_dry_run\" type=\"checkbox\" /> replay dry-run</label>
            <label><input id=\"batch_replay_failed_only\" type=\"checkbox\" /> replay failed-only</label>
            <select id=\"batch_replay_retry_max\">
              <option value=\"0\" selected>retry max: 0</option>
              <option value=\"1\">retry max: 1</option>
              <option value=\"2\">retry max: 2</option>
              <option value=\"3\">retry max: 3</option>
            </select>
            <input id=\"batch_replay_retry_backoff_ms\" type=\"number\" min=\"0\" max=\"5000\" step=\"50\" value=\"150\" placeholder=\"retry backoff ms\" />
            <input id=\"batch_replay_max_concurrency\" type=\"number\" min=\"1\" max=\"8\" step=\"1\" value=\"2\" placeholder=\"max concurrency\" />
            <button type=\"button\" onclick=\"applyReplayPreset('safe')\">Preset Safe</button>
            <button type=\"button\" onclick=\"applyReplayPreset('fast')\">Preset Fast</button>
            <button type=\"button\" onclick=\"applyReplayPreset('dryrun')\">Preset DryRun</button>
            <button type=\"button\" onclick=\"saveReplayDefaultsToProject()\">Save Replay Defaults</button>
          </div>
          <div class=\"project-batch-history-controls\">
            <select id=\"batch_history_action_filter\" onchange=\"resetBatchHistoryOffsetAndReload()\">
              <option value=\"\">action: all</option>
              <option value=\"batch_summary\">action: batch_summary</option>
              <option value=\"batch_validate\">action: batch_validate</option>
              <option value=\"batch_retry_failed\">action: batch_retry_failed</option>
            </select>
            <select id=\"batch_history_status_filter\" onchange=\"resetBatchHistoryOffsetAndReload()\">
              <option value=\"\">status: all</option>
              <option value=\"pass\">status: pass</option>
              <option value=\"partial\">status: partial</option>
              <option value=\"fail\">status: fail</option>
            </select>
            <select id=\"batch_history_page_size\" onchange=\"resetBatchHistoryOffsetAndReload()\">
              <option value=\"10\">page size: 10</option>
              <option value=\"20\" selected>page size: 20</option>
              <option value=\"50\">page size: 50</option>
            </select>
            <input id=\"batch_history_offset\" type=\"number\" min=\"0\" step=\"1\" value=\"0\" />
            <button type=\"button\" onclick=\"prevBatchHistoryPage()\">Prev Page</button>
            <button type=\"button\" onclick=\"nextBatchHistoryPage()\">Next Page</button>
          </div>
          <div class=\"project-board-summary\" id=\"project_batch_history_summary\">batch_history: -</div>
          <div class=\"project-board-summary\" id=\"project_batch_history_metrics\">batch_metrics: -</div>
          <div class=\"project-board-summary\" id=\"project_failed_queue_summary\">failed_queue: -</div>
          <div id=\"project_batch_history_list\" class=\"project-batch-history-list\"><div class=\"muted\">(none)</div></div>
          <pre id=\"project_batch_history\" style=\"margin: 0 0 8px 0; max-height: 120px; overflow: auto;\">(none)</pre>
          <pre id=\"project_failed_queue\" style=\"margin: 0 0 8px 0; max-height: 120px; overflow: auto;\">(none)</pre>
          <div class=\"project-session-list\" id=\"project_session_list\">
            <div class=\"muted\">(empty)</div>
          </div>
        </div>

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
        <label><input id=\"memory_enabled\" type=\"checkbox\" onchange=\"updateMemoryStatus()\" /> Enable project memory injection</label>
        <label>Project memory notes</label>
        <textarea id=\"memory_notes\" rows=\"5\" placeholder=\"记录该项目长期约束/偏好，例如目标波长范围、禁用骨架、数据来源优先级。\" oninput=\"updateMemoryStatus()\"></textarea>
        <div class=\"muted\" id=\"memory_status\">memory: disabled, chars=0</div>
        <div class=\"btn-row\">
          <button type=\"button\" onclick=\"clearMemoryNotes()\">Clear Memory Notes</button>
        </div>

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
              <button onclick=\"sendPendingResume()\">Resume With Patch</button>
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
        projects: [],
        batchHistory: [],
        batchHistoryMeta: {offset: 0, limit: 20, total: 0, has_more: false, action: '', status: ''},
        failedReplayQueue: {source_export_id: '', action: '', rows: [], count: 0, unique_task_count: 0, failure_reasons: []},
        sessionBoard: {
          filterText: '',
          health: 'all',
          sort: 'updated_desc',
          autoRefreshEnabled: false,
          refreshSeconds: 30,
          pinnedProjectIds: [],
          pinnedOnly: false,
          groupedView: false,
          batchLimit: 5,
        },
        sessionAutoRefreshTimer: null,
      };

      const PROMPT_HISTORY_LIMIT = 8;
      const SESSION_BOARD_KEY = 'agent4mat.ui.session_board.v1';

      const pendingFieldMeta = {
        property: {label: 'property', placeholder: 'plqy / lambda_em / stability'},
        range: {label: 'range', placeholder: '470+-12nm or 60-100'},
        n_structures: {label: 'n_structures', placeholder: 'e.g. 500', type: 'number'},
        prediction_model: {label: 'prediction_model', placeholder: 'e.g. unimol_lambda_plqy_v1'},
        predictor_id: {label: 'predictor_id', placeholder: 'e.g. unimol_lambda_plqy_v1'},
        generator_id: {label: 'generator_id', placeholder: 'e.g. reinvent4_lambda_em_v2'},
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

      function loadSessionBoardState() {
        const fallback = {
          filterText: '',
          health: 'all',
          sort: 'updated_desc',
          autoRefreshEnabled: false,
          refreshSeconds: 30,
          pinnedProjectIds: [],
          pinnedOnly: false,
          groupedView: false,
          batchLimit: 5,
        };
        try {
          const raw = localStorage.getItem(SESSION_BOARD_KEY);
          if (!raw) return fallback;
          const parsed = JSON.parse(raw);
          if (!parsed || typeof parsed !== 'object') return fallback;
          const filterText = String(parsed.filterText || '').trim().toLowerCase();
          const health = String(parsed.health || 'all').trim().toLowerCase();
          const sort = String(parsed.sort || 'updated_desc').trim().toLowerCase();
          const autoRefreshEnabled = Boolean(parsed.autoRefreshEnabled);
          const refreshSecondsRaw = Number(parsed.refreshSeconds || 30);
          const refreshSeconds = Number.isFinite(refreshSecondsRaw) ? Math.max(10, Math.min(120, Math.floor(refreshSecondsRaw))) : 30;
          const pinnedOnly = Boolean(parsed.pinnedOnly);
          const groupedView = Boolean(parsed.groupedView);
          const batchLimitRaw = Number(parsed.batchLimit || 5);
          const batchLimit = Number.isFinite(batchLimitRaw) ? Math.max(1, Math.min(20, Math.floor(batchLimitRaw))) : 5;
          const pinnedRaw = Array.isArray(parsed.pinnedProjectIds) ? parsed.pinnedProjectIds : [];
          const pinnedProjectIds = pinnedRaw
            .map((x) => String(x || '').trim())
            .filter((x) => Boolean(x))
            .slice(0, 200);
          return {filterText, health, sort, autoRefreshEnabled, refreshSeconds, pinnedOnly, groupedView, batchLimit, pinnedProjectIds};
        } catch (e) {
          return fallback;
        }
      }

      function saveSessionBoardState(v) {
        const payload = {
          filterText: String((v && v.filterText) || '').trim().toLowerCase(),
          health: String((v && v.health) || 'all').trim().toLowerCase(),
          sort: String((v && v.sort) || 'updated_desc').trim().toLowerCase(),
          autoRefreshEnabled: Boolean(v && v.autoRefreshEnabled),
          refreshSeconds: Number.isFinite(Number(v && v.refreshSeconds)) ? Math.max(10, Math.min(120, Math.floor(Number(v.refreshSeconds)))) : 30,
          pinnedOnly: Boolean(v && v.pinnedOnly),
          groupedView: Boolean(v && v.groupedView),
          batchLimit: Number.isFinite(Number(v && v.batchLimit)) ? Math.max(1, Math.min(20, Math.floor(Number(v.batchLimit)))) : 5,
          pinnedProjectIds: Array.isArray(v && v.pinnedProjectIds)
            ? (v.pinnedProjectIds
                .map((x) => String(x || '').trim())
                .filter((x) => Boolean(x))
                .slice(0, 200))
            : [],
        };
        try {
          localStorage.setItem(SESSION_BOARD_KEY, JSON.stringify(payload));
        } catch (e) {
          // ignore storage failures
        }
      }

      function applySessionBoardStateToControls(v) {
        const payload = v && typeof v === 'object' ? v : loadSessionBoardState();
        const filterEle = document.getElementById('session_filter_text');
        const healthEle = document.getElementById('session_filter_health');
        const sortEle = document.getElementById('session_sort_mode');
        const autoEle = document.getElementById('session_auto_refresh');
        const secEle = document.getElementById('session_refresh_seconds');
        const batchLimitEle = document.getElementById('session_batch_limit');
        if (filterEle) filterEle.value = String(payload.filterText || '');
        if (healthEle) healthEle.value = String(payload.health || 'all');
        if (sortEle) sortEle.value = String(payload.sort || 'updated_desc');
        if (autoEle) autoEle.checked = Boolean(payload.autoRefreshEnabled);
        if (secEle) secEle.value = String(payload.refreshSeconds || 30);
        if (batchLimitEle) batchLimitEle.value = String(payload.batchLimit || 5);
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
        if (Object.prototype.hasOwnProperty.call(opts, 'memory_enabled')) {
          document.getElementById('memory_enabled').checked = Boolean(opts.memory_enabled);
        }
        document.getElementById('memory_notes').value = String(project.memory_notes || '');
        if (opts.batch_replay_defaults && typeof opts.batch_replay_defaults === 'object') {
          applyBatchReplayOptions(opts.batch_replay_defaults);
        }
        updateMemoryStatus();
      }

      function applyBatchReplayOptions(raw) {
        const opts = (raw && typeof raw === 'object') ? raw : {};
        const retryMaxRaw = Number(opts.retry_max || 0);
        const retryMax = Number.isFinite(retryMaxRaw) ? Math.max(0, Math.min(3, Math.floor(retryMaxRaw))) : 0;
        const backoffRaw = Number(opts.retry_backoff_ms || 150);
        const retryBackoffMs = Number.isFinite(backoffRaw) ? Math.max(0, Math.min(5000, Math.floor(backoffRaw))) : 150;
        const concurrencyRaw = Number(opts.max_concurrency || 2);
        const maxConcurrency = Number.isFinite(concurrencyRaw) ? Math.max(1, Math.min(8, Math.floor(concurrencyRaw))) : 2;
        const dryRun = Boolean(opts.dry_run);
        const failedOnly = Boolean(opts.failed_only);

        const dryEle = document.getElementById('batch_replay_dry_run');
        if (dryEle) dryEle.checked = dryRun;
        const failedEle = document.getElementById('batch_replay_failed_only');
        if (failedEle) failedEle.checked = failedOnly;
        const retryEle = document.getElementById('batch_replay_retry_max');
        if (retryEle) retryEle.value = String(retryMax);
        const backoffEle = document.getElementById('batch_replay_retry_backoff_ms');
        if (backoffEle) backoffEle.value = String(retryBackoffMs);
        const concurrencyEle = document.getElementById('batch_replay_max_concurrency');
        if (concurrencyEle) concurrencyEle.value = String(maxConcurrency);
      }

      function replayPresetOptions(mode) {
        const m = String(mode || '').trim().toLowerCase();
        if (m === 'fast') {
          return {dry_run: false, failed_only: false, retry_max: 0, retry_backoff_ms: 0, max_concurrency: 6};
        }
        if (m === 'dryrun') {
          return {dry_run: true, failed_only: false, retry_max: 0, retry_backoff_ms: 0, max_concurrency: 1};
        }
        return {dry_run: false, failed_only: false, retry_max: 2, retry_backoff_ms: 200, max_concurrency: 2};
      }

      function applyReplayPreset(mode) {
        const opts = replayPresetOptions(mode);
        applyBatchReplayOptions(opts);
        renderJsonOut({status: 'pass', action: 'apply_replay_preset', preset: String(mode || 'safe'), replay_options: readBatchReplayOptions()});
      }

      async function saveReplayDefaultsToProject() {
        await saveProject();
        renderJsonOut({status: 'pass', action: 'save_replay_defaults', replay_options: readBatchReplayOptions()});
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

      async function apiDelete(url) {
        const resp = await fetch(url, {method: 'DELETE'});
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
        const memoryEnabled = document.getElementById('memory_enabled').checked;
        return {
          planner_provider: planner,
          catalog_path: catalog,
          web_search_enabled: Boolean(webEnabled),
          web_topk: Number.isFinite(webTopk) ? webTopk : 5,
          memory_enabled: Boolean(memoryEnabled),
          batch_replay_defaults: readBatchReplayOptions(),
        };
      }

      function collectMemoryNotes() {
        return String(document.getElementById('memory_notes').value || '');
      }

      function updateMemoryStatus() {
        const enabled = Boolean(document.getElementById('memory_enabled').checked);
        const notes = collectMemoryNotes().trim();
        const status = enabled ? 'enabled' : 'disabled';
        document.getElementById('memory_status').textContent = `memory: ${status}, chars=${notes.length}`;
      }

      function clearMemoryNotes() {
        document.getElementById('memory_notes').value = '';
        updateMemoryStatus();
      }

      async function refreshProjects() {
        const r = await apiGet('/api/projects?limit=120');
        renderJsonOut(r.data);
        const picker = document.getElementById('project_picker');
        while (picker.options.length > 1) picker.remove(1);
        const projects = Array.isArray(r.data.projects) ? r.data.projects : [];
        state.projects = projects;
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
        renderProjectSessionBoard(projects);
      }

      function readSessionBoardControls() {
        const filterText = String((document.getElementById('session_filter_text').value || '')).trim().toLowerCase();
        const health = String(document.getElementById('session_filter_health').value || 'all').trim().toLowerCase();
        const sort = String(document.getElementById('session_sort_mode').value || 'updated_desc').trim().toLowerCase();
        const autoRefreshEnabled = Boolean(document.getElementById('session_auto_refresh').checked);
        const refreshSecondsRaw = Number(document.getElementById('session_refresh_seconds').value || 30);
        const refreshSeconds = Number.isFinite(refreshSecondsRaw) ? Math.max(10, Math.min(120, Math.floor(refreshSecondsRaw))) : 30;
        const batchLimitRaw = Number(document.getElementById('session_batch_limit').value || 5);
        const batchLimit = Number.isFinite(batchLimitRaw) ? Math.max(1, Math.min(20, Math.floor(batchLimitRaw))) : 5;
        const pinnedOnly = Boolean(state.sessionBoard && state.sessionBoard.pinnedOnly);
        const groupedView = Boolean(state.sessionBoard && state.sessionBoard.groupedView);
        const pinnedProjectIds = Array.isArray(state.sessionBoard && state.sessionBoard.pinnedProjectIds)
          ? state.sessionBoard.pinnedProjectIds.slice()
          : [];
        return {filterText, health, sort, autoRefreshEnabled, refreshSeconds, batchLimit, pinnedOnly, groupedView, pinnedProjectIds};
      }

      function applySessionBoardControls() {
        state.sessionBoard = readSessionBoardControls();
        saveSessionBoardState(state.sessionBoard);
        ensureSessionAutoRefresh();
        renderProjectSessionBoard(state.projects || []);
      }

      function quickFilterFailedOnly() {
        const healthEle = document.getElementById('session_filter_health');
        if (healthEle) {
          healthEle.value = 'failed';
        }
        applySessionBoardControls();
      }

      function quickFilterByHealth(health) {
        const value = String(health || 'all').trim().toLowerCase();
        const allow = new Set(['all', 'failed', 'success', 'none']);
        const next = allow.has(value) ? value : 'all';
        const healthEle = document.getElementById('session_filter_health');
        if (healthEle) {
          healthEle.value = next;
        }
        applySessionBoardControls();
      }

      function quickSortPriority() {
        const sortEle = document.getElementById('session_sort_mode');
        if (sortEle) {
          sortEle.value = 'priority_desc';
        }
        applySessionBoardControls();
      }

      function clearSessionBoardControls() {
        applySessionBoardStateToControls({
          filterText: '',
          health: 'all',
          sort: 'updated_desc',
          autoRefreshEnabled: false,
          refreshSeconds: 30,
          batchLimit: 5,
          pinnedOnly: false,
          groupedView: false,
          pinnedProjectIds: Array.isArray(state.sessionBoard && state.sessionBoard.pinnedProjectIds)
            ? state.sessionBoard.pinnedProjectIds
            : [],
        });
        applySessionBoardControls();
      }

      function togglePinnedOnly() {
        const next = !Boolean(state.sessionBoard && state.sessionBoard.pinnedOnly);
        state.sessionBoard = {
          ...(state.sessionBoard || {}),
          pinnedOnly: next,
          pinnedProjectIds: Array.isArray(state.sessionBoard && state.sessionBoard.pinnedProjectIds)
            ? state.sessionBoard.pinnedProjectIds.slice()
            : [],
        };
        saveSessionBoardState(state.sessionBoard);
        renderProjectSessionBoard(state.projects || []);
      }

      function toggleSessionBoardGroupedView() {
        const next = !Boolean(state.sessionBoard && state.sessionBoard.groupedView);
        state.sessionBoard = {
          ...(state.sessionBoard || {}),
          groupedView: next,
        };
        saveSessionBoardState(state.sessionBoard);
        renderProjectSessionBoard(state.projects || []);
      }

      function toggleProjectPin(projectId) {
        const pid = String(projectId || '').trim();
        if (!pid) return;
        const current = Array.isArray(state.sessionBoard && state.sessionBoard.pinnedProjectIds)
          ? state.sessionBoard.pinnedProjectIds.slice()
          : [];
        const set = new Set(current);
        if (set.has(pid)) set.delete(pid);
        else set.add(pid);
        state.sessionBoard = {
          ...(state.sessionBoard || {}),
          pinnedProjectIds: Array.from(set).slice(0, 200),
        };
        saveSessionBoardState(state.sessionBoard);
        renderProjectSessionBoard(state.projects || []);
      }

      function onSessionAutoRefreshChanged() {
        state.sessionBoard = readSessionBoardControls();
        saveSessionBoardState(state.sessionBoard);
        ensureSessionAutoRefresh();
      }

      function ensureSessionAutoRefresh() {
        if (state.sessionAutoRefreshTimer) {
          clearInterval(state.sessionAutoRefreshTimer);
          state.sessionAutoRefreshTimer = null;
        }
        const controls = state.sessionBoard || {};
        if (!controls.autoRefreshEnabled) {
          return;
        }
        const sec = Number.isFinite(Number(controls.refreshSeconds)) ? Math.max(10, Math.min(120, Math.floor(Number(controls.refreshSeconds)))) : 30;
        state.sessionAutoRefreshTimer = setInterval(() => {
          void refreshProjects();
        }, sec * 1000);
      }

      function projectHealthStatus(row) {
        const st = String((row && row.runtime_health && row.runtime_health.status) || '').toLowerCase();
        if (st === 'failed') return 'failed';
        if (st === 'success') return 'success';
        return 'none';
      }

      function scoreProjectPriority(row) {
        const rh = (row && row.runtime_health && typeof row.runtime_health === 'object') ? row.runtime_health : {};
        const failed = Number(rh.failed_steps || 0);
        const dur = Number(rh.recent_duration_ms || 0);
        const ratio = Number(rh.success_ratio || 0);
        return (failed * 1000) + (dur / 1000) - (ratio * 100);
      }

      function computeSessionBoardRows(projects) {
        const controls = readSessionBoardControls();
        state.sessionBoard = controls;
        const baseRows = Array.isArray(projects) ? projects : [];
        const pinnedIds = new Set(
          Array.isArray(controls.pinnedProjectIds)
            ? controls.pinnedProjectIds
            : []
        );
        let rows = baseRows.slice();
        if (Boolean(controls.pinnedOnly)) {
          rows = rows.filter((row) => {
            const pid = String((row && row.project_id) || '').trim();
            return pid && pinnedIds.has(pid);
          });
        }
        if (controls.filterText) {
          rows = rows.filter((row) => {
            if (!row || typeof row !== 'object') return false;
            const pid = String(row.project_id || '').toLowerCase();
            const title = String(row.title || '').toLowerCase();
            const task = String(row.current_task_id || '').toLowerCase();
            const token = controls.filterText;
            return pid.includes(token) || title.includes(token) || task.includes(token);
          });
        }
        if (controls.health && controls.health !== 'all') {
          rows = rows.filter((row) => {
            const status = projectHealthStatus(row);
            if (controls.health === 'failed') return status === 'failed';
            if (controls.health === 'success') return status === 'success';
            if (controls.health === 'none') return status === 'none';
            return true;
          });
        }
        if (controls.sort === 'failed_desc') {
          rows.sort((a, b) => Number((b && b.runtime_health && b.runtime_health.failed_steps) || 0) - Number((a && a.runtime_health && a.runtime_health.failed_steps) || 0));
        } else if (controls.sort === 'success_ratio_asc') {
          rows.sort((a, b) => {
            const sa = Number((a && a.runtime_health && a.runtime_health.success_steps) || 0);
            const fa = Number((a && a.runtime_health && a.runtime_health.failed_steps) || 0);
            const sb = Number((b && b.runtime_health && b.runtime_health.success_steps) || 0);
            const fb = Number((b && b.runtime_health && b.runtime_health.failed_steps) || 0);
            const ra = (sa + fa) > 0 ? (sa / (sa + fa)) : 1;
            const rb = (sb + fb) > 0 ? (sb / (sb + fb)) : 1;
            return ra - rb;
          });
        } else if (controls.sort === 'priority_desc') {
          rows.sort((a, b) => scoreProjectPriority(b) - scoreProjectPriority(a));
        } else {
          rows.sort((a, b) => String((b && b.updated_at) || '').localeCompare(String((a && a.updated_at) || '')));
        }
        rows.sort((a, b) => {
          const ap = pinnedIds.has(String((a && a.project_id) || '').trim()) ? 1 : 0;
          const bp = pinnedIds.has(String((b && b.project_id) || '').trim()) ? 1 : 0;
          return bp - ap;
        });
        return {rows, controls, pinnedIds};
      }

      function readSessionBatchLimit() {
        const controls = state.sessionBoard || readSessionBoardControls();
        const raw = Number(controls.batchLimit || 5);
        return Number.isFinite(raw) ? Math.max(1, Math.min(20, Math.floor(raw))) : 5;
      }

      function recentSessionBatchRows(requireTask) {
        const payload = computeSessionBoardRows(state.projects || []);
        const rows = payload.rows || [];
        const limit = readSessionBatchLimit();
        const filtered = Boolean(requireTask)
          ? rows.filter((row) => String((row && row.current_task_id) || '').trim())
          : rows;
        const picked = filtered.slice(0, limit);
        return {payload, rows, picked, limit};
      }

      function summarizeBatchRow(row) {
        if (!row || typeof row !== 'object') return {};
        const runtime = (row.runtime_health && typeof row.runtime_health === 'object') ? row.runtime_health : {};
        return {
          project_id: String(row.project_id || '').trim(),
          task_id: String(row.current_task_id || '').trim(),
          title: String(row.title || '').trim(),
          health: String(runtime.status || 'none'),
          latest_failed_step: String(runtime.latest_failed_step || '').trim(),
          updated_at: String(row.updated_at || '').trim(),
        };
      }

      function readLatestBatchPayload() {
        try {
          const raw = sessionStorage.getItem('agent4mat.ui.latest_batch_payload');
          if (!raw) return null;
          const parsed = JSON.parse(raw);
          return parsed && typeof parsed === 'object' ? parsed : null;
        } catch (e) {
          return null;
        }
      }

      function storeLatestBatchPayload(payload) {
        try {
          sessionStorage.setItem('agent4mat.ui.latest_batch_payload', JSON.stringify(payload || {}));
        } catch (e) {
          // ignore storage failures
        }
      }

      async function persistBatchPayload(payload) {
        const body = payload && typeof payload === 'object' ? payload : {};
        const projectId = selectedProjectId();
        if (!projectId || !isSafeProjectId(projectId)) {
          return {status: 'fail', error: 'invalid project_id'};
        }
        const resp = await apiPost(`/api/projects/${encodeURIComponent(projectId)}/batch-export`, {payload: body});
        return resp.data || {status: 'fail', error: 'empty_response'};
      }

      function exportSessionBoardBatchResult() {
        void exportSessionBoardBatchResultPersisted();
      }

      async function exportSessionBoardBatchResultPersisted() {
        const batch = readLatestBatchPayload();
        if (!batch) {
          renderJsonOut({status: 'fail', error: 'no batch result to export'});
          return;
        }
        const payload = {
          exported_at: nowIso(),
          project_id: selectedProjectId(),
          batch_result: batch,
        };
        const saved = await persistBatchPayload(payload);
        renderJsonOut({
          status: String(saved.status || 'unknown'),
          action: 'batch_export_persisted',
          saved: saved,
          payload: payload,
        });
        await loadBatchHistory();
      }

      function readBatchHistoryControls() {
        const action = String((document.getElementById('batch_history_action_filter').value || '')).trim();
        const status = String((document.getElementById('batch_history_status_filter').value || '')).trim().toLowerCase();
        const limitRaw = Number(document.getElementById('batch_history_page_size').value || 20);
        const limit = Number.isFinite(limitRaw) ? Math.max(1, Math.min(100, Math.floor(limitRaw))) : 20;
        const offsetRaw = Number(document.getElementById('batch_history_offset').value || 0);
        const offset = Number.isFinite(offsetRaw) ? Math.max(0, Math.floor(offsetRaw)) : 0;
        return {action, status, limit, offset};
      }

      function setBatchHistoryOffset(offset) {
        const ele = document.getElementById('batch_history_offset');
        if (ele) {
          ele.value = String(Math.max(0, Number.isFinite(Number(offset)) ? Math.floor(Number(offset)) : 0));
        }
      }

      function resetBatchHistoryOffsetAndReload() {
        setBatchHistoryOffset(0);
        void loadBatchHistory();
      }

      function prevBatchHistoryPage() {
        const c = readBatchHistoryControls();
        setBatchHistoryOffset(Math.max(0, c.offset - c.limit));
        void loadBatchHistory();
      }

      function nextBatchHistoryPage() {
        const c = readBatchHistoryControls();
        const meta = state.batchHistoryMeta || {};
        const hasMore = Boolean(meta.has_more);
        if (!hasMore) return;
        setBatchHistoryOffset(c.offset + c.limit);
        void loadBatchHistory();
      }

      function renderBatchHistoryList(items) {
        const wrap = document.getElementById('project_batch_history_list');
        if (!wrap) return;
        wrap.innerHTML = '';
        const rows = Array.isArray(items) ? items : [];
        if (rows.length < 1) {
          const empty = document.createElement('div');
          empty.className = 'muted';
          empty.textContent = '(none)';
          wrap.appendChild(empty);
          return;
        }
        for (const item of rows) {
          if (!item || typeof item !== 'object') continue;
          const eid = String(item.export_id || '').trim();
          const metrics = (item.replay_metrics && typeof item.replay_metrics === 'object') ? item.replay_metrics : {};
          const okN = Number(metrics.ok_count || 0);
          const failN = Number(metrics.fail_count || 0);
          const skipN = Number(metrics.skipped_count || 0);
          const dryN = Number(metrics.dry_run_count || 0);
          const elapsedMs = Number(metrics.elapsed_ms || 0);
          const metricsText = (okN + failN + skipN + dryN) > 0
            ? ` | ok=${okN} fail=${failN} skipped=${skipN} dry=${dryN} elapsed_ms=${elapsedMs}`
            : '';
          const row = document.createElement('div');
          row.className = 'project-batch-history-item';
          row.textContent = `${String(item.created_at || '-')} | ${String(item.action || '-')} | status=${String(item.status || '-')} | count=${String(item.count || 0)} | export_id=${eid || '-'}${metricsText}`;
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = 'Use ID';
          btn.onclick = () => {
            const input = document.getElementById('batch_export_id');
            if (input) input.value = eid;
            renderJsonOut(item);
          };
          const compareBtn = document.createElement('button');
          compareBtn.type = 'button';
          compareBtn.textContent = 'Use As Compare';
          compareBtn.onclick = () => {
            const input = document.getElementById('batch_export_compare_id');
            if (input) input.value = eid;
            renderJsonOut(item);
          };
          row.appendChild(document.createElement('br'));
          row.appendChild(btn);
          row.appendChild(compareBtn);
          wrap.appendChild(row);
        }
      }

      async function loadBatchHistory() {
        const pid = selectedProjectId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        const c = readBatchHistoryControls();
        const qs = new URLSearchParams();
        qs.set('limit', String(c.limit));
        qs.set('offset', String(c.offset));
        if (c.action) qs.set('action', c.action);
        if (c.status) qs.set('status', c.status);
        const r = await apiGet(`/api/projects/${encodeURIComponent(pid)}/batch-exports?${qs.toString()}`);
        const items = Array.isArray(r.data && r.data.exports) ? r.data.exports : [];
        const total = Number((r.data && r.data.total_count) || items.length);
        const hasMore = Boolean(r.data && r.data.has_more);
        state.batchHistory = items;
        state.batchHistoryMeta = {offset: c.offset, limit: c.limit, total: total, has_more: hasMore, action: c.action, status: c.status};
        const head = document.getElementById('project_batch_history_summary');
        const box = document.getElementById('project_batch_history');
        const exportIdEle = document.getElementById('batch_export_id');
        const compareIdEle = document.getElementById('batch_export_compare_id');
        if (head) {
          const page = Math.floor(c.offset / c.limit) + 1;
          const latest = items.length > 0 ? items[0] : null;
          head.textContent = latest
            ? `batch_history: total=${total} | page=${page} | shown=${items.length} | has_more=${hasMore ? 'yes' : 'no'} | latest=${String(latest.action || '-')}/${String(latest.created_at || '-')}`
            : `batch_history: total=${total} | page=${page} | shown=0 | has_more=no`;
        }
        if (box) {
          box.textContent = items.length > 0 ? JSON.stringify(items.slice(0, 20), null, 2) : '(none)';
        }
        renderBatchHistoryMetrics(items);
        renderBatchHistoryList(items);
        if (exportIdEle && items.length > 0) {
          const current = String(exportIdEle.value || '').trim();
          const hit = items.some((x) => String((x && x.export_id) || '') === current);
          if (!current || !hit) {
            exportIdEle.value = String(items[0].export_id || '');
          }
        }
        if (compareIdEle && items.length > 1) {
          const current = String(compareIdEle.value || '').trim();
          const hit = items.some((x) => String((x && x.export_id) || '') === current);
          if (!current || !hit) {
            compareIdEle.value = String(items[1].export_id || items[0].export_id || '');
          }
        }
        return r;
      }

      function readBatchExportId() {
        const ele = document.getElementById('batch_export_id');
        const eid = String((ele && ele.value) || '').trim();
        return eid;
      }

      function readBatchCompareExportId() {
        const ele = document.getElementById('batch_export_compare_id');
        const eid = String((ele && ele.value) || '').trim();
        return eid;
      }

      function readBatchReplayOptions(forceFailedOnly) {
        const dryRun = Boolean(document.getElementById('batch_replay_dry_run').checked);
        const failedOnlyRaw = Boolean(document.getElementById('batch_replay_failed_only').checked);
        const retryMaxRaw = Number(document.getElementById('batch_replay_retry_max').value || 0);
        const retryMax = Number.isFinite(retryMaxRaw) ? Math.max(0, Math.min(3, Math.floor(retryMaxRaw))) : 0;
        const backoffRaw = Number(document.getElementById('batch_replay_retry_backoff_ms').value || 150);
        const retryBackoffMs = Number.isFinite(backoffRaw) ? Math.max(0, Math.min(5000, Math.floor(backoffRaw))) : 0;
        const concurrencyRaw = Number(document.getElementById('batch_replay_max_concurrency').value || 2);
        const maxConcurrency = Number.isFinite(concurrencyRaw) ? Math.max(1, Math.min(8, Math.floor(concurrencyRaw))) : 1;
        return {
          dry_run: dryRun,
          failed_only: Boolean(forceFailedOnly) ? true : failedOnlyRaw,
          retry_max: retryMax,
          retry_backoff_ms: retryBackoffMs,
          max_concurrency: maxConcurrency,
        };
      }

      function renderBatchHistoryMetrics(items) {
        const box = document.getElementById('project_batch_history_metrics');
        if (!box) return;
        const rows = Array.isArray(items) ? items : [];
        if (rows.length < 1) {
          box.textContent = 'batch_metrics: shown=0';
          return;
        }
        let passN = 0;
        let partialN = 0;
        let failN = 0;
        let totalElapsedMs = 0;
        let elapsedCnt = 0;
        let okN = 0;
        let errN = 0;
        let skippedN = 0;
        let dryN = 0;
        for (const row of rows) {
          if (!row || typeof row !== 'object') continue;
          const st = String(row.status || '').trim().toLowerCase();
          if (st === 'pass') passN += 1;
          else if (st === 'partial') partialN += 1;
          else if (st === 'fail') failN += 1;
          const m = (row.replay_metrics && typeof row.replay_metrics === 'object') ? row.replay_metrics : {};
          const elapsed = Number(m.elapsed_ms || 0);
          if (Number.isFinite(elapsed) && elapsed > 0) {
            totalElapsedMs += elapsed;
            elapsedCnt += 1;
          }
          okN += Number(m.ok_count || 0);
          errN += Number(m.fail_count || 0);
          skippedN += Number(m.skipped_count || 0);
          dryN += Number(m.dry_run_count || 0);
        }
        const avgElapsedMs = elapsedCnt > 0 ? Math.round(totalElapsedMs / elapsedCnt) : 0;
        box.textContent = `batch_metrics: shown=${rows.length} | pass=${passN} partial=${partialN} fail=${failN} | replay(ok/fail/skipped/dry)=${okN}/${errN}/${skippedN}/${dryN} | avg_elapsed_ms=${avgElapsedMs}`;
      }

      function renderFailedReplayQueue(queuePayload) {
        const head = document.getElementById('project_failed_queue_summary');
        const box = document.getElementById('project_failed_queue');
        const queue = (queuePayload && typeof queuePayload === 'object') ? queuePayload : {};
        const rows = Array.isArray(queue.rows) ? queue.rows : [];
        const sourceExportId = String(queue.source_export_id || '').trim();
        const action = String(queue.action || '').trim();
        const reasonRows = Array.isArray(queue.failure_reasons) ? queue.failure_reasons : [];
        const reasonText = reasonRows.slice(0, 4).map((x) => `${String((x && x.reason) || '-')}:${Number((x && x.count) || 0)}`).join(', ');
        if (head) {
          head.textContent = rows.length > 0
            ? `failed_queue: source=${sourceExportId || '-'} | action=${action || '-'} | count=${rows.length} | unique_tasks=${Number(queue.unique_task_count || 0)}${reasonText ? ` | reasons=${reasonText}` : ''}`
            : 'failed_queue: empty';
        }
        if (box) {
          box.textContent = rows.length > 0 ? JSON.stringify(rows.slice(0, 60), null, 2) : '(none)';
        }
      }

      async function loadFailedReplayQueueById() {
        const pid = selectedProjectId();
        const eid = readBatchExportId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!eid) {
          renderJsonOut({status: 'fail', error: 'missing export_id'});
          return;
        }
        const r = await apiGet(`/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(eid)}/failed-queue`);
        renderJsonOut(r.data);
        const q = (r.data && r.data.queue && typeof r.data.queue === 'object') ? r.data.queue : null;
        if (q) {
          state.failedReplayQueue = q;
          renderFailedReplayQueue(q);
        }
      }

      async function replayFailedQueueNow() {
        const pid = selectedProjectId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        const queue = (state.failedReplayQueue && typeof state.failedReplayQueue === 'object') ? state.failedReplayQueue : {};
        const sourceExportId = String(queue.source_export_id || readBatchExportId() || '').trim();
        if (!sourceExportId) {
          renderJsonOut({status: 'fail', error: 'no source export id for failed queue replay'});
          return;
        }
        const r = await apiPost(`/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(sourceExportId)}/replay`, {
          options: readBatchReplayOptions(true),
        });
        renderJsonOut(r.data);
        await loadBatchHistory();
        await refreshProjects();
        await loadFailedReplayQueueById();
      }

      async function compareBatchExportsById() {
        const pid = selectedProjectId();
        const primary = readBatchExportId();
        const other = readBatchCompareExportId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!primary || !other) {
          renderJsonOut({status: 'fail', error: 'missing export_id for compare'});
          return;
        }
        const r = await apiGet(`/api/projects/${encodeURIComponent(pid)}/batch-exports/compare?primary_export_id=${encodeURIComponent(primary)}&other_export_id=${encodeURIComponent(other)}`);
        renderJsonOut(r.data);
      }

      function downloadBatchExportById(format) {
        const pid = selectedProjectId();
        const eid = readBatchExportId();
        const fmt = String(format || 'json').trim().toLowerCase();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!eid) {
          renderJsonOut({status: 'fail', error: 'missing export_id'});
          return;
        }
        const q = new URLSearchParams();
        q.set('format', fmt === 'csv' ? 'csv' : 'json');
        const url = `/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(eid)}/download?${q.toString()}`;
        window.open(url, '_blank', 'noopener,noreferrer');
      }

      async function replayLatestBatchAction() {
        const pid = selectedProjectId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        const r = await apiPost(`/api/projects/${encodeURIComponent(pid)}/batch-exports/replay-latest`, {options: readBatchReplayOptions()});
        renderJsonOut(r.data);
        await loadBatchHistory();
        await refreshProjects();
      }

      async function replayFailedLatestBatchAction() {
        const pid = selectedProjectId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        const r = await apiPost(`/api/projects/${encodeURIComponent(pid)}/batch-exports/replay-latest`, {options: readBatchReplayOptions(true)});
        renderJsonOut(r.data);
        await loadBatchHistory();
        await refreshProjects();
      }

      async function viewBatchExportById() {
        const pid = selectedProjectId();
        const eid = readBatchExportId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!eid) {
          renderJsonOut({status: 'fail', error: 'missing export_id'});
          return;
        }
        const r = await apiGet(`/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(eid)}`);
        renderJsonOut(r.data);
      }

      async function replayBatchExportById() {
        const pid = selectedProjectId();
        const eid = readBatchExportId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!eid) {
          renderJsonOut({status: 'fail', error: 'missing export_id'});
          return;
        }
        const r = await apiPost(`/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(eid)}/replay`, {options: readBatchReplayOptions()});
        renderJsonOut(r.data);
        await loadBatchHistory();
        await refreshProjects();
      }

      async function replayFailedBatchExportById() {
        const pid = selectedProjectId();
        const eid = readBatchExportId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!eid) {
          renderJsonOut({status: 'fail', error: 'missing export_id'});
          return;
        }
        const r = await apiPost(`/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(eid)}/replay`, {options: readBatchReplayOptions(true)});
        renderJsonOut(r.data);
        await loadBatchHistory();
        await refreshProjects();
      }

      async function deleteBatchExportById() {
        const pid = selectedProjectId();
        const eid = readBatchExportId();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!eid) {
          renderJsonOut({status: 'fail', error: 'missing export_id'});
          return;
        }
        const r = await apiDelete(`/api/projects/${encodeURIComponent(pid)}/batch-exports/${encodeURIComponent(eid)}`);
        renderJsonOut(r.data);
        await loadBatchHistory();
      }

      function attachSessionCard(parent, row, pinnedIds, activeId) {
        if (!parent || !row || typeof row !== 'object') return;
        const pid = String(row.project_id || '').trim();
        if (!pid) return;
        const taskId = String(row.current_task_id || '').trim();
        const runtime = (row.runtime_health && typeof row.runtime_health === 'object') ? row.runtime_health : {};
        const successSteps = Number(runtime.success_steps || 0);
        const failedSteps = Number(runtime.failed_steps || 0);
        const totalSteps = Math.max(0, successSteps + failedSteps);
        const successRatio = totalSteps > 0 ? (successSteps / totalSteps) : 0;
        const runtimeSecRaw = Number((row.last_runtime && row.last_runtime.duration_ms) || 0);
        const runtimeSec = Number.isFinite(runtimeSecRaw) && runtimeSecRaw > 0 ? (runtimeSecRaw / 1000.0) : 0;
        const latestFailedStep = String((row.runtime_health && row.runtime_health.latest_failed_step) || '').trim();
        const latestFailedError = String((row.runtime_health && row.runtime_health.latest_failed_error) || '').trim();
        const healthObj = (row.runtime_health && typeof row.runtime_health === 'object') ? row.runtime_health : {};
        const healthStatus = String(healthObj.status || 'none').toLowerCase();
        const health = formatRuntimeHealth(healthObj || {});
        const updatedAt = String(row.updated_at || '-');
        const title = String(row.title || pid);

        const card = document.createElement('div');
        const isPinned = pinnedIds.has(pid);
        card.className = `project-session-item${pid === activeId ? ' active' : ''}${isPinned ? ' pinned' : ''}`;

        const head = document.createElement('div');
        head.className = 'project-session-head';
        const titleEle = document.createElement('div');
        titleEle.className = 'project-session-title';
        titleEle.textContent = title;
        const idEle = document.createElement('div');
        idEle.className = 'project-session-id';
        idEle.textContent = pid;
        head.appendChild(titleEle);
        head.appendChild(idEle);
        card.appendChild(head);

        const meta = document.createElement('div');
        meta.className = 'project-session-meta';
        meta.textContent = `task=${taskId || '-'} | health=${health} | updated=${updatedAt}`;
        card.appendChild(meta);
        const statusBadge = document.createElement('div');
        statusBadge.className = `project-session-status${healthStatus === 'failed' ? ' fail' : (healthStatus === 'success' ? ' pass' : '')}`;
        statusBadge.textContent = `status=${healthStatus || 'none'}`;
        card.appendChild(statusBadge);
        const failed = document.createElement('div');
        failed.className = 'project-session-failed';
        failed.textContent = latestFailedStep ? `latest_failed_step=${latestFailedStep}` : 'latest_failed_step=-';
        card.appendChild(failed);
        const failedErr = document.createElement('div');
        failedErr.className = 'project-session-error';
        failedErr.textContent = latestFailedError ? `failed_error=${latestFailedError}` : 'failed_error=-';
        card.appendChild(failedErr);
        const runtimeLine = document.createElement('div');
        runtimeLine.className = 'project-session-runtime';
        const ratioText = totalSteps > 0 ? `${Math.round(successRatio * 100)}%` : '-';
        const durationText = runtimeSec > 0 ? `${runtimeSec.toFixed(2)}s` : '-';
        const recordCount = Number(healthObj.record_count || 0);
        runtimeLine.textContent = `recent_duration=${durationText} | success_ratio=${ratioText} (${successSteps}/${totalSteps || 0}) | records=${recordCount}`;
        card.appendChild(runtimeLine);
        const progress = document.createElement('div');
        progress.className = 'project-session-progress';
        const progressBar = document.createElement('div');
        progressBar.className = 'project-session-progress-bar';
        progressBar.style.width = `${Math.max(0, Math.min(100, successRatio * 100))}%`;
        progress.appendChild(progressBar);
        card.appendChild(progress);

        const actions = document.createElement('div');
        actions.className = 'project-session-actions';
        const openBtn = document.createElement('button');
        openBtn.type = 'button';
        openBtn.textContent = 'Open';
        openBtn.onclick = () => {
          openProjectWorkspace(pid, {push: true});
        };
        actions.appendChild(openBtn);

        const pinBtn = document.createElement('button');
        pinBtn.type = 'button';
        pinBtn.textContent = isPinned ? 'Unpin' : 'Pin';
        pinBtn.onclick = () => {
          toggleProjectPin(pid);
        };
        actions.appendChild(pinBtn);

        const resumeBtn = document.createElement('button');
        resumeBtn.type = 'button';
        resumeBtn.textContent = 'Resume';
        resumeBtn.disabled = !taskId;
        resumeBtn.onclick = () => {
          resumeProjectTask(pid, taskId);
        };
        actions.appendChild(resumeBtn);

        const retryFailedBtn = document.createElement('button');
        retryFailedBtn.type = 'button';
        retryFailedBtn.textContent = 'Retry Failed';
        retryFailedBtn.disabled = !(taskId && latestFailedStep);
        retryFailedBtn.onclick = () => {
          retryProjectFailedStep(pid, taskId, latestFailedStep);
        };
        actions.appendChild(retryFailedBtn);

        const timelineBtn = document.createElement('button');
        timelineBtn.type = 'button';
        timelineBtn.textContent = 'Timeline';
        timelineBtn.disabled = !taskId;
        timelineBtn.onclick = () => {
          showProjectTimeline(pid, taskId);
        };
        actions.appendChild(timelineBtn);

        const copyTaskBtn = document.createElement('button');
        copyTaskBtn.type = 'button';
        copyTaskBtn.textContent = 'Copy Task ID';
        copyTaskBtn.disabled = !taskId;
        copyTaskBtn.onclick = () => {
          copyProjectTaskId(taskId);
        };
        actions.appendChild(copyTaskBtn);

        const summaryBtn = document.createElement('button');
        summaryBtn.type = 'button';
        summaryBtn.textContent = 'Summary';
        summaryBtn.disabled = !taskId;
        summaryBtn.onclick = () => {
          showProjectSummary(pid, taskId);
        };
        actions.appendChild(summaryBtn);

        const validateBtn = document.createElement('button');
        validateBtn.type = 'button';
        validateBtn.textContent = 'Validate';
        validateBtn.disabled = !taskId;
        validateBtn.onclick = () => {
          validateProjectTask(pid, taskId);
        };
        actions.appendChild(validateBtn);
        card.appendChild(actions);
        parent.appendChild(card);
      }

      function appendSessionBoardSection(wrap, label, rows, pinnedIds, activeId) {
        if (!wrap || !Array.isArray(rows) || rows.length < 1) return;
        const section = document.createElement('div');
        section.className = 'project-session-section';
        const head = document.createElement('div');
        head.className = 'project-session-section-head';
        head.textContent = `${label} (${rows.length})`;
        section.appendChild(head);
        for (const row of rows) {
          attachSessionCard(section, row, pinnedIds, activeId);
        }
        wrap.appendChild(section);
      }

      async function batchShowProjectSummary() {
        const payload = recentSessionBatchRows(true);
        const picked = payload.picked || [];
        const limit = payload.limit || 5;
        if (picked.length < 1) {
          renderJsonOut({status: 'fail', error: 'no task available in filtered set'});
          return;
        }
        const results = [];
        for (const row of picked) {
          const pid = String((row && row.project_id) || '').trim();
          const tid = String((row && row.current_task_id) || '').trim();
          const resp = await apiGet(`/api/task/${encodeURIComponent(tid)}/summary`);
          results.push({project_id: pid, task_id: tid, http_status: resp.status, data: resp.data});
        }
        const out = {
          status: 'pass',
          action: 'batch_summary',
          limit: limit,
          count: results.length,
          rows: picked.map((row) => summarizeBatchRow(row)),
          results: results,
          created_at: nowIso(),
        };
        storeLatestBatchPayload(out);
        renderJsonOut(out);
        await persistBatchPayload(out);
        await loadBatchHistory();
      }

      async function batchValidateProjectTask() {
        const payload = recentSessionBatchRows(true);
        const picked = payload.picked || [];
        const limit = payload.limit || 5;
        if (picked.length < 1) {
          renderJsonOut({status: 'fail', error: 'no task available in filtered set'});
          return;
        }
        const results = [];
        for (const row of picked) {
          const pid = String((row && row.project_id) || '').trim();
          const tid = String((row && row.current_task_id) || '').trim();
          const resp = await apiGet(`/api/task/${encodeURIComponent(tid)}/validate`);
          results.push({project_id: pid, task_id: tid, http_status: resp.status, data: resp.data});
        }
        const out = {
          status: 'pass',
          action: 'batch_validate',
          limit: limit,
          count: results.length,
          rows: picked.map((row) => summarizeBatchRow(row)),
          results: results,
          created_at: nowIso(),
        };
        storeLatestBatchPayload(out);
        renderJsonOut(out);
        await persistBatchPayload(out);
        await loadBatchHistory();
      }

      async function batchRetryFailedProjectStep() {
        const payload = recentSessionBatchRows(true);
        const picked = payload.picked || [];
        const limit = payload.limit || 5;
        if (picked.length < 1) {
          renderJsonOut({status: 'fail', error: 'no task available in filtered set'});
          return;
        }
        const retries = [];
        for (const row of picked) {
          const pid = String((row && row.project_id) || '').trim();
          const tid = String((row && row.current_task_id) || '').trim();
          const latestFailedStep = String((row && row.runtime_health && row.runtime_health.latest_failed_step) || '').trim();
          if (!latestFailedStep) {
            retries.push({project_id: pid, task_id: tid, status: 'skipped', reason: 'no_latest_failed_step'});
            continue;
          }
          const resp = await apiPost(`/api/task/${encodeURIComponent(tid)}/retry-failed-step`, {
            catalog_path: document.getElementById('catalog').value,
            failed_tool_name: latestFailedStep,
          });
          retries.push({
            project_id: pid,
            task_id: tid,
            failed_tool_name: latestFailedStep,
            http_status: resp.status,
            data: resp.data,
          });
        }
        const out = {
          status: 'pass',
          action: 'batch_retry_failed',
          limit: limit,
          count: retries.length,
          rows: picked.map((row) => summarizeBatchRow(row)),
          retries: retries,
          created_at: nowIso(),
        };
        storeLatestBatchPayload(out);
        renderJsonOut(out);
        await persistBatchPayload(out);
        await loadBatchHistory();
        await refreshProjects();
      }

      function renderProjectSessionBoard(projects) {
        const wrap = document.getElementById('project_session_list');
        if (!wrap) return;
        wrap.innerHTML = '';
        const summaryEle = document.getElementById('project_board_summary');
        const payload = computeSessionBoardRows(projects);
        const rows = payload.rows || [];
        const controls = payload.controls || readSessionBoardControls();
        const pinnedIds = payload.pinnedIds || new Set();

        if (summaryEle) {
          const total = rows.length;
          let failedN = 0;
          let successN = 0;
          let noneN = 0;
          let ratioSum = 0.0;
          let ratioCnt = 0;
          let pinnedN = 0;
          for (const row of rows) {
            if (!row || typeof row !== 'object') continue;
            const pid = String(row.project_id || '').trim();
            if (pid && pinnedIds.has(pid)) pinnedN += 1;
            const rh = (row.runtime_health && typeof row.runtime_health === 'object') ? row.runtime_health : {};
            const st = String(rh.status || 'none').toLowerCase();
            if (st === 'failed') failedN += 1;
            else if (st === 'success') successN += 1;
            else noneN += 1;
            const ratio = Number(rh.success_ratio || 0);
            if (Number.isFinite(ratio)) {
              ratioSum += ratio;
              ratioCnt += 1;
            }
          }
          const avgRatio = ratioCnt > 0 ? Math.round((ratioSum / ratioCnt) * 100) : 0;
          const mode = Boolean(controls.groupedView) ? 'grouped' : 'flat';
          const batchLimit = readSessionBatchLimit();
          summaryEle.textContent = `summary: total=${total} | pinned=${pinnedN} | failed=${failedN} | success=${successN} | none=${noneN} | avg_success_ratio=${avgRatio}% | mode=${mode} | batch_limit=${batchLimit}`;
          const failedBtn = document.querySelector(\"button[onclick=\\\"quickFilterByHealth('failed')\\\"]\");
          const successBtn = document.querySelector(\"button[onclick=\\\"quickFilterByHealth('success')\\\"]\");
          const noneBtn = document.querySelector(\"button[onclick=\\\"quickFilterByHealth('none')\\\"]\");
          if (failedBtn) failedBtn.textContent = `Failed Count (${failedN})`;
          if (successBtn) successBtn.textContent = `Success Count (${successN})`;
          if (noneBtn) noneBtn.textContent = `None Count (${noneN})`;
        }

        if (rows.length < 1) {
          const empty = document.createElement('div');
          empty.className = 'muted';
          empty.textContent = '(empty)';
          wrap.appendChild(empty);
          return;
        }

        const activeId = selectedProjectId();
        const cappedRows = rows.slice(0, 18);
        if (Boolean(controls.groupedView)) {
          const failedRows = [];
          const successRows = [];
          const noneRows = [];
          for (const row of cappedRows) {
            const st = projectHealthStatus(row);
            if (st === 'failed') failedRows.push(row);
            else if (st === 'success') successRows.push(row);
            else noneRows.push(row);
          }
          appendSessionBoardSection(wrap, 'Failed', failedRows, pinnedIds, activeId);
          appendSessionBoardSection(wrap, 'Success', successRows, pinnedIds, activeId);
          appendSessionBoardSection(wrap, 'None', noneRows, pinnedIds, activeId);
          if (wrap.childElementCount < 1) {
            const empty = document.createElement('div');
            empty.className = 'muted';
            empty.textContent = '(empty)';
            wrap.appendChild(empty);
          }
          return;
        }

        for (const row of cappedRows) {
          attachSessionCard(wrap, row, pinnedIds, activeId);
        }
      }


      async function openTopPrioritySession() {
        const rows = Array.isArray(state.projects) ? state.projects.slice() : [];
        if (rows.length < 1) {
          renderJsonOut({status: 'fail', error: 'no projects available'});
          return;
        }
        const scoreRow = (row) => {
          const rh = (row && row.runtime_health && typeof row.runtime_health === 'object') ? row.runtime_health : {};
          const failed = Number(rh.failed_steps || 0);
          const dur = Number(rh.recent_duration_ms || 0);
          const ratio = Number(rh.success_ratio || 0);
          return (failed * 1000) + (dur / 1000) - (ratio * 100);
        };
        rows.sort((a, b) => scoreRow(b) - scoreRow(a));
        const top = rows.find((r) => r && typeof r === 'object' && String(r.project_id || '').trim());
        if (!top) {
          renderJsonOut({status: 'fail', error: 'no valid project'});
          return;
        }
        const pid = String(top.project_id || '').trim();
        await openProjectWorkspace(pid, {push: true});
      }

      async function openNextFailedSession() {
        const rows = Array.isArray(state.projects) ? state.projects.slice() : [];
        if (rows.length < 1) {
          renderJsonOut({status: 'fail', error: 'no projects available'});
          return;
        }
        const failedRows = rows.filter((row) => {
          const st = String((row && row.runtime_health && row.runtime_health.status) || '').toLowerCase();
          return st === 'failed';
        });
        if (failedRows.length < 1) {
          renderJsonOut({status: 'fail', error: 'no failed project'});
          return;
        }
        failedRows.sort((a, b) => {
          const ad = Number((a && a.runtime_health && a.runtime_health.recent_duration_ms) || 0);
          const bd = Number((b && b.runtime_health && b.runtime_health.recent_duration_ms) || 0);
          if (bd !== ad) return bd - ad;
          return String((b && b.updated_at) || '').localeCompare(String((a && a.updated_at) || ''));
        });
        const top = failedRows.find((r) => r && typeof r === 'object' && String(r.project_id || '').trim());
        if (!top) {
          renderJsonOut({status: 'fail', error: 'no valid failed project'});
          return;
        }
        const pid = String(top.project_id || '').trim();
        await openProjectWorkspace(pid, {push: true});
      }

      async function openProjectWorkspace(projectId, opts) {
        const pid = String(projectId || '').trim();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        document.getElementById('project_id').value = pid;
        syncProjectPickerValue(pid);
        syncWorkspaceUrl(pid, opts && opts.push ? {push: true} : {});
        const hist = await loadHistory();
        const ok = hist && hist.status === 200 && hist.data && String(hist.data.status || '') === 'pass';
        if (!ok) {
          await saveProject();
        }
        await loadRunRuntime();
        await refreshProjects();
      }

      async function resumeProjectTask(projectId, taskId) {
        const pid = String(projectId || '').trim();
        const tid = String(taskId || '').trim();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!tid) {
          renderJsonOut({status: 'fail', error: 'project has no current_task_id'});
          return;
        }
        await openProjectWorkspace(pid, {push: true});
        const r = await apiPost('/api/resume', {
          task_id: tid,
          planner_provider: document.getElementById('planner').value,
          catalog_path: document.getElementById('catalog').value,
        });
        renderJsonOut(r.data);
        const status = String((r.data && r.data.status) || 'unknown');
        renderEvents([{stage: 'resume_project_task', status: status, operation: tid}]);
        await loadRunRuntime();
        await refreshProjects();
      }

      async function retryProjectFailedStep(projectId, taskId, failedToolName) {
        const pid = String(projectId || '').trim();
        const tid = String(taskId || '').trim();
        const failed = String(failedToolName || '').trim();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!tid) {
          renderJsonOut({status: 'fail', error: 'project has no current_task_id'});
          return;
        }
        if (!failed) {
          renderJsonOut({status: 'fail', error: 'project has no latest_failed_step'});
          return;
        }
        await openProjectWorkspace(pid, {push: true});
        const r = await apiPost(`/api/task/${encodeURIComponent(tid)}/retry-failed-step`, {
          catalog_path: document.getElementById('catalog').value,
          failed_tool_name: failed,
        });
        renderJsonOut(r.data);
        const status = String((r.data && r.data.status) || 'unknown');
        const op = String((r.data && r.data.retry_operation) || failed);
        renderEvents([{stage: 'retry_project_failed_step', status: status, operation: op}]);
        await loadRunRuntime();
        await refreshProjects();
      }

      async function showProjectTimeline(projectId, taskId) {
        const pid = String(projectId || '').trim();
        const tid = String(taskId || '').trim();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!tid) {
          renderJsonOut({status: 'fail', error: 'project has no current_task_id'});
          return;
        }
        await openProjectWorkspace(pid, {push: true});
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/timeline?sort=duration_desc`);
        renderJsonOut(r.data);
        if (r.data && Array.isArray(r.data.timeline_lines)) {
          document.getElementById('event_out').textContent = r.data.timeline_lines.join('\n');
        }
      }

      async function copyProjectTaskId(taskId) {
        const tid = String(taskId || '').trim();
        if (!tid) {
          renderJsonOut({status: 'fail', error: 'empty task_id'});
          return;
        }
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(tid);
            renderJsonOut({status: 'pass', copied_task_id: tid});
            return;
          }
        } catch (e) {
          // fall through
        }
        renderJsonOut({status: 'fail', error: 'clipboard_unavailable', task_id: tid});
      }

      async function showProjectSummary(projectId, taskId) {
        const pid = String(projectId || '').trim();
        const tid = String(taskId || '').trim();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!tid) {
          renderJsonOut({status: 'fail', error: 'project has no current_task_id'});
          return;
        }
        await openProjectWorkspace(pid, {push: true});
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/summary`);
        renderJsonOut(r.data);
      }

      async function validateProjectTask(projectId, taskId) {
        const pid = String(projectId || '').trim();
        const tid = String(taskId || '').trim();
        if (!pid || !isSafeProjectId(pid)) {
          renderJsonOut({status: 'fail', error: 'invalid project_id'});
          return;
        }
        if (!tid) {
          renderJsonOut({status: 'fail', error: 'project has no current_task_id'});
          return;
        }
        await openProjectWorkspace(pid, {push: true});
        const r = await apiGet(`/api/task/${encodeURIComponent(tid)}/validate`);
        renderJsonOut(r.data);
      }

      async function switchProjectFromPicker() {
        const picker = document.getElementById('project_picker');
        const pid = String(picker.value || '').trim();
        if (!pid) return;
        await openProjectWorkspace(pid, {push: true});
      }

      async function saveProject() {
        const projectId = selectedProjectId();
        const title = (document.getElementById('project_title').value || '').trim();
        const r = await apiPost('/api/projects', {
          project_id: projectId,
          title: title,
          options: collectOptions(),
          memory_notes: collectMemoryNotes(),
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
        await loadBatchHistory();
        renderFailedReplayQueue(state.failedReplayQueue);
        return r;
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
          memory_notes: collectMemoryNotes(),
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
        const pending = state.pendingInput && typeof state.pendingInput === 'object' ? state.pendingInput : {};
        const stage = String(pending.stage || '');
        if (sendNow && (stage === 'intake' || stage === 'approve' || stage === 'resume')) {
          await sendPendingResume();
          return;
        }
        setMessageInput(JSON.stringify(patch, null, 2));
        if (sendNow) {
          await sendChat(false);
        }
      }

      async function sendPendingResume() {
        const patch = collectPendingPatch();
        if (!patch || Object.keys(patch).length < 1) {
          renderJsonOut({status: 'fail', error: 'pending form has no values'});
          return;
        }
        const pid = selectedProjectId();
        const r = await apiPost('/api/chat/pending-submit', {
          project_id: pid,
          patch: patch,
          options: collectOptions(),
          memory_notes: collectMemoryNotes(),
        });
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
        updateMemoryStatus();
        bindComposerShortcuts();
        bindWorkspaceUrlNavigation();
        const savedSessionBoard = loadSessionBoardState();
        state.sessionBoard = savedSessionBoard;
        applySessionBoardStateToControls(savedSessionBoard);
        ensureSessionAutoRefresh();
        const urlProjectId = readProjectIdFromUrl();
        if (urlProjectId) {
          document.getElementById('project_id').value = urlProjectId;
        }
        await refreshProjects();
        const hist = await loadHistory();
        const ok = hist && hist.status === 200 && hist.data && String(hist.data.status || '') === 'pass';
        if (!ok) {
          await saveProject();
        }
        renderPromptHistory(currentProjectKey());
        await loadRunRuntime();
        renderFailedReplayQueue(state.failedReplayQueue);
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


def _normalize_resume_overrides(raw: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out

    def _pick_str(src: Dict[str, Any], key: str) -> str:
        return str(src.get(key) or "").strip()

    for key in ("candidate_data", "train_data", "prediction_model", "property", "range"):
        val = _pick_str(raw, key)
        if val:
            out[key] = val

    n_val = raw.get("n_structures")
    try:
        n = int(n_val)
    except Exception:
        n = 0
    if n > 0:
        out["n_structures"] = n

    predictor_id = _pick_str(raw, "predictor_id")
    generator_id = _pick_str(raw, "generator_id")
    if not predictor_id or not generator_id:
        for mk in ("model_preferences", "model_choice"):
            model = raw.get(mk)
            if not isinstance(model, dict):
                continue
            if not predictor_id:
                predictor_id = _pick_str(model, "predictor_id")
            if not generator_id:
                generator_id = _pick_str(model, "generator_id")
            if predictor_id and generator_id:
                break
    if predictor_id:
        out["predictor_id"] = predictor_id
    if generator_id:
        out["generator_id"] = generator_id
    return out


def _run_agent_resume(*, task_id: str, planner_provider: str, catalog_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    catalog = _resolve_catalog(catalog_path)
    ov = _normalize_resume_overrides(overrides)
    cli_args = [
        "agent-resume",
        "--workspace-root",
        str(REPO_ROOT),
        "--task-id",
        task_id,
        "--planner-provider",
        str(planner_provider or "rule_based_v1"),
        "--catalog",
        str(catalog),
    ]
    flag_pairs = (
        ("candidate_data", "--candidate-data"),
        ("train_data", "--train-data"),
        ("prediction_model", "--prediction-model"),
        ("property", "--property"),
        ("range", "--range"),
        ("n_structures", "--n-structures"),
        ("predictor_id", "--predictor-id"),
        ("generator_id", "--generator-id"),
    )
    for key, flag in flag_pairs:
        if key not in ov:
            continue
        value = ov.get(key)
        if key == "n_structures":
            try:
                v = int(value)
            except Exception:
                continue
            if v < 1:
                continue
            cli_args.extend([flag, str(v)])
            continue
        val_text = str(value or "").strip()
        if not val_text:
            continue
        cli_args.extend([flag, val_text])

    return _run_cli_command(
        cli_args=cli_args,
        ok_returncodes=[0, 2],
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


def _is_safe_export_id(export_id: str) -> bool:
    eid = str(export_id or "").strip()
    if not eid:
        return False
    if not TASK_ID_PATTERN.fullmatch(eid):
        return False
    if ".." in eid or "/" in eid or "\\" in eid:
        return False
    return True


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


def _ui_batch_exports_root(project_id: str) -> Path:
    p = (REPO_ROOT / BATCH_EXPORTS_DIR_REL / project_id).resolve()
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


def _normalize_memory_notes(raw: Any) -> str:
    text = str(raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > MAX_MEMORY_NOTES_CHARS:
        text = text[:MAX_MEMORY_NOTES_CHARS]
    return text


def _normalize_project_options(raw: Any) -> Dict[str, Any]:
    options = raw if isinstance(raw, dict) else {}
    planner = str(options.get("planner_provider") or "rule_based_v1").strip() or "rule_based_v1"
    catalog = str(options.get("catalog_path") or DEFAULT_CATALOG).strip() or DEFAULT_CATALOG
    web_enabled = bool(options.get("web_search_enabled", True))
    web_topk = _as_int(options.get("web_topk"), 5)
    web_topk = max(1, min(web_topk, 20))
    memory_enabled = bool(options.get("memory_enabled", False))
    batch_replay_defaults = _normalize_batch_replay_options(options.get("batch_replay_defaults"))
    return {
        "planner_provider": planner,
        "catalog_path": catalog,
        "web_search_enabled": web_enabled,
        "web_topk": web_topk,
        "memory_enabled": memory_enabled,
        "batch_replay_defaults": batch_replay_defaults,
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
        "memory_notes": "",
        "memory_updated_at": "",
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
        "memory_notes": _normalize_memory_notes(project.get("memory_notes")),
        "memory_updated_at": str(project.get("memory_updated_at") or ""),
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
            "success_ratio": 0.0,
            "latest_failed_step": "",
            "latest_failed_error": "",
            "recent_duration_ms": 0,
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
            "success_ratio": 0.0,
            "latest_failed_step": "",
            "latest_failed_error": "",
            "recent_duration_ms": 0,
        }
    records = execution.get("records") if isinstance(execution.get("records"), list) else []
    success_steps = 0
    failed_steps = 0
    latest_failed_step = ""
    latest_failed_error = ""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "") == "success":
            success_steps += 1
        else:
            failed_steps += 1
            latest_failed_step = str(rec.get("name") or latest_failed_step)
            err_txt = str(rec.get("error") or "").strip()
            if err_txt:
                latest_failed_error = err_txt[:240]
    total_steps = max(0, success_steps + failed_steps)
    success_ratio = (float(success_steps) / float(total_steps)) if total_steps > 0 else 0.0
    recent_duration_ms = 0
    started_raw = str(execution.get("started_at") or "").strip()
    ended_raw = str(execution.get("ended_at") or "").strip()
    if started_raw and ended_raw:
        try:
            started_dt = datetime.fromisoformat(started_raw)
            ended_dt = datetime.fromisoformat(ended_raw)
            dur = int((ended_dt - started_dt).total_seconds() * 1000)
            if dur > 0:
                recent_duration_ms = dur
        except Exception:
            recent_duration_ms = 0
    if recent_duration_ms <= 0:
        record_durations: List[int] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            st = str(rec.get("started_at") or "").strip()
            ed = str(rec.get("ended_at") or "").strip()
            if not st or not ed:
                continue
            try:
                st_dt = datetime.fromisoformat(st)
                ed_dt = datetime.fromisoformat(ed)
                d = int((ed_dt - st_dt).total_seconds() * 1000)
            except Exception:
                d = 0
            if d > 0:
                record_durations.append(d)
        if record_durations:
            recent_duration_ms = sum(record_durations)
    return {
        "status": str(execution.get("status") or "unknown"),
        "task_id": task_id,
        "record_count": len(records),
        "success_steps": success_steps,
        "failed_steps": failed_steps,
        "success_ratio": success_ratio,
        "latest_failed_step": latest_failed_step,
        "latest_failed_error": latest_failed_error,
        "recent_duration_ms": recent_duration_ms,
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
    payload["memory_notes"] = _normalize_memory_notes(payload.get("memory_notes"))
    payload["memory_updated_at"] = str(payload.get("memory_updated_at") or "").strip()
    if payload["memory_notes"] and not payload["memory_updated_at"]:
        payload["memory_updated_at"] = str(payload.get("updated_at") or "")
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
    project["memory_notes"] = _normalize_memory_notes(project.get("memory_notes"))
    project["memory_updated_at"] = str(project.get("memory_updated_at") or "").strip()
    if project["memory_notes"] and not project["memory_updated_at"]:
        project["memory_updated_at"] = str(project.get("updated_at") or "")
    if not project["memory_notes"]:
        project["memory_updated_at"] = ""
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


def _apply_project_memory_update(project: Dict[str, Any], memory_notes: Any, *, provided: bool) -> None:
    current_notes = _normalize_memory_notes(project.get("memory_notes"))
    project["memory_notes"] = current_notes
    current_updated = str(project.get("memory_updated_at") or "").strip()
    if not provided:
        project["memory_updated_at"] = current_updated
        return
    next_notes = _normalize_memory_notes(memory_notes)
    project["memory_notes"] = next_notes
    if next_notes != current_notes or (next_notes and not current_updated):
        project["memory_updated_at"] = _now_iso()
    elif not next_notes:
        project["memory_updated_at"] = ""
    else:
        project["memory_updated_at"] = current_updated


def _compose_intake_request_text(*, message: str, project: Dict[str, Any], options: Dict[str, Any]) -> Tuple[str, bool]:
    base = str(message or "").strip()
    if not base:
        return "", False
    if not bool(options.get("memory_enabled", False)):
        return base, False
    notes = _normalize_memory_notes(project.get("memory_notes"))
    if not notes:
        return base, False
    merged = f"{base}\n\nProject memory context:\n{notes}"
    return merged, True


def _batch_export_entry_path(project_id: str, export_id: str) -> Path:
    safe_export_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(export_id or "").strip()).strip("._") or "batch"
    return (_ui_batch_exports_root(project_id) / f"{safe_export_id}.json").resolve()


def _load_batch_export_entry(project_id: str, export_id: str) -> Optional[Dict[str, Any]]:
    if not _is_safe_project_id(project_id) or not _is_safe_export_id(export_id):
        return None
    path = _batch_export_entry_path(project_id, export_id)
    payload = _load_json_if_exists(path)
    if not isinstance(payload, dict):
        return None
    payload["path"] = str(path)
    payload["export_id"] = str(payload.get("export_id") or export_id)
    payload["project_id"] = str(payload.get("project_id") or project_id)
    return payload


def _delete_batch_export_entry(project_id: str, export_id: str) -> bool:
    if not _is_safe_project_id(project_id) or not _is_safe_export_id(export_id):
        return False
    path = _batch_export_entry_path(project_id, export_id)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _list_batch_export_entries(
    project_id: str,
    *,
    limit: int = 20,
    offset: int = 0,
    action_filter: str = "",
    status_filter: str = "",
) -> Tuple[List[Dict[str, Any]], int]:
    root = _ui_batch_exports_root(project_id)
    rows: List[Dict[str, Any]] = []
    for path in root.glob("*.json"):
        if not path.is_file():
            continue
        payload = _load_json_if_exists(path)
        if not isinstance(payload, dict):
            continue
        summary = _batch_export_summary(payload, export_id=str(payload.get("export_id") or path.stem), project_id=project_id)
        replay_metrics = payload.get("replay_metrics") if isinstance(payload.get("replay_metrics"), dict) else {}
        replay_options = payload.get("replay_options") if isinstance(payload.get("replay_options"), dict) else {}
        rows.append(
            {
                "export_id": str(summary.get("export_id") or path.stem),
                "project_id": str(summary.get("project_id") or project_id),
                "batch_type": str(payload.get("batch_type") or "unknown"),
                "created_at": str(summary.get("created_at") or payload.get("created_at") or ""),
                "path": str(path),
                "count": _as_int(summary.get("count"), 0),
                "limit": _as_int(summary.get("limit"), 0),
                "action": str(summary.get("action") or payload.get("action") or ""),
                "status": str(summary.get("status") or payload.get("status") or ""),
                "replay_metrics": replay_metrics,
                "replay_options": {
                    "dry_run": bool(replay_options.get("dry_run")),
                    "failed_only": bool(replay_options.get("failed_only")),
                    "retry_max": _as_int(replay_options.get("retry_max"), 0),
                    "retry_backoff_ms": _as_int(replay_options.get("retry_backoff_ms"), 0),
                    "max_concurrency": _as_int(replay_options.get("max_concurrency"), 1),
                },
            }
        )
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    act = str(action_filter or "").strip()
    st = str(status_filter or "").strip().lower()
    if act:
        rows = [row for row in rows if str(row.get("action") or "") == act]
    if st:
        rows = [row for row in rows if str(row.get("status") or "").lower() == st]
    total = len(rows)
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    return rows[safe_offset : safe_offset + safe_limit], total


def _save_batch_export_entry(project_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    export_id = str(payload.get("export_id") or "").strip() or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    entry = dict(payload)
    entry["export_id"] = export_id
    entry["project_id"] = project_id
    entry["created_at"] = str(entry.get("created_at") or _now_iso())
    entry["path"] = str(_batch_export_entry_path(project_id, export_id))
    path = _batch_export_entry_path(project_id, export_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def _batch_export_source_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload.get("batch_result")
    if isinstance(source, dict):
        return source
    return payload


def _batch_export_summary(payload: Dict[str, Any], *, export_id: str, project_id: str) -> Dict[str, Any]:
    source = _batch_export_source_payload(payload)
    rows = source.get("rows") if isinstance(source.get("rows"), list) else []
    results = source.get("results") if isinstance(source.get("results"), list) else []
    retries = source.get("retries") if isinstance(source.get("retries"), list) else []
    count_default = len(results) + len(retries)
    if count_default < 1:
        count_default = len(rows)
    return {
        "export_id": str(payload.get("export_id") or export_id),
        "project_id": str(payload.get("project_id") or project_id),
        "action": str(source.get("action") or payload.get("action") or ""),
        "status": str(source.get("status") or payload.get("status") or ""),
        "count": _as_int(source.get("count"), count_default),
        "limit": _as_int(source.get("limit"), len(rows)),
        "rows_count": len(rows),
        "results_count": len(results),
        "retries_count": len(retries),
        "created_at": str(payload.get("created_at") or source.get("created_at") or ""),
        "replayed_from_export_id": str(source.get("replayed_from_export_id") or payload.get("source_export_id") or ""),
    }


def _batch_export_compare_lines(primary: Dict[str, Any], other: Dict[str, Any], diff: Dict[str, Any]) -> List[str]:
    p_eid = str(primary.get("export_id") or "")
    o_eid = str(other.get("export_id") or "")
    out = [
        f"action {p_eid}={str(primary.get('action') or '')} vs {o_eid}={str(other.get('action') or '')}",
        f"status {p_eid}={str(primary.get('status') or '')} vs {o_eid}={str(other.get('status') or '')}",
        f"count {p_eid}={int(primary.get('count') or 0)} vs {o_eid}={int(other.get('count') or 0)} delta={int(primary.get('count') or 0) - int(other.get('count') or 0)}",
        f"rows {p_eid}={int(primary.get('rows_count') or 0)} vs {o_eid}={int(other.get('rows_count') or 0)} delta={int(primary.get('rows_count') or 0) - int(other.get('rows_count') or 0)}",
        f"changed_paths={int(diff.get('changed_count') or 0)} only_primary={int(diff.get('only_in_primary_count') or 0)} only_other={int(diff.get('only_in_other_count') or 0)}",
    ]
    return out


def _batch_export_download_filename(*, project_id: str, export_id: str, action: str, fmt: str) -> str:
    safe_project = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_id or "")).strip("._") or "project"
    safe_export = re.sub(r"[^A-Za-z0-9._-]+", "_", str(export_id or "")).strip("._") or "batch"
    safe_action = re.sub(r"[^A-Za-z0-9._-]+", "_", str(action or "")).strip("._") or "batch_export"
    ext = "csv" if str(fmt or "").lower() == "csv" else "json"
    return f"{safe_project}_{safe_action}_{safe_export}.{ext}"


def _batch_export_csv_text(payload: Dict[str, Any], *, export_id: str, project_id: str) -> str:
    source = _batch_export_source_payload(payload)
    action = str(source.get("action") or payload.get("action") or "")
    status = str(source.get("status") or payload.get("status") or "")
    created_at = str(payload.get("created_at") or source.get("created_at") or "")
    rows = source.get("rows") if isinstance(source.get("rows"), list) else []
    results = source.get("results") if isinstance(source.get("results"), list) else []
    retries = source.get("retries") if isinstance(source.get("retries"), list) else []
    flattened: List[Dict[str, Any]] = []
    for section_name, items in (("rows", rows), ("results", results), ("retries", retries)):
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            flattened.append(
                {
                    "section": section_name,
                    "index": idx,
                    "export_id": str(payload.get("export_id") or export_id),
                    "project_id": str(payload.get("project_id") or project_id),
                    "action": action,
                    "status": status,
                    "task_id": str(item.get("task_id") or ""),
                    "item_status": str(item.get("status") or ""),
                    "http_status": str(item.get("http_status") or ""),
                    "failed_tool_name": str(item.get("failed_tool_name") or ""),
                    "created_at": created_at,
                    "item_json": json.dumps(item, ensure_ascii=False),
                }
            )
    if not flattened:
        flattened.append(
            {
                "section": "meta",
                "index": 1,
                "export_id": str(payload.get("export_id") or export_id),
                "project_id": str(payload.get("project_id") or project_id),
                "action": action,
                "status": status,
                "task_id": "",
                "item_status": "",
                "http_status": "",
                "failed_tool_name": "",
                "created_at": created_at,
                "item_json": json.dumps(_batch_export_summary(payload, export_id=export_id, project_id=project_id), ensure_ascii=False),
            }
        )
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "section",
            "index",
            "export_id",
            "project_id",
            "action",
            "status",
            "task_id",
            "item_status",
            "http_status",
            "failed_tool_name",
            "created_at",
            "item_json",
        ],
    )
    writer.writeheader()
    for row in flattened:
        writer.writerow(row)
    return buf.getvalue()


def _extract_response_json_and_status(resp: Any) -> Tuple[int, Dict[str, Any]]:
    status = 200
    response_obj = resp
    if isinstance(resp, tuple):
        if len(resp) >= 2 and isinstance(resp[1], int):
            status = int(resp[1])
        response_obj = resp[0]
    try:
        if hasattr(response_obj, "status_code"):
            status = int(getattr(response_obj, "status_code"))
    except Exception:
        pass
    try:
        data = response_obj.get_json(silent=True) if hasattr(response_obj, "get_json") else None
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {}
    return status, data


def _normalize_batch_replay_options(raw: Any) -> Dict[str, Any]:
    body = raw if isinstance(raw, dict) else {}
    retry_max = max(0, min(_as_int(body.get("retry_max"), 0), 3))
    retry_backoff_ms = max(0, min(_as_int(body.get("retry_backoff_ms"), 150), 5000))
    max_concurrency = max(1, min(_as_int(body.get("max_concurrency"), 2), 8))
    return {
        "dry_run": bool(body.get("dry_run")),
        "failed_only": bool(body.get("failed_only")),
        "retry_max": retry_max,
        "retry_backoff_ms": retry_backoff_ms,
        "max_concurrency": max_concurrency,
    }


def _row_item_has_failure(item: Dict[str, Any]) -> bool:
    status_text = str(item.get("status") or "").strip().lower()
    if status_text in {"fail", "failed", "error", "missing"}:
        return True
    http_status = _as_int(item.get("http_status"), 0)
    if http_status >= 400:
        return True
    data = item.get("data")
    if isinstance(data, dict):
        data_status = str(data.get("status") or "").strip().lower()
        if data_status in {"fail", "failed", "error", "missing"}:
            return True
    return False


def _filter_failed_only_replay_rows(
    *,
    action: str,
    rows: List[Dict[str, Any]],
    source_batch: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int]:
    failed_task_ids: set[str] = set()
    failed_task_steps: set[Tuple[str, str]] = set()
    for bucket_name in ("results", "retries"):
        bucket = source_batch.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            if not _row_item_has_failure(item):
                continue
            tid = str(item.get("task_id") or "").strip()
            if not tid:
                continue
            failed_task_ids.add(tid)
            failed_step = str(item.get("failed_tool_name") or item.get("latest_failed_step") or "").strip()
            if failed_step:
                failed_task_steps.add((tid, failed_step))

    picked: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("task_id") or "").strip()
        if not tid:
            continue
        failed_step = str(row.get("failed_tool_name") or row.get("latest_failed_step") or "").strip()
        if (tid in failed_task_ids) or (failed_step and (tid, failed_step) in failed_task_steps):
            picked.append(row)

    if not picked and action == "batch_retry_failed":
        picked = [row for row in rows if isinstance(row, dict) and str(row.get("latest_failed_step") or row.get("failed_tool_name") or "").strip()]
    return picked, len(failed_task_ids)


def _batch_item_failure_reason(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "invalid_item"
    status_text = str(item.get("status") or "").strip().lower()
    if status_text in {"fail", "failed", "error", "missing"}:
        return status_text
    http_status = _as_int(item.get("http_status"), 0)
    if http_status >= 400:
        return f"http_{http_status}"
    data = item.get("data")
    if isinstance(data, dict):
        data_status = str(data.get("status") or "").strip().lower()
        if data_status in {"fail", "failed", "error", "missing"}:
            return data_status
        data_error = str(data.get("error") or "").strip().lower()
        if data_error:
            return data_error[:96]
    err = str(item.get("error") or "").strip().lower()
    if err:
        return err[:96]
    return "unknown_failure"


def _extract_failed_queue_rows_from_source_batch(
    *,
    source_batch: Dict[str, Any],
    project_id: str,
    source_export_id: str,
) -> Dict[str, Any]:
    action = str(source_batch.get("action") or "").strip()
    queue_rows: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for bucket_name in ("results", "retries"):
        bucket = source_batch.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            if not _row_item_has_failure(item):
                continue
            tid = str(item.get("task_id") or "").strip()
            if not tid:
                continue
            failed_step = str(item.get("failed_tool_name") or item.get("latest_failed_step") or "").strip()
            reason = _batch_item_failure_reason(item)
            row = {
                "task_id": tid,
                "project_id": str(item.get("project_id") or project_id),
                "latest_failed_step": failed_step,
                "failed_tool_name": failed_step,
                "source_export_id": source_export_id,
                "source_action": action,
                "source_bucket": bucket_name,
                "failure_reason": reason,
            }
            key = (tid, failed_step)
            if key in by_key:
                continue
            by_key[key] = row
            queue_rows.append(row)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if not queue_rows and action == "batch_retry_failed":
        rows = source_batch.get("rows")
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                tid = str(item.get("task_id") or "").strip()
                failed_step = str(item.get("latest_failed_step") or item.get("failed_tool_name") or "").strip()
                if not tid or not failed_step:
                    continue
                key = (tid, failed_step)
                if key in by_key:
                    continue
                row = {
                    "task_id": tid,
                    "project_id": str(item.get("project_id") or project_id),
                    "latest_failed_step": failed_step,
                    "failed_tool_name": failed_step,
                    "source_export_id": source_export_id,
                    "source_action": action,
                    "source_bucket": "rows",
                    "failure_reason": "previous_failed_step",
                }
                by_key[key] = row
                queue_rows.append(row)
                reason_counts["previous_failed_step"] = reason_counts.get("previous_failed_step", 0) + 1

    reason_ranked = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "status": "pass",
        "action": action,
        "source_export_id": source_export_id,
        "rows": queue_rows,
        "count": len(queue_rows),
        "unique_task_count": len({str(row.get("task_id") or "") for row in queue_rows}),
        "failure_reasons": [{"reason": k, "count": v} for k, v in reason_ranked[:12]],
    }


def _classify_batch_replay_response(status_code: int, data: Dict[str, Any]) -> Tuple[str, str]:
    if int(status_code) >= 400:
        return "fail", f"http_{int(status_code)}"
    if not isinstance(data, dict):
        return "pass", ""
    status_text = str(data.get("status") or "").strip().lower()
    if status_text in {"fail", "error", "missing"}:
        return "fail", status_text or "fail"
    return "pass", status_text or "pass"


def _invoke_batch_replay_action(*, action: str, task_id: str, failed_step: str) -> Tuple[int, Dict[str, Any]]:
    if action == "batch_summary":
        with app.test_request_context(f"/api/task/{task_id}/summary", method="GET"):
            return _extract_response_json_and_status(api_task_summary(task_id))
    if action == "batch_validate":
        with app.test_request_context(f"/api/task/{task_id}/validate", method="GET"):
            return _extract_response_json_and_status(api_task_validate(task_id))
    if action == "batch_retry_failed":
        with app.test_request_context(
            f"/api/task/{task_id}/retry-failed-step",
            method="POST",
            json={"failed_tool_name": failed_step, "catalog_path": DEFAULT_CATALOG},
        ):
            return _extract_response_json_and_status(api_task_retry_failed_step(task_id))
    return 500, {"status": "fail", "error": "unsupported_batch_export_action", "action": action}


def _run_batch_replay_row(*, action: str, row: Dict[str, Any], options: Dict[str, Any], project_id: str) -> Dict[str, Any]:
    tid = str(row.get("task_id") or "").strip()
    out: Dict[str, Any] = {
        "task_id": tid,
        "project_id": str(row.get("project_id") or project_id),
    }
    failed_step = str(row.get("latest_failed_step") or row.get("failed_tool_name") or "").strip()
    if failed_step:
        out["failed_tool_name"] = failed_step
    if not _is_safe_task_id(tid):
        out.update({"status": "skipped", "reason": "invalid_task_id", "attempts": 0, "retry_count": 0, "duration_ms": 0})
        return out
    if action == "batch_retry_failed" and not failed_step:
        out.update({"status": "skipped", "reason": "missing_failed_step", "attempts": 0, "retry_count": 0, "duration_ms": 0})
        return out
    max_attempts = max(1, 1 + int(options.get("retry_max") or 0))
    if bool(options.get("dry_run")):
        out.update(
            {
                "status": "dry_run",
                "reason": "dry_run_only",
                "attempts": 0,
                "retry_count": 0,
                "duration_ms": 0,
                "planned_attempts": max_attempts,
                "planned_action": action,
            }
        )
        return out

    attempt_logs: List[Dict[str, Any]] = []
    final_http_status = 500
    final_data: Dict[str, Any] = {}
    final_status = "fail"
    final_reason = "unknown"
    total_duration_ms = 0
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        http_status, data = _invoke_batch_replay_action(action=action, task_id=tid, failed_step=failed_step)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        total_duration_ms += max(0, elapsed_ms)
        status_tag, reason = _classify_batch_replay_response(http_status, data)
        attempt_logs.append(
            {
                "attempt": attempt,
                "http_status": int(http_status),
                "status": status_tag,
                "reason": reason,
                "duration_ms": max(0, elapsed_ms),
            }
        )
        final_http_status = int(http_status)
        final_data = data if isinstance(data, dict) else {}
        final_status = status_tag
        final_reason = reason
        if status_tag == "pass":
            break
        if attempt < max_attempts and int(options.get("retry_backoff_ms") or 0) > 0:
            time.sleep(int(options.get("retry_backoff_ms") or 0) / 1000.0)

    out.update(
        {
            "status": final_status,
            "reason": final_reason,
            "http_status": final_http_status,
            "data": final_data,
            "attempts": len(attempt_logs),
            "retry_count": max(0, len(attempt_logs) - 1),
            "attempt_logs": attempt_logs,
            "duration_ms": total_duration_ms,
        }
    )
    return out


def _summarize_batch_replay_items(items: List[Dict[str, Any]], *, elapsed_ms: int, options: Dict[str, Any], applied_concurrency: int) -> Dict[str, Any]:
    ok_count = 0
    fail_count = 0
    skipped_count = 0
    dry_run_count = 0
    attempts_total = 0
    retry_total = 0
    item_duration_ms = 0
    failed_task_ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().lower()
        if status == "pass":
            ok_count += 1
        elif status == "dry_run":
            dry_run_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            fail_count += 1
            tid = str(item.get("task_id") or "").strip()
            if tid:
                failed_task_ids.append(tid)
        attempts_total += max(0, _as_int(item.get("attempts"), 0))
        retry_total += max(0, _as_int(item.get("retry_count"), 0))
        item_duration_ms += max(0, _as_int(item.get("duration_ms"), 0))
    return {
        "ok_count": ok_count,
        "fail_count": fail_count,
        "skipped_count": skipped_count,
        "dry_run_count": dry_run_count,
        "attempts_total": attempts_total,
        "retry_count_total": retry_total,
        "elapsed_ms": max(0, int(elapsed_ms)),
        "item_duration_ms_total": item_duration_ms,
        "max_concurrency_requested": int(options.get("max_concurrency") or 1),
        "max_concurrency_applied": max(1, int(applied_concurrency)),
        "failed_task_ids": failed_task_ids[:20],
    }


def _replay_batch_export_payload(
    *,
    project_id: str,
    payload: Dict[str, Any],
    source_export_id: str,
    replay_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_batch = _batch_export_source_payload(payload)
    action = str(source_batch.get("action") or "").strip()
    rows = source_batch.get("rows") if isinstance(source_batch.get("rows"), list) else []
    replay_limit = max(1, min(_as_int(source_batch.get("limit"), len(rows) if rows else 5), 20))
    options = _normalize_batch_replay_options(replay_options)
    base_rows: List[Dict[str, Any]] = rows[:replay_limit]
    replay_rows = list(base_rows)
    if not replay_rows:
        derived: List[Dict[str, Any]] = []
        results = source_batch.get("results") if isinstance(source_batch.get("results"), list) else []
        retries = source_batch.get("retries") if isinstance(source_batch.get("retries"), list) else []
        for item in results:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("task_id") or "").strip()
            if tid:
                derived.append({"task_id": tid, "project_id": str(item.get("project_id") or project_id)})
        for item in retries:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("task_id") or "").strip()
            if tid:
                derived.append(
                    {
                        "task_id": tid,
                        "project_id": str(item.get("project_id") or project_id),
                        "latest_failed_step": str(item.get("failed_tool_name") or ""),
                    }
                )
        base_rows = derived[:replay_limit]
        replay_rows = list(base_rows)

    if action not in {"batch_summary", "batch_validate", "batch_retry_failed"}:
        return {"status": "fail", "error": "unsupported_batch_export_action", "action": action}

    failed_source_count = 0
    if bool(options.get("failed_only")):
        replay_rows, failed_source_count = _filter_failed_only_replay_rows(action=action, rows=replay_rows, source_batch=source_batch)

    started = time.perf_counter()
    replay_results: List[Dict[str, Any]] = []
    requested_concurrency = int(options.get("max_concurrency") or 1)
    if action in {"batch_summary", "batch_validate"} and requested_concurrency > 1 and not bool(options.get("dry_run")):
        with ThreadPoolExecutor(max_workers=requested_concurrency) as pool:
            future_map = {}
            for idx, row in enumerate(replay_rows):
                if not isinstance(row, dict):
                    continue
                future = pool.submit(_run_batch_replay_row, action=action, row=row, options=options, project_id=project_id)
                future_map[future] = idx
            indexed_results: List[Tuple[int, Dict[str, Any]]] = []
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    item = future.result()
                except Exception as exc:
                    item = {
                        "status": "fail",
                        "reason": f"internal_error:{type(exc).__name__}: {exc}",
                        "attempts": 0,
                        "retry_count": 0,
                        "duration_ms": 0,
                    }
                indexed_results.append((idx, item if isinstance(item, dict) else {"status": "fail", "reason": "invalid_row_result"}))
            indexed_results.sort(key=lambda x: x[0])
            replay_results = [item for _, item in indexed_results]
        applied_concurrency = requested_concurrency
    else:
        for row in replay_rows:
            if not isinstance(row, dict):
                continue
            replay_results.append(_run_batch_replay_row(action=action, row=row, options=options, project_id=project_id))
        applied_concurrency = 1 if action == "batch_retry_failed" else requested_concurrency

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    replay_metrics = _summarize_batch_replay_items(
        replay_results,
        elapsed_ms=elapsed_ms,
        options=options,
        applied_concurrency=applied_concurrency,
    )
    replay_metrics["base_rows_count"] = len(base_rows)
    replay_metrics["effective_rows_count"] = len(replay_rows)
    replay_metrics["failed_source_count"] = int(failed_source_count)
    replay_metrics["failed_only"] = bool(options.get("failed_only"))
    top_status = "pass" if int(replay_metrics.get("fail_count") or 0) < 1 else "partial"
    out = {
        "status": top_status,
        "action": action,
        "limit": replay_limit,
        "count": len(replay_results),
        "rows": replay_rows,
        "replayed_from_export_id": source_export_id,
        "created_at": _now_iso(),
        "replay_options": options,
        "replay_metrics": replay_metrics,
    }
    if action == "batch_retry_failed":
        out["retries"] = replay_results
    else:
        out["results"] = replay_results
    return out


def _normalize_import_project(raw: Dict[str, Any], *, project_id: str) -> Dict[str, Any]:
    base = _new_project_state(project_id, title=str(raw.get("title") or project_id), options=raw.get("options") if isinstance(raw.get("options"), dict) else {})
    if str(raw.get("created_at") or "").strip():
        base["created_at"] = str(raw.get("created_at"))
    base["memory_notes"] = _normalize_memory_notes(raw.get("memory_notes"))
    base["memory_updated_at"] = str(raw.get("memory_updated_at") or "").strip()
    if base["memory_notes"] and not base["memory_updated_at"]:
        base["memory_updated_at"] = str(raw.get("updated_at") or base.get("created_at") or "")
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


def _chat_resume_from_pending(
    *,
    project: Dict[str, Any],
    patch: Dict[str, Any],
    source_message: str,
) -> Dict[str, Any]:
    options = _normalize_project_options(project.get("options"))
    planner = str(options.get("planner_provider") or "rule_based_v1")
    catalog = str(options.get("catalog_path") or DEFAULT_CATALOG)
    pending = project.get("pending_input") if isinstance(project.get("pending_input"), dict) else {}
    stage = str(pending.get("stage") or "").strip()
    if stage not in {"intake", "approve", "resume"}:
        _append_message(
            project,
            role="assistant",
            kind="assistant",
            content=f"当前 pending stage={stage or '-'}，请使用普通 Send 或 Step 流程。",
        )
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "resume", "status": "fail", "reason": "invalid_pending_stage"}],
        }

    task_id = str(project.get("current_task_id") or "").strip()
    if not task_id:
        draft_path = _resolve_optional_path(project.get("task_draft_path"))
        if draft_path is not None:
            task_id = str(draft_path.parent.name or "").strip()
    if not task_id:
        _append_message(project, role="assistant", kind="assistant", content="pending continue 缺少 task_id。")
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "resume", "status": "fail", "reason": "missing_task_id"}],
        }

    started_at = datetime.now()
    resume = _run_agent_resume(
        task_id=task_id,
        planner_provider=planner,
        catalog_path=catalog,
        overrides=patch,
    )
    elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)

    if resume.get("status") != "pass":
        _append_message(project, role="assistant", kind="assistant", content=_assistant_cli_fail_text("agent-resume", resume))
        project["last_runtime"] = {"status": "failed", "duration_ms": elapsed_ms, "operation": "resume", "updated_at": _now_iso()}
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "resume", "status": "fail"}],
            "resume_result": resume.get("result"),
        }

    rr = resume.get("result") if isinstance(resume.get("result"), dict) else {}
    rr_status = str(rr.get("status") or "").strip()
    if rr_status == "need_user_input":
        pending_next = _pending_input_payload(
            stage="resume",
            missing_fields=rr.get("missing_fields"),
            questions=rr.get("questions"),
            task_draft_path=rr.get("task_draft_path"),
        )
        project["pending_input"] = pending_next
        _append_message(
            project,
            role="assistant",
            kind="assistant",
            content=_assistant_need_input_text(rr.get("missing_fields"), rr.get("questions")),
            meta={"resume_result": rr, "source_message": source_message},
        )
        project["last_runtime"] = {
            "status": "need_user_input",
            "duration_ms": elapsed_ms,
            "operation": "resume",
            "updated_at": _now_iso(),
        }
        project = _save_project_state(project)
        return {
            "status": "need_user_input",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "resume", "status": "need_user_input"}],
            "pending_input": pending_next,
            "resume_result": rr,
        }

    if rr_status != "success":
        _append_message(project, role="assistant", kind="assistant", content=f"agent-resume 返回未知状态: {rr_status or '(empty)'}")
        project["last_runtime"] = {"status": "failed", "duration_ms": elapsed_ms, "operation": "resume", "updated_at": _now_iso()}
        project = _save_project_state(project)
        return {
            "status": "fail",
            "project": _project_summary(project),
            "messages": _recent_messages(project),
            "events": [{"stage": "resume", "status": "fail", "reason": "unexpected_status"}],
            "resume_result": rr,
        }

    project["pending_input"] = {}
    project["current_task_id"] = str(rr.get("task_id") or task_id)
    task_path = _resolve_optional_path(rr.get("task_path"))
    if task_path is not None:
        project["task_json_path"] = str(task_path)
    request_path = _resolve_optional_path(rr.get("request_path"))
    if request_path is not None:
        project["request_path"] = str(request_path)
    run_label = str(rr.get("run_label") or "")
    result_dir = str(rr.get("result_dir") or "")
    _append_message(
        project,
        role="assistant",
        kind="assistant",
        content=(
            f"已根据补充字段继续执行: status=success"
            f"\nrun_label={run_label}"
            f"\nresult_dir={result_dir}"
        ),
        meta={"resume_result": rr, "source_message": source_message},
    )
    project["last_runtime"] = {
        "status": "success",
        "duration_ms": elapsed_ms,
        "operation": "resume",
        "run_label": run_label,
        "result_dir": result_dir,
        "updated_at": _now_iso(),
    }
    project = _save_project_state(project)
    return {
        "status": "pass",
        "project": _project_summary(project),
        "messages": _recent_messages(project),
        "events": [{"stage": "resume", "status": "success"}],
        "resume_result": rr,
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
        intake_request_text, memory_injected = _compose_intake_request_text(message=message, project=project, options=options)
        if memory_injected:
            _append_message(
                project,
                role="system",
                kind="memory_context",
                content="Project memory injected into intake request.",
                meta={"memory_chars": len(_normalize_memory_notes(project.get("memory_notes")))},
            )
        intake = _run_agent_intake(task_id=task_id, request_text=intake_request_text, web_topk=web_topk, enable_web_search=web_enabled)
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
    memory_notes_provided = "memory_notes" in body
    memory_notes = body.get("memory_notes")
    if not project_id:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(project_id):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400

    project = _load_project_state(project_id)
    if not isinstance(project, dict):
        project = _new_project_state(project_id, title=title, options=options if isinstance(options, dict) else {})
        _apply_project_memory_update(project, memory_notes, provided=memory_notes_provided)
    else:
        if title:
            project["title"] = title
        if isinstance(options, dict):
            merged = dict(project.get("options") or {})
            merged.update(options)
            project["options"] = merged
        _apply_project_memory_update(project, memory_notes, provided=memory_notes_provided)
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


@app.post("/api/projects/<project_id>/batch-export")
def api_project_batch_export(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    body = request.get_json(silent=True) or {}
    payload = body.get("payload")
    if not isinstance(payload, dict):
        return jsonify({"status": "fail", "error": "missing payload"}), 400
    saved = _save_batch_export_entry(pid, payload)
    project = _load_project_state(pid)
    if not isinstance(project, dict):
        project = _new_project_state(pid, title=pid, options={})
    _append_message(
        project,
        role="system",
        kind="batch_export",
        content=f"Batch export saved: {saved.get('export_id')}",
        meta={"batch_export": saved},
    )
    _save_project_state(project)
    return jsonify({"status": "pass", "project_id": pid, "batch_export": saved})


@app.get("/api/projects/<project_id>/batch-exports")
def api_project_batch_exports(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    limit = _as_int(request.args.get("limit"), 20)
    limit = max(1, min(limit, 100))
    offset = _as_int(request.args.get("offset"), 0)
    offset = max(0, min(offset, 200000))
    action_filter = str(request.args.get("action") or "").strip()
    status_filter = str(request.args.get("status") or "").strip().lower()
    if action_filter and not _safe_filter_token(action_filter):
        return jsonify({"status": "fail", "error": "invalid action filter"}), 400
    if status_filter and status_filter not in {"pass", "partial", "fail"}:
        return jsonify({"status": "fail", "error": "invalid status filter"}), 400
    exports, total_count = _list_batch_export_entries(
        pid,
        limit=limit,
        offset=offset,
        action_filter=action_filter,
        status_filter=status_filter,
    )
    return jsonify(
        {
            "status": "pass",
            "project_id": pid,
            "limit": limit,
            "offset": offset,
            "action_filter": action_filter,
            "status_filter": status_filter,
            "count": len(exports),
            "total_count": total_count,
            "has_more": (offset + len(exports)) < total_count,
            "exports": exports,
        }
    )


@app.get("/api/projects/<project_id>/batch-exports/compare")
def api_project_batch_exports_compare(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    primary_export_id = str(request.args.get("primary_export_id") or "").strip()
    other_export_id = str(request.args.get("other_export_id") or "").strip()
    if not primary_export_id or not other_export_id:
        return jsonify({"status": "fail", "error": "missing primary_export_id/other_export_id"}), 400
    if not _is_safe_export_id(primary_export_id) or not _is_safe_export_id(other_export_id):
        return jsonify({"status": "fail", "error": "invalid export_id"}), 400
    if primary_export_id == other_export_id:
        return jsonify({"status": "fail", "error": "other_export_id must differ from primary_export_id"}), 400
    primary = _load_batch_export_entry(pid, primary_export_id)
    other = _load_batch_export_entry(pid, other_export_id)
    if not isinstance(primary, dict) or not isinstance(other, dict):
        return (
            jsonify(
                {
                    "status": "missing",
                    "error": "batch_export_not_found",
                    "project_id": pid,
                    "primary_export_id": primary_export_id,
                    "other_export_id": other_export_id,
                    "primary_exists": isinstance(primary, dict),
                    "other_exists": isinstance(other, dict),
                }
            ),
            404,
        )
    primary_source = _batch_export_source_payload(primary)
    other_source = _batch_export_source_payload(other)
    diff = _artifact_diff_payload(primary_source, other_source)
    primary_summary = _batch_export_summary(primary, export_id=primary_export_id, project_id=pid)
    other_summary = _batch_export_summary(other, export_id=other_export_id, project_id=pid)
    return jsonify(
        {
            "status": "pass",
            "project_id": pid,
            "primary_export_id": primary_export_id,
            "other_export_id": other_export_id,
            "primary": primary_summary,
            "other": other_summary,
            "diff": diff,
            "compare_lines": _batch_export_compare_lines(primary_summary, other_summary, diff),
        }
    )


@app.post("/api/projects/<project_id>/batch-exports/replay-latest")
def api_project_batch_exports_replay_latest(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    exports, _ = _list_batch_export_entries(pid, limit=1)
    if not exports:
        return jsonify({"status": "missing", "error": "no_batch_export", "project_id": pid}), 200
    body = request.get_json(silent=True) or {}
    replay_options = body.get("options") if isinstance(body.get("options"), dict) else {}
    latest = exports[0]
    export_id = str(latest.get("export_id") or "").strip()
    payload = _load_batch_export_entry(pid, export_id)
    if not isinstance(payload, dict):
        return jsonify({"status": "fail", "error": "invalid_batch_export", "project_id": pid, "export_id": export_id}), 200
    out = _replay_batch_export_payload(
        project_id=pid,
        payload=payload,
        source_export_id=str(payload.get("export_id") or export_id),
        replay_options=replay_options,
    )
    if str(out.get("status") or "") == "fail":
        return jsonify(out), 200
    replay_entry = _save_batch_export_entry(
        pid,
        {
            **out,
            "replayed_at": _now_iso(),
            "source_export_id": str(payload.get("export_id") or export_id),
        },
    )
    return jsonify(
        {
            "status": "pass",
            "project_id": pid,
            "source": latest,
            "batch_export": replay_entry,
            "action": str(out.get("action") or ""),
            "replay_status": str(out.get("status") or ""),
        }
    )


@app.get("/api/projects/<project_id>/batch-exports/<export_id>")
def api_project_batch_export_detail(project_id: str, export_id: str):
    pid = str(project_id or "").strip()
    eid = str(export_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not _is_safe_export_id(eid):
        return jsonify({"status": "fail", "error": "invalid export_id"}), 400
    payload = _load_batch_export_entry(pid, eid)
    if not isinstance(payload, dict):
        return jsonify({"status": "missing", "error": "batch_export_not_found", "project_id": pid, "export_id": eid}), 404
    return jsonify({"status": "pass", "project_id": pid, "export_id": eid, "batch_export": payload})


@app.get("/api/projects/<project_id>/batch-exports/<export_id>/failed-queue")
def api_project_batch_export_failed_queue(project_id: str, export_id: str):
    pid = str(project_id or "").strip()
    eid = str(export_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not _is_safe_export_id(eid):
        return jsonify({"status": "fail", "error": "invalid export_id"}), 400
    payload = _load_batch_export_entry(pid, eid)
    if not isinstance(payload, dict):
        return jsonify({"status": "missing", "error": "batch_export_not_found", "project_id": pid, "export_id": eid}), 404
    source_batch = _batch_export_source_payload(payload)
    out = _extract_failed_queue_rows_from_source_batch(source_batch=source_batch, project_id=pid, source_export_id=eid)
    return jsonify(
        {
            "status": "pass",
            "project_id": pid,
            "export_id": eid,
            "queue": out,
        }
    )


@app.get("/api/projects/<project_id>/batch-exports/<export_id>/download")
def api_project_batch_export_download(project_id: str, export_id: str):
    pid = str(project_id or "").strip()
    eid = str(export_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not _is_safe_export_id(eid):
        return jsonify({"status": "fail", "error": "invalid export_id"}), 400
    payload = _load_batch_export_entry(pid, eid)
    if not isinstance(payload, dict):
        return jsonify({"status": "missing", "error": "batch_export_not_found", "project_id": pid, "export_id": eid}), 404
    fmt = str(request.args.get("format") or "json").strip().lower()
    if fmt not in {"json", "csv"}:
        return jsonify({"status": "fail", "error": "invalid format", "supported": ["json", "csv"]}), 400
    source = _batch_export_source_payload(payload)
    action = str(source.get("action") or payload.get("action") or "batch_export")
    filename = _batch_export_download_filename(project_id=pid, export_id=eid, action=action, fmt=fmt)
    if fmt == "csv":
        csv_text = _batch_export_csv_text(payload, export_id=eid, project_id=pid)
        return Response(
            csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return Response(
        json_text,
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/projects/<project_id>/batch-exports/<export_id>/replay")
def api_project_batch_export_replay(project_id: str, export_id: str):
    pid = str(project_id or "").strip()
    eid = str(export_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not _is_safe_export_id(eid):
        return jsonify({"status": "fail", "error": "invalid export_id"}), 400
    payload = _load_batch_export_entry(pid, eid)
    if not isinstance(payload, dict):
        return jsonify({"status": "missing", "error": "batch_export_not_found", "project_id": pid, "export_id": eid}), 404
    body = request.get_json(silent=True) or {}
    replay_options = body.get("options") if isinstance(body.get("options"), dict) else {}
    out = _replay_batch_export_payload(
        project_id=pid,
        payload=payload,
        source_export_id=eid,
        replay_options=replay_options,
    )
    if str(out.get("status") or "") == "fail":
        return jsonify(out), 200
    replay_entry = _save_batch_export_entry(pid, {**out, "replayed_at": _now_iso(), "source_export_id": eid})
    return jsonify(
        {
            "status": "pass",
            "project_id": pid,
            "source_export_id": eid,
            "batch_export": replay_entry,
            "action": str(out.get("action") or ""),
            "replay_status": str(out.get("status") or ""),
        }
    )


@app.delete("/api/projects/<project_id>/batch-exports/<export_id>")
def api_project_batch_export_delete(project_id: str, export_id: str):
    pid = str(project_id or "").strip()
    eid = str(export_id or "").strip()
    if not pid:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(pid):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not _is_safe_export_id(eid):
        return jsonify({"status": "fail", "error": "invalid export_id"}), 400
    deleted = _delete_batch_export_entry(pid, eid)
    if not deleted:
        return jsonify({"status": "missing", "error": "batch_export_not_found", "project_id": pid, "export_id": eid}), 404
    return jsonify({"status": "pass", "project_id": pid, "export_id": eid, "deleted": True})


@app.post("/api/chat/send")
def api_chat_send():
    body = request.get_json(silent=True) or {}
    project_id = str(body.get("project_id") or "").strip()
    message = str(body.get("message") or "").strip()
    new_task = bool(body.get("new_task"))
    options = body.get("options")
    memory_notes_provided = "memory_notes" in body
    memory_notes = body.get("memory_notes")

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
    _apply_project_memory_update(project, memory_notes, provided=memory_notes_provided)

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


@app.post("/api/chat/pending-submit")
def api_chat_pending_submit():
    body = request.get_json(silent=True) or {}
    project_id = str(body.get("project_id") or "").strip()
    patch = body.get("patch") if isinstance(body.get("patch"), dict) else {}
    options = body.get("options")
    memory_notes_provided = "memory_notes" in body
    memory_notes = body.get("memory_notes")

    if not project_id:
        return jsonify({"status": "fail", "error": "missing project_id"}), 400
    if not _is_safe_project_id(project_id):
        return jsonify({"status": "fail", "error": "invalid project_id"}), 400
    if not isinstance(patch, dict) or len(patch) < 1:
        return jsonify({"status": "fail", "error": "missing patch"}), 400

    project = _load_project_state(project_id)
    if not isinstance(project, dict):
        return jsonify({"status": "fail", "error": "project_not_found"}), 404
    if isinstance(options, dict):
        merged_options = dict(project.get("options") or {})
        merged_options.update(options)
        project["options"] = merged_options
    _apply_project_memory_update(project, memory_notes, provided=memory_notes_provided)

    out = _chat_resume_from_pending(
        project=project,
        patch=patch,
        source_message=json.dumps(patch, ensure_ascii=False),
    )
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
    overrides = _normalize_resume_overrides(body)
    if not task_id:
        return jsonify({"status": "fail", "error": "missing task_id"}), 400
    if not _is_safe_task_id(task_id):
        return jsonify({"status": "fail", "error": "invalid task_id"}), 400
    return jsonify(_run_agent_resume(task_id=task_id, planner_provider=planner, catalog_path=catalog, overrides=overrides))


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
