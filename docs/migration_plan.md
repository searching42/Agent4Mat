# Migration plan (workspace/scripts -> oled-agent)

## 已完成
- 建立可运行骨架：CLI + runner + stage 接口 + manifest。
- 提供 demo pipeline 并验证运行输出。
- 完成 `workspace/scripts` 复用审计。

## 下一步建议（执行顺序）
1. 先迁移纯逻辑 stage（compose/filter/explain），不碰远端 adapter。
2. 迁移 intake/prelaunch gate 作为 policy 模块。
3. 把 `run_current_default_shortlist_policy.py` 拆解为 runner 子图。
4. 最后再接 Uni-Mol / REINVENT4 远端 adapter。

## 迁移目标定义
- 所有核心流程必须支持：
  - `python -m oled_agent.cli run --config ...`
  - 输出 `runs/<run_id>/manifest.json`
  - 不依赖会话记忆
- 所有 gate 判定必须有 machine-readable JSON 输出。

## 验收标准
- 一条最小流水线：输入候选 -> 多目标排序 -> topN -> report。
- 一条带 gate 流水线：intake/prelaunch -> baseline -> B1/max1 -> decision JSON。
- 回归：至少保留 1 组 acceptance replay。
