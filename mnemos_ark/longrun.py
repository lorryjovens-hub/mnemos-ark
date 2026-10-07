"""长时程基准 — 连续任务压力协议（口径借鉴 TencentDB Agent Memory）。

TDAM 的 SWE-bench 口径是「一个会话连续 50 个任务」，模拟真实长时程
Agent 的上下文累积压力——比单问单答残酷得多，也更贴近生产。本模块
把这个协议做成可复现的基准：

  synthesize_long_run()   合成连任务流（真实数据集可直接构造 LongRunTask）
  run_long_horizon()      双策略对照：full-history（全量历史灌入）vs
                          status-first（bootstrap + 按需检索/跳转）

全程记录 token 轨迹（不是只报终点），并标记 full-history 装爆窗口的
首个任务号——窗口溢出后它的"全量上下文"承诺已经物理破产。

诚实声明：合成任务的检索难度可控（证据标题/正文含关键词），真实数据
的难度分布更偏斜；结论请以自有数据跑出的轨迹为准。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .memory import DLSMemory, estimate_tokens

__all__ = ["LongRunTask", "synthesize_long_run", "run_long_horizon"]


@dataclass
class LongRunTask:
    task_id: str
    question: str
    answer: str
    evidence_text: str
    noise_texts: list[str] = field(default_factory=list)


def synthesize_long_run(n_tasks: int = 50, noise_per_task: int = 2,
                        seed: int = 42) -> list[LongRunTask]:
    """合成连任务流：每个任务 1 条证据记录 + N 条噪声记录。"""
    rng = random.Random(seed)
    topics = ["登录模块", "缓存层", "消息队列", "支付网关", "索引优化",
              "日志管道", "权限系统", "调度器", "配置中心", "导出服务"]
    tasks: list[LongRunTask] = []
    for i in range(n_tasks):
        topic = topics[i % len(topics)]
        answer = f"答案-{i:03d}-{rng.randrange(16 ** 4):04x}"
        evidence = (
            f"关于{topic}的第 {i} 号决策记录：最终采用方案 {answer}，"
            f"原因是压测显示尾延迟下降 {20 + i % 30}%，回滚预案已验证。")
        noise = [
            f"{topic}相关的第 {j} 次闲聊：天气不错，咖啡续杯，会议改期。"
            for j in range(noise_per_task)]
        tasks.append(LongRunTask(
            task_id=f"T{i:03d}",
            question=f"{topic}第 {i} 号决策最终采用的方案是什么？",
            answer=answer, evidence_text=evidence, noise_texts=noise))
    return tasks


def run_long_horizon(dls: DLSMemory, tasks: list[LongRunTask],
                     k: int = 5, project: str = "longrun",
                     window_tokens: int = 128000,
                     status_every: int = 5) -> dict:
    """双策略长时程对照，返回 token 轨迹 + 命中统计。

    full-history：每个任务的上下文 = 至今全部记录正文（历史越攒越厚）。
    status-first：每 status_every 个任务滚动一次 status，任务上下文 =
    status 全文 + 检索 top-k（命中则额外 jump 证据全文）。
    """
    full_cum = 0
    sf_cum = 0
    traj_full: list[int] = []
    traj_sf: list[int] = []
    hits = 0
    overflow_at: int | None = None
    evidence_ids: list[str] = []

    for idx, task in enumerate(tasks):
        evid = dls.add_lesson(
            title=f"{task.task_id} 决策记录", mistake="（历史事实）",
            correction=task.evidence_text, project=project,
            provenance="longrun", verify=False)
        evidence_ids.append(evid.id)
        for noise in task.noise_texts:
            dls.add_lesson(title=f"{task.task_id} 闲聊", mistake=noise,
                           correction="（无实质内容）", project=project,
                           provenance="longrun", verify=False)

        # full-history：全量历史计入上下文
        bodies = dls.list_records(project=project)
        full_ctx_tokens = sum(estimate_tokens(r.body) for r in bodies)
        full_cum += full_ctx_tokens
        traj_full.append(full_ctx_tokens)
        if overflow_at is None and full_ctx_tokens > window_tokens:
            overflow_at = idx + 1

        # status-first：滚动 status + 检索 top-k
        if (idx + 1) % status_every == 0 or idx == 0:
            dls.add_status(
                title=f"{project} 进度 {idx + 1}/{len(tasks)}",
                summary=f"已完成 {idx + 1} 个任务",
                next_steps=[f"任务 {t.task_id}" for t in tasks[idx + 1:idx + 4]],
                project=project, provenance="longrun", verify=False)
        boot = dls.bootstrap(project, budget_tokens=600)
        sf_tokens = estimate_tokens(str(boot.get("status") or ""))
        hits_rows = dls.search(task.question.split("最终")[0], project=project,
                               limit=k)
        sf_tokens += sum(estimate_tokens(r.body[:400]) for r in hits_rows)
        if evid.id in {r.id for r in hits_rows}:
            hits += 1
            sf_tokens += estimate_tokens(task.evidence_text)  # jump 证据全文
        sf_cum += sf_tokens
        traj_sf.append(sf_tokens)

    n = len(tasks)
    return {
        "tasks": n,
        "hit_rate_status_first": round(hits / n, 4) if n else 0.0,
        "tokens_full_history": {
            "total": full_cum,
            "end": traj_full[-1] if traj_full else 0,
            "max": max(traj_full, default=0),
            "trajectory_sampled": traj_full[:: max(1, n // 10)],
        },
        "tokens_status_first": {
            "total": sf_cum,
            "end": traj_sf[-1] if traj_sf else 0,
            "max": max(traj_sf, default=0),
            "trajectory_sampled": traj_sf[:: max(1, n // 10)],
        },
        "window_overflow_at_task": overflow_at,
        "window_tokens": window_tokens,
        "savings_ratio": round(sf_cum / full_cum, 4) if full_cum else 0.0,
    }
