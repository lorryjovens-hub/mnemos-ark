"""LongMemEval-V2 基准接入（记忆检索层评测）。

LongMemEval（https://xiaowu0162.github.io/longmemeval-v2/）考的是
「跨会话记忆：状态追踪 / 坑点 / 多跳」——正是 DLS 的主场。本模块评的是
**记忆检索层**：给定问题，正确的证据会话能否被检索进上下文、花多少 token。

诚实的近似声明：
- haystack 会话**原文**入库（provenance=longmemeval-haystack），代替
  「蒸馏后的 DEC/LES」——被测的检索层（词法/语义/混合 + status-first
  装配）与生产完全一致，只是入库形态更粗；
- 证据判定优先用数据集的 evidence 标注字段，缺失时用「答案子串出现在
  会话文本」近似（文档化的启发式，不是精确标注）；
- 本模块**不评 LLM 答题准确率**，只评记忆管线的命中与 token 经济；
  答题评测可在其上叠加任意 LLM provider。

数据格式（官方 JSONL，字段宽松兼容 V1/V2）：
  {"question_id": ..., "question_type": ..., "question": ...,
   "answer": ..., "haystack": [[{"role","content"}, ...], ...],
   "evidence_session_ids"?: [...] 或 "evidence_indices"?: [...] }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .memory import DLSMemory, estimate_tokens

__all__ = [
    "LongMemEvalQuestion",
    "load_longmemeval",
    "ingest_question",
    "evaluate",
]


@dataclass
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    haystack: list[list[dict]] = field(default_factory=list)
    evidence_indices: list[int] = field(default_factory=list)

    def session_text(self, idx: int) -> str:
        parts = []
        for msg in self.haystack[idx]:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if content:
                parts.append(str(content))
        return "\n".join(parts)


def load_longmemeval(path: str | Path) -> list[LongMemEvalQuestion]:
    """加载官方 JSONL（宽松解析：缺字段用空默认，坏行跳过并计数）。"""
    questions: list[LongMemEvalQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        haystack = row.get("haystack") or []
        evidence = row.get("evidence_session_ids") or row.get("evidence_indices") or []
        try:
            evidence_idx = [int(i) for i in evidence]
        except (TypeError, ValueError):
            evidence_idx = []
        questions.append(LongMemEvalQuestion(
            question_id=str(row.get("question_id", f"q{len(questions)}")),
            question_type=str(row.get("question_type", "unknown")),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            haystack=[list(s) for s in haystack],
            evidence_indices=evidence_idx,
        ))
    return questions


def _resolve_evidence(q: LongMemEvalQuestion, session_ids: Sequence[str]) -> list[str]:
    """证据记录 id：显式标注优先；缺失时用答案子串启发式。"""
    if q.evidence_indices:
        return [session_ids[i] for i in q.evidence_indices if i < len(session_ids)]
    if not q.answer:
        return []
    return [sid for sid, idx in zip(session_ids, range(len(session_ids)))
            if q.answer and q.answer in q.session_text(idx)]


def ingest_question(dls: DLSMemory, q: LongMemEvalQuestion,
                    project: str = "longmemeval") -> list[str]:
    """把一条问题的 haystack 会话入库，返回会话记录 id 列表（顺序对应）。"""
    session_ids: list[str] = []
    for idx, _session in enumerate(q.haystack):
        rec = dls.add(
            type="lesson",
            title=f"[{q.question_id}] 会话 {idx}",
            body=q.session_text(idx)[:4000],
            project=project,
            payload={"kind": "haystack-session", "question_id": q.question_id,
                     "session_index": idx},
            provenance="longmemeval-haystack",
            verify=False,
        )
        session_ids.append(rec.id)
    return session_ids


def evaluate(dls: DLSMemory, questions: Iterable[LongMemEvalQuestion],
             k: int = 5, project: str = "longmemeval",
             strategy: str = "lexical") -> dict:
    """评测：hit@k / MRR / token 经济（对照全量注入）。

    strategy: "lexical"（词法）或 "hybrid"（词法+向量，无 embedder 时自动
    降级词法并计入 degraded）。
    """
    total = hit = 0
    rr_sum = 0.0
    tokens_full = tokens_strategy = 0
    by_type: dict[str, dict] = {}
    degraded = False

    for q in questions:
        session_ids = ingest_question(dls, q, project=project)
        if not session_ids:
            continue
        evidence = _resolve_evidence(q, session_ids)
        total += 1

        full_text = "\n".join(q.session_text(i) for i in range(len(q.haystack)))
        tokens_full += estimate_tokens(full_text)

        if strategy == "hybrid":
            try:
                hits = dls.hybrid_search(q.question, top_k=k, project=project)
            except Exception:
                hits = dls.search(q.question, project=project, limit=k)
                degraded = True
        else:
            hits = dls.search(q.question, project=project, limit=k)

        ranked_ids = [r.id for r in hits]
        tokens_strategy += estimate_tokens(
            "\n".join(r.body for r in hits)) + estimate_tokens(q.question)

        rank = next((i + 1 for i, rid in enumerate(ranked_ids) if rid in evidence), 0)
        if rank and rank <= k:
            hit += 1
            rr_sum += 1.0 / rank

        bucket = by_type.setdefault(q.question_type, {"n": 0, "hit": 0})
        bucket["n"] += 1
        if rank and rank <= k:
            bucket["hit"] += 1

    return {
        "questions": total,
        "hit_at_k": round(hit / total, 4) if total else 0.0,
        "mrr": round(rr_sum / total, 4) if total else 0.0,
        "k": k,
        "strategy": strategy,
        "tokens_full_injection": tokens_full,
        "tokens_strategy": tokens_strategy,
        "token_ratio": round(tokens_strategy / tokens_full, 4) if tokens_full else 0.0,
        "by_type": by_type,
        "degraded": degraded,
    }
