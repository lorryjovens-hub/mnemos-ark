"""符号画布（symbolic canvas）—— 长任务的 token 压缩层。

借鉴 TencentDB Agent Memory 的 Mermaid 画布思想（README 明确致意）：
长任务最大的 token 黑洞是工具日志。画布把**重内容整体外置**到
``refs/{node_id}.md``，只在上下文里留一张带 node_id 的**符号图**
（几百 token），需要细节时按 node_id 精准取回——用符号换 token，
用寻址保追溯。

与 DLS 的关系：canvas 管**任务内**（一次长任务的执行轨迹），
DLS 管**跨会话**（DEC/LES/STA 结构化沉淀）。两者共用同一个 home
目录与同一条溯源契约：DLS 记录的 sources 可指向 canvas 的 node_id，
``drill_down`` 沿链验证到 refs/ 原文。

不变量：offload 永不丢原文——符号图只是索引，原文永远可按 id 取回。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .memory import default_home, estimate_tokens

__all__ = ["CanvasNode", "TaskCanvas", "CanvasError"]


class CanvasError(ValueError):
    """画布契约违规（重复 id、悬空引用、自环等）。"""


@dataclass
class CanvasNode:
    node_id: str
    kind: str
    label: str
    summary: str
    status: str = "pending"
    tokens: int = 0
    created_at: float = 0.0


@dataclass
class CanvasEdge:
    src: str
    dst: str
    relation: str = "follows"
    created_at: float = 0.0


_SAFE_LABEL = re.compile(r'[\[\]"\n\r]+')


class TaskCanvas:
    """任务画布：refs/ 原文外置 + node_id 符号图 + Mermaid 顶层。"""

    def __init__(self, home: str | Path | None = None):
        self.home = Path(home) if home else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.refs_dir = self.home / "refs"
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.home / "dls.db"), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ 写入

    def offload(self, label: str, content: str, kind: str = "log",
                summary: str = "", status: str = "pending") -> CanvasNode:
        """外置一份重内容（工具输出/日志/代码块），返回符号节点。

        原文落 ``refs/{node_id}.md``；节点只存 label + summary（各 ≤120 字）。
        """
        if not label.strip():
            raise CanvasError("label 不能为空")
        node_id = self._next_id()
        now = time.time()
        node = CanvasNode(
            node_id=node_id, kind=kind, label=label.strip()[:120],
            summary=(summary or label).strip()[:120], status=status,
            tokens=estimate_tokens(content), created_at=now,
        )
        path = self.refs_dir / f"{node_id}.md"
        path.write_text(f"# [{node_id}] {node.label}\n\n{content}\n",
                        encoding="utf-8")
        with self._db:
            self._db.execute(
                "INSERT INTO canvas_nodes (node_id, kind, label, summary,"
                " status, tokens, created_at) VALUES (?,?,?,?,?,?,?)",
                (node.node_id, node.kind, node.label, node.summary,
                 node.status, node.tokens, node.created_at))
        return node

    def link(self, src: str, dst: str, relation: str = "follows") -> CanvasEdge:
        if src == dst:
            raise CanvasError("禁止自环")
        for ref in (src, dst):
            if self.get(ref) is None:
                raise CanvasError(f"悬空引用: {ref}")
        edge = CanvasEdge(src=src, dst=dst, relation=relation,
                          created_at=time.time())
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO canvas_edges (src, dst, relation,"
                " created_at) VALUES (?,?,?,?)",
                (edge.src, edge.dst, edge.relation, edge.created_at))
        return edge

    def set_status(self, node_id: str, status: str) -> None:
        with self._db:
            cur = self._db.execute(
                "UPDATE canvas_nodes SET status=? WHERE node_id=?",
                (status, node_id))
        if cur.rowcount == 0:
            raise CanvasError(f"节点不存在: {node_id}")

    # ------------------------------------------------------------------ 读取

    def get(self, node_id: str) -> CanvasNode | None:
        row = self._db.execute(
            "SELECT * FROM canvas_nodes WHERE node_id=?",
            (node_id,)).fetchone()
        if row is None:
            return None
        return CanvasNode(
            node_id=row["node_id"], kind=row["kind"], label=row["label"],
            summary=row["summary"], status=row["status"], tokens=row["tokens"],
            created_at=row["created_at"])

    def recall(self, node_id: str) -> str:
        """按 node_id 精准取回原文（O(1)）——下钻不变量的落点。"""
        path = self.refs_dir / f"{node_id}.md"
        if not path.exists():
            raise CanvasError(f"原文缺失（溯源链断裂）: {node_id}")
        return path.read_text(encoding="utf-8")

    def exists(self, node_id: str) -> bool:
        return (self.refs_dir / f"{node_id}.md").exists()

    def neighbors(self, node_id: str) -> dict:
        out = [dict(r) for r in self._db.execute(
            "SELECT dst AS peer, relation FROM canvas_edges WHERE src=?",
            (node_id,)).fetchall()]
        into = [dict(r) for r in self._db.execute(
            "SELECT src AS peer, relation FROM canvas_edges WHERE dst=?",
            (node_id,)).fetchall()]
        return {"out": out, "in": into}

    # ------------------------------------------------------------------ 顶层

    def to_mermaid(self) -> str:
        """符号图的 Mermaid 表示（顶层注意力，token 极小）。"""
        rows = self._db.execute(
            "SELECT node_id, label, status FROM canvas_nodes"
            " ORDER BY created_at").fetchall()
        lines = ["graph LR"]
        for row in rows:
            label = _SAFE_LABEL.sub(" ", row["label"])
            mark = {"done": " ok", "failed": " fail"}.get(row["status"], "")
            lines.append(f'    {row["node_id"]}["{label}{mark}"]')
        edges = self._db.execute(
            "SELECT src, dst, relation FROM canvas_edges"
            " ORDER BY created_at").fetchall()
        for edge in edges:
            arrow = "-->" if edge["relation"] == "follows" else "-.->"
            lines.append(f"    {edge['src']} {arrow} {edge['dst']}")
        return "\n".join(lines)

    def pack(self, max_nodes: int = 40) -> str:
        """顶层注入块：符号图 + 节点指针表（id · 摘要 · 状态）。"""
        rows = self._db.execute(
            "SELECT * FROM canvas_nodes ORDER BY created_at LIMIT ?",
            (max_nodes,)).fetchall()
        total = self._db.execute("SELECT COUNT(*) n FROM canvas_nodes").fetchone()["n"]
        lines = ["[Task Canvas] 任务画布（原文外置于 refs/，按 node_id 取回）",
                 self.to_mermaid()]
        if rows:
            lines.append("")
            for row in rows:
                lines.append(f"  · {row['node_id']} [{row['status']}] "
                             f"{row['summary']}")
        if total > max_nodes:
            lines.append(f"  … 另有 {total - max_nodes} 个节点（canvas.recall 取回）")
        return "\n".join(lines)

    def stats(self) -> dict:
        nodes = self._db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(tokens),0) t"
            " FROM canvas_nodes").fetchone()
        return {
            "nodes": nodes["n"],
            "offloaded_tokens": nodes["t"],
            "packed_tokens_estimate": estimate_tokens(self.pack()),
            "compression_ratio": round(
                estimate_tokens(self.pack()) / nodes["t"], 4) if nodes["t"] else 0.0,
        }

    # ------------------------------------------------------------------ 内部

    def _next_id(self) -> str:
        with self._db:
            row = self._db.execute(
                "SELECT COALESCE(MAX(n),0) n FROM canvas_seq").fetchone()
            n = row["n"] + 1
            self._db.execute("INSERT INTO canvas_seq (n) VALUES (?)", (n,))
        return f"N-{n:04d}"

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS canvas_nodes (
                node_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'log',
                label TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                tokens INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS canvas_edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                relation TEXT NOT NULL DEFAULT 'follows',
                created_at REAL NOT NULL,
                PRIMARY KEY (src, dst, relation)
            );
            CREATE TABLE IF NOT EXISTS canvas_seq (n INTEGER NOT NULL);
            """
        )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "TaskCanvas":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
