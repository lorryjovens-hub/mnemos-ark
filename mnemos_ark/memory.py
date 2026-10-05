"""DLS 结构化记忆引擎 — Decisions / Lessons / Status 三库互索引。

设计目标（对应 LAAP_MEMORY_SCOPED_DESIGN 的作用域隔离方案）：

1. **高度结构化的三类记忆**，而非自由文本摘要：
   - DEC（decisions）决策史 —— 长文，含 context/options/chosen/rationale/outcome
   - LES（lessons）错题本   —— 中文，含 mistake/correction/rule_of_thumb
   - STA（status）工程现状  —— 短，含 current_state/next_steps/open_questions
2. **互索引**：records 之间用类型化边（supports/refutes/supersedes/caused/
   derived_from/fixes/relates/part_of）连接，双向可查。
3. **fresh session 从 status 出发**：bootstrap() 只返回「当前 status 全文 +
   一环指针 + 二环编号」，其余记忆按需按编号精准跳转（jump/get），不整库
   灌入上下文。
4. **U 形曲线装箱**：pack_context() 把注意力预算倾斜给首尾（status 全文、
   最近条目全文），中段只留指针（id + 标题 + 一句话），符合 primacy-recency
   注意力分布。

存储：SQLite 单写入面（records + links + FTS5）+ Markdown 镜像（记忆即文件，
可读、可版本控制）。零重型依赖：stdlib only；FTS5 不可用时降级 LIKE 并记录
降级事件（不静默）。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Literal

__all__ = [
    "RecordType",
    "LinkRelation",
    "DLSRecord",
    "DLSLink",
    "DLSError",
    "DLSMemory",
]

RecordType = Literal["decision", "lesson", "status"]
LinkRelation = Literal[
    "supports", "refutes", "supersedes", "caused",
    "derived_from", "fixes", "relates", "part_of",
]
_RENDER_MODE = Literal["full", "pointer", "id"]

_PREFIX = {"decision": "DEC", "lesson": "LES", "status": "STA"}
_VALID_RELATIONS = frozenset({
    "supports", "refutes", "supersedes", "caused",
    "derived_from", "fixes", "relates", "part_of",
})
_VALID_STATUS = frozenset({"active", "superseded", "archived"})


class DLSError(ValueError):
    """DLS 引擎的契约违规（非法类型/关系/编号、自环等）。"""


@dataclass
class DLSRecord:
    id: str
    type: RecordType
    project: str
    domain: str
    title: str
    body: str = ""
    payload: dict = field(default_factory=dict)
    status: str = "active"
    confidence: float = 1.0
    tags: list = field(default_factory=list)
    provenance: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def one_liner(self) -> str:
        """指针模式的一句话摘要：优先 payload 摘要字段，其次正文首行。"""
        for key in ("summary", "rule_of_thumb", "chosen", "correction"):
            text = self.payload.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip().splitlines()[0][:120]
        first = self.body.strip().splitlines()[0] if self.body.strip() else ""
        return first[:120]


@dataclass
class DLSLink:
    src: str
    dst: str
    relation: LinkRelation
    note: str = ""
    created_at: float = 0.0


def _now() -> float:
    return time.time()


def estimate_tokens(text: str) -> int:
    """粗略 token 估算（CJK/英文混合）。工程约定：1 token ≈ 2.5 字符。"""
    return max(1, int(len(text) / 2.5))


def default_home() -> Path:
    """DLS 数据目录：LAAP_DLS_HOME 环境变量优先，缺省 ~/.laap/dls。"""
    env = os.environ.get("LAAP_DLS_HOME")
    return Path(env).expanduser() if env else Path.home() / ".laap" / "dls"


class DLSMemory:
    """Decisions / Lessons / Status 三库互索引记忆引擎。"""

    def __init__(self, home: str | Path | None = None,
                 verifier: Callable[[str], dict] | None = None):
        self.home = Path(home) if home else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.md_root = self.home / "md"
        self.md_root.mkdir(parents=True, exist_ok=True)
        self._verifier = verifier
        self._fts_ok = True
        self._degrade_events: list[str] = []
        self._db = sqlite3.connect(str(self.home / "dls.db"), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ 写入面

    def add(self, type: RecordType, title: str, body: str = "",
            project: str = "_global", domain: str = "general",
            payload: dict | None = None, confidence: float = 1.0,
            tags: Iterable[str] = (), provenance: str = "",
            verify: bool = True) -> DLSRecord:
        """统一写入面。所有记忆都从这里进，自动过验证钩子 + Markdown 镜像。"""
        if type not in _PREFIX:
            raise DLSError(f"非法记忆类型: {type!r}")
        if not title.strip():
            raise DLSError("标题不能为空")
        if not 0.0 <= confidence <= 1.0:
            raise DLSError(f"confidence 必须在 [0,1]: {confidence}")
        if verify and self._verifier is not None:
            verdict = self._verifier(f"{title}\n{body}".strip())
            if isinstance(verdict, dict) and verdict.get("state") == "error":
                raise DLSError(f"验证门拒绝入库: {verdict.get('evidence', '')}")

        rec_id = self._next_id(type)
        now = _now()
        record = DLSRecord(
            id=rec_id, type=type, project=project, domain=domain,
            title=title.strip(), body=body, payload=payload or {},
            confidence=float(confidence), tags=sorted(set(tags)),
            provenance=provenance, created_at=now, updated_at=now,
        )
        with self._db:
            self._db.execute(
                "INSERT INTO records (id, type, project, domain, title, body,"
                " payload, status, confidence, tags, provenance,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.id, record.type, record.project, record.domain,
                 record.title, record.body, json.dumps(record.payload, ensure_ascii=False),
                 record.status, record.confidence, json.dumps(record.tags, ensure_ascii=False),
                 record.provenance, record.created_at, record.updated_at),
            )
            self._index_text(record)
        self._write_markdown(record)
        return record

    def add_decision(self, title: str, context: str, chosen: str, rationale: str,
                     options_considered: Iterable[str] = (), **kw) -> DLSRecord:
        payload = {
            "context": context, "chosen": chosen, "rationale": rationale,
            "options_considered": list(options_considered),
            "revisit_trigger": kw.pop("revisit_trigger", ""),
            "expected_outcome": kw.pop("expected_outcome", ""),
        }
        return self.add("decision", title, body=rationale, payload=payload, **kw)

    def add_lesson(self, title: str, mistake: str, correction: str,
                   trigger: str = "", severity: str = "medium", **kw) -> DLSRecord:
        payload = {
            "mistake": mistake, "correction": correction,
            "trigger": trigger, "severity": severity,
            "rule_of_thumb": kw.pop("rule_of_thumb", correction.splitlines()[0] if correction else ""),
        }
        body = f"错：{mistake}\n正：{correction}"
        return self.add("lesson", title, body=body, payload=payload, **kw)

    def add_status(self, title: str, summary: str, current_state: dict | None = None,
                   next_steps: Iterable[str] = (), open_questions: Iterable[str] = (),
                   supersede_previous: bool = True, **kw) -> DLSRecord:
        """工程现状快照。默认取代同项目上一份 status（旧的转 superseded）。"""
        payload = {
            "summary": summary,
            "current_state": current_state or {},
            "next_steps": list(next_steps),
            "open_questions": list(open_questions),
            "last_verified": _now(),
        }
        rec = self.add("status", title, body=summary, payload=payload, **kw)
        if supersede_previous:
            with self._db:
                self._db.execute(
                    "UPDATE records SET status='superseded', updated_at=? "
                    "WHERE type='status' AND project=? AND id<>? AND status='active'",
                    (_now(), rec.project, rec.id),
                )
        return rec

    def link(self, src: str, dst: str, relation: LinkRelation,
             note: str = "") -> DLSLink:
        if relation not in _VALID_RELATIONS:
            raise DLSError(f"非法关系类型: {relation!r}")
        if src == dst:
            raise DLSError("禁止自环")
        for ref in (src, dst):
            if self.get(ref) is None:
                raise DLSError(f"悬空引用: {ref}")
        edge = DLSLink(src=src, dst=dst, relation=relation,
                       note=note, created_at=_now())
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO links (src, dst, relation, note, created_at)"
                " VALUES (?,?,?,?,?)",
                (edge.src, edge.dst, edge.relation, edge.note, edge.created_at),
            )
        return edge

    def set_status(self, rec_id: str, status: str) -> None:
        if status not in _VALID_STATUS:
            raise DLSError(f"非法生命周期状态: {status!r}")
        with self._db:
            cur = self._db.execute(
                "UPDATE records SET status=?, updated_at=? WHERE id=?",
                (status, _now(), rec_id),
            )
        if cur.rowcount == 0:
            raise DLSError(f"记录不存在: {rec_id}")

    # ------------------------------------------------------------------ 读取面

    def get(self, rec_id: str) -> DLSRecord | None:
        """按编号精准跳转（O(1)）。这是上下文经济的主通道。"""
        row = self._db.execute(
            "SELECT * FROM records WHERE id=?", (rec_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def jump(self, rec_id: str) -> str:
        """跳转并返回全文渲染。拿单条，不连带任何邻居。"""
        rec = self.get(rec_id)
        return self.render(rec, "full") if rec else f"[未找到 {rec_id}]"

    def links_from(self, rec_id: str) -> list[DLSLink]:
        rows = self._db.execute(
            "SELECT * FROM links WHERE src=? ORDER BY created_at DESC",
            (rec_id,)).fetchall()
        return [self._row_to_link(r) for r in rows]

    def links_to(self, rec_id: str) -> list[DLSLink]:
        rows = self._db.execute(
            "SELECT * FROM links WHERE dst=? ORDER BY created_at DESC",
            (rec_id,)).fetchall()
        return [self._row_to_link(r) for r in rows]

    def neighbors(self, rec_id: str) -> dict[str, list[dict]]:
        """互索引视图：出边 + 入边，每条带对端 id/标题。"""
        out, into = [], []
        for edge in self.links_from(rec_id):
            peer = self.get(edge.dst)
            out.append({"relation": edge.relation, "id": edge.dst,
                        "title": peer.title if peer else "?", "note": edge.note})
        for edge in self.links_to(rec_id):
            peer = self.get(edge.src)
            into.append({"relation": edge.relation, "id": edge.src,
                         "title": peer.title if peer else "?", "note": edge.note})
        return {"out": out, "in": into}

    def infer_project(self, query: str, default: str = "_global") -> str:
        """作用域路由（P2）：从查询推断激活的项目域，闲聊落 `_global`。

        匹配粒度是词级投票（英文词 + 中文二元组），而非整句子串——
        否则「Redis 连接池怎么选」命中不了标题「Redis 连接池选型」。
        规则：命中记录的 project 众数；平票取最近更新者；无命中回 default。
        这保证「同一项目组里说其他话」时，闲聊不会拉出项目记忆——
        它根本没有路由到那个桶。
        """
        terms = self._query_terms(query)
        if not terms:
            return default
        rows = self._db.execute(
            "SELECT project, title, body, tags, updated_at FROM records").fetchall()
        scored = []
        for row in rows:
            haystack = (row["title"] + " " + row["body"] + " " + row["tags"]).lower()
            score = sum(1 for t in terms if t in haystack)
            if score:
                scored.append((score, row["updated_at"], row["project"]))
        if not scored:
            return default
        best = max(s[0] for s in scored)
        top = [s for s in scored if s[0] == best]
        top.sort(key=lambda s: s[1], reverse=True)
        return top[0][2]

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        terms = [w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)]
        cn = re.findall(r"[\u4e00-\u9fff]", query)
        terms.extend(cn[i] + cn[i + 1] for i in range(len(cn) - 1))
        return sorted(set(terms))

    # ------------------------------------------------------- 睡眠蒸馏（P3）

    def distill_events(self, events: Iterable[dict], project: str = "_global",
                       link_to: str | None = None) -> list[DLSRecord]:
        """零 LLM 蒸馏：把结构化事件流编译成 DEC/LES 并入库。

        事件 schema（调用方如 structured_failure / error_reflection 产）：
          {kind: "decision"|"lesson", title, ...}
          decision: context/chosen/rationale/options_considered
          lesson:   mistake/correction/trigger/severity
        未知 kind 不静默丢弃，跳过并由调用方统计。
        """
        created: list[DLSRecord] = []
        for event in events:
            kind = event.get("kind")
            title = str(event.get("title", "")).strip()
            if not title:
                continue
            if kind == "decision":
                rec = self.add_decision(
                    title=title, context=event.get("context", ""),
                    chosen=event.get("chosen", ""),
                    rationale=event.get("rationale", ""),
                    options_considered=event.get("options_considered", []),
                    project=project, provenance="sleep-distill")
            elif kind == "lesson":
                rec = self.add_lesson(
                    title=title, mistake=event.get("mistake", ""),
                    correction=event.get("correction", ""),
                    trigger=event.get("trigger", ""),
                    severity=event.get("severity", "medium"),
                    rule_of_thumb=event.get("rule_of_thumb", ""),
                    project=project, provenance="sleep-distill")
            else:
                continue
            if link_to:
                self.link(link_to, rec.id, "derived_from")
            created.append(rec)
        return created

    @staticmethod
    def build_distill_prompt(day_notes: str) -> str:
        """LLM 路径的蒸馏提示（由调用方注入 laap.llm provider 执行）。

        引擎自身不调 LLM（分层不变）；产出必须是可被 distill_events
        消费的 JSON 事件数组。
        """
        return (
            "从以下工作日志中提炼结构化记忆事件。只输出 JSON 数组，不要其它文本。\n"
            "每项形如：\n"
            '{"kind":"decision","title":"...","context":"...","chosen":"...",'
            '"rationale":"...","options_considered":["..."]}\n'
            '或 {"kind":"lesson","title":"...","mistake":"...",'
            '"correction":"...","trigger":"...","severity":"high|medium|low",'
            '"rule_of_thumb":"一句话口诀"}\n'
            "只收录有长期价值的决策与教训，闲聊与流水账丢弃。\n\n"
            f"工作日志：\n{day_notes}"
        )

    def current_status(self, project: str = "_global") -> DLSRecord | None:
        row = self._db.execute(
            "SELECT * FROM records WHERE type='status' AND project=? "
            "AND status='active' ORDER BY updated_at DESC LIMIT 1",
            (project,)).fetchone()
        return self._row_to_record(row) if row else None

    def search(self, query: str, type: RecordType | None = None,
               project: str | None = None, limit: int = 20) -> list[DLSRecord]:
        """词法检索。FTS5 只对分词友好文本有效；CJK 查询或 FTS5 缺失时走 LIKE。

        工程注记：fts5 unicode61 把无空格的中文连续串索引成单 token，
        中文子串查询必然 0 命中，因此 CJK 查询直接走 LIKE，不算降级。
        """
        filter_sql, params = "", []
        if type:
            filter_sql += " AND records.type=?"
            params.append(type)
        if project:
            filter_sql += " AND records.project=?"
            params.append(project)
        if self._fts_ok and not _has_cjk(query):
            try:
                phrase = '"' + query.replace('"', '""') + '"'
                rows = self._db.execute(
                    "SELECT records.* FROM records_fts "
                    "JOIN records ON records.id = records_fts.id "
                    "WHERE records_fts MATCH ?" + filter_sql +
                    " ORDER BY bm25(records_fts) LIMIT ?",
                    [phrase, *params, limit]).fetchall()
                if rows:
                    return [self._row_to_record(r) for r in rows]
            except sqlite3.OperationalError as exc:
                self._fts_ok = False
                self._degrade_events.append(f"FTS5 不可用，降级 LIKE: {exc}")
        like = f"%{query}%"
        rows = self._db.execute(
            "SELECT * FROM records WHERE (title LIKE ? OR body LIKE ? OR tags LIKE ?)" +
            filter_sql + " ORDER BY updated_at DESC LIMIT ?",
            [like, like, like, *params, limit]).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------- 冷启动与 U 形曲线装箱

    def bootstrap(self, project: str = "_global",
                  budget_tokens: int = 1200,
                  max_ring1: int = 10, max_ring2: int = 20) -> dict:
        """fresh session 入口：status 全文 → 一环指针 → 二环编号，按预算裁剪。

        扇出封顶（P4 评测暴露的真问题）：status 直连上百条时，环列表
        自身就是 token 灾难，因此 ring1/ring2 限量，溢出的以计数摘要。

        返回结构：
          status   当前 status 全文渲染（完整 payload）
          ring1    与 status 直接相连的条目（pointer 模式，限量 max_ring1）
          ring2    一环邻居的邻居（id + 标题，限量 max_ring2）
          overflow  被限量截断的一环/二环计数
          index    项目内三库计数
          usage    token 估算与裁剪情况
        """
        status = self.current_status(project)
        if status is None:
            rows = self._db.execute(
                "SELECT * FROM records WHERE project=? ORDER BY updated_at DESC LIMIT 1",
                (project,)).fetchone()
            status = self._row_to_record(rows) if rows else None
        if status is None:
            return {"status": None, "ring1": [], "ring2": [],
                    "index": self._index_counts(project),
                    "usage": {"budget": budget_tokens, "used": 0, "trimmed": 0}}

        used = estimate_tokens(self.render(status, "full"))
        ring1, ring2, trimmed = [], [], 0
        overflow = {"ring1": 0, "ring2": 0}
        seen = {status.id}
        for edge in self.links_from(status.id) + self.links_to(status.id):
            peer_id = edge.dst if edge.src == status.id else edge.src
            if peer_id in seen:
                continue
            seen.add(peer_id)
            peer = self.get(peer_id)
            if peer is None:
                continue
            if len(ring1) >= max_ring1:
                overflow["ring1"] += 1
                continue
            cost = estimate_tokens(self.render(peer, "pointer"))
            if used + cost <= budget_tokens:
                ring1.append({"relation": edge.relation, "id": peer.id,
                              "title": peer.title, "line": peer.one_liner(),
                              "type": peer.type})
                used += cost
            else:
                ring1.append({"relation": edge.relation, "id": peer.id,
                              "title": peer.title, "type": peer.type,
                              "trimmed": True})
                trimmed += 1
            for edge2 in self.links_from(peer_id) + self.links_to(peer_id):
                far_id = edge2.dst if edge2.src == peer_id else edge2.src
                if far_id in seen:
                    continue
                seen.add(far_id)
                if len(ring2) >= max_ring2:
                    overflow["ring2"] += 1
                    continue
                far = self.get(far_id)
                if far is not None:
                    ring2.append({"id": far.id, "title": far.title,
                                  "type": far.type})
        return {
            "status": self.render(status, "full"),
            "ring1": ring1,
            "ring2": ring2,
            "overflow": overflow,
            "index": self._index_counts(project),
            "usage": {"budget": budget_tokens, "used": used, "trimmed": trimmed},
        }

    def pack_context(self, record_ids: Iterable[str],
                     budget_tokens: int = 2000,
                     recent_tail: int = 2) -> str:
        """U 形曲线装箱：呈现顺序 = 首（全文）→ 中段（指针）→ 尾（全文），
        而预算分配顺序 = 首 → 尾 → 中段（注意力 U 形：首尾优先吃预算）。
        超预算时中段先降级为编号，尾部再降级为指针。"""
        records = [r for rid in record_ids if (r := self.get(rid)) is not None]
        if not records:
            return ""
        head = records[0]
        tail_ids = {r.id for r in records[-recent_tail:]} if len(records) > 1 else set()
        tail = [r for r in records if r.id in tail_ids and r.id != head.id]
        middle = [r for r in records[1:] if r.id not in tail_ids]

        levels: dict[str, str] = {head.id: "full"}
        used = estimate_tokens(self.render(head, "full"))
        for rec in tail:  # 尾部优先吃预算
            cost = estimate_tokens(self.render(rec, "full"))
            if used + cost <= budget_tokens:
                levels[rec.id] = "full"
                used += cost
            else:
                levels[rec.id] = "pointer"
        for rec in middle:  # 中段只拿指针的零头
            cost = estimate_tokens(self.render(rec, "pointer"))
            if used + cost <= budget_tokens:
                levels[rec.id] = "pointer"
                used += cost
            else:
                levels[rec.id] = "id"

        presented = [head, *middle, *tail]  # 呈现顺序：首 → 中段 → 尾
        return "\n\n---\n\n".join(self.render(r, levels[r.id]) for r in presented)

    def render(self, record: DLSRecord, mode: _RENDER_MODE = "full") -> str:
        if mode == "id":
            return f"[{record.id}]"
        if mode == "pointer":
            return f"[{record.id}] ({record.type}) {record.title} — {record.one_liner()}"
        lines = [f"## [{record.id}] {record.title}",
                 f"类型: {record.type} | 项目: {record.project} | 置信: {record.confidence:.2f}"
                 f" | 状态: {record.status}"]
        if record.tags:
            lines.append(f"标签: {', '.join(record.tags)}")
        if record.body:
            lines.append("")
            lines.append(record.body)
        for key, value in record.payload.items():
            if value and key not in ("summary",) or (key == "summary" and not record.body):
                rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                lines.append(f"- {key}: {rendered}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 内部

    def _next_id(self, type: RecordType) -> str:
        with self._db:
            row = self._db.execute(
                "SELECT n FROM seq WHERE type=?", (type,)).fetchone()
            n = (row["n"] if row else 0) + 1
            self._db.execute(
                "INSERT INTO seq (type, n) VALUES (?,?) "
                "ON CONFLICT(type) DO UPDATE SET n=excluded.n", (type, n))
        return f"{_PREFIX[type]}-{n:04d}"

    def _index_counts(self, project: str) -> dict:
        rows = self._db.execute(
            "SELECT type, COUNT(*) n FROM records WHERE project=? GROUP BY type",
            (project,)).fetchall()
        counts = {"decision": 0, "lesson": 0, "status": 0}
        for row in rows:
            counts[row["type"]] = row["n"]
        return counts

    def _index_text(self, record: DLSRecord) -> None:
        if not self._fts_ok:
            return
        try:
            self._db.execute(
                "INSERT INTO records_fts (id, title, body, tags) VALUES (?,?,?,?)",
                (record.id, record.title, record.body, " ".join(record.tags)))
        except sqlite3.OperationalError as exc:
            self._fts_ok = False
            self._degrade_events.append(f"FTS5 写入不可用，降级 LIKE: {exc}")

    def _write_markdown(self, record: DLSRecord) -> Path:
        project_dir = self.md_root / _slug(record.project)
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / f"{record.id}.md"
        front = {
            "id": record.id, "type": record.type, "project": record.project,
            "domain": record.domain, "status": record.status,
            "confidence": record.confidence, "tags": record.tags,
            "provenance": record.provenance,
            "created_at": record.created_at, "updated_at": record.updated_at,
            "payload": record.payload,
        }
        text = "---\n" + json.dumps(front, ensure_ascii=False, indent=2) + "\n---\n\n"
        text += f"# [{record.id}] {record.title}\n\n{record.body}\n"
        path.write_text(text, encoding="utf-8")
        return path

    def _row_to_record(self, row: sqlite3.Row) -> DLSRecord:
        return DLSRecord(
            id=row["id"], type=row["type"], project=row["project"],
            domain=row["domain"], title=row["title"], body=row["body"],
            payload=json.loads(row["payload"]), status=row["status"],
            confidence=row["confidence"], tags=json.loads(row["tags"]),
            provenance=row["provenance"], created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> DLSLink:
        return DLSLink(src=row["src"], dst=row["dst"], relation=row["relation"],
                       note=row["note"], created_at=row["created_at"])

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                project TEXT NOT NULL DEFAULT '_global',
                domain TEXT NOT NULL DEFAULT 'general',
                title TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 1.0,
                tags TEXT NOT NULL DEFAULT '[]',
                provenance TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_project ON records(project, type);
            CREATE TABLE IF NOT EXISTS links (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                relation TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                PRIMARY KEY (src, dst, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);
            CREATE TABLE IF NOT EXISTS seq (
                type TEXT PRIMARY KEY,
                n INTEGER NOT NULL
            );
            """
        )
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5("
                "id UNINDEXED, title, body, tags)")
        except sqlite3.OperationalError as exc:
            self._fts_ok = False
            self._degrade_events.append(f"FTS5 不可用，检索降级 LIKE: {exc}")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "DLSMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text.strip())
    return cleaned or "_global"
