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

try:
    import sqlite_vec as _sqlite_vec  # 可选加速：vec0 KNN（万条级）
except Exception:  # 缺席是合法状态，blob 余弦是完整兑底
    _sqlite_vec = None

from .embeddings import cosine

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
    sources: list = field(default_factory=list)
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
                 verifier: Callable[[str], dict] | None = None,
                 embedder: Callable | None = None):
        self.home = Path(home) if home else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.md_root = self.home / "md"
        self.md_root.mkdir(parents=True, exist_ok=True)
        self._verifier = verifier
        self._embedder = embedder  # 语义检索挂载点（EmbeddingProvider 协议）
        self._vec_ok = _sqlite_vec is not None  # sqlite-vec 加速后端可用性
        self._vec_dim: int | None = None
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
            sources: Iterable[str] = (),
            verify: bool = True) -> DLSRecord:
        """统一写入面。所有记忆都从这里进，自动过验证钩子 + Markdown 镜像。

        ``sources`` 是下钻不变量的锚点：溯源链（记录 id / 画布 node_id /
        任意外部引用 id），``drill_down`` 沿链验证到原文。
        """
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
            provenance=provenance, sources=sorted(set(sources)),
            created_at=now, updated_at=now,
        )
        with self._db:
            self._db.execute(
                "INSERT INTO records (id, type, project, domain, title, body,"
                " payload, status, confidence, tags, provenance, sources,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.id, record.type, record.project, record.domain,
                 record.title, record.body, json.dumps(record.payload, ensure_ascii=False),
                 record.status, record.confidence, json.dumps(record.tags, ensure_ascii=False),
                 record.provenance, json.dumps(record.sources, ensure_ascii=False),
                 record.created_at, record.updated_at),
            )
            self._index_text(record)
        self._write_markdown(record)
        self._index_embedding(record)
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
                   supersede_previous: bool = True,
                   extra: dict | None = None, **kw) -> DLSRecord:
        """工程现状快照。默认取代同项目上一份 status（旧的转 superseded）。

        ``extra`` 合并进 payload（如场景蒸馏层的 scenarios 块）。
        """
        payload = {
            "summary": summary,
            "current_state": current_state or {},
            "next_steps": list(next_steps),
            "open_questions": list(open_questions),
            "last_verified": _now(),
        }
        if extra:
            payload.update(extra)
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

    def list_records(self, project: str | None = None,
                     type: RecordType | None = None,
                     limit: int | None = None) -> list[DLSRecord]:
        """按过滤条件列出记录（基准与审计用；不检索不打分）。"""
        sql = "SELECT * FROM records"
        conds, params = [], []
        if project:
            conds.append("project=?")
            params.append(project)
        if type:
            conds.append("type=?")
            params.append(type)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY updated_at"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_record(r) for r in
                self._db.execute(sql, params).fetchall()]

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
                       link_to: str | None = None,
                       require_sources: bool = True) -> list[DLSRecord]:
        """零 LLM 蒸馏：把结构化事件流编译成 DEC/LES 并入库。

        事件 schema（调用方如 structured_failure / error_reflection 产）：
          {kind: "decision"|"lesson", title, source: "溯源 id", ...}
          decision: context/chosen/rationale/options_considered
          lesson:   mistake/correction/trigger/severity

        **下钻不变量**（v0.2）：require_sources=True 时每个事件必须携带
        source（画布 node_id / 记录 id / 外部引用 id）。两阶段先验后写：
        任一违规整批拒收——蒸馏是写入路径，不猜、不半写。
        未知 kind 与空标题跳过（非违规，不入库）。
        """
        items: list[tuple[dict, str, str, list[str]]] = []
        violations: list[str] = []
        for event in events:
            kind = event.get("kind")
            title = str(event.get("title", "")).strip()
            if not title or kind not in ("decision", "lesson"):
                continue
            raw = event.get("source") or event.get("sources") or []
            srcs = [raw] if isinstance(raw, str) else list(raw)
            srcs = [s for s in (str(x).strip() for x in srcs) if s]
            if require_sources and not srcs:
                violations.append(title)
            items.append((event, kind, title, srcs))
        if violations:
            raise DLSError(
                f"下钻不变量违规：{len(violations)} 条事件缺少 source"
                f"（{'；'.join(violations[:3])}）—— 整批拒收")

        created: list[DLSRecord] = []
        for event, kind, title, srcs in items:
            if kind == "decision":
                rec = self.add_decision(
                    title=title, context=event.get("context", ""),
                    chosen=event.get("chosen", ""),
                    rationale=event.get("rationale", ""),
                    options_considered=event.get("options_considered", []),
                    project=project, provenance="sleep-distill", sources=srcs)
            else:
                rec = self.add_lesson(
                    title=title, mistake=event.get("mistake", ""),
                    correction=event.get("correction", ""),
                    trigger=event.get("trigger", ""),
                    severity=event.get("severity", "medium"),
                    rule_of_thumb=event.get("rule_of_thumb", ""),
                    project=project, provenance="sleep-distill", sources=srcs)
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
            "只收录有长期价值的决策与教训，闲聊与流水账丢弃。\n"
            "每条必须带 source 字段（引用日志中的来源标记或原文片段 id），"
            "没有来源的条目不要输出。\n\n"
            f"工作日志：\n{day_notes}"
        )

    def drill_down(self, rec_id: str, max_depth: int = 5) -> dict:
        """下钻不变量验证：沿 sources 链走到原文，报告每一跳是否可解析。

        source 两类：记录 id（本库 records，递归下钻）与画布/外部 id
        （如 N-0001，按 ``refs/{id}.md`` 文件契约解析）。自身即原文
        （无 sources）视为可解析——不变量约束的是“抽象必须可溯源”，
        不是“每条记录都必须有上游”。
        """
        visited: set[str] = set()
        chain: list[dict] = []
        broken: list[str] = []

        def walk(node_id: str, depth: int) -> None:
            if node_id in visited or depth > max_depth:
                return
            visited.add(node_id)
            rec = self.get(node_id)
            if rec is not None:
                chain.append({"id": node_id, "kind": rec.type,
                              "resolved": True, "title": rec.title})
                for src in rec.sources:
                    walk(src, depth + 1)
            else:
                path = self.home / "refs" / f"{node_id}.md"
                resolved = path.exists()
                chain.append({"id": node_id, "kind": "ref",
                              "resolved": resolved, "title": node_id})
                if not resolved:
                    broken.append(node_id)

        walk(rec_id, 0)
        return {"record_id": rec_id, "chain": chain, "broken": broken,
                "chain_ok": not broken and bool(chain)}

    def distill_day(self, day_notes: str, project: str = "_global",
                    provider: Callable | None = None,
                    link_to: str | None = None,
                    scenarios: bool = False) -> list[DLSRecord]:
        """睡眠蒸馏的 LLM 路径：日志文本 → LLM 提炼事件 → DEC/LES 入库。

        provider 只需实现 ``complete(prompt) -> str``（见 mnemos_ark.llm，
        内置 OpenAI 兼容适配器）。provider 为 None 时显式报错——蒸馏是
        写入路径，不猜。解析失败同样报错（parse_event_array 不容错入库）。

        ``scenarios=True`` 启用场景蒸馏层（金字塔 L2，吸收自 TencentDB
        Agent Memory 的分层思想）：第二遍 LLM 把事件归纳为场景块，
        聚合成一条 status（payload.scenarios）并与其成员事件互索引
        （成员 --part_of--> status）。返回列表尾部即该 status。
        """
        if provider is None:
            raise DLSError("distill_day 需要 LLM provider（mnemos_ark.llm.OpenAICompatProvider）")
        from .llm import build_scenario_prompt, parse_event_array
        raw = provider.complete(self.build_distill_prompt(day_notes))
        try:
            events = parse_event_array(raw)
        except RuntimeError as exc:
            raise DLSError(f"蒸馏输出解析失败: {exc}") from exc
        created = self.distill_events(events, project=project, link_to=link_to)
        if not scenarios or not created:
            return created

        raw_scen = provider.complete(build_scenario_prompt(
            json.dumps(events, ensure_ascii=False)))
        try:
            scen_list = parse_event_array(raw_scen)
        except RuntimeError as exc:
            raise DLSError(f"场景蒸馏输出解析失败: {exc}") from exc
        blocks = []
        member_ids: set[str] = set()
        for scen in scen_list:
            if not isinstance(scen, dict) or not str(scen.get("title", "")).strip():
                continue
            members = []
            for m in scen.get("members", []):
                try:
                    rec = created[int(m)]
                except (ValueError, IndexError, TypeError):
                    continue
                members.append(rec.id)
                member_ids.add(rec.id)
            blocks.append({
                "title": str(scen["title"]).strip(),
                "situation": scen.get("situation", ""),
                "pattern": scen.get("pattern", ""),
                "response": scen.get("response", ""),
                "members": members,
            })
        if not blocks:
            return created
        sta = self.add_status(
            title=f"场景蒸馏 {time.strftime('%Y-%m-%d', time.localtime())}",
            summary=f"{len(blocks)} 个场景块（金字塔 L2）",
            project=project, provenance="sleep-distill-scenario",
            sources=sorted(member_ids),
            extra={"scenarios": blocks}, supersede_previous=False)
        for rid in member_ids:
            self.link(rid, sta.id, "part_of")
        created.append(sta)
        return created

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
        like_terms = self._query_terms(query) or [query]
        rows = self._db.execute(
            "SELECT * FROM records" + filter_sql.replace(" AND ", " WHERE ", 1)
            if filter_sql else "SELECT * FROM records",
            params).fetchall()
        scored = []
        for row in rows:
            haystack = (row["title"] + " " + row["body"] + " " + row["tags"]).lower()
            score = sum(1 for t in like_terms if t in haystack)
            if score:
                scored.append((score, row["updated_at"], row))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [self._row_to_record(s[2]) for s in scored[:limit]]

    # --------------------------------------- 语义检索（可插拔向量层）

    @property
    def embedder(self):
        return self._embedder

    @property
    def vector_backend(self) -> str:
        """当前向量后端能力：``sqlite-vec``（KNN 加速）或 ``blob``（全表余弦兑底）。

        报告的是可用能力而非激活态——vec0 表在首次写入嵌入时建。"""
        return "sqlite-vec" if self._vec_ok else "blob"

    def _ensure_vec_table(self, dim: int) -> bool:
        if not self._vec_ok:
            return False
        if self._vec_dim == dim:
            return True
        try:
            self._db.enable_load_extension(True)
            _sqlite_vec.load(self._db)
            if self._vec_dim is not None:
                self._db.execute("DROP TABLE IF EXISTS embeddings_vec")
                self._degrade_events.append(
                    f"向量维度变更 {self._vec_dim}->{dim}，vec0 索引已重建")
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vec "
                f"USING vec0(embedding float[{dim}])")
            self._vec_dim = dim
            return True
        except Exception as exc:
            self._vec_ok = False
            self._degrade_events.append(
                f"sqlite-vec 不可用，向量检索降级 blob 余弦: {exc}")
            return False

    def _vec_upsert(self, record_id: str, vec: list[float]) -> None:
        row = self._db.execute(
            "SELECT rowid FROM embeddings WHERE id=?", (record_id,)).fetchone()
        if row is None or not self._ensure_vec_table(len(vec)):
            return
        with self._db:
            self._db.execute(
                "DELETE FROM embeddings_vec WHERE rowid=?", (row["rowid"],))
            self._db.execute(
                "INSERT INTO embeddings_vec(rowid, embedding) VALUES (?,?)",
                (row["rowid"], _sqlite_vec.serialize_float32(vec)))

    def _index_embedding(self, record: DLSRecord) -> None:
        if self._embedder is None:
            return
        vec = self._embedder.embed([f"{record.title}\n{record.body}"])[0]
        blob = json.dumps([round(float(x), 6) for x in vec]).encode("utf-8")
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO embeddings (id, dim, vector) VALUES (?,?,?)",
                (record.id, len(vec), blob))
        self._vec_upsert(record.id, vec)

    def rebuild_embeddings(self) -> int:
        """挂载/更换 embedder 后全量重建向量索引，返回重建条数。"""
        if self._embedder is None:
            raise DLSError("未挂载 embedder：语义检索需要先传入 EmbeddingProvider")
        rows = self._db.execute("SELECT id, title, body FROM records").fetchall()
        with self._db:
            self._db.execute("DELETE FROM embeddings")
            for row in rows:
                vec = self._embedder.embed(
                    [f"{row['title']}\n{row['body']}"])[0]
                blob = json.dumps([round(float(x), 6) for x in vec]).encode("utf-8")
                self._db.execute(
                    "INSERT INTO embeddings (id, dim, vector) VALUES (?,?,?)",
                    (row["id"], len(vec), blob))
        self._db.execute("DROP TABLE IF EXISTS embeddings_vec")
        self._vec_dim = None
        for row in rows:
            vec = self._embedder.embed([f"{row['title']}\n{row['body']}"])[0]
            self._vec_upsert(row["id"], vec)
        return len(rows)

    def semantic_search(self, query: str, top_k: int = 10,
                        type: RecordType | None = None,
                        project: str | None = None) -> list[DLSRecord]:
        """向量语义检索。sqlite-vec 可用时走 KNN，否则 blob 全表余弦。
        未挂 embedder 时显式报错，不静默降级。"""
        if self._embedder is None:
            raise DLSError("未挂载 embedder：语义检索需要先传入 EmbeddingProvider")
        qvec = self._embedder.embed([query])[0]

        if self._vec_ok and self._vec_dim == len(qvec):
            try:
                knn = self._db.execute(
                    "SELECT rowid, distance FROM embeddings_vec "
                    "WHERE embedding MATCH ? AND k = ?",
                    (_sqlite_vec.serialize_float32(qvec), top_k * 4)).fetchall()
                scored = []
                for hit in knn:
                    row = self._db.execute(
                        "SELECT * FROM records WHERE id=("
                        "SELECT id FROM embeddings WHERE rowid=?)",
                        (hit["rowid"],)).fetchone()
                    if row is None:
                        continue
                    if type and row["type"] != type:
                        continue
                    if project and row["project"] != project:
                        continue
                    scored.append((-float(hit["distance"]), row["updated_at"], row))
                scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
                return [self._row_to_record(s[2]) for s in scored[:top_k]]
            except sqlite3.OperationalError as exc:
                self._degrade_events.append(
                    f"vec0 KNN 查询失败，降级 blob 余弦: {exc}")

        rows = self._db.execute(
            "SELECT e.id, e.vector, r.* FROM embeddings e "
            "JOIN records r ON r.id = e.id").fetchall()
        scored = []
        for row in rows:
            if type and row["type"] != type:
                continue
            if project and row["project"] != project:
                continue
            vec = json.loads(row["vector"])
            score = cosine(qvec, vec) if len(vec) == len(qvec) else 0.0
            scored.append((score, row["updated_at"], row))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [self._row_to_record(s[2]) for s in scored[:top_k]]

    def hybrid_search(self, query: str, top_k: int = 10,
                      type: RecordType | None = None,
                      project: str | None = None) -> list[DLSRecord]:
        """词法 + 语义混合检索（RRF 融合）。无 embedder 时降级词法并记录事件。"""
        lexical = self.search(query, type=type, project=project, limit=top_k * 3)
        if self._embedder is None:
            self._degrade_events.append("hybrid_search 无 embedder，降级词法")
            return lexical[:top_k]
        semantic = self.semantic_search(query, top_k=top_k * 3,
                                        type=type, project=project)
        fused: dict[str, float] = {}
        records: dict[str, DLSRecord] = {}
        for rank, rec in enumerate(lexical):
            fused[rec.id] = fused.get(rec.id, 0.0) + 1.0 / (60 + rank + 1)
            records[rec.id] = rec
        for rank, rec in enumerate(semantic):
            fused[rec.id] = fused.get(rec.id, 0.0) + 1.0 / (60 + rank + 1)
            records[rec.id] = rec
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [records[rid] for rid, _ in ordered[:top_k]]

    # --------------------------------------- U 形曲线装箱

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
            "provenance": record.provenance, "sources": record.sources,
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
            sources=json.loads(row["sources"]) if "sources" in row.keys() else [],
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
                sources TEXT NOT NULL DEFAULT '[]',
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
        cols = {r["name"] for r in self._db.execute(
            "PRAGMA table_info(records)").fetchall()}
        if "sources" not in cols:
            self._db.execute(
                "ALTER TABLE records ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
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
