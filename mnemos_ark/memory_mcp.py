"""DLS 结构化记忆的 MCP 工具面。

与 laap-cognitive 服务器其它 *_mcp_tools.py 同构：纯函数 + JSON 字符串返回，
便于直接注册为 MCP 工具。所有工具共用一个模块级 DLSMemory 实例（同一数据
目录），参数中的 home 可指向测试目录。
"""

from __future__ import annotations

import json
from typing import Any

from .memory import DLSMemory, DLSError, default_home

_INSTANCES: dict[str, DLSMemory] = {}


def _engine(home: str | None = None) -> DLSMemory:
    key = home or str(default_home())
    if key not in _INSTANCES:
        _INSTANCES[key] = DLSMemory(home=home)
    return _INSTANCES[key]


def _ok(**kw: Any) -> str:
    return json.dumps(kw, ensure_ascii=False)


def _err(exc: Exception) -> str:
    return json.dumps({"error": str(exc)}, ensure_ascii=False)


def dls_add_record(type: str, title: str, body: str = "", project: str = "_global",
                   payload_json: str = "{}", confidence: float = 1.0,
                   tags: str = "", provenance: str = "", home: str = "") -> str:
    """写入一条 DLS 记录（type ∈ decision/lesson/status）。tags 逗号分隔。"""
    try:
        rec = _engine(home or None).add(
            type=type, title=title, body=body, project=project,
            payload=json.loads(payload_json or "{}"), confidence=confidence,
            tags=[t for t in tags.split(",") if t.strip()], provenance=provenance)
        return _ok(stored=True, id=rec.id, type=rec.type)
    except (DLSError, json.JSONDecodeError) as exc:
        return _err(exc)


def dls_add_decision(title: str, context: str, chosen: str, rationale: str,
                     options_considered: str = "", project: str = "_global",
                     home: str = "") -> str:
    """记录一条决策史（长文）：背景/备选/选择/理由。"""
    try:
        rec = _engine(home or None).add_decision(
            title=title, context=context, chosen=chosen, rationale=rationale,
            options_considered=[o for o in options_considered.split("|") if o.strip()],
            project=project)
        return _ok(stored=True, id=rec.id)
    except DLSError as exc:
        return _err(exc)


def dls_add_lesson(title: str, mistake: str, correction: str,
                   trigger: str = "", severity: str = "medium",
                   rule_of_thumb: str = "", project: str = "_global",
                   home: str = "") -> str:
    """记录一条错题本（中文）：触发/错误/纠正/口诀。"""
    try:
        rec = _engine(home or None).add_lesson(
            title=title, mistake=mistake, correction=correction,
            trigger=trigger, severity=severity, rule_of_thumb=rule_of_thumb,
            project=project)
        return _ok(stored=True, id=rec.id)
    except DLSError as exc:
        return _err(exc)


def dls_add_status(title: str, summary: str, next_steps: str = "",
                   open_questions: str = "", project: str = "_global",
                   home: str = "") -> str:
    """记录工程现状快照（短文）。默认取代同项目上一份 status。"""
    try:
        rec = _engine(home or None).add_status(
            title=title, summary=summary,
            next_steps=[s for s in next_steps.split("|") if s.strip()],
            open_questions=[q for q in open_questions.split("|") if q.strip()],
            project=project)
        return _ok(stored=True, id=rec.id, superseded_previous=True)
    except DLSError as exc:
        return _err(exc)


def dls_link(src: str, dst: str, relation: str, note: str = "",
             home: str = "") -> str:
    """建立互索引边。relation ∈ supports/refutes/supersedes/caused/derived_from/fixes/relates/part_of。"""
    try:
        edge = _engine(home or None).link(src=src, dst=dst, relation=relation, note=note)
        return _ok(linked=True, src=edge.src, dst=edge.dst, relation=edge.relation)
    except DLSError as exc:
        return _err(exc)


def dls_jump(rec_id: str, home: str = "") -> str:
    """按编号精准跳转，返回单条全文（不连带邻居）——上下文经济主通道。"""
    return _engine(home or None).jump(rec_id)


def dls_neighbors(rec_id: str, home: str = "") -> str:
    """查一条记录的互索引（出边+入边），用于按需展开下一跳。"""
    try:
        return _ok(**_engine(home or None).neighbors(rec_id))
    except DLSError as exc:
        return _err(exc)


def dls_bootstrap(project: str = "_global", budget_tokens: int = 1200,
                  home: str = "") -> str:
    """fresh session 冷启动：status 全文 → 一环指针 → 二环编号。"""
    try:
        result = _engine(home or None).bootstrap(project=project,
                                                 budget_tokens=budget_tokens)
        return json.dumps(result, ensure_ascii=False)
    except DLSError as exc:
        return _err(exc)


def dls_search(query: str, type: str = "", project: str = "",
               limit: int = 20, home: str = "") -> str:
    """词法检索（FTS5，缺失时降级 LIKE）。type 为空搜全库。"""
    try:
        recs = _engine(home or None).search(
            query=query, type=type or None, project=project or None, limit=limit)
        return _ok(results=[{"id": r.id, "type": r.type, "title": r.title,
                             "line": r.one_liner()} for r in recs])
    except DLSError as exc:
        return _err(exc)


def dls_pack_context(rec_ids: str, budget_tokens: int = 2000,
                     home: str = "") -> str:
    """U 形曲线装箱：逗号分隔的编号列表 → 首尾全文 + 中段指针的上下文块。"""
    try:
        ids = [i.strip() for i in rec_ids.split(",") if i.strip()]
        text = _engine(home or None).pack_context(ids, budget_tokens=budget_tokens)
        return _ok(context=text)
    except DLSError as exc:
        return _err(exc)


def dls_infer_project(query: str, home: str = "") -> str:
    """作用域路由（P2）：从查询推断激活项目域，闲聊落 _global。"""
    try:
        project = _engine(home or None).infer_project(query)
        return _ok(project=project)
    except DLSError as exc:
        return _err(exc)


def dls_drill_down(rec_id: str, home: str = "") -> str:
    """下钻不变量验证：沿 sources 溯源链走到原文，报告每一跳可解析性。"""
    try:
        return json.dumps(_engine(home or None).drill_down(rec_id),
                          ensure_ascii=False)
    except DLSError as exc:
        return _err(exc)


def dls_export_pack(project: str, require_chain_ok: bool = True,
                    home: str = "") -> str:
    """导出团队记忆包（治理门：溯源可验证才出门，被拒附理由）。"""
    try:
        from .sharing import SharePolicy, export_pack
        pack = export_pack(_engine(home or None), project,
                           SharePolicy(require_chain_ok=require_chain_ok))
        return json.dumps(pack, ensure_ascii=False)
    except (DLSError, ValueError) as exc:
        return _err(exc)


def dls_import_pack(pack_json: str, target_project: str = "",
                    home: str = "") -> str:
    """导入团队记忆包（验哈希 → 策略裁决 → 去重 → 重新编号 → 边重映射）。"""
    try:
        from .sharing import import_pack
        pack = json.loads(pack_json)
        result = import_pack(_engine(home or None), pack,
                             target_project=target_project or None)
        return json.dumps(result, ensure_ascii=False)
    except (DLSError, ValueError) as exc:
        return _err(exc)


def register_dls_tools(mcp_server: Any) -> int:
    """向 FastMCP 注册 DLS 记忆工具，返回注册数量。

    与 world_prior_mcp.register_world_prior_tools 同构；生产实例的
    数据目录由 LAAP_DLS_HOME 决定，缺省 ~/.laap/dls。
    """
    if mcp_server is None:
        return 0

    @mcp_server.tool()
    async def dls_add_record(type: str, title: str, body: str = "",
                             project: str = "_global", payload_json: str = "{}",
                             confidence: float = 1.0, tags: str = "",
                             provenance: str = "") -> str:
        """写入一条 DLS 结构化记忆（type ∈ decision/lesson/status）。

        三库：DEC 决策史（长）/ LES 错题本（中）/ STA 工程现状（短），
        自动落 SQLite + Markdown 镜像。tags 逗号分隔。
        """
        return _run(_IMPL["add_record"], type=type, title=title, body=body,
                    project=project, payload_json=payload_json,
                    confidence=confidence, tags=tags, provenance=provenance)

    @mcp_server.tool()
    async def dls_add_decision(title: str, context: str, chosen: str,
                               rationale: str, options_considered: str = "",
                               project: str = "_global") -> str:
        """记录一条决策史：背景/备选（|分隔）/选择/理由，长期保留可被推翻不覆盖。"""
        return _run(_IMPL["add_decision"], title=title, context=context, chosen=chosen,
                    rationale=rationale, options_considered=options_considered,
                    project=project)

    @mcp_server.tool()
    async def dls_add_lesson(title: str, mistake: str, correction: str,
                             trigger: str = "", severity: str = "medium",
                             rule_of_thumb: str = "",
                             project: str = "_global") -> str:
        """记录一条错题本：触发/错误/纠正/一句话口诀。"""
        return _run(_IMPL["add_lesson"], title=title, mistake=mistake,
                    correction=correction, trigger=trigger, severity=severity,
                    rule_of_thumb=rule_of_thumb, project=project)

    @mcp_server.tool()
    async def dls_add_status(title: str, summary: str, next_steps: str = "",
                             open_questions: str = "",
                             project: str = "_global") -> str:
        """记录工程现状快照（短文）。默认取代同项目上一份 status。"""
        return _run(_IMPL["add_status"], title=title, summary=summary,
                    next_steps=next_steps, open_questions=open_questions,
                    project=project)

    @mcp_server.tool()
    async def dls_link(src: str, dst: str, relation: str, note: str = "") -> str:
        """建立互索引边。relation ∈ supports/refutes/supersedes/caused/derived_from/fixes/relates/part_of。"""
        return _run(_IMPL["link"], src=src, dst=dst, relation=relation, note=note)

    @mcp_server.tool()
    async def dls_jump(rec_id: str) -> str:
        """按编号精准跳转，返回单条全文（不连带邻居）——上下文经济主通道。"""
        return _run(_IMPL["jump"], rec_id=rec_id)

    @mcp_server.tool()
    async def dls_neighbors(rec_id: str) -> str:
        """查一条记录的互索引（出边+入边），用于按需展开下一跳。"""
        return _run(_IMPL["neighbors"], rec_id=rec_id)

    @mcp_server.tool()
    async def dls_bootstrap(project: str = "_global",
                            budget_tokens: int = 1200) -> str:
        """fresh session 冷启动：status 全文 → 一环指针 → 二环编号，按预算裁剪。"""
        return _run(_IMPL["bootstrap"], project=project, budget_tokens=budget_tokens)

    @mcp_server.tool()
    async def dls_search(query: str, type: str = "", project: str = "",
                         limit: int = 20) -> str:
        """DLS 记忆检索（FTS5/中文 LIKE）。type 为空搜全库。"""
        return _run(_IMPL["search"], query=query, type=type, project=project,
                    limit=limit)

    @mcp_server.tool()
    async def dls_pack_context(rec_ids: str, budget_tokens: int = 2000) -> str:
        """U 形曲线装箱：逗号分隔编号 → 首尾全文 + 中段指针的上下文块。"""
        return _run(_IMPL["pack_context"], rec_ids=rec_ids,
                    budget_tokens=budget_tokens)

    @mcp_server.tool()
    async def dls_infer_project(query: str) -> str:
        """作用域路由（P2）：从查询推断激活项目域，闲聊落 _global。"""
        return _run(_IMPL["infer_project"], query=query)

    @mcp_server.tool()
    async def dls_drill_down(rec_id: str) -> str:
        """下钻不变量验证：沿 sources 溯源链走到原文（记录 id / refs/ 文件契约）。"""
        return _run(_IMPL["drill_down"], rec_id=rec_id)

    @mcp_server.tool()
    async def dls_export_pack(project: str, require_chain_ok: bool = True) -> str:
        """导出团队记忆包（治理铁律：溯源可验证才出门，被拒附理由）。"""
        return _run(_IMPL["export_pack"], project=project,
                    require_chain_ok=require_chain_ok)

    @mcp_server.tool()
    async def dls_import_pack(pack_json: str, target_project: str = "") -> str:
        """导入团队记忆包（验哈希防篡改 → 策略裁决 → 去重 → 边重映射）。"""
        return _run(_IMPL["import_pack"], pack_json=pack_json,
                    target_project=target_project)

    return 14


def _run(fn: Any, **kw: Any) -> str:
    """统一执行器：剥离 home 生产缺省，异常归入 JSON 错误。"""
    kw.setdefault("home", "")
    try:
        return fn(**kw)
    except Exception as exc:  # 工具面不向上抛，全部 JSON 化
        return _err(exc)


# 实现函数表：注册层的内层工具与模块函数同名，Python 作用域会遮蔽全局名，
# 因此统一走这张表取实现（避免自递归与协程泄漏）。
_IMPL: dict[str, Any] = {
    "add_record": dls_add_record,
    "add_decision": dls_add_decision,
    "add_lesson": dls_add_lesson,
    "add_status": dls_add_status,
    "link": dls_link,
    "jump": dls_jump,
    "neighbors": dls_neighbors,
    "bootstrap": dls_bootstrap,
    "search": dls_search,
    "pack_context": dls_pack_context,
    "infer_project": dls_infer_project,
    "drill_down": dls_drill_down,
    "export_pack": dls_export_pack,
    "import_pack": dls_import_pack,
}
