"""任务路由器的 MCP 工具面 + 资源解析器接线。

注册函数 ``register_task_router_tools(mcp)`` 与 laap-cognitive 服务器其它
``*_mcp_tools.register_*_tools`` 同构。三个外部解析器的接入策略：

- skill_resolver：调用本机 skill-router 语义路由（路径由环境变量
  ``LAAP_SKILL_ROUTER`` 指定，缺省从 laap 包位置相对解析），失败降级规则表
- graph_resolver：调用资产图谱注册表（存在才接），失败降级空
- memory_resolver：DLS 引擎的 infer_project（作用域路由，P2）
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .memory import DLSMemory, DLSError
from .router import TaskRouter

logger = logging.getLogger(__name__)

_router: TaskRouter | None = None


def _skill_router_script() -> Path | None:
    env = os.environ.get("LAAP_SKILL_ROUTER")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "skills" / "skill-router" / "scripts" / "route.py"
    return p if p.exists() else None


def _resolve_skills(task: str) -> list[str]:
    """调 skill-router 语义路由，解析输出里的 skill 名（best-effort）。"""
    script = _skill_router_script()
    if script is None:
        raise FileNotFoundError("skill-router 不可用（LAAP_SKILL_ROUTER 未配置）")
    proc = subprocess.run(
        [sys.executable, str(script), task], capture_output=True,
        text=True, timeout=30, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"route.py 退出码 {proc.returncode}")
    names: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip().lstrip("*-· ").strip()
        if line and " " not in line and "/" not in line and len(line) < 40:
            names.append(line)
    return names[:8]


def _resolve_graph(task: str) -> list[str]:
    """资产图谱文件线索（存在注册表才接）。"""
    repo_root = Path(__file__).resolve().parents[2]
    registry = repo_root / "asset_graph_engine" / "asset_registry.json"
    if not registry.exists():
        raise FileNotFoundError("asset_registry.json 不存在")
    data = json.loads(registry.read_text(encoding="utf-8"))
    domains = [str(k) for k in (data.get("domains") or {}).keys()]
    hits = [d for d in domains if d.split("_")[0] in task][:5]
    return [f"asset_graph_engine 域: {h}" for h in hits]


def get_task_router() -> TaskRouter:
    global _router
    if _router is None:
        mem = DLSMemory()

        def _mem_resolve(task: str) -> str:
            return mem.infer_project(task)

        _router = TaskRouter(
            skill_resolver=_resolve_skills,
            graph_resolver=_resolve_graph,
            memory_resolver=_mem_resolve,
        )
    return _router


def route_task(task: str, project: str = "") -> str:
    """路由一个任务，返回定位/路由/计划/验收契约 JSON。"""
    try:
        return json.dumps(get_task_router().route(task, project).to_dict(),
                          ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def dls_sleep_distill(day_notes_json: str, project: str = "_global",
                      link_to: str = "") -> str:
    """睡眠蒸馏（P3）：结构化事件数组 → DEC/LES 入库。

    day_notes_json 是 JSON 数组（零 LLM 路径，事件由结构化失败/会话钩子产生）。
    原始文本日志请先用 DLSMemory.build_distill_prompt 走 LLM 产出事件数组。
    """
    try:
        events = json.loads(day_notes_json or "[]")
        if not isinstance(events, list):
            raise DLSError("day_notes_json 必须是事件数组")
        created = DLSMemory().distill_events(
            events, project=project, link_to=link_to or None)
        return json.dumps(
            {"distilled": len(created), "ids": [r.id for r in created]},
            ensure_ascii=False)
    except (DLSError, json.JSONDecodeError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def register_task_router_tools(mcp_server: Any) -> int:
    """向 FastMCP 注册任务路由器工具，返回注册数量。"""
    if mcp_server is None:
        logger.warning("register_task_router_tools: mcp_server is None")
        return 0

    @mcp_server.tool()
    async def laap_route_task(task: str, project: str = "") -> str:
        """LAAP 任务路由器：问题精准定位 → skills/MCP/子代理精准路由。

        输出执行契约：problem（分类/实体）+ route（skills/mcp_tools/agents/
        file_hints）+ plan（定位/执行/沉淀）+ verification（宪章/oracle/测试）
        + memory（项目域与回写类型）。解析器失败时 degraded 字段显式记录。

        Args:
            task: 任务描述（一句话或一段话）。
            project: 可选项目域；缺省由 DLS 记忆推断。
        """
        return route_task(task, project)

    @mcp_server.tool()
    async def laap_sleep_distill(day_notes_json: str, project: str = "_global",
                                 link_to: str = "") -> str:
        """睡眠蒸馏（P3）：结构化事件数组 → DEC/LES 记忆入库。

        事件 schema 见 DLSMemory.distill_events；原始日志请先用
        build_distill_prompt 走 LLM 产出事件数组再调本工具。
        """
        return dls_sleep_distill(day_notes_json, project, link_to)

    return 2
