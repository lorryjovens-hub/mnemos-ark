"""任务路由器 + DLS P2/P3 测试：分类、路由、作用域防污染、蒸馏、注册面。"""

from __future__ import annotations

import json

import pytest

from mnemos_ark.router import TaskRouter
from mnemos_ark import router_mcp as trm
from mnemos_ark import memory_mcp as mcp
from mnemos_ark.memory import DLSMemory


class FakeMCP:
    """最小 FastMCP 替身：收集 @tool() 装饰的函数。"""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


# ────────────────────────────────────────────── 分类与实体

class TestClassify:
    def test_frontend_backend_debug(self):
        r = TaskRouter()
        assert r.classify("把这个落地页的组件样式改一下")[0][0] == "frontend"
        assert r.classify("重构记忆引擎的 API 接口")[0][0] == "backend"
        assert r.classify("程序报错了，帮我排查这个 bug")[0][0] == "debug"

    def test_entity_extraction(self):
        r = TaskRouter()
        ents = r.extract_entities("修复 laap/cognition/dls_memory.py 的 search 函数")
        assert any("dls_memory.py" in p for p in ents["paths"])
        assert "search" in ents["keywords"]


# ────────────────────────────────────────────── 路由契约

class TestRoute:
    def test_frontend_routes_to_charter_skill_stack(self):
        route = TaskRouter().route("做一个官网落地页").to_dict()
        skills = route["route"]["skills"]
        assert skills[:3] == ["taste-skill", "emil-design-eng", "impeccable"]
        assert any("VIS-01" in o for o in route["verification"]["oracles"])
        assert any("Lucide" in c for c in route["verification"]["charter"])
        assert route["memory"]["write_back"] == "decision"

    def test_debug_routes_to_lesson_and_root_cause(self):
        route = TaskRouter().route("排查这个报错为什么不工作").to_dict()
        assert route["memory"]["write_back"] == "lesson"
        assert any("根因" in step for step in route["plan"])
        assert any("回归测试" in t for t in route["verification"]["tests"])

    def test_plan_ends_with_memory_writeback(self):
        route = TaskRouter().route("写一篇调研报告").to_dict()
        assert "DLS" in route["plan"][-1]

    def test_resolvers_inject_and_degrade(self):
        router = TaskRouter(
            skill_resolver=lambda t: ["custom-skill"],
            graph_resolver=lambda t: ["file: x.py"],
            memory_resolver=lambda t: "proj-x")
        d = router.route("做个页面").to_dict()
        assert d["route"]["skills"][0] == "custom-skill"
        assert d["route"]["file_hints"] == ["file: x.py"]
        assert d["memory"]["project"] == "proj-x"
        assert d["degraded"] == []

    def test_failing_resolver_degrades_loudly(self):
        def boom(t):
            raise RuntimeError("炸了")
        d = TaskRouter(skill_resolver=boom).route("做个页面").to_dict()
        assert any("skill_resolver 失败" in x for x in d["degraded"])
        assert d["route"]["skills"]  # 规则表兜底仍在

    def test_no_resolver_declared(self):
        d = TaskRouter().route("做个页面").to_dict()
        assert any("未接入" in x for x in d["degraded"])


# ────────────────────────────────────────────── P2 作用域防污染

class TestScopeIsolation:
    def test_chatter_does_not_enter_project_bucket(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "dls")
        dls.add_decision("Redis 连接池选型", context="项目", chosen="Hikari",
                         rationale="稳", project="web-proj")
        dls.add_lesson("今天好累", mistake="熬夜", correction="早睡",
                       project="_global")
        # 项目问题 → 项目桶；闲聊 → _global，绝不拉出项目记忆
        assert dls.infer_project("Redis 连接池怎么选") == "web-proj"
        assert dls.infer_project("今天好累啊") in ("_global",)
        boot = dls.bootstrap(dls.infer_project("今天好累啊"))
        assert "Redis" not in json.dumps(boot, ensure_ascii=False)
        dls.close()

    def test_search_scoped_by_inferred_project(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "dls")
        dls.add_lesson("A 项目的坑", mistake="x", correction="y", project="pa")
        proj = dls.infer_project("A 项目的坑")
        hits = dls.search("坑", project=proj)
        assert all(h.project == proj for h in hits)
        dls.close()


# ────────────────────────────────────────────── P3 蒸馏

class TestDistill:
    def test_distill_events_creates_and_links(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "dls")
        sta = dls.add_status("现状", "s", project="p")
        events = [
            {"kind": "decision", "title": "选了方案 A", "context": "c",
             "chosen": "A", "rationale": "r", "options_considered": ["A", "B"]},
            {"kind": "lesson", "title": "踩了坑 X", "mistake": "m",
             "correction": "c", "rule_of_thumb": "口诀"},
            {"kind": "gossip", "title": "闲聊不入库"},
        ]
        created = dls.distill_events(events, project="p", link_to=sta.id)
        assert [r.type for r in created] == ["decision", "lesson"]
        assert created[0].provenance == "sleep-distill"
        links = dls.links_from(sta.id)
        assert sum(1 for e in links if e.relation == "derived_from") == 2
        dls.close()

    def test_distill_prompt_schema(self):
        prompt = DLSMemory.build_distill_prompt("今天修了 bug")
        assert '"kind":"decision"' in prompt and '"kind":"lesson"' in prompt
        assert "只输出 JSON 数组" in prompt


# ────────────────────────────────────────────── 注册面

class TestRegistration:
    def test_register_dls_tools_count(self):
        fake = FakeMCP()
        assert mcp.register_dls_tools(fake) == 11
        assert "dls_bootstrap" in fake.tools and "dls_jump" in fake.tools

    def test_register_task_router_tools_count(self):
        fake = FakeMCP()
        assert trm.register_task_router_tools(fake) == 2
        assert "laap_route_task" in fake.tools

    def test_registered_jump_is_not_recursive(self, tmp_path, monkeypatch):
        """内层工具与模块函数同名——必须走实现表，否则自递归。"""
        fake = FakeMCP()
        mcp.register_dls_tools(fake)
        engine = DLSMemory(home=tmp_path / "dls")
        monkeypatch.setattr(mcp, "_engine", lambda home=None: engine)
        import asyncio
        out = asyncio.run(fake.tools["dls_jump"]("DEC-9999"))
        assert "未找到" in out
        engine.close()

    def test_route_task_end_to_end(self, monkeypatch):
        monkeypatch.setattr(trm, "_router", TaskRouter())  # 隔离子进程解析器
        out = trm.route_task("修复登录接口报错", project="p")
        data = json.loads(out)
        assert data["problem"]["primary_type"] in ("debug", "backend")
        assert data["memory"]["project"] == "p"
