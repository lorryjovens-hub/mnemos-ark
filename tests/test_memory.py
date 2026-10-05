"""DLS 结构化记忆引擎测试：三库写入、互索引、status 引导、U 形装箱。"""

from __future__ import annotations

import json

import pytest

from mnemos_ark.memory import DLSMemory, DLSError
from mnemos_ark import memory_mcp as mcp


@pytest.fixture()
def dls(tmp_path):
    engine = DLSMemory(home=tmp_path / "dls")
    yield engine
    engine.close()


def _sample_project(dls: DLSMemory):
    """造一个小型工程记忆组：1 决策 + 2 错题 + 1 现状，全部互索引。"""
    dec = dls.add_decision(
        title="会话记忆走 SQLite 单写入面",
        context="4 套并行记忆实现互相打架",
        chosen="vault_bridge 作为唯一写入口",
        rationale="统一打作用域标签的前提是单一写入面",
        options_considered=["继续四套并行", "全部重写", "收敛到单写入面"],
        project="mem-proj")
    les1 = dls.add_lesson(
        title="v1 用边际分布测结构",
        mistake="打乱标签得到 +0.0000 却误判为无结构",
        correction="用互信息而不是边际分布",
        rule_of_thumb="测结构用互信息", project="mem-proj")
    les2 = dls.add_lesson(
        title="FTS5 不可用时曾静默降级",
        mistake="检索悄悄变差无人发觉",
        correction="降级必须记录事件",
        rule_of_thumb="降级不静默", project="mem-proj")
    sta = dls.add_status(
        title="mem-proj 工程现状",
        summary="单写入面已收敛，检索层待接路由",
        next_steps=["接入 memory_router", "扩 LongMemEval 样本"],
        open_questions=["多模态记忆怎么办"], project="mem-proj")
    dls.link(sta.id, dec.id, "part_of")
    dls.link(dec.id, les1.id, "caused")
    dls.link(dec.id, les2.id, "fixes")
    return dec, les1, les2, sta


class TestWrite:
    def test_ids_sequential_by_type(self, dls):
        a = dls.add("decision", "d1")
        b = dls.add("decision", "d2")
        c = dls.add("lesson", "l1")
        assert (a.id, b.id, c.id) == ("DEC-0001", "DEC-0002", "LES-0001")

    def test_rejects_invalid_type_title_confidence(self, dls):
        with pytest.raises(DLSError):
            dls.add("gossip", "x")
        with pytest.raises(DLSError):
            dls.add("decision", "   ")
        with pytest.raises(DLSError):
            dls.add("decision", "x", confidence=1.5)

    def test_verifier_hook_rejects_error_state(self, tmp_path):
        engine = DLSMemory(home=tmp_path / "dls",
                           verifier=lambda text: {"state": "error", "evidence": "幻觉"})
        with pytest.raises(DLSError, match="验证门拒绝"):
            engine.add("decision", "假决策", body="假的")
        engine.close()

    def test_markdown_mirror_written(self, dls, tmp_path):
        rec = dls.add("lesson", "镜像测试", body="正文", project="p1", tags=["t"])
        md = tmp_path / "dls" / "md" / "p1" / f"{rec.id}.md"
        assert md.exists()
        text = md.read_text(encoding="utf-8")
        assert text.startswith("---")
        assert f'"{rec.id}"' in text
        assert "正文" in text

    def test_status_supersedes_previous(self, dls):
        s1 = dls.add_status("现状1", "第一版", project="p2")
        s2 = dls.add_status("现状2", "第二版", project="p2")
        assert dls.get(s1.id).status == "superseded"
        assert dls.get(s2.id).status == "active"
        assert dls.current_status("p2").id == s2.id


class TestLinks:
    def test_bidirectional_index(self, dls):
        dec, les1, _, sta = _sample_project(dls)
        out = dls.neighbors(sta.id)["out"]
        assert any(e["id"] == dec.id and e["relation"] == "part_of" for e in out)
        into = dls.neighbors(dec.id)["in"]
        assert any(e["id"] == sta.id for e in into)
        assert any(e["id"] == les1.id for e in dls.neighbors(dec.id)["out"])

    def test_rejects_dangling_and_self_loop(self, dls):
        rec = dls.add("decision", "x")
        with pytest.raises(DLSError, match="悬空"):
            dls.link(rec.id, "DEC-9999", "relates")
        with pytest.raises(DLSError, match="自环"):
            dls.link(rec.id, rec.id, "relates")
        with pytest.raises(DLSError, match="非法关系"):
            dls.link(rec.id, rec.id, "gossip")


class TestBootstrap:
    def test_status_first_with_rings(self, dls):
        dec, les1, les2, sta = _sample_project(dls)
        boot = dls.bootstrap("mem-proj", budget_tokens=2000)
        assert f"[{sta.id}]" in boot["status"]
        ring1_ids = {r["id"] for r in boot["ring1"]}
        assert dec.id in ring1_ids
        ring2_ids = {r["id"] for r in boot["ring2"]}
        assert les1.id in ring2_ids and les2.id in ring2_ids
        assert boot["index"] == {"decision": 1, "lesson": 2, "status": 1}

    def test_budget_trim_marks_pointer(self, dls):
        dec, _, _, sta = _sample_project(dls)
        boot = dls.bootstrap("mem-proj", budget_tokens=1)
        entry = next(r for r in boot["ring1"] if r["id"] == dec.id)
        assert entry.get("trimmed") is True
        assert boot["usage"]["trimmed"] >= 1

    def test_empty_project(self, dls):
        boot = dls.bootstrap("nothing-here")
        assert boot["status"] is None
        assert boot["usage"]["used"] == 0


class TestJumpAndPack:
    def test_jump_returns_single_full_record(self, dls):
        dec, _, _, _ = _sample_project(dls)
        text = dls.jump(dec.id)
        assert f"[{dec.id}]" in text
        assert "vault_bridge" in text
        assert "LES-" not in text  # 不连带邻居

    def test_pack_context_u_shape(self, dls):
        dec, les1, les2, sta = _sample_project(dls)
        order = [sta.id, dec.id, les1.id, les2.id]
        text = dls.pack_context(order, budget_tokens=400)
        head, tail = text.split("\n\n---\n\n")[0], text.split("\n\n---\n\n")[-1]
        assert head.startswith(f"## [{sta.id}]")   # 首部全文
        assert f"## [{les2.id}]" in tail           # 尾部全文
        assert f"[{dec.id}]" in text               # 中段至少留指针

    def test_pack_respects_budget_by_degrading(self, dls):
        _dec, les1, les2, sta = _sample_project(dls)
        many = [sta.id] + [r.id for r in (les1, les2)]
        tiny = dls.pack_context(many, budget_tokens=2)
        assert f"[{les1.id}]" in tiny  # 降级为指针/编号后仍可见编号


class TestSearch:
    def test_fts_or_like_fallback(self, dls):
        _sample_project(dls)
        hits = dls.search("互信息", project="mem-proj")
        assert any(r.title.startswith("v1 用边际分布") for r in hits)
        assert dls.search("互信息", type="lesson")

    def test_search_scope_filter(self, dls):
        _sample_project(dls)
        dls.add("lesson", "无关项目的互信息教训", project="other")
        hits = dls.search("互信息", project="mem-proj")
        assert all(r.project == "mem-proj" for r in hits)


class TestMcpSurface:
    def test_mcp_end_to_end(self, tmp_path):
        home = str(tmp_path / "dls")
        created = json.loads(mcp.dls_add_decision(
            title="走 remote MCP", context="stdio 不跨机", chosen="remote",
            rationale="手机要连", options_considered="stdio|remote",
            project="mc", home=home))
        assert created["id"] == "DEC-0001"
        sta = json.loads(mcp.dls_add_status(
            title="mc 现状", summary="刚起步", next_steps="写 UI",
            project="mc", home=home))
        mcp.dls_link(sta["id"], created["id"], "part_of", home=home)
        boot = json.loads(mcp.dls_bootstrap("mc", home=home))
        assert created["id"] in json.dumps(boot, ensure_ascii=False)
        jumped = mcp.dls_jump(created["id"], home=home)
        assert "remote" in jumped
        neighbors = json.loads(mcp.dls_neighbors(created["id"], home=home))
        assert neighbors["in"][0]["id"] == sta["id"]
        packed = json.loads(mcp.dls_pack_context(
            f"{sta['id']},{created['id']}", budget_tokens=100, home=home))
        assert f"[{sta['id']}]" in packed["context"]
