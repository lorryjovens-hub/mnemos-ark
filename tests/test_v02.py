"""v0.2 测试：符号画布 / 下钻不变量 / 场景蒸馏层 / 长时程基准。"""

from __future__ import annotations

import json

import pytest

from mnemos_ark.canvas import TaskCanvas, CanvasError
from mnemos_ark.llm import CallableProvider, build_scenario_prompt
from mnemos_ark.memory import DLSMemory, DLSError
from mnemos_ark.longrun import run_long_horizon, synthesize_long_run


# ────────────────────────────────── 符号画布

class TestCanvas:
    def test_offload_recall_roundtrip(self, tmp_path):
        cv = TaskCanvas(home=tmp_path / "h")
        heavy = "超长工具日志 " * 500
        node = cv.offload("步骤1 抓取数据", heavy, kind="tool-output")
        assert (tmp_path / "h" / "refs" / f"{node.node_id}.md").exists()
        assert heavy in cv.recall(node.node_id)
        assert node.tokens > 1000
        cv.close()

    def test_pack_stays_small_while_offload_is_big(self, tmp_path):
        cv = TaskCanvas(home=tmp_path / "h")
        for i in range(20):
            cv.offload(f"步骤{i}", "详细日志内容 " * 300, summary=f"第{i}步完成")
        stats = cv.stats()
        assert stats["offloaded_tokens"] > 20 * 500
        assert stats["packed_tokens_estimate"] < stats["offloaded_tokens"] * 0.05
        assert stats["compression_ratio"] < 0.05
        cv.close()

    def test_mermaid_and_links(self, tmp_path):
        cv = TaskCanvas(home=tmp_path / "h")
        a = cv.offload("抓取", "log-a")
        b = cv.offload("校验", "log-b", status="done")
        cv.link(a.node_id, b.node_id, "follows")
        mmd = cv.to_mermaid()
        assert "graph LR" in mmd
        assert a.node_id in mmd and b.node_id in mmd
        assert f"{a.node_id} --> {b.node_id}" in mmd
        assert "ok" in mmd
        nb = cv.neighbors(a.node_id)
        assert nb["out"][0]["peer"] == b.node_id
        cv.close()

    def test_contract_violations(self, tmp_path):
        cv = TaskCanvas(home=tmp_path / "h")
        a = cv.offload("x", "y")
        with pytest.raises(CanvasError, match="自环"):
            cv.link(a.node_id, a.node_id)
        with pytest.raises(CanvasError, match="悬空"):
            cv.link(a.node_id, "N-9999")
        with pytest.raises(CanvasError, match="原文缺失"):
            cv.recall("N-404")
        cv.close()

    def test_set_status_and_pack_overflow_note(self, tmp_path):
        cv = TaskCanvas(home=tmp_path / "h")
        nodes = [cv.offload(f"步骤{i}", "x" * 50) for i in range(6)]
        cv.set_status(nodes[0].node_id, "done")
        pack = cv.pack(max_nodes=3)
        assert "另有 3 个节点" in pack
        cv.close()


# ────────────────────────────────── 下钻不变量

class TestDrillDown:
    def test_chain_resolves_to_ref_file(self, tmp_path):
        home = tmp_path / "h"
        cv = TaskCanvas(home=home)
        dls = DLSMemory(home=home)
        node = cv.offload("原始日志", "这里是最原始的证据文本")
        rec = dls.add_lesson("蒸馏出的教训", mistake="m", correction="c",
                             sources=[node.node_id])
        report = dls.drill_down(rec.id)
        assert report["chain_ok"] is True
        assert report["chain"][1]["id"] == node.node_id
        assert report["chain"][1]["resolved"] is True
        cv.close()
        dls.close()

    def test_broken_chain_reported(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        rec = dls.add_lesson("悬空的", mistake="m", correction="c",
                             sources=["N-9999"])
        report = dls.drill_down(rec.id)
        assert report["chain_ok"] is False
        assert report["broken"] == ["N-9999"]
        dls.close()

    def test_recursive_chain(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        leaf = dls.add_lesson("叶子", mistake="m", correction="c",
                              sources=["EXT-1"])
        mid = dls.add_decision("中层", context="c", chosen="x", rationale="r",
                               sources=[leaf.id])
        top = dls.add_status("顶层", "s", sources=[mid.id], project="p")
        report = dls.drill_down(top.id)
        ids = {c["id"] for c in report["chain"]}
        assert {top.id, mid.id, leaf.id, "EXT-1"} <= ids
        assert report["chain_ok"] is False  # EXT-1 无 refs 文件
        dls.close()

    def test_distill_requires_source_and_is_atomic(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        events = [
            {"kind": "lesson", "title": "有来源", "mistake": "m",
             "correction": "c", "source": "N-0001"},
            {"kind": "lesson", "title": "没来源", "mistake": "m",
             "correction": "c"},
        ]
        with pytest.raises(DLSError, match="下钻不变量违规"):
            dls.distill_events(events, project="p")
        assert dls.list_records(project="p") == []  # 整批拒收，零半写
        ok = dls.distill_events(events, project="p", require_sources=False)
        assert len(ok) == 2
        dls.close()

    def test_record_without_sources_is_primary(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        rec = dls.add_lesson("一手记录", mistake="m", correction="c")
        report = dls.drill_down(rec.id)
        assert report["chain_ok"] is True and len(report["chain"]) == 1
        dls.close()


# ────────────────────────────────── 场景蒸馏层（金字塔 L2）

class TestScenarioLayer:
    def _provider(self):
        events = [
            {"kind": "lesson", "title": "缓存穿透", "mistake": "空值未缓存",
             "correction": "缓存空值", "source": "N-0001"},
            {"kind": "lesson", "title": "缓存击穿", "mistake": "热点过期",
             "correction": "互斥重建", "source": "N-0002"},
        ]
        scenarios = [
            {"title": "缓存失效三连", "situation": "高并发读场景",
             "pattern": "缓存失效引发连锁回源",
             "response": "空值缓存 + 互斥重建 + 熔断",
             "members": [0, 1]},
        ]

        def complete(prompt: str) -> str:
            if "场景块" in prompt:
                return json.dumps(scenarios, ensure_ascii=False)
            return json.dumps(events, ensure_ascii=False)

        return CallableProvider(complete)

    def test_two_pass_distill_creates_status_with_scenarios(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        created = dls.distill_day("今天处理了缓存穿透和击穿", project="cache",
                                  provider=self._provider(), scenarios=True)
        types = [r.type for r in created]
        assert types == ["lesson", "lesson", "status"]
        sta = created[-1]
        blocks = sta.payload["scenarios"]
        assert blocks[0]["title"] == "缓存失效三连"
        assert set(blocks[0]["members"]) == {created[0].id, created[1].id}
        for ev in created[:2]:
            assert any(e["relation"] == "part_of" for e in dls.neighbors(ev.id)["out"])
        report = dls.drill_down(sta.id)
        assert report["chain_ok"] is False  # 源头 N-0001/N-0002 无 refs 文件
        dls.close()

    def test_scenario_prompt_shape(self):
        prompt = build_scenario_prompt("[]")
        assert "场景块" in prompt and '"members"' in prompt


# ────────────────────────────────── 长时程基准

class TestLongRun:
    def test_synthesize_shapes(self):
        tasks = synthesize_long_run(n_tasks=5, noise_per_task=2, seed=1)
        assert len(tasks) == 5 and tasks[0].answer in tasks[0].evidence_text
        assert len(tasks[0].noise_texts) == 2

    def test_long_horizon_status_first_saves_tokens(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        tasks = synthesize_long_run(n_tasks=12, noise_per_task=2, seed=7)
        result = run_long_horizon(dls, tasks, k=3, window_tokens=100000)
        full = result["tokens_full_history"]
        sf = result["tokens_status_first"]
        assert full["total"] > sf["total"]
        assert result["savings_ratio"] < 1.0
        assert 0.0 <= result["hit_rate_status_first"] <= 1.0
        assert len(full["trajectory_sampled"]) >= 5
        dls.close()

    def test_window_overflow_flag(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        tasks = synthesize_long_run(n_tasks=8, noise_per_task=3, seed=3)
        result = run_long_horizon(dls, tasks, k=2, window_tokens=200)
        assert result["window_overflow_at_task"] is not None
        dls.close()
