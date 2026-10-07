"""v0.3 测试：sqlite-vec 向量后端（KNN+兜底）与跨 Agent 治理共享。"""

from __future__ import annotations

import json

import pytest

from mnemos_ark.embeddings import HashingEmbedder
from mnemos_ark.memory import DLSMemory, DLSError
from mnemos_ark.sharing import (
    SharePolicy, export_pack, import_pack, verify_pack)
from mnemos_ark import memory as memory_mod


# ────────────────────────────────── sqlite-vec 向量后端

class TestVecBackend:
    def test_sqlite_vec_backend_active(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        dls = DLSMemory(home=tmp_path / "dls", embedder=HashingEmbedder(dim=32))
        assert dls.vector_backend == "sqlite-vec"
        dls.add_lesson("连接池选型", mistake="max_connections 太小",
                       correction="按并发调大", project="p")
        dls.add_lesson("草莓果酱", mistake="糖放少了", correction="加糖", project="p")
        hits = dls.semantic_search("connection pool sizing", project="p")
        assert hits and "连接池" in hits[0].title
        dls.close()

    def test_blob_fallback_when_vec_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory_mod, "_sqlite_vec", None)
        dls = DLSMemory(home=tmp_path / "dls", embedder=HashingEmbedder(dim=32))
        assert dls.vector_backend == "blob"
        dls.add_lesson("连接池选型", mistake="m", correction="c", project="p")
        hits = dls.semantic_search("connection pool", project="p")
        assert hits and "连接池" in hits[0].title
        dls.close()

    def test_rebuild_and_dim_change(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        dls = DLSMemory(home=tmp_path / "dls", embedder=HashingEmbedder(dim=32))
        dls.add_lesson("A", mistake="m", correction="c")
        n = dls.rebuild_embeddings()
        assert n == 1
        dls.close()


# ────────────────────────────────── 跨 Agent 治理共享

def _seed(dls: DLSMemory) -> tuple[str, str]:
    good = dls.add_decision(
        "选了单写入面", context="四套打架", chosen="vault_bridge",
        rationale="统一作用域标签", project="team",
        confidence=0.9, provenance="internal-note", sources=["N-0001"])
    weak = dls.add_lesson("低置信教训", mistake="m", correction="c",
                          project="team", confidence=0.2)
    return good.id, weak.id


class TestSharing:
    def test_export_governance_gate(self, tmp_path):
        home = tmp_path / "h"
        dls = DLSMemory(home=home)
        good_id, weak_id = _seed(dls)
        (home / "refs").mkdir(parents=True, exist_ok=True)
        (home / "refs" / "N-0001.md").write_text("原文证据", encoding="utf-8")
        pack = export_pack(dls, "team")
        ids = {r["id"] for r in pack["records"]}
        assert good_id in ids and weak_id not in ids
        rejected = {r["id"]: r["reason"] for r in pack["rejected"]}
        assert "置信度" in rejected[weak_id]
        good = next(r for r in pack["records"] if r["id"] == good_id)
        assert good["provenance"] == ""  # redact 生效
        assert pack["hash"]
        dls.close()

    def test_export_rejects_broken_chain(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        dls.add_lesson("坏链", mistake="m", correction="c",
                       project="team", sources=["N-9999"])
        pack = export_pack(dls, "team")
        assert pack["records"] == []
        assert "下钻不变量" in pack["rejected"][0]["reason"]
        ok = export_pack(dls, "team", SharePolicy(require_chain_ok=False))
        assert len(ok["records"]) == 1
        dls.close()

    def test_verify_pack_detects_tamper(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        _seed(dls)
        pack = export_pack(dls, "team", SharePolicy(require_chain_ok=False))
        assert verify_pack(pack)["ok"] is True
        pack["records"][0]["title"] = "被篡改的标题"
        verdict = verify_pack(pack)
        assert verdict["ok"] is False and "篡改" in verdict["reason"]
        dls.close()

    def test_import_roundtrip_remaps_ids_and_links(self, tmp_path):
        home_a, home_b = tmp_path / "a", tmp_path / "b"
        dls_a = DLSMemory(home=home_a)
        good_id, _ = _seed(dls_a)
        les = dls_a.add_lesson("相关教训", mistake="m", correction="c",
                               project="team", sources=["N-0001"])
        dls_a.link(good_id, les.id, "caused")
        pack = export_pack(dls_a, "team", SharePolicy(require_chain_ok=False))

        dls_b = DLSMemory(home=home_b)
        result = import_pack(dls_b, pack, target_project="team-b")
        assert result["imported"] == 2  # good + les（weak 低置信度被裁决拦下）
        assert good_id in result["id_map"] and result["id_map"][good_id]
        new_ids = set(result["id_map"].values())
        assert all(i in {r.id for r in dls_b.list_records(project="team-b")}
                   for i in new_ids)
        linked = {e.dst for e in dls_b.links_from(result["id_map"][good_id])}
        assert linked & new_ids  # 边已重映射
        # 去重：重复导入不双写
        again = import_pack(dls_b, pack, target_project="team-b")
        assert again["imported"] == 0 and again["skipped"] == 2
        dls_a.close()
        dls_b.close()

    def test_import_rejects_tampered_pack(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        _seed(dls)
        pack = export_pack(dls, "team", SharePolicy(require_chain_ok=False))
        pack["records"][0]["confidence"] = 9.9
        with pytest.raises(DLSError, match="校验失败"):
            import_pack(dls, pack, target_project="x")
        dls.close()

    def test_share_log_written(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "h")
        _seed(dls)
        export_pack(dls, "team", SharePolicy(require_chain_ok=False))
        rows = dls._db.execute("SELECT * FROM share_log").fetchall()
        assert rows and rows[0]["direction"] == "export"
        dls.close()

    def test_mcp_pack_tools(self, tmp_path, monkeypatch):
        from mnemos_ark import memory_mcp as mcp
        home = str(tmp_path / "h")
        engine = DLSMemory(home=home)
        monkeypatch.setattr(mcp, "_engine", lambda h=None: engine)
        _seed(engine)
        out = json.loads(mcp.dls_export_pack("team", require_chain_ok=False,
                                             home=home))
        assert out["format"] == "mnemos-ark-pack/1"
        res = json.loads(mcp.dls_import_pack(json.dumps(out),
                                             target_project="t2", home=home))
        assert res["imported"] == 1  # 仅 good 过治理门（weak 0.2 置信度被拦）
        engine.close()
