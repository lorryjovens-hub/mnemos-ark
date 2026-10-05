"""Roadmap 三件的测试：可插拔向量层 / LLM 蒸馏适配器 / LongMemEval 基准。"""

from __future__ import annotations

import json

import pytest

from mnemos_ark.embeddings import HashingEmbedder, CallableEmbedder, cosine
from mnemos_ark.llm import CallableProvider, parse_event_array
from mnemos_ark.memory import DLSMemory, DLSError
from mnemos_ark.benchmark import evaluate, load_longmemeval
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "longmemeval_sample.jsonl"


@pytest.fixture()
def dls(tmp_path):
    engine = DLSMemory(home=tmp_path / "dls", embedder=HashingEmbedder(dim=64))
    yield engine
    engine.close()


# ────────────────────────────────── 语义检索挂载点

class TestEmbeddings:
    def test_hashing_embedder_deterministic(self):
        emb = HashingEmbedder(dim=32)
        a = emb.embed(["hello world"])[0]
        b = emb.embed(["hello world"])[0]
        assert a == b and len(a) == 32
        assert abs(sum(x * x for x in a) - 1.0) < 1e-6

    def test_cosine_similarity_ordering(self):
        emb = HashingEmbedder(dim=64)
        q, near, far = emb.embed([
            "redis connection pool sizing",
            "redis connection pool tuning",
            "strawberry jam recipe"])[0:3]
        assert cosine(q, near) > cosine(q, far)

    def test_semantic_search_finds_record(self, dls):
        dls.add_lesson("连接池选型", mistake="max_connections 太小",
                       correction="按并发调大", project="p")
        dls.add_lesson("草莓果酱", mistake="糖放少了", correction="加糖", project="p")
        hits = dls.semantic_search("connection pool sizing", project="p")
        assert hits and "连接池" in hits[0].title

    def test_semantic_search_requires_embedder(self, tmp_path):
        bare = DLSMemory(home=tmp_path / "bare")
        with pytest.raises(DLSError, match="未挂载 embedder"):
            bare.semantic_search("anything")
        bare.close()

    def test_hybrid_degrades_loudly_without_embedder(self, tmp_path):
        bare = DLSMemory(home=tmp_path / "bare")
        bare.add_lesson("测试", mistake="m", correction="c")
        hits = bare.hybrid_search("测试")
        assert hits and any("hybrid_search 无 embedder" in e
                            for e in bare._degrade_events)
        bare.close()

    def test_hybrid_fusion(self, dls):
        dls.add_lesson("SQLite WAL 模式", mistake="锁冲突", correction="开 WAL")
        dls.add_lesson("无关条目", mistake="x", correction="y")
        hits = dls.hybrid_search("SQLite WAL")
        assert hits and "SQLite" in hits[0].title

    def test_rebuild_embeddings(self, dls):
        dls.add_lesson("先建的", mistake="m", correction="c")
        n = dls.rebuild_embeddings()
        assert n == 1


# ────────────────────────────────── LLM 蒸馏适配器

class TestLLMAdapter:
    def test_parse_event_array_plain_and_fenced(self):
        events = [{"kind": "lesson", "title": "t"}]
        raw = json.dumps(events, ensure_ascii=False)
        assert parse_event_array(raw) == events
        assert parse_event_array(f"```json\n{raw}\n```") == events
        assert parse_event_array(f"前置杂音 {raw} 后置杂音") == events

    def test_parse_event_array_rejects_garbage(self):
        with pytest.raises(RuntimeError, match="找不到 JSON"):
            parse_event_array("我觉得今天天气不错")
        with pytest.raises(RuntimeError, match="为空"):
            parse_event_array("   ")

    def test_distill_day_end_to_end(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "dls")
        payload = json.dumps([
            {"kind": "decision", "title": "走单写入面", "context": "四套打架",
             "chosen": "vault_bridge", "rationale": "统一作用域标签"},
            {"kind": "lesson", "title": "FTS5 中文陷阱", "mistake": "整句成单 token",
             "correction": "CJK 走 LIKE", "rule_of_thumb": "中文不进 FTS"},
        ], ensure_ascii=False)
        provider = CallableProvider(lambda prompt: f"```json\n{payload}\n```")
        created = dls.distill_day("今天收敛了写入面，还修了 FTS5", project="p",
                                  provider=provider)
        assert [r.type for r in created] == ["decision", "lesson"]
        assert created[0].provenance == "sleep-distill"

    def test_distill_day_requires_provider_and_clean_json(self, tmp_path):
        dls = DLSMemory(home=tmp_path / "dls")
        with pytest.raises(DLSError, match="需要 LLM provider"):
            dls.distill_day("日志")
        bad = CallableProvider(lambda p: "不是 JSON")
        with pytest.raises(DLSError, match="解析失败"):
            dls.distill_day("日志", provider=bad)
        dls.close()


# ────────────────────────────────── LongMemEval-V2 基准

class TestBenchmark:
    def test_loader_reads_fixture(self):
        questions = load_longmemeval(FIXTURE)
        assert len(questions) == 3
        assert questions[0].question_type == "semantic_coreference"
        assert questions[0].haystack and questions[0].session_text(0)

    def test_evaluate_hits_and_tokens(self, dls):
        questions = load_longmemeval(FIXTURE)
        result = evaluate(dls, questions, k=2, strategy="lexical")
        assert result["questions"] == 3
        assert 0.0 <= result["hit_at_k"] <= 1.0
        assert result["tokens_full_injection"] > result["tokens_strategy"] > 0
        assert "semantic_coreference" in result["by_type"]

    def test_hybrid_strategy(self, dls):
        questions = load_longmemeval(FIXTURE)
        result = evaluate(dls, questions, k=2, strategy="hybrid")
        assert result["strategy"] == "hybrid"
        assert result["degraded"] is False
