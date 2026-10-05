"""可插拔向量层（语义检索挂载点）。

设计：引擎本体零重型依赖，向量能力通过 `EmbeddingProvider` 协议挂载——
生产可挂 sentence-transformers / API 嵌入，测试与离线用确定性哈希嵌入。
不挂载时语义检索显式降级（记录降级事件），不静默。

三个实现：
  HashingEmbedder        零依赖确定性嵌入（字符 n-gram 哈希），离线/测试用
  OpenAICompatEmbedder   任意 OpenAI 兼容 /v1/embeddings 端点（stdlib urllib）
  CallableEmbedder       包一个任意函数（对接 sentence-transformers 等）
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from typing import Callable, Protocol, Sequence

__all__ = [
    "EmbeddingProvider",
    "HashingEmbedder",
    "OpenAICompatEmbedder",
    "CallableEmbedder",
    "cosine",
]


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量文本 → 向量列表（与输入等长）。"""
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """字符 n-gram 哈希嵌入（确定性、零依赖）。

    不是真语义嵌入——它保证短语/词形相近的文本向量相近，
    用于测试、离线兜底与"无模型环境"的可复现基线。
    """

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (2, 4)):
        self.dim = dim
        self.ngram_range = ngram_range

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        lowered = text.lower()
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(max(0, len(lowered) - n + 1)):
                gram = lowered[i:i + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class CallableEmbedder:
    """包一个任意 embed 函数（如 sentence-transformers 的 encode）。"""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]], dim: int = 384):
        self._fn = fn
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._fn(list(texts))


class OpenAICompatEmbedder:
    """任意 OpenAI 兼容 /v1/embeddings 端点（DeepSeek / Ollama / vLLM / 代理）。"""

    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 dim: int = 1024, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        vectors = [r.get("embedding", []) for r in rows]
        if len(vectors) != len(texts):
            raise RuntimeError(f"嵌入返回数量不符: {len(vectors)} != {len(texts)}")
        return vectors
