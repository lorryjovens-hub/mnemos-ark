"""睡眠蒸馏的 LLM provider 适配器。

引擎分层不变：mnemos_ark 本身不绑定任何 LLM SDK，只认一个三行协议
`complete(prompt) -> str`。提供两个实现：

  CallableProvider       包任意函数（测试 / 自定义后端）
  OpenAICompatProvider   任意 OpenAI 兼容 /v1/chat/completions（stdlib urllib，
                         DeepSeek / Ollama / vLLM / MiMo 通用）

`parse_event_array` 负责把模型输出的 JSON 数组抠出来（容忍 ```json 围栏
与前后杂音），失败抛 DLSError——蒸馏是记忆写入路径，宁可报错不猜。
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, Protocol, Sequence

__all__ = [
    "LLMProvider",
    "CallableProvider",
    "OpenAICompatProvider",
    "parse_event_array",
]


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class CallableProvider:
    """包一个任意 complete 函数。"""

    def __init__(self, fn: Callable[[str], str]):
        self._fn = fn

    def complete(self, prompt: str) -> str:
        return self._fn(prompt)


class OpenAICompatProvider:
    """OpenAI 兼容 chat/completions 端点（零依赖，urllib 直连）。"""

    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 temperature: float = 0.2, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"LLM 响应结构异常: {exc}") from exc


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_event_array(text: str) -> list:
    """从模型输出中提取 JSON 事件数组。

    依次尝试：整体解析 → ```json 围栏 → 第一个 '[' 到最后一个 ']'。
    任何一步解析不出 list 都抛 RuntimeError（蒸馏写入路径不猜）。
    """
    if not text or not text.strip():
        raise RuntimeError("LLM 输出为空")
    candidates = [text.strip()]
    candidates.extend(m.strip() for m in _FENCE_RE.findall(text))
    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    raise RuntimeError("LLM 输出中找不到 JSON 事件数组")
