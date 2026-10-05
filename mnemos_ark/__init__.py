"""Mnemos Ark — status-first structured memory & precision routing for AI agents.

三库记忆（Decisions / Lessons / Status）互索引 + 按编号精准跳转 +
U 形曲线装箱 + 任务路由器。零重型依赖，stdlib only。
"""

from .memory import DLSMemory, DLSRecord, DLSLink, DLSError, estimate_tokens
from .router import TaskRouter, TaskRoute

__version__ = "0.1.0"

__all__ = [
    "DLSMemory",
    "DLSRecord",
    "DLSLink",
    "DLSError",
    "TaskRouter",
    "TaskRoute",
    "estimate_tokens",
    "__version__",
]
