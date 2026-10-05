"""LAAP 任务路由器 — 问题精准定位 → 资源精准路由 → 执行契约。

设计目标（Lorry 2026-10-06 提出）：
拿到任务时「精准定位问题并精准解决」，做开发时「专业级解决前后端」，
并且**自主分析、调用专业的 MCP 与 skills 高质量完成**。

路由器是一张决策表 + 一套执行契约，不是 LLM：
1. **定位（locate）**：任务分类 → 实体抽取（文件/模块/关键词）→ 定位步骤
2. **路由（route）**：skills（语义路由可插拔）/ MCP 工具 / 子代理 / 文件线索
3. **契约（contract）**：执行计划 + 开发宪章检查 + 验收 oracle + 记忆回写

外部解析器全部可插拔（skill_router / graph / memory），核心逻辑纯函数化，
测试零 I/O。任何解析器缺失都在输出 degraded 字段里显式记录，不静默。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

__all__ = ["TaskRoute", "TaskRouter", "TaskType"]

# ── 任务类型词表（规则分类器的证据源，可扩展）─────────────────────────
_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "frontend": ("页面", "前端", "ui", "界面", "组件", "样式", "css", "落地页",
                 "网页", "动画", "交互", "排版", "公众号排版", "harness", "react", "vue"),
    "backend": ("后端", "接口", "api", "服务", "数据库", "sql", "并发", "架构",
                "重构", "模块", "引擎", "mcp", "sdk", "协议", "守护进程"),
    "debug": ("报错", "bug", "调试", "修复", "不工作", "失败", "异常", "崩溃",
              "排查", "为什么错", "traceback", "error"),
    "data": ("数据", "excel", "csv", "分析", "统计", "图表", "报表", "实验数据"),
    "research": ("调研", "论文", "研究", "综述", "对比", "可行性", "现状", "文献"),
    "content": ("写文章", "公众号", "推文", "长文", "文案", "稿件", "排版文章"),
    "deploy": ("部署", "发布", "上线", "迁移", "服务器", "域名", "dns", "cloudflare",
               "容器", "docker"),
    "media": ("生图", "图片", "封面", "视频", "配图", "插图", "头像", "海报"),
}

# ── 资源路由表（我们的专业工作方式，编码为可执行决策）─────────────────
_UI_SKILL_STACK = ["taste-skill", "emil-design-eng", "impeccable"]  # DEV_CHARTER 硬规则
_COMMON_VERIFY = {
    "charter": ["DEV_CHARTER 门（core/dev_charter_gate.py）"],
    "oracles": ["python asset_graph_engine/behavior_oracle/run_oracles.py"],
}

_ROUTE_TABLE: dict[str, dict] = {
    "frontend": {
        "skills": [*_UI_SKILL_STACK, "frontend-design", "gzh-design"],
        "mcp_tools": ["codegraph_explore（改前看影响面）", "vision-bridge_describe（VIS-01 视觉验收）",
                      "beautify_get-html-style-guide"],
        "agents": [],
        "locate": ["用 codegraph_explore 找到组件/页面入口与调用者",
                   "读现有设计 token（asset-vault/tokens），禁自造色板"],
        "plan": ["盘点现有组件与 token → 实现（Lucide 图标、焦点环、无 alert）",
                 "本地渲染 → 截图 → vision-bridge 视觉验收（层级/色彩/间距）"],
        "charter_extra": ["RED-01 禁 emoji 图标（用 Lucide SVG）",
                          "RED-04 必须有焦点环", "禁紫渐变白底 AI 腔", "禁三等宽卡片行"],
        "oracles": _COMMON_VERIFY["oracles"] + ["VIS-01 视觉验收（vision_describe 判定写入报告）"],
        "tests": ["UI 冒烟：页面可渲染、交互可触发、无 console error"],
        "write_back": "decision",
    },
    "backend": {
        "skills": ["code-review", "tdd", "plan-eng-review"],
        "mcp_tools": ["codegraph_explore/callers（先看架构与影响面）",
                      "laap-cognitive_lcrp_review（变更审查门）",
                      "laap-cognitive_code_audit"],
        "agents": [],
        "locate": ["codegraph_explore 定位模块边界与调用链",
                   "找到契约（接口/数据结构）再动手"],
        "plan": ["先写失败测试（TDD）→ 实现 → 重构",
                 "批量重构后导入级冒烟（py_compile 抓不住运行时炸弹）"],
        "charter_extra": ["DEBT-01 硬编码字面量与静默损坏为零",
                          "新 stub 必须 raise NotImplementedError 或写明 no-op 语义"],
        "oracles": _COMMON_VERIFY["oracles"],
        "tests": ["pytest 目标模块 + 受影响模块", "导入级冒烟"],
        "write_back": "decision",
    },
    "debug": {
        "skills": ["investigate", "systematic-debugging", "diagnose"],
        "mcp_tools": ["codegraph_explore（从症状到根因）", "laap-cognitive_truth_ground"],
        "agents": [],
        "locate": ["复现 → 读错误信息 → 检查假设（铁律：无根因不修）",
                   "二分定位：最近变更 / 环境 / 数据"],
        "plan": ["最小复现 → 根因假设 → 针对性修复 → 回归测试钉住"],
        "charter_extra": ["修复必须带回归测试", "失败两次换方向并记录教训"],
        "oracles": _COMMON_VERIFY["oracles"],
        "tests": ["回归测试（先失败后通过）"],
        "write_back": "lesson",
    },
    "data": {
        "skills": ["xlsx", "com-excel-editor", "data-report", "d3-visualization"],
        "mcp_tools": ["vision-bridge_analyze-chart", "office_read-document"],
        "agents": [],
        "locate": ["先探查数据形状（行数/字段/缺失）再下结论"],
        "plan": ["探查 → 清洗 → 分析 → 可视化 → 诚实标注不确定性"],
        "charter_extra": ["小样本偏差必须标注", "数据不含的结论不得外推"],
        "oracles": _COMMON_VERIFY["oracles"],
        "tests": ["数字抽查核对"],
        "write_back": "decision",
    },
    "research": {
        "skills": ["knowledge-base", "quiet-musing", "argumentative-writing"],
        "mcp_tools": ["laap-cognitive_research_deep / research_search",
                      "laap-cognitive_truth_ground（防幻觉三态）", "web_search/web_fetch"],
        "agents": ["hanako-2 / butter（独立视角交叉验证重要结论）"],
        "locate": ["先搜综述再下钻论文；区分一手/二手来源"],
        "plan": ["多源调研 → 特征矩阵 → 可借鉴/短板 → 带 URL 引用"],
        "charter_extra": ["事实边界分档标注（工程事实/未验证/只有框架）"],
        "oracles": _COMMON_VERIFY["oracles"],
        "tests": ["关键论断过 truth_ground"],
        "write_back": "decision",
    },
    "content": {
        "skills": ["laap-content-pipeline", "argumentative-writing", "gzh-design"],
        "mcp_tools": ["media_generate-image / free-image-gen", "office_html-to-pdf"],
        "agents": [],
        "locate": ["选题角度先行；只借角度不搬正文"],
        "plan": ["选题→调研→写作→排版→配图→四查（字数/图片内联/合规/渲染）"],
        "charter_extra": ["图片 base64 内联", "SVG 禁用 id 属性", "强调用黑底白字"],
        "oracles": _COMMON_VERIFY["oracles"],
        "tests": ["发布前四查脚本全绿"],
        "write_back": "decision",
    },
    "deploy": {
        "skills": ["cloudflare-deploy", "careful", "land-and-deploy"],
        "mcp_tools": ["laap-cognitive_world_predict/calibrate（发布前风险预测）"],
        "agents": [],
        "locate": ["确认现役目录与线上实际内容一致再动手", "备份配置与凭据引用（不入库）"],
        "plan": ["灰度 → 健康探针 → 回滚预案（先写好再发）"],
        "charter_extra": ["不可逆/外向操作先经 Lorry 确认"],
        "oracles": _COMMON_VERIFY["oracles"] + ["健康探针（health_probe 或等价）"],
        "tests": ["部署后冒烟：关键路径可用"],
        "write_back": "decision",
    },
    "media": {
        "skills": ["free-image-gen", "free-image-sources", "canvas-design"],
        "mcp_tools": ["media_generate-image", "media_describe-options"],
        "agents": [],
        "locate": ["确定用途（封面/插图）→ 尺寸与版权约束"],
        "plan": ["生成/检索 → 魔数验格式 → 压缩 → 版权记录"],
        "charter_extra": ["只收 PD/CC0/CC-BY", "下载后必须验魔数防伪 jpg"],
        "oracles": _COMMON_VERIFY["oracles"],
        "tests": ["图片格式魔数校验"],
        "write_back": "decision",
    },
}

_DEFAULT_ROUTE = {
    "skills": ["quiet-musing"],
    "mcp_tools": ["laap-cognitive_laap_before_turn（拉认知上下文）"],
    "agents": [],
    "locate": ["先复述问题确认理解", "第一性原理拆解到不证自明处"],
    "plan": ["拆解 → 实现 → 验证 → 沉淀"],
    "charter_extra": [],
    "oracles": _COMMON_VERIFY["oracles"],
    "tests": [],
    "write_back": "decision",
}

_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+|[\w./-]+\.(?:py|ts|tsx|js|jsx|md|json|html|css|sql)")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")


@dataclass
class TaskRoute:
    task: str
    types: list[str]
    primary_type: str
    entities: dict = field(default_factory=dict)
    route: dict = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "problem": {
                "task": self.task,
                "types": self.types,
                "primary_type": self.primary_type,
                "entities": self.entities,
            },
            "route": self.route,
            "plan": self.plan,
            "verification": self.verification,
            "memory": self.memory,
            "degraded": self.degraded,
        }


class TaskRouter:
    """任务路由器。resolvers 全部可插拔；缺失时 degraded 显式记录。"""

    def __init__(self,
                 skill_resolver: Callable[[str], list[str]] | None = None,
                 graph_resolver: Callable[[str], list[str]] | None = None,
                 memory_resolver: Callable[[str], str] | None = None):
        self._skill_resolver = skill_resolver
        self._graph_resolver = graph_resolver
        self._memory_resolver = memory_resolver

    # -------------------------------------------------------------- 定位

    def classify(self, task: str) -> list[tuple[str, int]]:
        """规则分类：按关键词证据打分，返回 [(type, score)] 降序。"""
        lowered = task.lower()
        scores: list[tuple[str, int]] = []
        for task_type, keywords in _TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lowered)
            if score:
                scores.append((task_type, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores

    def extract_entities(self, task: str) -> dict:
        return {
            "paths": sorted(set(_PATH_RE.findall(task)))[:10],
            "keywords": sorted(set(_TOKEN_RE.findall(task)))[:15],
        }

    # -------------------------------------------------------------- 路由

    def route(self, task: str, project: str = "") -> TaskRoute:
        scored = self.classify(task)
        types = [t for t, _ in scored] or ["general"]
        primary = types[0]
        table = _ROUTE_TABLE.get(primary, _DEFAULT_ROUTE)
        degraded: list[str] = []

        skills = list(table["skills"])
        if self._skill_resolver is not None:
            try:
                extras = [s for s in self._skill_resolver(task) if s not in skills]
                skills = extras + skills  # 语义命中的排前面
            except Exception as exc:  # 解析器失败不阻塞路由
                degraded.append(f"skill_resolver 失败: {exc}")
        else:
            degraded.append("skill_resolver 未接入（用规则表）")

        file_hints: list[str] = []
        if self._graph_resolver is not None:
            try:
                file_hints = self._graph_resolver(task)
            except Exception as exc:
                degraded.append(f"graph_resolver 失败: {exc}")
        else:
            degraded.append("graph_resolver 未接入（无文件影响面提示）")

        if not project and self._memory_resolver is not None:
            try:
                project = self._memory_resolver(task) or "_global"
            except Exception as exc:
                degraded.append(f"memory_resolver 失败: {exc}")

        route = {
            "skills": skills,
            "mcp_tools": table["mcp_tools"],
            "agents": table["agents"],
            "file_hints": file_hints,
        }
        plan = [f"[定位] {step}" for step in table["locate"]] + \
               [f"[执行] {step}" for step in table["plan"]] + \
               ["[沉淀] 把决策/教训写入 DLS（dls_add_decision / dls_add_lesson）"]
        verification = {
            "charter": _COMMON_VERIFY["charter"] + table["charter_extra"],
            "oracles": table["oracles"],
            "tests": table["tests"],
        }
        memory = {
            "project": project or "_global",
            "write_back": table["write_back"],
            "fresh_session": f"dls_bootstrap(project={project or '_global'}) 后按编号 jump",
        }
        return TaskRoute(
            task=task, types=types, primary_type=primary,
            entities=self.extract_entities(task), route=route, plan=plan,
            verification=verification, memory=memory, degraded=degraded,
        )
