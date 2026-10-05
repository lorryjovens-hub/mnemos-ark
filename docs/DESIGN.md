# LAAP 结构化记忆引擎（DLS）— 设计与实现

> 日期：2026-10-06 · 状态：**核心已落地（16 测试全绿 + 行为 oracle 8/8 PASS）**
> 实现：`laap/cognition/dls_memory.py` + `laap/cognition/dls_memory_mcp_tools.py`
> 测试：`tests/memory/test_dls_memory.py`

---

## 0. 一句话

**Decisions（决策史，长）/ Lessons（错题本，中）/ Status（工程现状，短）三库互索引；
fresh session 从 status 出发，其余记忆按编号精准跳转、U 形曲线装箱——
分析跨域，注意力不跨域。**

---

## 1. 问题定义

1. **上下文污染**：同一项目组里聊别的事，闲聊内容混进项目上下文，检索被稀释。
2. **token 经济**：每次把长文本整体注入是 O(全文)；记忆一多就装不下、也看不清。
3. **注意力 U 形曲线**（primacy-recency / lost-in-the-middle）：长上下文里模型
   对首尾注意力最强、中段最弱——长文灌中段是最差的注入策略。

## 2. 调研结论（2025-2026 论文与开源，详细矩阵见附录）

**业界已有的最好做法**：
- **status-first 冷启动**：Memobase 的 Profile（结构化现状）+ Event 时间线（按需查）
  优于"塞一段摘要"；Letta Memory Block 常驻 core block、recall/archival 按需取。
- **失效而非删除**：Zep/Graphiti 事实边带 valid_at/invalid_at，矛盾使旧边失效但保留
  溯源（对应我们的 `superseded` 生命周期）。
- **睡眠期专职记忆加工**：Letta sleep-time compute 把记忆编辑放后台，主 agent 零开销。
- **固定情景 schema**：LangMem Episode 的 observation→thoughts→action→result。
- **token 预算化**：Mem0 2026 把 token/query 指标化（~6.9k/查询，较全上下文省 90%）。

**业界普遍的三个空白（本设计直接瞄准）**：
1. 没人做真正的**决策日志/错题本**——置信度、证伪条件、生效范围几乎无人建模；
2. **id 级按需寻址缺位**——多数系统仍是"检索→全量注入"，没有 memory_id→单条取回
   →顺引用跳转的分页协议；
3. **互索引只有两个极端**——纯向量（无因果/派生/反驳语义）或全量知识图谱（构建贵、
   过时快）；轻量类型化边的混合索引没人做好（LongMemEval-V2 的"状态追踪/坑点"考项）。

## 3. DLS 三库设计

### 3.1 记录模型（`records` 表，SQLite 单写入面 + Markdown 镜像）

公共字段：`id`（DEC-0001 / LES-0001 / STA-0001）、`type`、`project`（作用域命名
空间）、`domain`、`title`、`body`、`payload`（类型化 JSON）、`status`
（active/superseded/archived）、`confidence`、`tags`、`provenance`、时间戳。

| 类型 | 长度 | payload 核心字段 | 生命周期 |
|---|---|---|---|
| **DEC 决策史** | 长 | context / options_considered / chosen / rationale / expected_outcome / **revisit_trigger** | 长期；被新决策 `superseded`，不覆盖 |
| **LES 错题本** | 中 | trigger / mistake / correction / **rule_of_thumb** / severity | 长期；可 `fixes` 回指决策 |
| **STA 工程现状** | 短 | summary / current_state / next_steps / open_questions / last_verified | **滚动取代**：新 status 自动把旧的转 superseded |

Markdown 镜像（记忆即文件）：`md/{project}/{ID}.md`，frontmatter 全字段化，
人可读、可 git 版本控制、失去运行时仍可迁移。

### 3.2 互索引（`links` 表，类型化有向边）

关系集：`supports / refutes / supersedes / caused / derived_from / fixes / relates / part_of`。

- 典型形状：`STA --part_of--> DEC --caused--> LES`、`新DEC --supersedes--> 旧DEC`、
  `LES --fixes--> DEC`。
- 双向可查（`links_from` / `links_to` / `neighbors`），禁止自环与悬空引用。
- 与 Graphiti 的对齐：`superseded` 边 = 双时序失效；旧记录保留，溯源不断。

### 3.3 写入门控

统一写入面 `DLSMemory.add()`（对应"4 套并行记忆实现收敛到单入口"的缺口），
可注入 `verifier` 钩子（对接 `memory_verification_gate`），`state=error` 直接拒收。
**作用域标签 `project` 在写入时强制确定**——这是防污染的第一道闸。

## 4. 冷启动协议：从 status 出发

```
fresh session
   │
   ├─ bootstrap(project, budget)      ← 只读一份 status 全文
   │     ├─ status   全文（current_state / next_steps / open_questions）
   │     ├─ ring1    一环：status 直连的 DEC/LES（指针：id+标题+一句话）
   │     ├─ ring2    二环：仅 id+标题
   │     └─ usage    token 预算与裁剪统计
   │
   ├─ jump(DEC-0042)                  ← 按编号 O(1) 精准跳转，只取这一条
   ├─ neighbors(DEC-0042)             ← 需要展开时才看下一跳
   └─ pack_context([...ids], budget)  ← 多条合并时 U 形装箱
```

对照业界：这是 Memobase profile-first + Letta archival-工具 + LangMem store.get(id)
的合体，并补上他们都没有的**指针环协议**（ring1/ring2 分层展开）。

## 5. U 形曲线装箱（`pack_context`）

- **呈现顺序**：首条全文 → 中段指针 → 尾部全文（注意力 U 形：首尾吃预算）。
- **预算分配顺序**：head → tail → middle；超预算时中段先降级为纯编号，尾部再降为指针。
- 指针 = `[DEC-0042] (decision) 标题 — 一句话`，全文按需 `jump`。
- token 估算启发式 1 token ≈ 2.5 字符（CJK/EN 混合），可后续换真 tokenizer。

## 6. 与既有记忆系统集成（落点）

| 既有零件 | DLS 的关系 |
|---|---|
| `memory_verification_gate` | 作为 `verifier` 钩子接入 `add()` |
| `memory_router`（Schema→Entity） | 域路由决定 `project` 桶，检索先路由后精排（防跨域污染的检索侧） |
| `laap_before_turn` | 注入改为 status-first：`bootstrap()` 产物 + 需要时才 jump |
| `sleep_consolidate` | 睡眠期把当日情景**编译成 DEC/LES**（借鉴 Letta sleep-time agent） |
| `session_memory_*` 会话桥 | 会话桥负责原始流，DLS 负责蒸馏后的结构化沉淀，双层分工 |
| MCP 服务器 | `dls_memory_mcp_tools.py` 已按 `*_mcp_tools.py` 同构备好，待注册 |

## 7. 实现现状（2026-10-06）

- `laap/cognition/dls_memory.py` — 引擎：写入面/互索引/搜索/引导/装箱/Markdown 镜像
- `laap/cognition/dls_memory_mcp_tools.py` — MCP 工具面 9 个函数
- `tests/memory/test_dls_memory.py` — **16 用例全绿**（写入契约、互索引、supersede、
  bootstrap 环与裁剪、jump 单条、U 形装箱、CJK 检索、作用域过滤、MCP 端到端）
- 行为 oracle：**8/8 PASS**（含 DEBT-01 零硬编码门禁）
- 零重型依赖（stdlib），FTS5 缺失自动降级 LIKE 且记录事件（不静默）

**实测中修掉的两个真 bug**（有价值的工程注记）：
1. SQLite FTS5 unicode61 把中文连续串索引成**单 token**，中文子串查询必然 0 命中
   → CJK 查询直接走 LIKE；英文/代码走 FTS5 短语查询。
2. U 形装箱 v1 把尾部排到了中段之后（呈现顺序≠预算顺序）→ 拆成"预算分配序"与
   "呈现序"两阶段。

## 8. 路线图

- **P1 · 接入**：注册 9 个 dls_* 工具进 laap-cognitive MCP；`laap_before_turn` 注入
  status-first 块；`memory_verification_gate` 接入写入面。
- **P2 · 域隔离**：`memory_router` 域表与 `project` 桶对齐；检索侧先路由后精排，
  闲聊只进 `_global`/日常域，不响应项目查询（A/B 测污染率）。
- **P3 · 睡眠编译**：`sleep_consolidate` 增加"当日→DEC/LES 蒸馏"步骤，决策沉淀
  不再依赖我手动写。
- **P4 · 评测**：LongMemEval-V2 的"状态追踪/坑点"考项 + BEAM@10M 口径按 token 计分，
  对比：全量注入 vs status-first+jump 的 token 消耗与命中率。

## 9. 诚实边界

- v1 无向量检索（词法+标签+互索引）；语义召回可挂 `memory_ranker`，但要先过域路由。
- token 为启发式估算；预算裁剪的精确度依赖估算精度。
- 互索引展开深度到 ring2 为止；更深的图传播由 `causal_memory_graph` 负责，不重复造。
- MCP 工具面已就绪但尚未注册进服务器（需登记 + 重启引擎）。

---

## 附录：调研特征矩阵（精简）

| 对象 | 结构化记录 | 互索引 | 冷启动 | 按需跳转 | token 经济 |
|---|---|---|---|---|---|
| Mem0 | 自由文本事实+算子 | graph 版实体边 | 无状态层 | 三信号融合 | ~6.9k tok/query |
| Zep/Graphiti | 三元组+双时序 | 时序 KG，失效不删除 | 线程+图装配 | 语义+BM25+图遍历 | 亚秒装配 |
| Letta/MemGPT | 自编辑 Memory Block | 弱 | **core block 常驻** | recall/archival 工具 | sleep-time 后台压缩 |
| LangMem | Episode 四段式 | 弱 | procedural 常驻 | store.get(ns,id) | 规则重写 |
| Memobase | Profile+事件时间线 | 弱 | **profile 注入最佳** | 时间线按需 | 覆盖式合并 |
| A-MEM | Zettelkasten 笔记 | 动态链接+反向演化 | 无 | 顺链接跳转 | 无 |
| LongMemEval-V2 | 考"状态快照/坑点" | — | — | — | 按 token 计分 |

来源：arxiv 2504.19413（Mem0）、2501.13956（Zep）、2502.12110（A-MEM）、
2502.14802（HippoRAG 2）、2504.13171（sleep-time）、letta.com/blog/sleep-time-compute、
memobase GitHub、langchain-ai.github.io/langmem、mem0.ai/blog/state-of-ai-agent-memory-2026、
xiaowu0162.github.io/longmemeval-v2。
