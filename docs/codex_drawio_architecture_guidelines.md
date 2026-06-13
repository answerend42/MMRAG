# Codex + draw.io 软件设计架构图指导原则

> 目的：让 Codex 在使用 `drawio-skill` 或直接生成 `.drawio` XML 时，先完成“架构建模”，再画图。不要只把组件堆成流程图；必须按视图、边界、分层、职责、数据流、部署关系来表达系统设计。

适用场景：软件系统架构图、AI/RAG 系统架构图、微服务架构图、云部署架构图、数据/检索管线图、运行时调用链、组件图、部署图。

---

## 1. 总原则：先建模，再画图

Codex 在画架构图前，必须先输出或内部构造一个 `Architecture Model`，再根据模型生成 draw.io。

### 1.1 必须先回答 6 个问题

1. **图的受众是谁？** 产品/导师/开发/运维/安全/评审。
2. **图的视角是什么？** 上下文、容器、组件、运行时、部署、数据、运维。
3. **系统边界在哪里？** 哪些是本系统，哪些是外部系统、第三方 API、云服务或用户。
4. **分层逻辑是什么？** UI/API、编排、领域服务、检索/工具、数据存储、可观测性、安全治理。
5. **主路径是什么？** 请求从哪里进来，经过哪些层，最终如何返回结果。
6. **图要证明什么设计决策？** 例如：自适应检索、多模态索引、证据校验、可追溯引用、SLA 控制。

### 1.2 严禁直接画“大而全单页图”

除非用户明确要求一页图，否则复杂系统必须拆成多页：

| 页 | 视图 | 目的 |
|---|---|---|
| Page 1 | C1 System Context | 说明系统与用户、外部系统、数据源的关系 |
| Page 2 | C2 Container / Logical Layered Architecture | 说明系统内部高层容器、职责分布、主要技术选择 |
| Page 3 | C3 Component View | 说明某个关键容器内部组件，例如 RAG Orchestrator |
| Page 4 | Runtime / Sequence Flow | 说明一次典型请求如何执行、哪里分支、哪里循环 |
| Page 5 | Data & Indexing View | 说明离线解析、索引构建、在线检索之间的关系 |
| Page 6 | Deployment / Infrastructure View | 说明服务、数据库、队列、模型网关如何部署 |
| Page 7 | Observability / Security Overlay | 说明鉴权、审计、Tracing、指标、反馈闭环 |

> 若只能画一页，优先画 Page 2：C2 + 分层 + 主数据流。

---

## 2. 参考标准：Codex 应按这些业界画法组织图

### 2.1 首选：C4 Model

C4 是软件架构图的首选抽象方法。使用下面的缩放层级：

| C4 层级 | 画什么 | 什么时候用 |
|---|---|---|
| C1 Context | 用户、外部系统、当前系统边界 | 开题、汇报、需求评审 |
| C2 Container | 应用、服务、数据存储、队列、模型网关 | 最常用的软件设计架构图 |
| C3 Component | 某个容器内部模块和职责 | 解释核心服务内部设计 |
| C4 Code | 类、接口、函数 | 只有代码设计评审才画 |

执行规则：

- C1 不画内部模块。
- C2 不画函数、类、表字段。
- C3 只 zoom in 一个容器，不要把全系统所有组件混在一起。
- Deployment 信息不要塞进 C2，单独画 Deployment View。
- 每张图只讲一个故事。

### 2.2 辅助：arc42 视图分类

当用户要“技术设计文档配图”时，用 arc42 的视图组织：

- Context & Scope：系统边界和外部接口。
- Building Block View：静态模块分解。
- Runtime View：关键场景的运行时行为。
- Deployment View：基础设施和软件到节点的映射。
- Crosscutting Concepts：安全、日志、监控、配置、权限、缓存、错误处理。
- Architecture Decisions：关键设计决策和取舍。

### 2.3 辅助：UML Component / Deployment

当图需要更正式时：

- 用 **Component Diagram** 表达模块、接口、依赖。
- 用 **Deployment Diagram** 表达物理节点、运行环境、制品、网络连接。
- 不要在 Deployment 图里画业务流程；Deployment 关注“组件部署在哪里、如何通信”。

### 2.4 云架构图参考

如果使用 AWS/Azure/GCP：

- 使用官方图标或 draw.io 内置云图标。
- 云服务图标旁必须写服务名，不要只放图标。
- 先画网络/账号/区域边界，再放服务。
- 不要把云服务图标当作自研服务图标。
- 数据流箭头必须标注协议或事件类型。

---

## 3. 输出之前必须生成 Architecture Model

Codex 应先从用户描述、README、技术文档、代码目录中提取如下模型。这个模型可以临时存在，也可以输出到 `architecture-model.json`。

```json
{
  "system_name": "...",
  "purpose": "...",
  "stakeholders": ["user", "developer", "operator", "security reviewer"],
  "quality_goals": ["latency", "traceability", "scalability", "security"],
  "actors": [
    {"id": "user", "name": "End User", "type": "person", "description": "..."}
  ],
  "external_systems": [
    {"id": "llm_api", "name": "LLM/VLM Provider", "type": "external_system", "trust": "external"}
  ],
  "containers": [
    {
      "id": "rag_orchestrator",
      "name": "RAG Orchestrator",
      "layer": "orchestration",
      "type": "service",
      "responsibility": "Plan, route, control retrieval loops, and call generator",
      "technology": "Python / LangGraph",
      "interfaces": ["REST", "gRPC", "internal events"]
    }
  ],
  "data_stores": [
    {"id": "vector_db", "name": "Vector DB", "type": "vector_store", "technology": "Qdrant/Milvus"}
  ],
  "connectors": [
    {
      "from": "api_gateway",
      "to": "rag_orchestrator",
      "label": "POST /answer",
      "protocol": "HTTPS/JSON",
      "sync_async": "sync",
      "data": "UserQuery"
    }
  ],
  "deployment_nodes": [
    {"id": "k8s", "name": "Kubernetes Cluster", "zone": "private", "runs": ["api_gateway", "rag_orchestrator"]}
  ],
  "cross_cutting": {
    "security": ["AuthN/AuthZ", "RBAC", "PII redaction"],
    "observability": ["OpenTelemetry", "logs", "metrics", "traces"],
    "governance": ["citation verification", "audit trail", "feedback loop"]
  }
}
```

---

## 4. 标准分层模板

### 4.1 通用软件系统分层

架构图从上到下或从左到右排列，不能随意散点布局。

```text
[External Actors / External Systems]
        ↓
[Client / Channel Layer]
        ↓
[API Edge Layer]
        ↓
[Application / Orchestration Layer]
        ↓
[Domain Services / Core Capabilities]
        ↓
[Integration / Tools / Workers]
        ↓
[Data / Index / Storage Layer]

右侧或底部横切：Security / Observability / Governance / CI-CD
```

### 4.2 AI / RAG 系统推荐分层

```text
[User / Admin / Evaluation User]
        ↓
[Client & API]
  Web UI, SDK, API Gateway, Session Manager
        ↓
[Planning & Orchestration]
  Query Planner, Need-Retrieval Decision, Modality Router,
  Retrieval Plan, Reflection Loop Controller, Budget Controller
        ↓
[Retrieval & Tool Execution]
  Text Retriever, Page/Image Retriever, Document Retriever,
  Audio/Video Retriever, SQL/KG Executor, OCR, ASR, VLM Caption,
  Web Search, Tool Registry
        ↓
[Evidence Fusion & Verification]
  Candidate Merge, Reranker, Deduplicator, Evidence Pack,
  Evidence Verifier, Citation Mapper, Conflict Detector
        ↓
[Generation & Response]
  Direct Generator, Grounded Answer Generator,
  Answer Verifier, Response Composer, Citation Renderer
        ↓
[Data & Index]
  Object Store, Metadata DB, Text Index, Sparse/BM25 Index,
  Vector DB, Multi-vector Page Index, SQL DB, KG, Cache

Side band:
[Security / Governance / Observability]
  Auth/RBAC, PII Guard, Audit Log, Telemetry, Evaluation, Feedback, SLA/Cost
```

### 4.3 多模态 RAG 的专用页面建议

#### Page 1：System Context

必须包含：

- End User / Domain Expert
- Admin / Data Curator
- 多模态 RAG System 边界
- Enterprise Data Sources：PDF、扫描件、图片、表格、音频、视频、数据库、知识图谱
- External Model Provider：LLM/VLM/Embedding/Reranker
- Web Search / External APIs
- Monitoring / Evaluation Platform

#### Page 2：Online Query Path Layered Architecture

主线必须是：

```text
User Query
→ API Gateway / Session Manager
→ Query Planner
→ Need Retrieval?
→ Direct Generator 或 Modality Router
→ Parallel Retrievers / Tools
→ Candidate Merge & Rerank
→ Evidence Verifier
→ Answer Generator
→ Grounded Answer + Citations
```

Reflection / Retry Loop 不要画成穿过全图的乱线，应放在 Orchestration 层右侧，用外圈回路连接 `Evidence Verifier → Query Planner`。

#### Page 3：Offline Ingestion & Indexing

主线必须是：

```text
Raw Data Sources
→ Connectors / Crawlers
→ Document Parser / OCR / ASR / VLM Caption / Table Parser / Chart Parser
→ Normalized Elements
→ Chunking / Region Extraction / Metadata Enrichment
→ Embedding / Sparse Index / Multi-vector Index
→ Vector DB / BM25 / Metadata DB / Object Store / SQL / KG
```

在线查询路径和离线建索引路径不要混在一张图里，除非用明确 swimlane 分开。

#### Page 4：Runtime Sequence

用编号箭头表达一次请求：

1. User sends query.
2. Planner classifies intent and decides whether retrieval is needed.
3. Router chooses modalities.
4. Retrievers run in parallel.
5. Reranker and verifier select evidence.
6. Generator writes answer with citations.
7. Verifier checks answer support.
8. If confidence is low, retry loop starts; otherwise return.

#### Page 5：Deployment

必须表达：

- Public zone / private zone / data zone。
- API 服务、Orchestrator 服务、Retriever 服务、Parser Workers、Model Gateway、Vector DB、Object Store、Metadata DB、Monitoring。
- 每个服务的运行环境：Kubernetes、VM、Serverless、Batch Job。
- 同步/异步边界：REST/gRPC、queue、event bus。
- 安全边界：VPC、private subnet、secret manager、RBAC。

---

## 5. draw.io 视觉规范

### 5.1 页面大小与布局

- 默认画布：`2200 × 1400`，横向。
- 简单图：≤ 12 个节点。
- 中等图：13–25 个节点。
- 复杂图：必须拆页，不要超过 35 个节点。
- 节点对齐到 10px 网格。
- 主路径从左到右或从上到下，不能混用。
- 每层之间至少留 `180–240px`，给箭头走线。
- 每行/列之间预留 `80px` routing corridor，不放组件。

### 5.2 容器规则

- 用 `swimlane` 或 `container=1` 表达层、子系统、信任边界。
- 子节点必须设置 `parent="containerId"`，不要只是视觉上放在大框上。
- 系统边界使用粗边框。
- 外部系统使用灰色。
- 安全/信任边界使用虚线或淡红色边框。

### 5.3 颜色语义

| 类型 | 颜色建议 | 说明 |
|---|---|---|
| Client / API | 蓝色 | 用户入口、API 网关、前端 |
| Orchestration / Planner | 紫色 | Agent、Planner、Router、Loop Controller |
| Retrieval / Tools | 青色 | 检索器、OCR、ASR、VLM、SQL/KG 工具 |
| Evidence / Verification | 橙色 | Rerank、Verifier、Citation、Conflict check |
| Generation | 绿色 | Generator、Answer Writer、Response Composer |
| Data Stores | 浅灰或绿色 cylinder | DB、索引、对象存储、缓存 |
| External Systems | 灰色 | 第三方 API、外部模型、外部数据源 |
| Security / Risk | 红色或淡红 | Auth、PII、Policy、风险点 |

### 5.4 形状语义

| 形状 | 用途 |
|---|---|
| Rounded Rectangle | 服务、模块、组件 |
| Cylinder | 数据库、索引、对象存储、缓存 |
| Hexagon / Cloud | 外部系统、外部服务 |
| Diamond | 判断点，例如 Need Retrieval? |
| Swimlane | 分层、子系统、部署区域 |
| Dashed Container | 信任边界、可选模块、未来规划 |
| Note / Callout | 设计决策、约束、SLA、风险 |

### 5.5 连线语义

| 线型 | 用途 |
|---|---|
| 实线箭头 | 同步调用，例如 REST/gRPC |
| 虚线箭头 | 异步事件、队列、回调 |
| 点线箭头 | 观测、日志、指标、trace |
| 粗箭头 | 主请求路径 |
| 细箭头 | 辅助依赖 |
| 双向箭头 | 仅在确实双向协议时使用 |

每条边必须有标签，至少包含一种：

- 协议：`HTTPS`, `gRPC`, `SQL`, `Vector Search`, `BM25`, `Kafka Event`
- 数据：`UserQuery`, `Subqueries`, `Candidates`, `EvidencePack`, `GroundedAnswer`
- 语义：`retrieve`, `rerank`, `verify`, `generate`, `audit`

错误示例：

```text
Service A → Service B
```

正确示例：

```text
API Gateway → RAG Orchestrator: HTTPS POST /answer, UserQuery
Modality Router → Text Retriever: retrieve(subqueries), topK=50
Evidence Verifier → Query Planner: low confidence, retry plan
```

---

## 6. Codex 生成 draw.io 的工作流

Codex 必须按以下流程执行：

### Step 1：解析输入

从用户描述、Markdown 设计文档、代码目录、README、配置文件中提取：

- 组件名
- 组件职责
- 技术栈
- 输入输出
- 数据存储
- 外部依赖
- 同步/异步通信方式
- 关键运行时场景
- 部署环境
- 安全和可观测性需求

### Step 2：选择视图

按需求选择视图，不要默认只画流程图。

```text
用户说“总体架构” → C1 + C2，默认输出 Page 1 和 Page 2
用户说“系统怎么跑起来” → Runtime View
用户说“部署架构” → Deployment View
用户说“模块设计” → C3 Component View
用户说“数据怎么进索引” → Data & Indexing View
用户说“给论文/开题汇报” → C1 + C2 + Runtime + Evaluation/Observability
```

### Step 3：分层排布

先分层，再放组件：

1. External Actors / Systems
2. Client/API
3. Orchestration
4. Domain / Retrieval / Tools
5. Evidence / Verification / Generation
6. Data / Index / Storage
7. Cross-cutting side band

### Step 4：放置节点

- 同层组件横向排列。
- 主链路组件居中。
- 高频交互组件靠近放置。
- 数据库放在最底层，靠近使用它的服务。
- 外部系统放在系统边界外侧。
- 循环控制放在右侧，不要横穿主链路。

### Step 5：连线

- 先画主链路，后画辅助依赖。
- 连线使用正交线。
- 每条线必须有标签。
- 多条线连接同一节点时，必须分散 entry/exit 点，避免重叠。
- 跨层长线尽量绕外侧 corridor。

### Step 6：添加图例和说明

每页必须包含：

- Title：图名 + 视图类型。
- Scope：本图表达什么，不表达什么。
- Legend：颜色、形状、线型含义。
- Notes：关键设计决策、假设、SLA、风险。

### Step 7：自检

生成后必须自检：

- 是否有明确系统边界？
- 是否每个组件都有职责？
- 是否每条边都有协议或数据语义？
- 是否混入了不同抽象层级？例如 C2 里出现函数名。
- 是否把部署细节混进逻辑架构图？
- 是否所有数据存储都被至少一个服务读写？
- 是否外部系统位于系统边界外？
- 是否主路径可以一眼从入口看到输出？
- 是否存在交叉线、重叠、断线、标签被遮挡？
- 是否有图例？

---

## 7. 多模态 RAG 架构图的推荐节点清单

当用户没有给出具体组件时，Codex 可使用以下默认节点，但必须按实际文档删减。

### 7.1 Online Query Path

- User / Domain Expert
- Web UI / SDK / API Client
- API Gateway
- Session Manager
- Query Planner
- Need Retrieval Decision
- Direct Generator
- Modality Router
- Retrieval Plan Builder
- Text Retriever
- Image/Page Retriever
- Document Retriever
- Audio/Video Retriever
- SQL/KG Executor
- Tool Registry
- OCR Tool
- ASR Tool
- VLM Caption Tool
- Table Parser
- Chart Parser
- Web Search Tool
- Candidate Merge
- Cross-modal Reranker
- Evidence Pack Builder
- Evidence Verifier
- Citation Mapper
- Reflection / Retry Loop Controller
- Grounded Answer Generator
- Answer Verifier
- Response Composer
- Telemetry Collector
- Feedback Collector

### 7.2 Data & Index Layer

- Raw Object Store
- Parsed Artifact Store
- Metadata DB
- Text Index / BM25
- Dense Vector DB
- Multi-vector Page Index
- Image Region Index
- Audio/Video Segment Index
- SQL Database
- Knowledge Graph
- Cache
- Evaluation Dataset Store
- Audit Log Store

### 7.3 Offline Ingestion Path

- Data Connector / Crawler
- Document Parser
- Layout Parser
- OCR Worker
- ASR Worker
- Keyframe Extractor
- Table Parser
- Chart Parser
- Chunker
- Region Extractor
- Embedding Worker
- Index Builder
- Metadata Enricher
- Quality Checker

### 7.4 Cross-cutting

- AuthN/AuthZ / RBAC
- Tenant / Permission Filter
- PII Redaction
- Secret Manager
- Observability: Logs, Metrics, Traces
- Cost / Latency Budget Controller
- Evaluation / Ragas / Phoenix
- Human Feedback / Data Governance

---

## 8. 多模态 RAG 一页总图模板

当用户只要一张图时，按以下结构画：

```text
Title: Multimodal RAG System - C2 Layered Architecture

┌──────────────────────────────────────────────────────────────┐
│ External Actors / Systems                                    │
│ User, Admin, Enterprise Data Sources, External LLM/VLM APIs   │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Client & API Layer                                           │
│ Web UI / SDK → API Gateway → Session Manager                 │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Planning & Orchestration Layer                               │
│ Query Planner → Need Retrieval? → Modality Router            │
│ Reflection / Retry Loop Controller                           │
└──────────────────────────────────────────────────────────────┘
               ↙              ↓                ↘
┌──────────────────────────────────────────────────────────────┐
│ Retrieval & Tool Layer                                       │
│ Text / Page / Image / Doc / Audio / Video / SQL / KG         │
│ OCR / ASR / VLM Caption / Table / Chart / Web Search         │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Evidence Fusion & Verification                               │
│ Candidate Merge → Reranker → Evidence Pack → Verifier        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Generation & Citation Layer                                  │
│ Direct/Grounded Generator → Answer Verifier → Citation Render │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Data & Index Layer                                           │
│ Object Store, Metadata DB, BM25, Vector DB, Page Index, KG   │
└──────────────────────────────────────────────────────────────┘

Right side vertical band:
Security / Governance / Observability / Evaluation / Feedback
```

---

## 9. Codex 可直接使用的提示词模板

把下面这段发给 Codex，让它生成 draw.io：

```text
请使用 drawio-skill 生成软件架构图。不要直接画图，先根据我的文档抽取 Architecture Model，再生成多页 .drawio。

必须遵循：
1. 使用 C4 + arc42 视图方法。
2. 至少生成这些页面：
   - Page 1: C1 System Context
   - Page 2: C2 Layered Architecture
   - Page 3: Runtime Flow / Sequence
   - Page 4: Data & Indexing View
   - Page 5: Deployment View
3. 每页必须有 Title、Scope、Legend、Notes。
4. 使用明确分层：External → Client/API → Orchestration → Retrieval/Tools → Evidence/Verification → Generation → Data/Index。
5. 每个组件必须写职责，不能只有名词。
6. 每条箭头必须写协议或数据语义。
7. 同步调用用实线，异步事件用虚线，观测/日志用点线。
8. 外部系统必须放在系统边界外。
9. 数据库/索引用 cylinder；服务用 rounded rectangle；层/边界用 swimlane/container。
10. 不要把逻辑架构、运行时流程、部署拓扑混在一张图里。
11. 生成 .drawio 源文件，并导出 PNG/SVG。如果 CLI 不可用，至少输出 .drawio XML。
12. 完成后自检：分层、边界、职责、箭头标签、图例、线条交叉、标签遮挡。

系统主题：多模态检索问答 / Multimodal RAG。
核心组件包括：Query Planner、Need Retrieval Decision、Modality Router、Text/Image/Page/Document/Audio/Video/SQL/KG Retriever、Tool Registry、OCR/ASR/VLM/Table/Chart Parser、Candidate Merge、Reranker、Evidence Verifier、Answer Generator、Citation Mapper、Memory Store、Telemetry。

请优先画出“在线查询路径”和“离线索引构建路径”的区别。在线查询路径不要与离线 ingestion 混线；必要时拆页。
```

---

## 10. 可放进 AGENTS.md 的简短版

如果不想整份文件都放进 Codex，可把下面内容放到项目根目录 `AGENTS.md`：

```md
## Architecture diagram rules for Codex

When generating draw.io architecture diagrams:

1. First build an Architecture Model: actors, external systems, containers, components, data stores, connectors, deployment nodes, cross-cutting concerns.
2. Use C4 as the default method: C1 Context, C2 Container, C3 Component, plus Runtime and Deployment views when needed.
3. Do not draw a single huge diagram. Split pages by view: Context, Layered Architecture, Component, Runtime Flow, Data/Indexing, Deployment, Observability/Security.
4. Use explicit layers: External → Client/API → Orchestration → Domain/Retrieval/Tools → Evidence/Verification → Generation → Data/Index; place Security/Observability/Governance as a side band.
5. Every node must have a responsibility and technology label when known.
6. Every edge must be labeled with protocol, data type, or operation; use solid for sync, dashed for async, dotted for telemetry.
7. Use draw.io parent-child containment for groups/swimlanes; do not merely place shapes on top of large boxes.
8. Keep logical architecture, runtime flow, and deployment topology separate unless the user explicitly requests a one-page overview.
9. Add Title, Scope, Legend, and Notes to every page.
10. Self-check before final output: boundary clarity, consistent abstraction level, no crossed lines through shapes, no clipped labels, no unlabeled arrows, no external system inside the system boundary.
```

---

## 11. drawio-skill 的补充约束

`drawio-skill` 负责生成 XML、布局、导出和自检，但架构图质量主要取决于“画什么”和“按什么视图画”。因此 Codex 在调用 skill 前必须补充以下约束：

- Architecture preset 只能决定样式，不能代替架构建模。
- 必须先输出视图选择和分层模型，再调用 skill。
- 使用 `swimlane` 表达层，不要使用普通大矩形冒充层。
- 使用 parent-child containment，保证 draw.io 中拖动层时子组件跟随。
- 对复杂图使用多页面，不要依赖自动布局硬塞到一页。
- 生成后用 PNG 视觉检查：重叠、断线、线穿节点、标签截断、箭头堆叠。

---

## 12. 常见错误与修正

| 错误 | 表现 | 修正 |
|---|---|---|
| 没有系统边界 | 外部 API 和内部服务混在一起 | 加 System Boundary，外部系统放外侧 |
| 没有分层 | 组件散点摆放 | 用 swimlane 分层 |
| 箭头无标签 | 不知道调用/数据含义 | 每条边写协议/数据/动作 |
| 抽象层级混乱 | C2 里出现函数、表字段 | 拆到 C3 或 Code View |
| 运行时和部署混画 | 图里既有请求步骤又有 K8s 节点 | 拆 Runtime 和 Deployment |
| 线条穿过节点 | 自动布局造成混乱 | 重新分层，增加 corridor 或 waypoint |
| 数据层太抽象 | 只有一个 DB | 区分 Object Store、Metadata DB、BM25、Vector DB、Page Index、KG |
| RAG 图缺少 verifier | 只有 retrieve + generate | 加 Evidence Verifier、Answer Verifier、Citation Mapper |
| 多模态图硬拼模态 | 所有检索器并列但无路由 | 加 Modality Router 和条件/预算控制 |
| 离线/在线混乱 | ingestion 和 query 混在同一主链路 | 拆页或用 swimlane 明确分开 |

---

## 13. 参考入口

- C4 Model: https://c4model.com/
- C4 diagrams: https://c4model.com/diagrams
- C4 container diagram: https://c4model.com/diagrams/container
- arc42: https://arc42.org/
- arc42 overview: https://arc42.org/overview
- UML 2.5.1 specification: https://www.omg.org/spec/UML/2.5.1/About-UML
- UML deployment diagram explanation: https://www.ibm.com/docs/en/rational-soft-arch/9.7.0?topic=diagrams-deployment
- draw.io layers: https://www.drawio.com/doc/layers
- draw.io examples/templates: https://www.drawio.com/example-diagrams
- AWS reference architecture diagrams: https://aws.amazon.com/architecture/reference-architecture-diagrams/
- Azure architecture icons and diagram guidance: https://learn.microsoft.com/en-us/azure/architecture/icons/
- Google Cloud Architecture Center: https://docs.cloud.google.com/architecture
- OpenAI Codex AGENTS.md guidance: https://developers.openai.com/codex/guides/agents-md
- OpenAI Codex Skills guidance: https://developers.openai.com/codex/skills
- drawio-skill repository: https://github.com/Agents365-ai/drawio-skill
