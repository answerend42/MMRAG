# 系统架构

MMRAG 当前是一个 PicRetrieve 优先的多模态 Agent 原型。它不是一个“大而全”的 RAG 平台，而是先把一个真实模块跑通：图片召回。后续模块通过同一套规划、路由、证据和注册表契约逐步接入。

## 分层视角

当前架构可以按五层理解：

| 层级 | 当前组件 | 责任 |
| --- | --- | --- |
| 用户层 | 浏览器界面 | 输入文本、上传图片、观察流水线进度和召回结果。 |
| Agent 层 | LangGraph StateGraph | 编排 Planner、Router、PicRetrieve、Reranker、Verifier、Retry、Answer。 |
| 平台层 | FastAPI / SSE / 模块注册表 | 暴露 API、推送节点事件、管理 Retriever 扩展点。 |
| 数据层 | PicRetrieve 索引和图片语料 | 提供图片向量、元数据、图片文件和召回结果。 |
| 基础设施层 | LM Studio / 本地模型 / 本地文件系统 | 提供 OpenAI 兼容 LLM、CLIP 模型文件和运行环境。 |

## 运行链路

```text
浏览器界面
  -> FastAPI Agent API / SSE
  -> LangGraph StateGraph
  -> Retriever Registry
  -> ImageRetriever Adapter
  -> PicRetrieve CLIP Index
  -> EvidenceCard
  -> LLM Verifier / Answer
```

这条链路里，真正的 Agent 执行由 `mrag/graph.py` 中的 `build_picretrieve_graph` 定义。前端只负责输入、展示和事件追踪。

## C4 视图

LikeC4 是当前架构图的事实来源。

| C4 层级 | LikeC4 视图 | 导出 PNG |
| --- | --- | --- |
| C1 系统上下文 | `index` | `docs/likec4/png/index.png` |
| C2 容器图 | `c2Containers` | `docs/likec4/png/c2Containers.png` |
| C3 组件图 | `c3WebComponents` | `docs/likec4/png/c3WebComponents.png` |
| C3 组件图 | `c3AgentComponents` | `docs/likec4/png/c3AgentComponents.png` |
| C3 组件图 | `c3RegistryComponents` | `docs/likec4/png/c3RegistryComponents.png` |
| C4 动态图 | `c4QueryFlow` | `docs/likec4/png/c4QueryFlow.png` |

当前还没有维护代码级 C4 图。原因是原型仍在快速变化，C3 组件已经能准确映射到 `mrag/nodes/*`，而 LangGraph 原生导出图更适合展示节点级控制流。

## LangGraph 流水线

当前主路径：

```text
picretrieve_planner
  -> modality_router
  -> picretrieve
  -> reranker
  -> picretrieve_verifier
  -> picretrieve_answer
```

可选重试路径：

```text
picretrieve_verifier
  -> picretrieve_reflection
  -> picretrieve
```

`picretrieve_reflection` 只有在 Verifier 判断证据不足，并且仍有 retry budget 时才会执行。

## 输入策略

MMRAG 支持三种输入形态：

- 纯文本：Planner 生成视觉检索 query，PicRetrieve 执行 text-to-image。
- 纯图片：上传图片直接作为 query image，PicRetrieve 执行 image-to-image。
- 图文混合：LLM 先理解文字和图片，再决定是否只用图片、只用文本，或执行显式混合检索。

重要规则：

> “按这张图找”“类似这张”“similar to this image”这类指令会被视为图片参考指令，不会自动拆成并行文本任务。

只有 Planner 明确返回 `hybrid_text_image` 时，系统才会同时使用文本和图片检索。

## Planner 契约

Planner 调用本地 OpenAI 兼容模型，并要求严格 JSON schema 输出。典型结构如下：

```json
{
  "intent": "visual_localization",
  "retrieval_mode": "image_direct",
  "rewritten_queries": [],
  "confidence": 0.8
}
```

设计约束：

- 模型只负责返回结构化计划。
- UI rationale 由后端根据 `retrieval_mode` 和执行状态生成。
- JSON 解析失败时进入可解释 fallback，并把错误写入状态。

这比让本地模型自由输出完整解释更稳定，尤其适合 LM Studio 这类本地多模态模型服务。

## 召回契约

`modality_router` 把 `QueryPlan` 转成 `RetrievalPlan`：

```text
RetrievalPlan
  query
  tasks[]
    modality
    subquery
    top_k
    metadata_filter
    query_image?
```

`retrievers_node` 通过 `dict[Modality, Retriever]` 分发任务。

当前注册表：

```text
Modality.IMAGE -> ImageRetriever -> PicRetrieve
```

未来增加新模块时，优先扩展 Retriever 协议和注册表，不要把模块分支写死在 LangGraph 主流程中。

## 可观测性

前端通过 `/api/ask/stream` 接收 Server-Sent Events。每个阶段事件包含：

- 节点 ID。
- 执行状态。
- 人类可读摘要。
- 关键输出项和元数据。
- 当前节点第几次运行。
- 节点耗时。
- 端到端累计耗时。

这个设计足够轻量，不依赖外部 tracing 后端，同时能清楚展示 Agent 到底在做什么。

## 架构源码

- LikeC4 源码：`architecture/likec4/mmrag.c4`
- LikeC4 JSON 导出：`docs/likec4/model.json`
- LikeC4 PNG 导出：`docs/likec4/png/`
- LangGraph 源码：`mrag/graph.py`
- LangGraph Mermaid：`docs/langgraph/picretrieve_graph.mmd`
- LangGraph PNG：`docs/langgraph/picretrieve_graph.png`
