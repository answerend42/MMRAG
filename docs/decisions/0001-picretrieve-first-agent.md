# ADR 0001：PicRetrieve 优先的 Agent 原型

## 状态

已接受。

## 背景

MMRAG 的目标是构建一个多模块的多模态 RAG / Agent 系统。长期看，它可能包含图片、文本、文档、表格、SQL、知识图谱、音频和视频等模块。

但在早期阶段，如果同时铺开所有模块，系统很容易变成一组没有真实召回能力的 UI 和抽象。因此当前选择先把一个模块做实：图片召回。

PicRetrieve 已经提供 CLIP 图片检索能力，适合作为第一个模块接入 LangGraph。与此同时，系统需要保留清晰的扩展点，避免后续增加模块时推翻现有设计。

## 决策

当前原型采用 PicRetrieve 优先架构：

- 使用 `build_picretrieve_graph` 作为主执行图。
- 使用本地 OpenAI 兼容多模态 LLM 做 Planner、Verifier 和 Answer。
- 使用 `QueryPlan`、`RetrievalPlan`、`EvidenceCard` 作为跨模块契约。
- 使用 `dict[Modality, Retriever]` 作为召回模块注册表。
- 当前只注册 `Modality.IMAGE -> ImageRetriever -> PicRetrieve`。
- 前端必须展示 LangGraph 每一步的状态、输出摘要和耗时。
- 使用 LikeC4 维护 C4 架构图，使用脚本导出 LangGraph 原生图。

## 结果

好处：

- 第一个模块是真实可运行的，不是 mock。
- 前端能直观看到 Agent 在每一步做了什么。
- PicRetrieve 的细节被封装在 `ImageRetriever` 中。
- 后续模块可以复用现有路由和证据契约。
- 架构图和真实代码保持相对同步。

代价：

- 当前系统还不是完整多模块平台。
- Web 前端暂时内嵌在 FastAPI 文件中，复杂度上升后需要拆分。
- 本地数据准备仍依赖 PicRetrieve 的索引和模型文件。
- LLM 质量受本地模型和 LM Studio 配置影响。

## 后续方向

- 增加小型可公开 demo 数据集，降低首次运行门槛。
- 为 Retriever 注册表增加更多模块样例。
- 拆分前端工程，保留现有 SSE 事件契约。
- 增加更细的 LangGraph 节点测试和端到端测试。
- 在 LikeC4 中补充新增模块的 C3 视图和动态图。
