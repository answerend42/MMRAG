# 开发指南

这份文档面向想修改 MMRAG 的开发者。当前仓库的原则是：保持原型小、真实、可观察，避免在第一个模块还没稳定前引入过多抽象。

## 开发环境

安装依赖：

```bash
uv sync --extra dev
npm install
```

启动应用：

```bash
npm run app:dev
```

启动架构图预览：

```bash
npm run arch:dev
```

## 代码结构

```text
mrag/
├── graph.py                 # LangGraph 图定义
├── state.py                 # MMRAGState
├── web.py                   # FastAPI、前端页面、SSE
├── models/                  # QueryPlan、RetrievalPlan、EvidenceCard 等契约
├── nodes/                   # LangGraph 节点
├── retrievers/              # Retriever 协议和 PicRetrieve 适配器
└── utils/                   # LLM 和 JSON 工具
```

PicRetrieve 子项目位于：

```text
PicRetrieve/
```

架构源码位于：

```text
architecture/likec4/
```

## 质量检查

```bash
npm run check
```

该命令会执行：

- `uv run ruff check mrag tests scripts`
- `uv run python -m pytest tests/test_picretrieve_graph.py PicRetrieve/tests/test_retrieval.py`

提交代码前至少运行一次。

## 架构图检查

```bash
npm run arch:validate
```

如果修改了 LikeC4 源码，也建议重新导出 PNG：

```bash
npm run arch:export:png
```

如果修改了 LangGraph 流程，重新导出原生图：

```bash
npm run graph:export
```

## 增加新召回模块

新增模块应沿着现有契约扩展：

1. 在 `mrag/models/evidence.py` 中确认或增加 `Modality`。
2. 实现 `mrag/retrievers/base.py` 中的 Retriever 协议。
3. 在运行时注册到 `dict[Modality, Retriever]`。
4. 让 `modality_router` 生成对应的 `RetrievalTask`。
5. 保持输出统一为 `EvidenceCard`。
6. 补充测试和 LikeC4 架构图。

不要把某个模块的特殊逻辑直接塞进 `build_picretrieve_graph` 主路径。模块差异应该尽量收敛在 Retriever 适配器和 Router 规则里。

## 修改 Planner

Planner 的目标是稳定地产生结构化计划，而不是生成长篇解释。修改时注意：

- 优先保持 JSON schema 严格。
- 不要依赖模型自由文本中的某个句子。
- 图文混合输入必须区分“图片作为参考”和“文本、图片并行检索”。
- 本地模型失败时要保留可解释 fallback。

相关文件：

```text
mrag/nodes/picretrieve_planner.py
mrag/utils/json_utils.py
mrag/utils/llm.py
```

## 修改前端

当前前端内嵌在 `mrag/web.py` 中，目的是让原型启动简单。修改时优先保持：

- 输入区模拟主流大模型的图文混合输入。
- 流水线每一步都能看到状态、输出摘要和耗时。
- 图片召回结果可直接预览。
- UI 不依赖外部构建步骤。

如果未来前端复杂度明显上升，再考虑拆分为独立前端工程。

## 提交建议

提交前检查：

```bash
npm run check
npm run arch:validate
```

如果改了文档链接，也建议运行简单链接检查。不要提交本地索引、模型权重、虚拟环境或 `node_modules/`。
