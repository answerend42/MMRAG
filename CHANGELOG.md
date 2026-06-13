# 更新日志

本项目遵循轻量级更新日志格式。当前版本仍处在原型阶段，条目以实际能力演进为主。

## 未发布

- 统一维护文档为简体中文。
- 强化 GitHub 首页展示：徽章入口、架构预览图、快速开始和文档导航。
- 补充 C4 / LikeC4、LangGraph、API、数据和开发文档。
- 明确 PicRetrieve 优先的 Agent 架构决策。

## 0.1.0

- 搭建基于 LangGraph 的 PicRetrieve 图片召回 Agent 原型。
- 接入本地 OpenAI 兼容 LLM，用于 Planner、Verifier 和 Answer。
- 支持文本查图、以图搜图和图文混合输入。
- 提供 FastAPI + 浏览器前端演示。
- 前端通过 SSE 展示流水线进度、节点输出和耗时。
- 使用 LikeC4 维护 C4 架构图。
- 支持导出 LangGraph 原生 Mermaid / PNG 图。
