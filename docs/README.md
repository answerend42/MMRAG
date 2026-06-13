# 文档总览

这里存放 MMRAG 当前维护中的中文文档。文档目标是让新参与者可以快速理解：系统怎么跑、架构怎么分层、图片召回模块怎么接入 LangGraph，以及后续模块应该沿着哪些契约扩展。

## 推荐阅读顺序

1. [快速开始](getting-started.md)：安装依赖、配置 LM Studio、启动前后端演示。
2. [运行配置](configuration.md)：环境变量、端口、路径、默认模型和运行时行为。
3. [数据与索引](data.md)：PicRetrieve 本地数据结构，以及为什么不把索引和图片提交到仓库。
4. [系统架构](architecture.md)：C4 分层、LangGraph 流程、路由契约和模块扩展方式。
5. [API 文档](api.md)：FastAPI 端点、请求体、响应体和 SSE 事件。
6. [开发指南](development.md)：测试、格式化、架构图导出和贡献流程。
7. [LikeC4 架构图维护](likec4_architecture.md)：如何维护架构即代码。

## 决策记录

- [ADR 0001：PicRetrieve 优先的 Agent 原型](decisions/0001-picretrieve-first-agent.md)

## 架构产物

这些产物会被提交到仓库，方便直接在 GitHub 上预览：

- `docs/likec4/model.json`
- `docs/likec4/png/*.png`
- `docs/langgraph/picretrieve_graph.mmd`
- `docs/langgraph/picretrieve_graph.png`

## 历史资料

`docs/` 中保留了部分早期研究笔记、draw.io 参考和实验仪表盘。这些文件用于保留上下文，不代表当前维护文档的主线。当前主线以本页列出的文档为准。
