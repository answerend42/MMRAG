# 贡献指南

感谢你关注 MMRAG。这个项目还处在原型阶段，但我们希望它从一开始就保持清晰、可运行、可解释。

## 项目方向

当前优先级：

- 把 PicRetrieve 图片召回模块做成真实可演示的 Agent 模块。
- 让 LangGraph 每一步都能在前端被观察和解释。
- 用 LikeC4 维护简体中文 C4 架构图。
- 保留后续模块扩展所需的最小契约。

暂时不优先：

- 大规模平台化。
- 多租户、鉴权、部署编排。
- 复杂前端工程拆分。
- 与当前图片召回无关的大型重构。

## 本地开发

```bash
uv sync --extra dev
npm install
```

启动演示应用：

```bash
npm run app:dev
```

启动架构图预览：

```bash
npm run arch:dev
```

## 提交前检查

```bash
npm run check
npm run arch:validate
```

如果修改了 LangGraph 流程：

```bash
npm run graph:export
```

如果修改了 LikeC4 源码：

```bash
npm run arch:export:png
```

## 文档要求

- 面向仓库维护的文档请使用简体中文。
- 代码标识、API 字段、命令和文件路径保持原文。
- 修改架构边界时，同步更新 `docs/architecture.md` 和 `architecture/likec4/mmrag.c4`。
- 修改接口时，同步更新 `docs/api.md`。

## 代码风格

- 优先沿用现有结构，不为未来可能性提前制造复杂抽象。
- 新模块通过 Retriever 协议和注册表接入，不要硬编码到主流程。
- LLM 输出优先使用结构化 JSON，不依赖自由文本解析。
- 注释只写对理解有帮助的内容，避免重复代码本身。

## Pull Request 建议

PR 描述请说明：

- 解决了什么问题。
- 改了哪些主要文件。
- 如何验证。
- 是否影响架构图、API 或数据格式。

不要提交：

- `PicRetrieve/data/`
- 模型权重
- 本地图片语料
- `.venv/`
- `node_modules/`
- 缓存目录
