# 快速开始

这份指南用于从一个干净的仓库开始，启动本地 PicRetrieve Agent 演示系统。

## 前置条件

- Python 3.11 或更高版本。
- `uv`。
- Node.js 和 npm。
- LM Studio，或其他 OpenAI 兼容的 Chat Completions 服务。
- 本地 PicRetrieve 索引、CLIP 模型和图片语料，位于 `PicRetrieve/data/`。

## 安装依赖

```bash
uv sync --extra dev
npm install
```

Python 依赖由 `uv` 管理，PicRetrieve 以 editable 方式从 `PicRetrieve/` 子目录接入。LikeC4 由 npm 管理，用于架构图预览和导出。

## 配置本地 LLM

演示系统默认连接 LM Studio：

```bash
export MRAG_LLM_BASE_URL=http://127.0.0.1:1234/v1
export MRAG_LLM_MODEL=google/gemma-4-26b-a4b
export MRAG_LLM_API_KEY=lm-studio
```

如果不设置这些变量，Web 应用会使用同样的默认值。模型需要支持 OpenAI 兼容的 `/v1/chat/completions`，最好支持多模态输入和 JSON schema 输出。

## 准备 PicRetrieve 数据

仓库不会提交完整图片索引、模型权重或图片语料。一个可运行的本地环境通常需要：

```text
PicRetrieve/data/index.sqlite
PicRetrieve/data/embeddings.npy
PicRetrieve/data/ids.npy
PicRetrieve/data/models/openai_clip-vit-base-patch32/
PicRetrieve/data/image_root.txt
```

`image_root.txt` 可选。它用于指定真实图片语料目录；如果不存在，系统会尝试使用 `PicRetrieve/data/corpus/images`。

更多细节见 [数据与索引](data.md)。

## 启动演示应用

```bash
npm run app:dev
```

打开：

[http://127.0.0.1:8010/](http://127.0.0.1:8010/)

页面会显示输入框、图片上传入口、模块状态、LangGraph 流水线和召回结果。查询时前端会通过 SSE 追踪每一步的状态和耗时。

## 健康检查

```bash
curl http://127.0.0.1:8010/api/status
```

健康状态通常包含：

- `ready: true`：PicRetrieve 图片召回器已加载。
- `llm_ready: true`：本地模型服务可用，并且配置的模型存在。
- `modules` 包含 `picretrieve`。
- `graph_nodes` 包含 `picretrieve_planner`、`modality_router`、`picretrieve`、`picretrieve_answer`。

如果 `ready` 或 `llm_ready` 为 `false`，优先检查 `/api/status` 返回的 `errors` 字段。

## 查看架构图

启动 LikeC4 预览：

```bash
npm run arch:dev
```

打开：

[http://127.0.0.1:5173/](http://127.0.0.1:5173/)

## 运行检查

```bash
npm run check
```

该命令会运行 Ruff、MMRAG 核心测试和 PicRetrieve 召回测试。
