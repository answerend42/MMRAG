# API 文档

MMRAG 的 Web 层由 FastAPI 提供，前端页面和 Agent API 在同一个进程中运行。当前 API 面向本地演示和调试，不做公网鉴权。

## `GET /`

返回浏览器演示页面。

页面能力：

- 文本输入和图片上传在同一个输入区域中完成。
- 展示 LM Studio 与 PicRetrieve 的运行状态。
- 通过 SSE 追踪 LangGraph 每个节点的进度、输出摘要和耗时。
- 展示召回图片、得分、路径和最终回答。

## `GET /api/status`

返回运行时状态。

示例响应：

```json
{
  "ready": true,
  "llm_ready": true,
  "llm_model": "google/gemma-4-26b-a4b",
  "llm_base_url": "http://127.0.0.1:1234/v1",
  "images": 1280,
  "model": "openai_clip-vit-base-patch32",
  "data_dir": "/path/to/MMRAG/PicRetrieve/data",
  "errors": []
}
```

关键字段：

| 字段 | 说明 |
| --- | --- |
| `ready` | PicRetrieve 图片召回器是否已加载。 |
| `llm_ready` | 本地 LLM 是否可访问，并且模型名匹配。 |
| `modules` | 当前可用模块，现阶段主要是 `picretrieve`。 |
| `graph_nodes` | 前端展示流水线所需的 LangGraph 节点列表。 |
| `errors` | 初始化时收集到的错误信息。 |

## `POST /api/ask`

一次性执行 Agent，并在流程完成后返回完整结果。

请求体：

```json
{
  "query": "找和这张图风格相似的图片",
  "input_images": [
    {
      "name": "reference.jpg",
      "mime_type": "image/jpeg",
      "size": 123456,
      "data_url": "data:image/jpeg;base64,..."
    }
  ],
  "top_k": 8,
  "max_retries": 1,
  "modules": ["picretrieve"]
}
```

字段约束：

| 字段 | 约束 |
| --- | --- |
| `query` | 可为空，最大 1000 字符。 |
| `input_images` | 最多 3 张；当 `query` 为空时至少需要 1 张图片。 |
| `top_k` | 1 到 50。 |
| `max_retries` | 0 到 3。 |
| `modules` | 当前必须包含 `picretrieve`。 |

响应体包含：

- `answer`：最终回答。
- `agent_plan`：Planner 输出的结构化计划。
- `agent_verification`：Verifier 输出的结构化判断。
- `trace`：执行过的节点。
- `stage_timings`：每个节点耗时。
- `evidence_cards`：召回结果卡片。
- `latency_ms`：端到端耗时。

## `POST /api/ask/stream`

以 Server-Sent Events 方式执行 Agent。前端默认使用这个接口，因为它可以逐步展示流水线进度。

事件类型：

| 事件 | 说明 |
| --- | --- |
| `pipeline` | 流水线初始化，包含节点列表、输入摘要和模型信息。 |
| `stage_start` | 某个节点开始执行。 |
| `stage_done` | 某个节点执行完成，包含摘要、元数据和节点耗时。 |
| `stage_route` | 后端推断的下一节点，用于前端高亮。 |
| `result` | 最终结果，与 `/api/ask` 的结果结构一致。 |
| `error` | 执行失败时返回错误信息。 |

每个 `stage_done` 通常包含：

```json
{
  "node": "picretrieve",
  "title": "PicRetrieve 召回完成",
  "detail": "执行图片召回，返回 8 张候选图片。",
  "items": ["image-1 · 0.912 · corpus/a.jpg"],
  "meta": {"top_score": 0.912},
  "duration_ms": 324,
  "elapsed_ms": 1842
}
```

## `GET /api/images/{item_id}`

返回召回结果对应的图片文件。

安全策略：

- `item_id` 必须来自 PicRetrieve 索引。
- 解析出的图片路径必须位于允许的图片根目录下。
- 找不到文件时返回 `404`。

## 错误码

| 状态码 | 场景 |
| --- | --- |
| `400` | 输入为空、图片格式不合法、未启用 `picretrieve`。 |
| `404` | 图片不存在或不在允许目录中。 |
| `500` | Agent 执行异常。 |
| `503` | LangGraph 运行时未初始化。 |
