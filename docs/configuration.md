# 运行配置

MMRAG 默认是本地优先的原型系统。配置项尽量少，核心是本地 LLM、PicRetrieve 数据目录和两个开发端口。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MRAG_LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI 兼容服务地址，LM Studio 默认可用。 |
| `MRAG_LLM_MODEL` | `google/gemma-4-26b-a4b` | Planner、Verifier 和 Answer 节点使用的模型名。 |
| `MRAG_LLM_API_KEY` | `lm-studio` | OpenAI SDK 需要的 API key，本地服务通常只需要占位值。 |

示例：

```bash
export MRAG_LLM_BASE_URL=http://127.0.0.1:1234/v1
export MRAG_LLM_MODEL=google/gemma-4-26b-a4b
export MRAG_LLM_API_KEY=lm-studio
```

## 默认端口

| 服务 | 命令 | 地址 |
| --- | --- | --- |
| Agent 演示应用 | `npm run app:dev` | `http://127.0.0.1:8010/` |
| LikeC4 预览 | `npm run arch:dev` | `http://127.0.0.1:5173/` |
| LM Studio | 用户手动启动 | `http://127.0.0.1:1234/v1` |

## PicRetrieve 路径

Web 运行时会从项目根目录解析以下路径：

```text
PicRetrieve/data/
PicRetrieve/data/models/openai_clip-vit-base-patch32/
PicRetrieve/data/image_root.txt
```

如果 `PicRetrieve/data/image_root.txt` 存在，文件内容会被当作图片语料根目录。否则系统会尝试使用：

```text
PicRetrieve/data/corpus/images
```

路径解析逻辑在 `mrag/web.py` 中，主要用于让命令从项目根目录或相邻目录执行时都能找到本地数据。

## LLM 可用性检查

启动时，系统会访问：

```text
{MRAG_LLM_BASE_URL}/models
```

如果响应中包含 `MRAG_LLM_MODEL`，则 `llm_ready` 为 `true`，LangGraph 节点会调用真实 LLM。否则系统仍可启动，但 Planner、Verifier 或 Answer 可能退回到规则逻辑或错误提示。

## JSON 输出策略

Planner 使用严格 JSON schema 调用本地模型。这样做是为了降低本地模型输出半截 JSON、附带解释文本或字段漂移的概率。

当前约束是：

- Planner 只返回结构化字段，不返回自由发挥的长段 rationale。
- UI 展示用的说明由后端根据 `retrieval_mode` 和执行状态生成。
- 如果模型仍然输出无效 JSON，系统会记录错误并进入可解释的 fallback。

## 前端上传限制

`/api/ask` 和 `/api/ask/stream` 支持：

- `query` 最大 1000 字符。
- `input_images` 最多 3 张。
- 单张图片最大 8 MB。
- 图片必须以 `data:image/...` Data URL 形式提交。

这些限制是为了让本地演示保持轻量，并避免浏览器端请求体过大。

## 推荐本地启动顺序

1. 启动 LM Studio，并加载配置中的模型。
2. 确认 PicRetrieve 数据位于 `PicRetrieve/data/`。
3. 执行 `npm run app:dev`。
4. 打开 `http://127.0.0.1:8010/`。
5. 用 `/api/status` 检查 `ready` 和 `llm_ready`。
