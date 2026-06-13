# PicRetrieve 图片检索接口

> 范围：只覆盖图片检索分支，用于 Dify / FastGPT 通过 HTTP 调用。当前实现已经具备可对接接口、网页试用页、本地索引和检索结果图片访问能力。

## 是否足够

按“专注相似图片查找算法，根据当前图片检索相关图片，可先用文件夹尝试，图片可在 Hugging Face 找”这个要求，当前工作已经足够做图片检索分支展示。效果说明只使用 Hugging Face 上有真实标注的标准 benchmark，不用 50k 级演示语料、硬件商品图或某个具体品类做精度背书：

- 已支持 Hugging Face 标注图片数据集离线索引。
- 已支持上传图片并返回相似图片。
- 已支持返回结果图片文件，便于 Dify / FastGPT 展示。
- 已支持 Hugging Face 标准 benchmark 数据，当前聚焦 Flickr30k 和 CUB200。
- 已有 Flickr30k 文搜图指标和 CUB200 图搜图指标。
- 已有 FastAPI，可直接作为 Dify / FastGPT 的 HTTP 工具接口。

## 工具本身做了什么

PicRetrieve 是一个本地图片检索工具，核心能力是“给一张图片，从已有图片库里找相似图片”。它现在做了这些事情：

- 从 Hugging Face 标注数据集中准备图片库。
- 扫描图片文件、目录、sidecar JSON/TXT、EXIF 等元信息。
- 使用 CLIP 把图片编码成向量。
- 把图片向量保存到 NumPy 文件，把元信息保存到 SQLite。
- 查询时接收一张上传图片，并编码成查询向量。
- 用余弦相似度从图片库中找出相似图片。
- 可选融合元信息关键词。
- 返回排序后的结果、分数、路径、元信息和图片访问地址。
- 提供 Web 页面、本地 CLI 和 HTTP API。
- 提供标准 benchmark 指标，用于说明模型检索效果。

简单说，它不是 Dify / FastGPT 的替代品，而是给它们提供“图片检索能力”的后端工具。Dify / FastGPT 负责工作流、对话和结果组织，PicRetrieve 负责图片索引、相似度计算和返回图片结果。

## 当前 benchmark 结果

当前效果说明只使用有真实标注的标准 benchmark，不使用 50k 级演示语料、商品图、型号词或弱监督标签做精度依据。

| 任务 | 数据集 | 规模 | 主要指标 |
|---|---|---:|---|
| 文搜图 | `clip-benchmark/wds_flickr30k` test | 1000 张图 / 5000 条 caption | Recall@1 0.5928，Recall@5 0.8424，Recall@10 0.9018，MRR 0.7037 |
| 图搜图 | `mteb/cub200_retrieval` | 5794 query / 5794 corpus | Hit@1 0.5195，Hit@5 0.8029，Hit@10 0.8835，MRR 0.6389 |
| 近重复图搜图 | Flickr30k 确定性变换 | 1000 张图 / 4000 个变体 query | Recall@1 1.0000 |

说明：CUB200 每个 query 有多张相关图，所以 `recall@10` 不是主要展示指标；更适合看 `Hit@K`、`MRR` 和 `Precision@K`。

## 图片检索架构图

![PicRetrieve 图片检索架构图](picretrieve_image_architecture.png)

架构图只描述当前图片检索链路：HF 标注数据集进入离线索引，写入本地向量库、SQLite 元信息库和文件映射；在线服务接收查询图片后做 CLIP 编码、向量检索、可选分数融合，并返回 JSON 与图片访问地址。

## 服务启动

```bash
PICRETRIEVE_MODEL_NAME=data/models/openai_clip-vit-base-patch32 \
PICRETRIEVE_DATA_DIR=data/flickr30k_index \
PICRETRIEVE_IMAGE_ROOT=data/flickr30k_test/images \
PICRETRIEVE_DEVICE=auto \
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

如果 Dify / FastGPT 与服务不在同一台机器，不能填 `127.0.0.1`，需要使用局域网 IP、内网穿透地址或部署后的域名。

设备可用 `PICRETRIEVE_DEVICE` 指定：

- `cpu`：强制 CPU。
- `cuda` 或 `cuda:0`：使用 CUDA。
- `auto`：自动选择可用设备。

## Dify / FastGPT 对接接口

### 健康检查

```http
GET /health
```

返回示例：

```json
{
  "status": "ok",
  "device": "cuda",
  "model": "data/models/openai_clip-vit-base-patch32",
  "items": 1000
}
```

### 图片检索

```http
POST /search/image
Content-Type: multipart/form-data
```

参数：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---:|---:|---:|---|
| `file` | file | 是 | - | 查询图片 |
| `top_k` | int | 否 | 10 | 返回结果数量 |
| `profile` | string | 否 | `general` | `general` 或 `document` |
| `metadata_query` | string | 否 | - | 可选元信息关键词 |
| `candidate_k` | int | 否 | 200 | 候选召回数量 |

curl 示例：

```bash
curl -X POST http://127.0.0.1:8000/search/image \
  -F "file=@/path/to/query.jpg" \
  -F "top_k=5" \
  -F "profile=general"
```

返回示例：

```json
{
  "query_type": "image",
  "top_k": 5,
  "results": [
    {
      "id": 123,
      "rel_path": "flickr30k/test/s0000001.jpg",
      "score": 0.812345,
      "visual_score": 0.79321,
      "metadata_score": 0.0,
      "width": 512,
      "height": 512,
      "metadata": {}
    }
  ]
}
```

Dify / FastGPT 展示图片时，用返回的 `id` 拼接：

```text
http://服务地址:8000/files/{id}
```

例如：

```text
http://127.0.0.1:8000/files/123
```

### 可选：文本搜图

如果工作流里需要“用户文字描述找图”，可以调用：

```http
POST /search/text
Content-Type: application/json
```

```json
{
  "text": "A black and white dog is running in the grass.",
  "top_k": 5,
  "profile": "general",
  "candidate_k": 200
}
```

## OpenAPI

FastAPI 已自动提供 OpenAPI：

```text
http://服务地址:8000/openapi.json
```

Dify / FastGPT 如果支持导入 OpenAPI，可以直接导入该地址；如果只支持 HTTP 节点，则按上面的 `/search/image` multipart 表单配置即可。
