# 本地相似图片检索 MVP - Codex 实现说明

> 目标：在本地 MacBook 上实现一个最小可复现的“以图搜图/以文搜图/元信息混合检索”服务。不要训练模型、不要租显卡、不要接云向量数据库。先把一个文件夹里的图片跑通，并通过 HTTP API 暴露给后续 Agent 调用。

## 1. 一句话需求

给定一张查询图片，从本地图片文件夹中返回最相关的图片；同时支持用文件名、目录名、EXIF、sidecar JSON/TXT、可选 OCR 文本做补充检索，解决发票、专利、合同、模板类图片“视觉上都很像，但关键字段不同”的问题。

## 2. 硬约束

- 必须能在 Apple Silicon MacBook 本地运行；没有 CUDA 也不能失败。
- 第一版只做离线索引 + 在线检索 + FastAPI API，不做前端。
- 第一版默认模型使用 Hugging Face Transformers 的 `openai/clip-vit-base-patch32`，因为最稳、最容易复现。
- 支持 `mps`，但必须自动 fallback 到 `cpu`。
- 不要接 Qdrant、Milvus、Pinecone、Weaviate 等外部服务。
- 50k 张以内先用 NumPy 暴力余弦相似度；后续再换 FAISS/hnswlib/LanceDB。
- OCR 是可选增强项：没有安装 Tesseract 或 RapidOCR 时，索引过程不能失败。
- API 不允许任意读取机器上的文件；只能索引配置中允许的图片根目录。

## 3. 推荐项目结构

```text
image-retrieval-mvp/
  README.md
  requirements.txt
  .env.example
  app/
    __init__.py
    config.py
    embedder.py
    metadata.py
    index_store.py
    retrieval.py
    api.py
    cli.py
  scripts/
    save_hf_images.py        # 可选：把 Hugging Face image dataset 保存成普通文件夹
  tests/
    test_metadata.py
    test_index_store.py
    test_retrieval.py
  data/
    .gitkeep
  samples/
    .gitkeep
```

## 4. Python 依赖

第一版依赖保持少：

```txt
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic>=2
pydantic-settings>=2
pillow>=10
numpy>=1.26
tqdm>=4.66
torch>=2.3
transformers>=4.45
```

可选依赖，不要放进必需路径：

```txt
pytesseract>=0.3.10
rapidocr-onnxruntime>=1.4
huggingface_hub>=0.25
datasets>=3.0
```

## 5. 数据组织方式

先用普通文件夹验证，不依赖 Hugging Face：

```text
samples/
  hard_disk/
    001.jpg
    002.jpg
  gpu/
    001.jpg
  cpu/
    001.jpg
  robot/
    001.jpg
```

可选 sidecar 元信息：

```text
samples/invoice/invoice_001.png
samples/invoice/invoice_001.json
samples/invoice/invoice_001.txt
```

`invoice_001.json` 示例：

```json
{
  "doc_type": "invoice",
  "vendor": "AMD",
  "invoice_no": "INV-2026-001",
  "date": "2026-04-20",
  "amount": "1280.00",
  "currency": "USD"
}
```

## 6. 索引设计

### 6.1 SQLite 表

```sql
CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  rel_path TEXT NOT NULL,
  filename TEXT NOT NULL,
  folder TEXT,
  ext TEXT,
  width INTEGER,
  height INTEGER,
  size_bytes INTEGER,
  mtime REAL,
  metadata_json TEXT,
  ocr_text TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
  filename,
  folder,
  rel_path,
  metadata_text,
  ocr_text,
  content=''
);
```

不要依赖 FTS 外部内容表触发器，第一版手动插入 `items_fts` 即可。插入时确保 `rowid = items.id`。

### 6.2 向量文件

保存两个 NumPy 文件：

```text
data/embeddings.npy   # float32, shape = [N, D], 每行已 L2 normalize
data/ids.npy          # int64, shape = [N], 与 embeddings 行对齐
```

检索时加载到内存，执行：

```python
scores = embeddings @ query_embedding
```

50k 张、512 维向量约 100MB，MacBook 可以接受。后续超过 50k 再引入 FAISS 或 hnswlib。

## 7. Embedding 模块接口

实现 `app/embedder.py`：

```python
class ClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None): ...
    def encode_images(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray: ...
    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...
```

要求：

- 使用 `CLIPModel` + `CLIPProcessor`。
- 自动选择设备：`mps` > `cuda` > `cpu`，但本项目不能依赖 CUDA。
- 对输出向量做 L2 normalize，并转成 `np.float32`。
- `mps` 报错时打印 warning 并 fallback 到 `cpu` 重跑当前 batch。
- 图片统一 `RGB`。

## 8. 元信息模块

实现 `app/metadata.py`：

- 扫描扩展名：`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`。
- 从文件系统提取：文件名、相对路径、父目录、扩展名、大小、mtime。
- 用 Pillow 提取：width、height、EXIF；EXIF tag 转成人类可读名称。
- 读取同名 sidecar：`.json`, `.txt`, `.md`。
- （pHash 已移除，不再需要 imagehash）
- OCR 函数只做可选：
  - `ocr_mode="none"`：不跑。
  - `ocr_mode="tesseract"`：如果 `pytesseract` 或系统 tesseract 不可用，返回空字符串，不抛出致命异常。

## 9. 检索逻辑

### 9.1 视觉检索

输入查询图片，算 CLIP image embedding，然后对 `embeddings.npy` 做余弦相似度。取 top `candidate_k`，默认 100。

### 9.2 元信息检索

输入 `metadata_query`，用 SQLite FTS5 查：

```sql
SELECT rowid, bm25(items_fts) AS rank
FROM items_fts
WHERE items_fts MATCH ?
ORDER BY rank
LIMIT ?;
```

第一版不追求复杂分词。需要一个很小的清洗函数：去掉危险字符，仅保留中英文、数字、下划线、短横线、空格，并用空格连接 token。


### 9.4 分数融合

提供两个 profile：

```python
PROFILES = {
  "general": {"visual": 0.85, "metadata": 0.15},
  "document": {"visual": 0.45, "metadata": 0.55}
}
```

最终分数：

```python
final_score = w_visual * visual_score_norm + w_meta * metadata_score_norm
```

归一化建议：

- `visual_score_norm = (cosine + 1) / 2`
- `metadata_score_norm` 第一版用名次归一化：第 1 名 1.0，第 2 名 0.9，线性衰减到 0。
- （pHash 已移除，不再使用）

## 10. CLI

实现 `app/cli.py`，支持：

```bash
python -m app.cli index \
  --image-dir ./samples \
  --data-dir ./data \
  --model-name openai/clip-vit-base-patch32 \
  --batch-size 16 \
  --ocr none \
  --reset

python -m app.cli search-image \
  --image ./query.jpg \
  --data-dir ./data \
  --top-k 10 \
  --profile general

python -m app.cli search-text \
  --text "robot arm" \
  --data-dir ./data \
  --top-k 10
```

## 11. FastAPI API

实现 `app/api.py`。

### 11.1 `GET /health`

返回：

```json
{"status":"ok","device":"mps","model":"openai/clip-vit-base-patch32","items":1234}
```

### 11.2 `POST /search/image`

Multipart form：

- `file`: 查询图片
- `top_k`: 默认 10
- `profile`: `general` 或 `document`
- `metadata_query`: 可选，例如 `AMD INV-2026`
- `candidate_k`: 默认 100

返回：

```json
{
  "query_type": "image",
  "top_k": 10,
  "results": [
    {
      "id": 12,
      "rel_path": "gpu/001.jpg",
      "score": 0.93,
      "visual_score": 0.91,
      "metadata_score": 0.5,
      "width": 1024,
      "height": 768,
      "metadata": {"folder": "gpu"}
    }
  ]
}
```

### 11.3 `POST /search/text`

JSON body：

```json
{
  "text": "graphics card with three fans",
  "metadata_query": "gpu",
  "top_k": 10,
  "profile": "general"
}
```

逻辑：用 CLIP text embedding 做视觉空间检索，同时用 `metadata_query` 或 `text` 做 FTS 检索。

### 11.4 `GET /items/{id}`

返回单个 item 的完整元信息。

### 11.5 `GET /files/{id}`

返回图片文件。必须校验 path 在允许的 image root 下，避免 path traversal。

## 12. Hugging Face 图片样本脚本，可选

实现一个通用脚本即可，不要绑定具体数据集：

```bash
python scripts/save_hf_images.py \
  --dataset DATASET_ID \
  --split 'train[:100]' \
  --image-column image \
  --label-column label \
  --out ./samples/hf
```

脚本行为：

- 用 `datasets.load_dataset` 读取有 image column 的数据集。
- 把 PIL image 保存成 JPG。
- 如果有 label，按 label 建子目录；否则保存到 `samples/hf/all/`。
- 失败时给出清晰错误，不影响主项目。

## 13. README 必须包含的复现命令

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 准备 samples/ 文件夹后：
python -m app.cli index --image-dir ./samples --data-dir ./data --reset
python -m app.cli search-image --image ./samples/gpu/001.jpg --data-dir ./data --top-k 5
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

curl：

```bash
curl -X POST http://127.0.0.1:8000/search/image \
  -F "file=@./samples/gpu/001.jpg" \
  -F "top_k=5" \
  -F "profile=general"
```

文档类检索示例：

```bash
curl -X POST http://127.0.0.1:8000/search/image \
  -F "file=@./samples/invoice/invoice_query.png" \
  -F "top_k=5" \
  -F "profile=document" \
  -F "metadata_query=AMD INV-2026-001"
```

## 14. 测试要求

至少实现：

- `test_metadata.py`
  - 能扫描图片。
  - 能读取 sidecar JSON/TXT。
  - 没有 EXIF 或 OCR 时不报错。
- `test_index_store.py`
  - SQLite 表创建成功。
  - 插入 items 后 FTS 能搜到文件名或 sidecar 文本。
  - `embeddings.npy` 与 `ids.npy` 行数一致。
- `test_retrieval.py`
  - 用固定小矩阵验证 cosine top-k 排序。
  - 验证分数融合 profile 行为。
  - 验证分数融合 profile 行为。

## 15. 验收标准

项目完成时必须满足：

- 可以在没有 GPU 的 Mac 上跑通索引和检索。
- 只放 20 张图片也可以正常工作。
- `POST /search/image` 能返回 JSON 列表。
- 对硬盘、显卡、CPU、机器人这类图片区分度较高的集合，用同类图片 query 时 top-5 大部分是同类。
- 对发票/专利模板类图片，如果提供 sidecar/OCR 元信息，`profile=document` 能明显把元信息匹配的图片排到前面。
- README 里的命令从零开始可执行。

## 16. 明确不要做

- 不要训练或微调 CLIP。
- 不要做 Web UI。
- 不要引入外部向量数据库。
- 不要实现权限系统；只做本地 localhost API。
- 不要把图片存进 SQLite BLOB；SQLite 只存路径和元信息。
- 不要为了 Hugging Face 样本下载阻塞主功能。

## 17. 后续可选升级，不属于第一版

- 超过 50k 图片后接 FAISS、hnswlib 或 LanceDB。
- 把默认模型切换到 SigLIP2 或 MobileCLIP2。
- 用 DINOv2 增强 image-to-image instance-level 检索。
- 文档页检索升级到 ColPali/ColQwen2，但这会显著增加模型体积和运行成本，不建议进入 MVP。
- 增加缩略图缓存和一个极简 Web UI。
