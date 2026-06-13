# 图片召回实验：数据获取与本地语料构建补充说明（给 Codex 实现）

> 目的：给“本地相似图片检索 MVP”补上可复现的数据获取模块。第一版不要依赖大规模数据，也不要让用户手工到处找硬盘、显卡、CPU、发票、专利图片。实现后，用户只需要运行一个脚本，就能得到一个可被现有索引器消费的本地图片文件夹和 `metadata.jsonl`。

## 1. 目标边界

实现一个非常小的数据引导模块：

```text
scripts/bootstrap_data.py
scripts/evaluate_smoke.py
```

`bootstrap_data.py` 负责从公开 Hugging Face 数据集或 URL 元数据中抽样图片，保存成本地普通文件夹；`evaluate_smoke.py` 负责做最小可用的检索冒烟评测。检索服务本身只读取本地文件夹，不直接依赖 Hugging Face。

第一版默认只做三类数据：

1. `hardware`：CPU、GPU/video-card、internal-hard-drive 等硬件产品图片，带品牌、名称、规格等元信息。
2. `documents`：扫描文档图片，测试模板相似但内容不同的情况。
3. `receipts_optional`：发票/收据类样本，后续用于 OCR 和结构化字段检索。

不要在第一版引入 LAION、CC12M、Open Images、机器人视频数据集这类大数据源。它们对“本地可复现子模块”来说过重。

## 2. 默认数据源

### 2.1 Hardware 默认源：`Doshiba/pcpartpicker-parts-dataset`

用途：硬盘、显卡、CPU、主板、内存、电源、机箱等产品图片召回。

数据特点：

- 数据集字段包含 `category`、`name`、`brand`、`url`、`image_url`、`price_eur`、`rating_count`、`specs`。
- 类别包括 `cpu`、`cpu-cooler`、`motherboard`、`memory`、`internal-hard-drive`、`video-card`、`power-supply`、`case`。
- 图片没有直接重分发在数据集中，而是以 `image_url` 字段提供，脚本需要下载图片并缓存到本地。
- 这是第一版最适合的默认源，因为它同时解决了“图片”和“元信息”两个问题。

默认只取：

```text
cpu: 60
video-card: 60
internal-hard-drive: 60
motherboard: 30
memory: 30
```

这样总量约 240 张，足够验证图片召回、文本召回、元信息混合召回。

### 2.2 Document 默认源：`vaclavpechtor/rvl_cdip-small-200`

用途：测试文档模板类图片，例如 letter、memo、email、advertisement 等。第一版不要上完整 RVL-CDIP，先用 small-200 子集。

默认只取：

```text
每个 label 20 张，最多 320 张
```

保存时把 `label` 写入 metadata。没有 OCR 时也可以测试“视觉模板相似”的问题。

### 2.3 Receipt 可选源：`Voxel51/consolidated_receipt_dataset`

用途：测试收据/发票类图片的 OCR 字段、金额、商户名等元信息检索。

这个源使用 FiftyOne 的 Hugging Face loader，依赖比前两个重，所以第一版可做成可选：

```bash
python scripts/bootstrap_data.py receipts --out data/corpus --max-samples 100
```

如果用户没有安装 `fiftyone`，脚本应提示“跳过 receipts，可先跑 hardware/documents”。不要让主流程失败。

### 2.4 Patent 可选源：`danaaubakirova/patfig`

用途：专利图、技术图、流程图、灰度图检索。这个源有专利图片、caption、reference numeral、publication number 等元信息，但约 2GB+，且是非商业许可。第一版不要默认下载，只保留命令入口。

## 3. 输出目录契约

所有数据源都要被转换成统一的普通文件夹格式：

```text
data/corpus/
  images/
    pcpartpicker/cpu/493350.jpg
    pcpartpicker/video-card/123456.jpg
    rvl_cdip/letter/000001.jpg
    rvl_cdip/email/000002.jpg
  metadata.jsonl
```

`metadata.jsonl` 每行一个 JSON，至少包含：

```json
{
  "file_name": "images/pcpartpicker/cpu/493350.jpg",
  "dataset": "Doshiba/pcpartpicker-parts-dataset",
  "category": "cpu",
  "title": "Intel Core i9-14900K",
  "brand": "Intel",
  "source_id": "493350",
  "source_url": "https://fr.pcpartpicker.com/product/...",
  "image_url": "https://cdna.pcpartpicker.com/...jpg",
  "text": "Intel Core i9-14900K cpu Intel Core Count 24 TDP 125 W ...",
  "metadata": {
    "price_eur": 459.9,
    "rating_count": 41,
    "specs": {
      "Core Count": "24",
      "TDP": "125 W"
    }
  },
  "license_note": "For local research/MVP only; do not redistribute downloaded images without checking source terms."
}
```

其中 `text` 字段是给 SQLite FTS 用的拼接文本，规则：

```text
text = title + category + brand + flattened(specs) + optional OCR/caption
```

检索模块后续只需要读取：

- 图片路径：`file_name`
- 语义补充文本：`text`
- 结构化字段：`metadata`

## 4. CLI 设计

### 4.1 下载硬件数据

```bash
python scripts/bootstrap_data.py hardware \
  --out data/corpus \
  --per-category 60 \
  --categories cpu video-card internal-hard-drive motherboard memory \
  --max-image-size 768 \
  --seed 42
```

行为：

- 用 `datasets.load_dataset("Doshiba/pcpartpicker-parts-dataset", split="train", streaming=True)` 读取行。
- 按 `category` 过滤。
- 下载 `image_url`。
- 用 Pillow 校验图片，统一转 RGB，最长边缩放到 `--max-image-size`。
- 保存到 `data/corpus/images/pcpartpicker/<category>/<source_id>.jpg`。
- 写 sidecar JSON：同名 `.json`。
- 追加到 `data/corpus/metadata.jsonl`。
- 下载失败、图片损坏、字段缺失时跳过，不中断。

### 4.2 下载文档数据

```bash
python scripts/bootstrap_data.py documents \
  --out data/corpus \
  --dataset rvl_cdip_small \
  --per-label 20 \
  --max-image-size 1024
```

行为：

- 用 `datasets.load_dataset("vaclavpechtor/rvl_cdip-small-200", split="train")` 读取。
- 每个 label 最多保存 `--per-label` 张。
- TIFF/PNG/JPG 都统一转成 RGB JPG。
- 保存到 `data/corpus/images/rvl_cdip/<label>/<idx>.jpg`。
- metadata 写入 `dataset`、`category=label`、`title`、`text`。

### 4.3 下载收据数据（可选）

```bash
python scripts/bootstrap_data.py receipts \
  --out data/corpus \
  --max-samples 100
```

行为：

- 尝试导入 `fiftyone`。
- 如果不可用，打印安装提示并正常退出，exit code 为 0。
- 用 `fiftyone.utils.huggingface.load_from_hub("Voxel51/consolidated_receipt_dataset", max_samples=...)`。
- 保存图片和可用字段，例如 `num_items`、`subtotal_price`、`service_price` 等。
- 把字段拼进 `text`，用于验证“视觉模板相似但金额、字段不同”的检索。

## 5. requirements 追加

在 `requirements.txt` 中把数据下载依赖设为可选段落，避免主服务变重：

```txt
# data bootstrap, optional for experiments
requests>=2.32
huggingface_hub>=0.25
datasets>=3.0

# optional receipt dataset explorer
# fiftyone>=1.0
```

不要把 `fiftyone` 放进默认必装依赖；它只用于 receipt 可选数据。

## 6. 核心实现伪代码

```python
from datasets import load_dataset
from PIL import Image
import requests, io, json, hashlib
from pathlib import Path


def safe_image_id(row: dict) -> str:
    raw = str(row.get("source_id") or row.get("product_tag") or row.get("name") or row.get("image_url"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def download_image(url: str, max_size: int) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "image-retrieval-mvp/0.1"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_size, max_size))
        return img
    except Exception:
        return None


def flatten_specs(specs: dict | None) -> str:
    if not isinstance(specs, dict):
        return ""
    parts = []
    for k, v in specs.items():
        if v is not None:
            parts.append(f"{k} {v}")
    return " ".join(parts)


def build_text(row: dict) -> str:
    return " ".join(str(x) for x in [
        row.get("name", ""),
        row.get("category", ""),
        row.get("brand", ""),
        flatten_specs(row.get("specs")),
    ] if x)
```

## 7. 冒烟评测设计

第一版不需要人工标注大规模 ground truth。用弱监督即可：

### 7.1 文本到图片检索

自动生成 `data/queries/smoke_queries.jsonl`：

```json
{"query": "AMD Ryzen CPU", "must_category": "cpu", "must_brand": "AMD"}
{"query": "NVIDIA RTX graphics card", "must_category": "video-card"}
{"query": "2TB SSD internal hard drive", "must_category": "internal-hard-drive"}
{"query": "scanned email document", "must_category": "email"}
{"query": "receipt with subtotal price", "must_category": "receipt"}
```

指标先用简单的：

```text
category_hit@10 = top10 里是否有 must_category
brand_hit@10 = 如果 query 指定 must_brand，top10 里是否有 must_brand
```

### 7.2 图片到图片检索

从每个 category 抽 3 张作为 query，正例定义为同 category。指标：

```text
same_category_hit@5
same_category_hit@10
```

注意：这不是严肃 benchmark，只是确保管线工作。真正的业务评估后续再做人工标注。

## 8. 与现有索引器的衔接

现有 `app/metadata.py` 需要增强：

1. 如果 `data/corpus/metadata.jsonl` 存在，按 `file_name` 合并元信息。
2. 如果图片旁边有同名 `.json`，读取 sidecar 元信息。
3. 如果图片旁边有同名 `.txt`，读取 OCR/说明文本。
4. 最终写入 SQLite：

```text
items.path
items.metadata_json
items.metadata_text
```

`metadata_text` 应包含：

```text
目录名 + 文件名 + metadata.jsonl.text + sidecar.json flatten + sidecar.txt
```

## 9. 推荐复现命令

```bash
# 1. 安装
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 拉一个很小的硬件样本集
python scripts/bootstrap_data.py hardware --out data/corpus --per-category 30

# 3. 可选：拉文档样本
python scripts/bootstrap_data.py documents --out data/corpus --per-label 10

# 4. 建索引
python -m app.cli index --root data/corpus/images --metadata data/corpus/metadata.jsonl

# 5. 启动 API
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload

# 6. 冒烟评测
python scripts/evaluate_smoke.py --queries data/queries/smoke_queries.jsonl --top-k 10
```

## 10. 验收标准

- `python scripts/bootstrap_data.py hardware --per-category 10` 能在普通 MacBook 上成功生成至少 30 张图片。
- 每张图片都有 sidecar `.json`，总目录有 `metadata.jsonl`。
- `metadata.jsonl` 中 `file_name` 路径真实存在。
- 索引器可以直接索引 `data/corpus/images`。
- 文本查询 `NVIDIA RTX graphics card`、`AMD Ryzen CPU`、`2TB SSD internal hard drive` 能返回合理类别。
- API `/search/image` 和 `/search/text` 不关心数据源，只返回统一 schema。
- 下载失败不能导致任务整体失败。
- 所有外部数据默认只用于本地实验，不把下载后的图片打包提交到 Git。

## 11. 不要做的事

- 不要把 Hugging Face dataset loader 写进在线检索 API。
- 不要默认下载 10GB 级数据。
- 不要要求用户申请 token 或租 GPU。
- 不要先做复杂标注平台。
- 不要为了机器人样例去处理视频数据；如果真需要 robot 类，后续单独从小型图片数据集或用户本地图片补 20-50 张即可。

## 12. 参考来源

- Hugging Face ImageFolder 文档：`https://huggingface.co/docs/datasets/en/image_dataset`
- Hugging Face Streaming 文档：`https://huggingface.co/docs/datasets/en/stream`
- PCPartPicker parts dataset：`https://huggingface.co/datasets/Doshiba/pcpartpicker-parts-dataset`
- RVL-CDIP small-200：`https://huggingface.co/datasets/vaclavpechtor/rvl_cdip-small-200`
- Consolidated Receipt Dataset：`https://huggingface.co/datasets/Voxel51/consolidated_receipt_dataset`
- PatFig：`https://huggingface.co/datasets/danaaubakirova/patfig`
