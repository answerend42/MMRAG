# 数据与索引

MMRAG 提交的是代码、测试、文档和轻量级架构产物，不提交本地图片索引、图片语料或模型权重。这样仓库保持可读、可克隆，也避免把大文件和隐私数据放进 Git 历史。

## 本地数据目录

当前 PicRetrieve 模块默认读取：

```text
PicRetrieve/data/
```

一个可运行环境通常包含：

```text
PicRetrieve/data/
├── index.sqlite
├── embeddings.npy
├── ids.npy
├── image_root.txt
└── models/
    └── openai_clip-vit-base-patch32/
```

字段说明：

| 文件或目录 | 说明 |
| --- | --- |
| `index.sqlite` | PicRetrieve 的图片元数据和索引数据库。 |
| `embeddings.npy` | 图片向量。 |
| `ids.npy` | 向量与图片条目的 ID 映射。 |
| `models/openai_clip-vit-base-patch32/` | 本地 CLIP 模型文件。 |
| `image_root.txt` | 可选，内容是一行本机图片语料目录绝对路径。 |

## 图片语料位置

如果存在 `image_root.txt`，系统会读取其中的路径作为图片根目录。示例：

```text
/Users/example/Datasets/my-image-corpus
```

如果不存在，系统会尝试使用：

```text
PicRetrieve/data/corpus/images
```

召回结果中的图片通过 `/api/images/{item_id}` 暴露给前端。后端会检查图片路径是否位于允许的图片根目录下，避免任意文件读取。

## 为什么不提交这些数据

- 图片语料可能包含版权或隐私信息。
- 模型权重和向量文件体积较大，不适合进入 Git 历史。
- 本地实验经常重建索引，提交会制造大量无意义 diff。
- 开源仓库应该优先保证代码和文档可审查。

## 重新准备索引

PicRetrieve 子项目中保留了自己的文档和测试。当前 MMRAG 只要求它最终提供一个可用的本地索引目录。准备数据时请优先确认：

```bash
ls PicRetrieve/data
```

至少应该看到 `index.sqlite`、向量文件和模型目录。

## 检查运行状态

启动应用后执行：

```bash
curl http://127.0.0.1:8010/api/status
```

重点看：

- `ready`：PicRetrieve 是否可用。
- `images`：索引中的图片数量。
- `data_dir`：当前解析到的数据目录。
- `errors`：缺失文件、导入失败或运行时初始化失败的具体原因。

## Git 忽略策略

仓库应持续忽略以下内容：

```text
PicRetrieve/data/
.venv/
node_modules/
__pycache__/
.pytest_cache/
.ruff_cache/
```

如果未来需要发布小型示例数据集，建议新增单独的 `examples/` 或 `demo-data/`，并明确数据来源和许可证，而不是把本地实验索引直接提交。
