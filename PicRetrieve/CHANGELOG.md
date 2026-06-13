# 更新日志

## 0.1.0 — 2025-06-27

### 新增
- 基于 CLIP (openai/clip-vit-base-patch32) 的图片/文本向量编码
- 本地索引：SQLite 元信息 + FTS5 全文检索 + NumPy 向量存储
- 混合检索：视觉相似度 + 元信息关键词检索，支持可配置 profile 权重
- FastAPI HTTP 服务：`/search/image`、`/search/text`、`/health`、`/items/{id}`、`/files/{id}`
- 内置单页 Web 试用界面
- CLI 工具：`picretrieve index`、`picretrieve search-image`、`picretrieve search-text`
- 标准 benchmark 评测：Flickr30k 文搜图、MTEB CUB200 图搜图、近重复图搜图
- 弱监督召回评测与冒烟测试
- 数据引导脚本（Hugging Face → 本地 JPG）
- 完整的项目基础设施：MIT 许可证、Docker 部署、ruff 代码规范、结构化日志
