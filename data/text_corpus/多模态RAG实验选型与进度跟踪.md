# 多模态 RAG 实验选型与进度跟踪

本文档用于承接架构图中不再展示的技术选型、实验对比和进度状态。架构图只保留功能职责、数据流和统一接口；具体方案在这里维护，便于后续持续更新。

## 相关视图

- 可视化看板：[mrag_experiment_dashboard.html](../../docs/mrag_experiment_dashboard.html)
- 结构化数据源：[mrag_experiment_registry.json](../../docs/mrag_experiment_registry.json)

## 状态约定

| 状态 | 含义 |
| --- | --- |
| TODO | 尚未准备数据或环境 |
| Planned | 已明确实验目标，待执行 |
| Running | 正在跑实验或整理结果 |
| Done | 已完成并有可复现实验记录 |
| Blocked | 缺数据、缺权限、缺环境或结果不可复现 |

## 总体实验原则

- 先做可复现 baseline，再增加复杂组件。
- 每次只改变一个关键变量，例如 OCR 引擎、chunk 粒度、caption 开关、reranker 开关。
- 每条实验记录都要保留数据集版本、代码版本、参数、耗时、成本、指标和结论。
- 所有分支最终都要输出统一的 EvidenceCard：`modality`、`source_id`、`locator`、`content_ref`、`score`、`raw_reference`、`confidence`。

## 实验看板

| 编号 | 数据类型 | 关键处理难点 | 候选方案 | Baseline | 主要指标 | 状态 | 结论/下一步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-01 | 文本 | chunk 粒度、元数据过滤、混合检索 | BM25；Dense；Hybrid；Reranker | BM25 only | Recall@K；MRR；nDCG；Evidence Recall | Planned | 先固定小规模标注集，比较 chunk size 和 reranker 开关 |
| I-01 | 图像/页面 | OCR、caption、bbox/region 定位 | PaddleOCR；Mistral OCR；Qwen2.5-VL；InternVL3；ColPali；ColQwen2 | OCR RAG | Page Hit；Region Accuracy；Citation Precision | TODO | 先准备扫描页和截图样例，标注页码与区域 |
| D-01 | 文档 | 版面解析、跨页证据、表格/图表抽取 | Docling；PaddleOCR；MinerU；Unstructured；Mistral OCR | OCR-only document RAG | Evidence Page Hit；Multi-page Coverage；Citation Precision | TODO | 财报、报表和普通 PDF 分开评估 |
| A-01 | 音频 | ASR、说话人、时间戳对齐 | Whisper；FunASR；PaddleSpeech；speaker diarization | ASR transcript retrieval | Timestamp Hit；Speaker Accuracy；WER Impact | TODO | 先用会议录音样例建立转写和时间戳证据 |
| V-01 | 视频 | 抽帧、场景切分、字幕对齐、关键帧证据 | FFmpeg；OpenCV；ASR subtitles；keyframe caption | ASR-only video retrieval | Segment Hit；Event Localization；Latency；Cost | TODO | 先比较固定间隔抽帧和 shot-based 抽帧 |
| TB-01 | 表格 | 表头识别、单元格定位、指标口径 | 表格结构抽取；单元格引用；指标口径映射 | CSV/table text retrieval | Cell Hit；Answer Correctness；Traceability | TODO | 财报表格和业务报表需要单独标注 |
| DB-01 | 数据库 | schema 理解、Text-to-SQL、权限、结果追溯 | PostgreSQL；DuckDB；Warehouse；schema vector index；SQL validation | schema-only Text-to-SQL | Execution Accuracy；Result Correctness；Access Safety | TODO | 先准备只读测试库和问题集合 |
| KG-01 | 知识图谱 | 实体链接、关系路径、子图证据 | entity linking；path retrieval；subgraph extraction；triple store | entity keyword retrieval | Path Hit；Relation Accuracy；Traceability | TODO | 先定义实体、关系和路径深度上限 |
| F-01 | 融合与重排 | 多分支候选统一、去重、跨模态排序 | RRF；Score Fusion；cross-modal rerank | branch-local topK concat | Evidence Recall；Answer Faithfulness；Latency | Planned | 等至少两个分支 baseline 可用后再启动 |

## EvidenceCard 字段草案

| 字段 | 说明 | 示例 |
| --- | --- | --- |
| `modality` | 证据来源类型 | `text`、`image`、`audio`、`sql` |
| `source_id` | 原始数据或资产 ID | 文档 ID、图片 ID、音频 ID、表名 |
| `locator` | 可定位引用 | chunk id、page+bbox、start/end time、row id、KG path |
| `content_ref` | 可供生成器读取的内容引用 | 文本片段、OCR 块、caption、SQL 结果、子图 |
| `score` | 召回或重排分数 | normalized score |
| `raw_reference` | 回溯到原始文件或查询的引用 | file path、object key、SQL query、graph query |
| `confidence` | 分支或 verifier 置信度 | high、medium、low |
| `notes` | 风险、冲突或人工备注 | OCR low confidence、SQL validation failed |

## 实验记录模板

```text
实验编号：
日期：
数据版本：
代码版本：
目标：
变量：
参数：
指标：
结果：
结论：
下一步：
```
