# Codex + draw.io 修改指导（针对当前多模态 RAG 检索分支图）

> 目的：让 Codex 基于现有 `.drawio` 文件进行**结构化重构**，重点解决以下问题：
>
> 1. **画布太小，线与节点重合**；
> 2. **单个框内塞入过多候选技术，使用斜杠 `/` 串联，导致难读**；
> 3. **页面虽然分了模块，但“主干流程”和“可选技术”没有层级分离**；
> 4. **同一页同时承担流程图、技术选型表、实验矩阵、备注，信息密度过高**。
>
> 本文不是一般性的画图建议，而是给 Codex 的**可执行修改规范**。

---

## 0. 先给 Codex 的任务定义

请让 Codex **不要从头随意重画**，而是：

1. 读取现有 `multimodal_rag_retrieval_branch_experiments.drawio`；
2. 保留原有主题、分支、主要技术内容；
3. 按照本文规则进行**版式重构**与**块级拆分**；
4. 优先提升：
   - 可读性
   - 分层性
   - 技术选项的可比较性
   - 页面之间的一致性
5. 如某一页信息过多，允许**拆页**，不要强行塞进一张图。

一句话原则：

> **先把“主流程”画清楚，再把“候选技术”拆成可比较的小块，最后再放实验项与注释。**

---

## 1. 对当前图的诊断（Codex 必须先理解）

当前图已经有一些优点：

- 已经按 modality branch 分页；
- 已经有统一的列标题（Input / Processing / Index / Retrieval / Experiment / Output）；
- 已经有主框架页 + 分支页；
- 已经有 scope、legend、notes。

但目前最主要的问题是：

### 1.1 画布与布局问题

- 页面宽度和高度不足，导致：
  - 主流程框之间间距不够；
  - 连线只能贴近节点边缘；
  - 连线文字标签（如 `normalize / embed`, `retrieve(topK, filters)`）挤压在框体附近；
  - 下方 `Legend / Notes / 中心接口 / 后续实验原则` 等说明块和上方主流程争抢空间。

### 1.2 粒度问题

例如这种写法：

- `BM25: Elasticsearch / OpenSearch`
- `Dense/Sparse: BGE-M3 / Qwen3-Embedding / OpenAI`
- `Hybrid: RRF / score fusion`

这些内容放在一个框里，虽然信息上完整，但**视觉上不适合框图**：

- 候选项之间没有边界；
- 比较关系不明显；
- 难以表达“这是同一类中的多个备选方案”；
- 斜杠 `/` 容易把“并列备选”和“组合关系”混在一起。

### 1.3 语义层级问题

当前页把这些不同层次的东西揉在一起：

1. 主流程阶段（Input → Processing → Index → Retrieval → Experiment → Output）
2. 阶段内部的技术类别（如 OCR / caption / region extraction）
3. 技术类别下的候选实现（如 PaddleOCR、Mistral OCR、Qwen2.5-VL）
4. 实验矩阵
5. 注释与原则

这样会让图看起来像“所有信息并排放上去”，而不是“有主干、有展开”。

---

## 2. 重构总原则（必须遵守）

### 原则 A：主流程层 与 技术选型层 分离

每一页都要分成两个层次：

- **主流程层（必有）**：只保留该 branch 的 6 个一级阶段；
- **技术选型层（展开）**：每个一级阶段内部，用多个小块列出候选方案。

也就是说：

- 一级阶段 = 大框 / 容器；
- 候选技术 = 容器内部的小卡片；
- **禁止再把多个备选项用斜杠硬塞进一行文本。**

### 原则 B：1 个框只表达 1 个概念

一个框中最多表达以下三类之一：

1. 一个阶段（如 `Index Options`）
2. 一个技术类别（如 `Dense / Sparse Retrieval`）
3. 一个具体方案（如 `BGE-M3`）

**禁止一个框同时混合“类别 + 多个具体实现 + 组合方式 + 备注”。**

### 原则 C：可选项必须“拆块”而不是“斜杠拼接”

把：

- `Elasticsearch / OpenSearch`

改成：

- `BM25`
  - `Elasticsearch`
  - `OpenSearch`

把：

- `BGE-M3 / Qwen3-Embedding / OpenAI`

改成：

- `Dense/Sparse Embedding`
  - `BGE-M3`
  - `Qwen3-Embedding`
  - `OpenAI Embeddings`

把：

- `RRF / score fusion`

改成：

- `Hybrid Fusion`
  - `RRF`
  - `Score Fusion`

### 原则 D：空间不够时优先拆页，不要缩字

禁止通过以下方式“硬塞”：

- 大幅缩小字号；
- 大量压缩行距；
- 让连接线穿过文本；
- 让多个卡片紧贴到几乎没有留白。

如果一页容纳不下：

- 先放大画布；
- 仍然不够，则拆成二级页面。

---

## 3. 画布与版式规范（重点解决重叠）

### 3.1 页面尺寸

对于每个 branch 页面，统一使用：

- **横向（landscape）**
- 推荐画布：
  - 优先 `A3 Landscape` 或更大；
  - 或 draw.io 自定义页面尺寸，例如：`2800 x 1800` 或 `3200 x 2000`。

如果 Codex 使用的是 XML / programmatic layout，请明确：

- 外边距至少 `60~80 px`
- 列间距至少 `40~60 px`
- 行间距至少 `50~70 px`
- 主容器之间留出 connector routing gutter

### 3.2 页面分区

每一页严格分为 4 个横向区域：

#### 区域 1：标题区

包含：

- Page title
- Scope

#### 区域 2：主流程区（最重要）

只放 6 个一级阶段：

1. Input / Data
2. Processing / Tools
3. Index / Storage Options
4. Retrieval Strategy
5. Experiment Options
6. Output Evidence

要求：

- 一字排开，保持等宽；
- 每个一级阶段使用**大容器**；
- 容器内部再放子卡片；
- 主流程区占据页面视觉重心。

#### 区域 3：公共说明区

放：

- 后续实验原则
- 中心接口 / Query Planner / Modality Router 说明

要求：

- 这些说明不要再与主流程框同一行竞争空间；
- 作为第二行大说明块，整齐排列。

#### 区域 4：页脚区

放：

- Legend
- Notes

要求：

- 固定在底部；
- 不要漂浮在中间；
- 尺寸可以较小，但要一致。

### 3.3 连线规则

- 主流程之间只画**一级阶段的连线**；
- 子卡片之间通常**不需要全部连线**；
- 如果一定要连，使用容器内部的短线，不要跨越整页。

必须遵守：

- 连线使用 orthogonal / elbow style；
- 连线优先走容器外侧的空白 gutter；
- **禁止连线穿过卡片正文**；
- 连线标签放在线段中点附近，不要贴到节点边缘。

建议：

- `parse / clean`
- `normalize / embed`
- `upsert index`
- `retrieve(topK, filters)`
- `standardize EvidenceCard`

这些标签保留，但要给它们预留空间。

---

## 4. 框体结构规范（重点解决“一个框里塞太多”）

## 4.1 一级阶段容器

一级阶段容器统一结构：

- 顶部：阶段标题（深色 header）
- 中间：若干子卡片（1~4 个为宜）
- 底部：如有必要，可放简短注释

一级阶段容器**不是纯文本框**，而应当是“父容器 + 子卡片”。

## 4.2 子卡片类型

每个子卡片只承担一种角色：

### 类型 1：数据或输入描述卡
例如：

- 培训视频 / 会议录像 / 监控巡检
- PDF / Word / PPT / 扫描文档

### 类型 2：技术类别卡
例如：

- OCR
- Caption / VLM
- Dense Retrieval
- Hybrid Fusion

### 类型 3：具体方案卡
例如：

- PaddleOCR
- Mistral OCR
- ColPali
- BGE-M3
- Elasticsearch

### 类型 4：实验对比卡
例如：

- `ASR-only vs audio embedding`
- `page-level vs chunk-level vs element-level`

## 4.3 单卡片信息上限

每个卡片：

- 最多 4~5 行；
- 每行尽量短；
- 不要用自然语言写长句；
- 优先用“名词短语 + 少量补充”。

例如，不要写：

> BM25：Elasticsearch / OpenSearch Dense/Sparse：BGE-M3 / Qwen3-Embedding / OpenAI Hybrid：RRF / score fusion

要改成：

- 子卡片 1：`BM25`
  - `Elasticsearch`
  - `OpenSearch`
- 子卡片 2：`Dense / Sparse`
  - `BGE-M3`
  - `Qwen3-Embedding`
  - `OpenAI Embeddings`
- 子卡片 3：`Hybrid Fusion`
  - `RRF`
  - `Score Fusion`

---

## 5. 各类页面的具体重画规则

## 5.1 主框架页（总览页）

### 目标

主框架页不是技术清单页，而是**总流程 + 分支扩展关系页**。

### 必须保留

- User Query
- Query Planner
- Need Retrieval? 决策节点
- Modality Router
- 各 retrieval branch
- Candidate Merge & Rerank
- Evidence Verifier
- Answer Generator
- Reflection / Retry Loop

### 必须修改

1. **给左右与下方留更大空白**，避免分支线拥挤。
2. `Text / Image / Document / Audio / Video / Structured` 这些 branch 节点横向排布时要更均匀。
3. 从 `Modality Router` 到各 branch 的 plan 线：
   - 不要共线重叠；
   - 使用等间距下拉线。
4. `Candidate Merge & Rerank` 只接收 candidate 输入，不要让太多线交叉。
5. `Reflection / Retry Loop` 用一条独立颜色的反馈路径，避免和主流程混在一起。
6. 各分支节点内部只保留一句摘要：
   - `Text: BM25 / dense / hybrid`
   - `Document: page / chunk / element`
   - `Audio: ASR / speaker / timestamp`
   - 不要在总览页塞入太多技术细节。

### 总览页的原则

> 总览页只回答“系统怎么走”，不回答“每个分支的全部技术选项是什么”。

---

## 5.2 各 branch 页（Video / Audio / Document / Text / Image / Structured）

### 页面结构统一模板

每个 branch 页面统一采用以下结构：

#### 第 1 行：标题 + Scope
#### 第 2 行：6 个一级阶段大容器
#### 第 3 行：公共说明块
- 后续实验原则
- 中心接口说明
#### 第 4 行：Legend + Notes

### 每个一级阶段容器内部的拆分方式

#### A. Input / Data
通常 1 个卡片即可。

#### B. Processing / Tools
拆成 2~4 个技术类别卡，例如：

- OCR
- Parsing
- ASR
- Caption / VLM
- Speaker diarization
- Region extraction

每个技术类别卡下再放 1~3 个候选实现。

#### C. Index / Storage Options
这是当前最需要拆分的区域。

必须拆成多个子卡片，例如：

- Lexical Index
- Dense / Sparse Index
- Metadata Index
- Region / Element Index
- Segment / Timestamp Index

每个子卡片下再列具体实现，不再用 `/` 堆在一起。

#### D. Retrieval Strategy
拆成：

- 主检索路径
- 辅助过滤 / expansion
- rerank / fusion

#### E. Experiment Options
不要把所有实验项压成一段话，拆成：

- 对比实验
- Ablation
- Metrics

例如：

- `Baselines`
  - BM25 vs Dense vs Hybrid
- `Ablations`
  - with / without reranker
  - chunk size / overlap
- `Metrics`
  - Recall@K
  - MRR / nDCG
  - Evidence Recall

#### F. Output Evidence
拆成：

- 主证据字段
- 引用/定位字段
- 原始引用字段

---

## 6. 具体“拆块”示例（给 Codex 直接照着改）

以下示例最重要，因为它直接解决你指出的问题。

---

### 示例 1：Text Retrieval 页的 Index Options

#### 不要这样写

```text
BM25: Elasticsearch / OpenSearch
Dense/Sparse: BGE-M3 / Qwen3-Embedding / OpenAI
Hybrid: RRF / score fusion
```

#### 要改成这样

父容器：`Index / Storage Options`

子卡片 1：`Lexical Index`
- Elasticsearch
- OpenSearch

子卡片 2：`Dense / Sparse Embeddings`
- BGE-M3
- Qwen3-Embedding
- OpenAI Embeddings

子卡片 3：`Hybrid Fusion`
- RRF
- Score Fusion

如果空间仍不足：

- 把每个子卡片缩成“标题 + 2~3 个 bullet”；
- 或把 `Dense / Sparse Embeddings` 单独拆成两个子卡片：
  - Dense Embeddings
  - Sparse Embeddings

---

### 示例 2：Document Retrieval 页的 Parser Options

#### 不要这样写

```text
Docling + PaddleOCR
备选：minerU / Unstructured / Mistral OCR
layout parser、table/chart/formula parser
```

#### 要改成这样

父容器：`Processing / Tools`

子卡片 1：`Primary Parsing Stack`
- Docling
- PaddleOCR

子卡片 2：`Alternative Parsers`
- MinerU
- Unstructured
- Mistral OCR

子卡片 3：`Structure Extraction`
- Layout Parser
- Table Parser
- Chart / Formula Parser

---

### 示例 3：Image / Page Retrieval 页的 Preprocess & Tools

#### 不要这样写

```text
OCR: PaddleOCR / Mistral OCR
VLM caption: Qwen2.5-VL / InternVL3
region / patch extraction、bbox、质量检查
```

#### 要改成这样

父容器：`Processing / Tools`

子卡片 1：`OCR`
- PaddleOCR
- Mistral OCR

子卡片 2：`VLM Caption`
- Qwen2.5-VL
- InternVL3

子卡片 3：`Region Processing`
- Patch Extraction
- BBox Extraction
- Quality Check

---

### 示例 4：Audio Retrieval 页的 ASR & Segmentation

#### 不要这样写

```text
ASR: Whisper / FunASR / PaddleSpeech
speaker diarization
silence split, timestamp alignment
```

#### 要改成这样

父容器：`Processing / Tools`

子卡片 1：`ASR Engines`
- Whisper
- FunASR
- PaddleSpeech

子卡片 2：`Speaker Processing`
- Speaker Diarization

子卡片 3：`Segmentation`
- Silence Split
- Timestamp Alignment

---

### 示例 5：Experiment Matrix 拆分

#### 不要这样写

```text
BM25 vs dense vs hybrid
chunk size / overlap / section tree
with vs without reranker
指标: Recall@K / MRR / nDCG / Evidence Recall
```

#### 要改成这样

父容器：`Experiment Options`

子卡片 1：`Baselines`
- BM25
- Dense
- Hybrid

子卡片 2：`Ablations`
- Chunk Size / Overlap
- Section Tree
- With / Without Reranker

子卡片 3：`Metrics`
- Recall@K
- MRR
- nDCG
- Evidence Recall

---

## 7. 页面拆分策略（如果仍然太挤）

如果一个 branch 页面在按上述规则拆块之后仍然很拥挤，则改成 **1 + N** 的形式：

### Page A：Branch Overview
只保留：

- 6 个一级阶段
- 每个阶段 1~2 个摘要子卡片
- 公共说明块

### Page B：Technology Options
专门展开：

- Processing / Tools
- Index / Storage Options
- Retrieval Strategy

### Page C：Experiment Design
专门展开：

- Baselines
- Ablations
- Metrics
- Expected evidence schema

> 结论：如果页面开始出现“缩字体才能放下”的情况，就应该拆页，而不是继续压缩。

---

## 8. draw.io 层面必须执行的样式规范

### 8.1 统一样式

- 一级阶段容器：圆角矩形 / 带标题栏
- 子卡片：较小圆角矩形
- 数据存储：圆柱或数据库形状
- 决策节点：菱形（仅总览页需要）

### 8.2 颜色语义固定

沿用现有语义，但更严格：

- 蓝：输入 / 数据
- 紫：处理 / 规划
- 灰：索引 / 存储
- 青：召回 / 检索
- 橙：实验 / 融合 / 评估
- 绿：输出 / 证据
- 红：反馈 / retry / 风险提示

### 8.3 字体与字号

- 标题：较大，统一
- 一级容器标题：统一字号
- 子卡片正文：统一字号
- Notes / Legend：可略小，但不能小到难读

### 8.4 对齐规则

- 同一行容器顶对齐；
- 同类型子卡片等宽等高；
- 页面间相同区域的位置尽量一致。

---

## 9. 给 Codex 的直接执行要求（可以原样复制）

下面这段可以直接给 Codex：

---

请修改现有的 `multimodal_rag_retrieval_branch_experiments.drawio`，目标是提高可读性与分层结构，不要随意改变技术语义。

### 你必须完成的修改

1. **扩大每个页面的画布**，优先使用更大的横向页面，避免节点和连线拥挤。
2. **重构每个 branch 页面布局**，统一分为四个区域：
   - 标题区（Title + Scope）
   - 主流程区（6 个一级阶段）
   - 公共说明区（后续实验原则 + 中心接口说明）
   - 页脚区（Legend + Notes）
3. **一级阶段必须是父容器**，内部用多个子卡片表达内容，而不是纯文本大段堆叠。
4. **禁止把多个备选技术用斜杠 `/` 串联在同一个框里**。必须拆成多个小块或子卡片。
5. 对以下典型内容强制拆块：
   - `Elasticsearch / OpenSearch`
   - `BGE-M3 / Qwen3-Embedding / OpenAI`
   - `RRF / score fusion`
   - `PaddleOCR / Mistral OCR`
   - `Qwen2.5-VL / InternVL3`
   - `Whisper / FunASR / PaddleSpeech`
6. **主流程只连接一级阶段容器**，不要让大量子项之间跨页乱连。
7. **Experiment Matrix 必须拆成 Baselines / Ablations / Metrics 三类子卡片**。
8. 若某页仍然拥挤，则拆成：
   - branch overview page
   - technology options page
   - experiment design page
9. 总览页（主框架页）只表达系统流程与分支关系，不要塞过多技术细节。
10. 所有页面保持统一的颜色语义、标题风格、子卡片样式和对齐方式。

### 具体视觉目标

- 线不压字；
- 线不穿框；
- 框内不出现长串 `/` 连接的技术项；
- 每个框的内容能一眼看出是“阶段 / 类别 / 具体方案”中的哪一种；
- 同一类页面结构一致；
- 页面留白明显增加。

### 额外要求

- 优先保持内容完整性；
- 其次才是压缩页面数量；
- 如果必须取舍，优先“拆页”而不是“缩字”或“挤在一起”。

---

## 10. 验收清单（让 Codex 自检）

Codex 修改后，必须逐页检查：

### A. 画布与空间
- [ ] 是否存在明显的线与框重叠？
- [ ] 是否存在说明块挤压主流程区？
- [ ] 是否仍然需要缩小字体才能容纳内容？

### B. 结构层次
- [ ] 是否清楚区分了一级阶段容器和内部子卡片？
- [ ] 是否还存在“一个框里塞多个层次概念”的情况？

### C. 技术选型表达
- [ ] 是否还存在大量 `/` 串联的候选项？
- [ ] 候选技术是否已经拆成多个小块？
- [ ] 候选方案之间是否具备并列可比较性？

### D. 总览页
- [ ] 总览页是否只强调流程与分支？
- [ ] 是否避免在总览页塞过多技术细节？

### E. 一致性
- [ ] 各 branch 页面是否使用了统一模板？
- [ ] Legend / Notes / 公共说明块位置是否一致？
- [ ] 颜色与样式是否一致？

---

## 11. 最后一句话原则（最重要）

请让 Codex 记住：

> **架构图不是把信息都放上去，而是先决定“哪些信息在这一层该被看见”。**
>
> 对当前这组图来说，最重要的不是再增加技术名词，而是：
>
> 1. 把主流程层画清楚；
> 2. 把技术选型拆成多个可比较的小块；
> 3. 给连线和留白足够空间；
> 4. 空间不够就拆页，不要硬塞。

