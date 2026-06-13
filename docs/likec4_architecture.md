# LikeC4 架构图维护

MMRAG 使用 LikeC4 维护架构即代码。原因很简单：Agent 项目经常同时涉及前端、API、编排图、模型服务、召回模块和数据资产，如果只靠口头说明或临时截图，很快会和代码脱节。

## 架构源码

主文件：

```text
architecture/likec4/mmrag.c4
```

配置文件：

```text
architecture/likec4/likec4.config.json
```

导出产物：

```text
docs/likec4/model.json
docs/likec4/png/
docs/likec4/site/
```

## 当前 C4 视图

| 视图 | 目的 |
| --- | --- |
| `index` | C1 系统上下文，展示用户、MMRAG、本地 LLM 和 PicRetrieve。 |
| `c2Containers` | C2 容器图，展示 FastAPI、LangGraph、注册表和本地数据。 |
| `c3WebComponents` | C3 Web 组件图，展示浏览器界面和 Agent API / SSE。 |
| `c3AgentComponents` | C3 Agent 组件图，展示真实 LangGraph 节点。 |
| `c3RegistryComponents` | C3 注册表组件图，展示召回模块扩展点。 |
| `c4QueryFlow` | C4 动态图，展示一次查询的端到端调用顺序。 |

## 本地预览

```bash
npm run arch:dev
```

打开：

[http://127.0.0.1:5173/](http://127.0.0.1:5173/)

## 校验

```bash
npm run arch:validate
```

任何修改 `architecture/likec4/mmrag.c4` 的提交都应该通过这个校验。

## 格式化

```bash
npm run arch:format
```

如果 LikeC4 CLI 调整了格式，以工具输出为准。

## 导出 PNG

```bash
npm run arch:export:png
```

PNG 会写入：

```text
docs/likec4/png/
```

这些图会提交到仓库，方便 GitHub 直接展示。

## 导出 JSON

```bash
npm run arch:export:json
```

JSON 会写入：

```text
docs/likec4/model.json
```

它适合后续接入自动文档、架构审查或额外可视化工具。

## 导出静态站点

```bash
npm run arch:build
```

静态站点会写入：

```text
docs/likec4/site/
```

如果未来启用 GitHub Pages，可以直接使用该目录作为发布产物。

## 维护原则

- C4 图只表达真实存在或明确规划的边界，不画空泛概念。
- 元素名称优先使用简体中文，代码标识保留真实节点名。
- C1 和 C2 解释系统边界，C3 解释组件责任，动态图解释一次查询如何流动。
- 增加新召回模块时，先更新 `召回模块注册表` 视图，再更新查询动态图。
- 不在架构图里记录临时实验参数；这些内容应该进入文档或 issue。

## 与 LangGraph 原生图的关系

LikeC4 负责解释系统结构和团队沟通边界。LangGraph 原生图负责展示真实执行节点和控制流。

导出 LangGraph 图：

```bash
npm run graph:export
```

输出位置：

```text
docs/langgraph/picretrieve_graph.mmd
docs/langgraph/picretrieve_graph.png
```
