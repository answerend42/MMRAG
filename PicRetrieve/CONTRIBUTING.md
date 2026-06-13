# 贡献指南

欢迎贡献！以下是参与 PicRetrieve 开发的一些基本约定。

## 环境

- Python >= 3.11
- 使用 `uv` 管理依赖：`uv sync --extra dev`
- 运行测试：`uv run pytest`

## 代码规范

- 缩进：4 空格
- 编码：UTF-8
- 注释：中文 Doxygen 风格（`/** ... */` 或 `#! @brief`）
- 类型标注：全项目启用 `from __future__ import annotations`

提交前请确保：

```bash
uv run ruff check
uv run pytest
```

## 分支管理

- `main`：稳定分支，只能通过 PR 合并
- 功能开发和修复请在独立分支上进行，合并前保持测试通过
