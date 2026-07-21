# 领域文档

说明工程技能在探索代码库时应如何使用本仓库的领域文档。

## 探索前先阅读

- 根目录的 **`CONTEXT.md`**；或
- 根目录的 **`CONTEXT-MAP.md`**（如存在），它指向每个上下文对应的 `CONTEXT.md`；只读取与当前主题相关的文档
- **`docs/adr/`**，阅读与即将处理区域相关的 ADR。在多上下文仓库中，还需检查 `src/<context>/docs/adr/` 中的上下文级决策

若这些文件不存在，**静默继续**。不要报告其缺失，也不要主动建议创建它们。`/domain-modeling` 技能（可通过 `/grill-with-docs` 和 `/improve-codebase-architecture` 使用）会在术语或决策确实得到明确时按需创建它们。

## 文件结构

单一上下文仓库（大多数仓库）：

```
/
|- CONTEXT.md
|- docs/adr/
|  |- 0001-event-sourced-orders.md
|  `- 0002-postgres-for-write-model.md
`- src/
```

多上下文仓库（根目录存在 `CONTEXT-MAP.md`）：

```
/
|- CONTEXT-MAP.md
|- docs/adr/                          <- 系统级决策
`- src/
   |- ordering/
   |  |- CONTEXT.md
   |  `- docs/adr/                    <- 上下文级决策
   `- billing/
      |- CONTEXT.md
      `- docs/adr/
```

## 使用词汇表中的术语

当输出命名领域概念时（例如事项标题、重构建议、假设或测试名称），使用 `CONTEXT.md` 中定义的术语。不要改用词汇表明确避免的同义词。

若所需概念尚未出现在词汇表中，这意味着要么正在引入项目未使用的语言（应重新考虑），要么存在真实缺口（为 `/domain-modeling` 记录该缺口）。

## 标出 ADR 冲突

若输出与既有 ADR 冲突，应明确说明，而不是静默覆盖：

> _与 ADR-0007（基于事件的订单）冲突，但由于……值得重新讨论。_
