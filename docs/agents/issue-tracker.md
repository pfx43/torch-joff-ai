# 事项追踪器：本地 Markdown

本仓库的事项和规格（也可称为 PRD）以 Markdown 文件形式存放在 `.scratch/` 中。

## 约定

- 每个功能使用一个目录：`.scratch/<feature-slug>/`
- 规格文件为：`.scratch/<feature-slug>/spec.md`
- 实现事项每个工单一个文件，位于 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`；从 `01` 开始编号，不能将所有工单合并到单一文件中
- 分诊状态记录在各事项文件开头附近的 `Status:` 行中；角色字符串见 `triage-labels.md`
- 评论和对话历史追加到文件末尾的 `## Comments` 标题下

## 当技能要求“发布到事项追踪器”时

在 `.scratch/<feature-slug>/` 下创建新文件；如有需要，同时创建该目录。

## 当技能要求“获取相关工单”时

读取所引用路径的文件。用户通常会直接提供文件路径或事项编号。

## 路径探索操作

供 `/wayfinder` 使用。**地图**是一个文件，每个工单对应一个**子文件**。

- **地图**：`.scratch/<effort>/map.md`，记录 Notes、Decisions-so-far 和 Fog 正文
- **子工单**：`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 开始编号，正文包含待解决的问题。开头附近的 `Type:` 行记录工单类型（`research`、`prototype`、`grilling` 或 `task`），`Status:` 行记录 `claimed` 或 `resolved`
- **阻塞关系**：开头附近的 `Blocked by: NN, NN` 行。该行引用的所有文件均为 `resolved` 时，工单才解除阻塞
- **前沿工单**：扫描 `.scratch/<effort>/issues/`，选择开放、未阻塞且未认领的工单；编号最小者优先
- **认领**：开始工作前，将 `Status:` 设为 `claimed` 并保存
- **解决**：将答案追加到 `## Answer` 标题下，将 `Status:` 设为 `resolved`，再将上下文指针（摘要与链接）追加到 `map.md` 的 Decisions-so-far 中
