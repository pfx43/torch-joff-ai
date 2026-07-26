## Agent skills

### Issue tracker

事项和规格以 Markdown 文件形式存放在 `.scratch/` 下（目录按需创建；工单属于工作产物，创建后应纳入版本控制，不要加入 .gitignore）。详见 `docs/agents/事项追踪器.md`。

### Triage labels

使用五个默认的规范化分诊标签。详见 `docs/agents/分诊标签.md`。

### Domain docs

使用单一上下文文档布局。详见 `docs/agents/领域文档.md`。

### Skill 目录约定

`.claude/skills/` 与 `.agents/skills/` 是同一批技能面向不同工具的两份拷贝（Claude Code 读前者，其他 agent 工具读后者），当前内容逐字节相同。修改任一侧后必须同步另一侧，勿让两份拷贝分叉。

当前技能：`ieee-english-paper-polish`（IEEE 英文稿写作与润色，自建）、`paper-triage` / `expand-references` / `trace-citations`（Semantic Scholar 文献检索，来自 github.com/zongmin-yu/semantic-scholar-skills，自包含 vendored runtime，只依赖 Python 3.10+ 标准库）。

## 文献检索与 Zotero 工作流（强制）

**铁律：任何进入论文的文献，必须先入 Zotero，再被引用。** 严禁凭模型记忆写参考文献——`.bib` 条目只能来自 Zotero 导出或权威数据库返回的元数据。

流程：`/paper-triage` 或 `/expand-references` 或 `/trace-citations` 检索 → 相关论文立刻用 `zotero_add_by_doi` 等写入 Zotero 并归集合打标签 → 精读经 Zotero MCP（全文与 PDF 标注）→ `.bib` 从 Zotero 导出，投稿前核验条目真实性。**未入库的论文不得进入正文或 `.bib`。**

MCP 服务 `zotero` 配在 `~/.codex/config.toml`（Codex）与 `.mcp.json`（Claude Code）。读走本地 API（`ZOTERO_LOCAL=true`，需 Zotero 客户端运行并允许其他应用通信），写入走 Web API，需用户级环境变量 `ZOTERO_API_KEY` / `ZOTERO_LIBRARY_ID`；密钥只放系统环境变量，绝不写进仓库。

完整约定（集合划分、限流处理、与写作阶段的衔接）见 `CLAUDE.md` 的"文献检索与 Zotero 工作流"一节。
