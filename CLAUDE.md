@AGENTS.md
@1-1-assignment/AGENTS.md
@1-2-assignment/AGENTS.md

# Claude Code

本仓库规范分两层，**单一事实源，不要另建副本**：

- `AGENTS.md`（根）— 跨作业通用规则
- `1-1-assignment/AGENTS.md` — 1-1 作业专属（命令、文件地图、该作业的坑）
- `1-2-assignment/AGENTS.md` — 1-2 作业专属（命令、文件地图、竞态两态结论、变异自检要求）

上面几行导入已把三者载入。**只改对应的 `AGENTS.md`，不要在本文件里重复维护规则。**
新增作业时，在此追加一行 `@<新目录>/AGENTS.md`。

## Claude 专属补充

- 若规则未出现在上下文中，运行 `/context` 检查 **Memory files** 是否包含 `CLAUDE.md`。
- 本仓库未安装 ruff / mypy，不要声称跑过 lint 或类型检查。
- 具体的探针脚本与实测参考值见 `1-1-assignment/AGENTS.md` 第 2、3 节，直接照抄执行。
