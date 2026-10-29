<div align="center">

# SuperMemory

**给 Claude Code 和 Codex 共用的本地长期记忆。**

[![CI](https://github.com/Yuziki1227/supermemory/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuziki1227/supermemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![MCP](https://img.shields.io/badge/MCP-stdio-6E56CF)
![Storage](https://img.shields.io/badge/storage-SQLite%20FTS5-003B57)
![Local first](https://img.shields.io/badge/cloud-none%20by%20default-2E7D32)

[English](README.md) · 简体中文

</div>

SuperMemory 给编码助手一份能跨会话、跨上下文压缩的长期记忆。所有数据都放在一个
SQLite 文件里。检索是 BM25、向量、时近性、重要性、强化和pinned的混合打分，经
softmax 注意力层、MMR 多样性筛选，再按 token 预算截断。Claude Code 和 Codex 通过
MCP 读写同一份记忆。

```text
对话 ──► memory_remember ──► SQLite（分块 · FTS5 · 向量 · 版本链）
                                   │
新会话 ◄── memory_briefing ◄── 注意力加权检索 ◄── memory_recall
```

---

## 为什么需要它

编码助手每次会话都从零开始。`CLAUDE.md` 和 `AGENTS.md` 能放静态的项目规范，
却放不下工作中不断产生的决策、约束和半成品任务状态。上下文一旦被压缩，连这些
也会丢掉。结果是每隔几小时就要重新问一遍、重新踩一遍坑、重新争论一遍设计。

SuperMemory 把这条链路补上：

- 学到需要长期保留的信息时**立即** `memory_remember`；
- 下一次 **SessionStart** 自动收到按预算裁剪的交接简报；
- 需要细节时用 `memory_recall` 按需检索，不必把整库塞进上下文。

## 特性

- **本地优先**：一个 `~/.supermemory/memory.db`。默认配置不联网、不注册、不需要 API key。
- **两个 agent 共用一个大脑**：MCP stdio 服务器由 Claude Code 和 Codex 共享，其他 MCP 客户端也能接。
- **写入不会越积越乱**：精确去重、相似提示、不可变版本链、软删除、命名空间隔离、相关性门槛。
- **检索可解释、可调参**：六个特征加权 + softmax 注意力分布 + MMR 多样性。
- **检索质量可度量**：带标注的评测集计算 Recall@k、MRR、nDCG 并进入 CI；`--explain` 直接展示每条结果的打分来源。
- **中文优先分词**：SQLite 自带 tokenizer 检索不了中文，这里自建 CJK 单字 + 二元组索引，并拆开 `camelCase` / `snake_case` 标识符。
- **永不阻断会话**：钩子任何异常只写 stderr 并返回 0。记忆系统坏了，不影响正常写代码。

## 环境要求

| 要求 | 说明 |
|---|---|
| Python ≥ 3.10 | 测试套件在 3.14 上通过；避免使用 3.11+ 独占语法 |
| `numpy` ≥ 1.24 | 唯一的第三方运行时依赖 |
| 带 FTS5 的 SQLite | CPython 官方构建自带 |
| Claude Code 和/或 Codex | 可选，命令行可以独立使用 |

一键自检：

```bash
python -m supermemory doctor
```

## 快速开始

```bash
git clone https://github.com/Yuziki1227/supermemory.git
cd supermemory
python install.py            # 安装、注册 MCP、装钩子、自检
```

然后**重启 Claude Code / Codex**。第一次会话不会出现简报，因为数据库还是空的；
agent 写过几条记忆之后，下次会话就会自动带上。

其他安装模式：

```bash
python install.py --dry-run   # 只显示会改哪些文件
python install.py --dev       # 直接使用当前仓库，不复制
python install.py --uninstall # 移除配置，保留数据库
python install.py --purge     # 卸载并删除 ~/.supermemory
```

### 安装器会改什么

| 动作 | 位置 | 说明 |
|---|---|---|
| 复制程序 | `~/.supermemory/app/` | 之后仓库可以随意移动 |
| 注册 MCP（Claude Code） | `~/.claude.json` → `mcpServers.supermemory` | 用户级作用域 |
| 安装钩子（Claude Code） | `~/.claude/settings.json` → `hooks` | SessionStart、PreCompact、SessionEnd |
| 注册 MCP（Codex） | `~/.codex/config.toml` → `[mcp_servers.supermemory]` | |
| 写入使用说明 | `~/.claude/CLAUDE.md`（带标记块） | `--no-instructions` 可跳过 |

所有被修改的文件都会先备份到 `~/.supermemory/backups/`。合并是幂等的，重复运行
不会产生重复条目。

### 手动注册 MCP

不想运行安装器时，可以自己注册。

Claude Code（`~/.claude.json`，或项目级 `.mcp.json`）：

```json
{
  "mcpServers": {
    "supermemory": {
      "type": "stdio",
      "command": "python",
      "args": ["C:/Users/you/.supermemory/app/run.py", "mcp"],
      "env": { "PYTHONUTF8": "1" }
    }
  }
}
```

Codex（`~/.codex/config.toml`）：

```toml
[mcp_servers.supermemory]
command = "python"
args = ["C:/Users/you/.supermemory/app/run.py", "mcp"]
env = { PYTHONUTF8 = "1" }
startup_timeout_sec = 30
```

> Windows 下 `command` 指向 `python.exe`；`.cmd` 启动器只给人工命令行使用，安装器
> 会自动写好路径。

`pip install .` 也能安装命令行（`supermemory ...`），但 MCP 仍需手动注册；此时
服务器命令用 `python -m supermemory mcp`。

## 工作原理

### 1. 存储：分块、索引、版本链

长文本在进入 SQL 之前就被切块，优先级为：空行 → 行 → 句子 → 硬切。默认每块
260 tokens、重叠 48 tokens，因此每一行都是自包含、可独立检索的原子。

每行保存标题、正文、摘要、标签、来源、重要度、置信度、时间戳、向量、内容哈希
和版本指针。

**更新从不覆盖。** `memory_update` 写入新行，并把旧行标上 `superseded_by` 和
`valid_to`。默认检索只看当前版本，`include_superseded=true` 能看到完整历史。
这样"内容被改过"和"没人维护"不会混在一起。

**去重分两级：**

- 精确重复（同命名空间 + 同类型 + 同内容哈希）直接返回已有 id，不写新行；
- 相似但不完全相同（余弦 ≥ 0.90）**不自动合并**，只在结果里提示"可能是修订"。
  因为"X 用 Y"和"X 不用 Y"向量可能很近，自动合并会吃掉否定语义——这类系统最
  常见的死法就是静默覆盖。

### 2. 检索：注意力分布

每个候选记忆有六个特征：

| 特征 | 来源 | 默认权重 |
|---|---|---|
| `lexical` | FTS5 BM25，在候选集内归一化 | 0.36 |
| `vector` | 查询与记忆的余弦相似度 | 0.28 |
| `importance` | 写入时声明，乘置信度阻尼 | 0.13 |
| `recency` | 最后一次写入/召回后的指数衰减，半衰期 45 天 | 0.09 |
| `reinforcement` | `log(1 + 被召回次数)`，被用过且有用的记忆更容易再被选中 | 0.06 |
| `pinned` | 显式固定，永久加成 | 0.08 |

加权和是 `score`，然后：

```text
alpha_i = softmax(score_i / τ)      τ = 0.35
```

`alpha` 就是注意力分布：固定总量的概率质量按相关性分配给候选记忆，`τ` 控制
分布的锐利程度。这不是神经网络里的 attention head——本地跑推理不现实——而是
同一种数学形状的注意力加权，好处是完全可解释、可调参、零推理成本。

最终用 **MMR**（最大边际相关）挑选 top-k：每选一条，就对与已选内容高度相似的
候选打折扣，避免返回五条几乎一样的结果。再按 token 预算截断。

还有一个硬门槛：候选必须命中词法，或余弦 ≥ 0.15，否则淘汰。没有这道门槛，
仅靠"最近 + 重要"就能把无关记忆拖进结果——这是最简单的召回污染来源。

### 3. 中文为什么要自己分词

SQLite 的 `unicode61` 把一整句中文当成一个 token，`MATCH "检索"` 永远不命中。
SuperMemory 在写入和查询时使用同一个分词器：中文逐字 + 相邻二元组
（`检索层` → `检 索 层 检索 索层`），英文按 camelCase / snake_case 拆分
（`MemoryStore` → `memorystore memory store`）。二元组保住了短语，单字由 BM25
的 IDF 自动降权。

### 4. 嵌入

默认是**离线哈希嵌入**：词、标识符、中文 n-gram 用 blake2b 做特征哈希（1024
维），子线性词频加权，L2 归一化。没有下载、没有 API key、跨进程确定性——装完
即可用。

它和 BM25 有相关性，但补上了字符级相似度（拼写差异、词形变化、中英混写）。
想要真正的语义向量，可以切到任意 OpenAI 兼容的 `/v1/embeddings` 接口：

```bash
export SUPERMEMORY_EMBED_API_KEY=sk-...
python -m supermemory config --set embedder.provider=openai
python -m supermemory config --set embedder.model=text-embedding-3-small
python -m supermemory reembed
```

远程接口不可达时进程会**自动降级**到本地嵌入并在 stderr 告警——记忆服务不该
因为 key 过期就拒绝启动。

### 5. 命名空间

默认按当前项目目录名取命名空间（`SuperMemory/` → `supermemory`）。召回时同时
搜索该项目和 `global`；跨项目通用的知识用 `namespace="global"` 写入。
`namespace="*"` 搜索全部。

项目目录的解析顺序：

1. `SUPERMEMORY_NAMESPACE`（显式命名空间，`*` 表示全部）
2. `SUPERMEMORY_PROJECT_CWD`（钩子从 payload 写入）
3. `CLAUDE_PROJECT_DIR`（Claude Code 注入）
4. 进程当前工作目录

### 6. 会话交接

**SessionStart 钩子**注入简报：固定规则、进行中任务、关键决策、偏好与踩坑，
外加最近更新。它在所有会话来源上触发，**包括 `source=compact`**——这是上下文
被压缩后，唯一能合法把文本重新送到模型面前的时机。

**PreCompact 钩子不能注入上下文**（Claude Code 会拒绝它的 `hookSpecificOutput`），
所以它只往 stdout 打一行纯文本提醒。落盘仍然是 agent 自己的责任；真正的兜底是
压缩后的 SessionStart。

**SessionEnd 钩子**做维护：清理过期软删除、裁剪事件日志。

只靠检索不够——得在正确的时间点把记忆重新推到模型面前。

## MCP 工具

| 工具 | 用途 | 常用参数 |
|---|---|---|
| `memory_briefing` | 会话开始时调一次：项目状态 + 相关记忆，按预算裁剪 | `query`、`max_tokens`、`namespace` |
| `memory_remember` | 写入事实/决策/偏好/任务/坑，自动分块、去重、相似提示 | `content`、`kind`、`tags`、`importance`、`pinned`、`ttl_days` |
| `memory_recall` | 混合检索，返回 id、标题、摘要、分数的紧凑条目 | `query`、`limit`、`kinds`、`tags`、`max_tokens`、`explain`、`format` |
| `memory_get` | 按 id 取全文 | `ids` |
| `memory_update` | 修订，写新版本并保留历史 | `id`、`content`、`pinned` 等 |
| `memory_forget` | 默认软删除；`hard=true` 永久删除 | `ids`、`hard` |
| `memory_link` | 记忆之间建立 `refines` / `contradicts` / `depends-on` 等关系 | `src`、`dst`、`relation`、`weight` |
| `memory_stats` | 条数、类型分布、库大小、嵌入模型 | `format` |
| `memory_export` | 整库导出为 JSON | `path` |

工具描述本身写入了使用时机，模型不需要额外提示也知道什么时候该调。
`initialize` 还会返回一段 server instructions，说明"什么时候写、什么时候读、
怎么修订"。

## 命令行

`python -m supermemory ...` 与 `supermemory ...` 等价；后者需要把
`~/.supermemory/bin` 加入 `PATH`（安装器会创建）。

```bash
# 安装与健康检查
supermemory doctor
supermemory init
supermemory stats

# 写入与读取
supermemory remember "内容" -t "标题" -k decision --tag 架构 -i 0.9
supermemory recall "检索怎么做的"
supermemory recall "检索怎么做的" --explain
supermemory get M12
supermemory list --kind decision
supermemory briefing --query "实现 MCP"
supermemory bench bench/cases.jsonl       # 检索回归评测

# 修订与维护
supermemory update M12 --content "修正后的内容"
supermemory forget M12                     # 软删除
supermemory link M12 M7 --relation refines
supermemory maintenance

# 迁移与配置
supermemory export backup.json
supermemory import backup.json
supermemory reembed --provider openai
supermemory config
supermemory config --set ranking.temperature=0.5

# 集成入口
supermemory mcp                            # 运行 stdio MCP 服务器
supermemory hook session-start             # 钩子入口（从 stdin 读 JSON）
```

## 检索质量

两个工具让排序改动有据可依。

### 为什么这条排第一？

`supermemory recall "查询" --explain` 会在每条结果下追加打分分解：

```text
M3 [decision 0.63] 检索层
    检索层使用 BM25 与向量混合打分。
    ↳ lex 1.00 · vec 0.42 · imp 0.46 · rec 1.00 · rein 0.00 · pin 0 · score 0.626 · attn 1.0000
```

MCP 侧对应参数是 `memory_recall` 的 `"explain": true`。这几个特征就是
`ranking.py` 里加权求和的那六个，不存在"模型暗中决定"的部分。

### 评测集

`bench/cases.jsonl` 是自包含的语料 + 标注查询。每次运行会建一个临时数据库，
输出 Recall@k、MRR、nDCG@k、负例表现和令牌成本：

```bash
python -m supermemory bench bench/cases.jsonl
python -m supermemory bench bench/cases.jsonl --json
python -m supermemory bench bench/cases.jsonl --fail-under-recall 0.85 --fail-under-mrr 0.75
```

CI 会跑这套评测，排序改动导致召回退化会直接让构建失败。每修一个检索 bug
就补一条 case；文件格式见 [bench/README.md](bench/README.md)。评测默认使用内置
默认配置，想测调过的配置就加 `--config ~/.supermemory/config.json`。

## 配置

`~/.supermemory/config.json` 是可选的，未设置的项使用默认值。常用的几个：

```json
{
  "ranking": {
    "weights": { "lexical": 0.36, "vector": 0.28, "importance": 0.13,
                 "recency": 0.09, "reinforcement": 0.06, "pinned": 0.08 },
    "temperature": 0.35,
    "half_life_days": 45,
    "mmr_lambda": 0.22
  },
  "recall": { "default_limit": 8, "max_tokens": 1400, "vector_floor": 0.15 },
  "briefing": { "max_tokens": 900 },
  "store": { "chunk_max_tokens": 260, "namespace_mode": "cwd" }
}
```

- `temperature` 调小 → 注意力更集中在最高分项；调大 → 分布更均衡。
- `mmr_lambda` 调大 → 更强调多样性；`0` 关闭 MMR。
- `half_life_days` 调小 → 更偏向近期记忆。
- `namespace_mode`：`cwd`（默认）/ `fixed`（固定用 `fixed_namespace`）/ `none`
  （全部写到 `global`）。

环境变量：

| 变量 | 作用 |
|---|---|
| `SUPERMEMORY_HOME` | 数据目录（默认 `~/.supermemory`） |
| `SUPERMEMORY_NAMESPACE` | 强制指定命名空间；`*` 搜索全部 |
| `SUPERMEMORY_PROJECT_CWD` | 用于推导命名空间的项目目录 |
| `SUPERMEMORY_EMBED_PROVIDER` | `local`、`openai` 或 `none` |
| `SUPERMEMORY_EMBED_MODEL` / `_BASE_URL` / `_DIM` | 远程嵌入配置 |
| `SUPERMEMORY_EMBED_API_KEY` | 远程嵌入的 API key |
| `CLAUDE_PROJECT_DIR` | 推导命名空间时自动读取 |

## 目录结构

```text
supermemory/
  config.py       默认配置、路径、命名空间解析
  text.py         分词、token 估算、摘要、分块
  embed.py        本地哈希嵌入 / OpenAI 兼容嵌入
  db.py           schema、FTS5、连接
  ranking.py      注意力打分 + MMR
  core.py         MemoryStore：写入、检索、版本、简报、导出
  render.py       面向 token 的输出渲染
  mcp_server.py   零依赖 MCP stdio 服务器
  cli.py          命令行
  hooks.py        Claude Code 钩子
  bench.py        带标注的检索评测与指标
install.py        安装器
run.py            统一启动入口（MCP / 钩子 / CLI）
bench/            评测用例与格式说明
tests/            32 个测试，含真实 MCP 子进程握手
```

## 设计取舍

- **不做自动摘要压缩**：摘要只取首句，不做 LLM 生成。省掉一次推理调用，也避免
  摘要漂移导致"记忆内容和摘要不符"。需要语义压缩时，让 agent 自己写一条
  `kind=decision` 的记忆。
- **不自动合并相似记忆**：见上文，否定语义会被吃掉。
- **不自动衰减删除**：时近性只在检索时参与打分，不写回数据库。记忆不会因为你
  一阵子没用就消失。
- **钩子永不阻断**：任何异常都打印到 stderr 并返回 0。记忆系统坏了，不影响你
  正常写代码。
- **MCP 手写而非引 SDK**：整个服务器只依赖 CPython 标准库 + numpy，目标是任何
  装了 Python 的机器都能直接跑。
- **双语文档、单一代码库**：目前命令行输出为中文；每个功能在两份 README 中都有
  说明。

## 常见问题

**重启后没看到简报？**
数据库是空的（还没有任何记忆），或钩子没生效。跑 `python -m supermemory stats`
确认条数，手工验证钩子输出：

```bash
echo '{"source":"startup","cwd":"/path/to/project"}' | python -m supermemory hook session-start
```

**命名空间不对？**
命名空间按项目目录名取。stdio MCP 进程的 cwd 不保证是项目根目录，所以优先读
`SUPERMEMORY_PROJECT_CWD` → `CLAUDE_PROJECT_DIR` → 最后才是进程 cwd。仍不对时，
用 `SUPERMEMORY_NAMESPACE` 环境变量或在 MCP 配置的 `env` 里显式指定。

**中文检索不准？**
先确认 `doctor` 显示 FTS5 ✓。再检查是否是单字查询——分词器在存在二元组时会
丢弃单字。

**两个 agent 同时写会冲突吗？**
不会。数据库使用 WAL、30 秒 busy timeout，写入用 `BEGIN IMMEDIATE`。MCP 服务器、
钩子和命令行可以安全共享同一个文件。

**换电脑怎么迁移？**
复制 `~/.supermemory/memory.db`（连同 `-wal` 文件），或 `supermemory export` 后
在新机器 `supermemory import`。

**数据库变大？**
`supermemory maintenance` 清理过期软删除并压缩 FTS 索引。

**数据会联网吗？**
默认配置不会。只有启用 OpenAI 兼容嵌入时，记忆文本才会发送到你配置的接口。

## 开发

```bash
python -m unittest discover -s tests -v
python -m supermemory bench bench/cases.jsonl
python install.py --dev       # 直接用仓库，不复制
```

没有第三方测试框架，没有构建步骤。测试通过临时 `SUPERMEMORY_HOME` 隔离，不会
碰你的真实记忆库。CI 在 Linux（Python 3.10–3.14）和 Windows 上同时运行单元测试
与检索评测。

架构、不可破坏的约定和贡献流程见 [AGENTS.md](AGENTS.md)；快速上手见
[CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE) © 2026 Yuziki1227
