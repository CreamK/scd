# SCD

SCD（Similar Code Detector）是一个基于 AI 的代码相似度检测工具，用来比较两个代码仓库在目录结构、文件职责和函数实现上的相似性。

它适合用于代码复用审查、疑似重复实现排查、迁移项目对比、外包交付验收，或在大型仓库之间快速定位可能存在相似逻辑的代码。

## 功能特点

- 扫描两个仓库中的源代码文件，并自动忽略常见构建产物、依赖目录和测试数据。
- 使用 LLM 为文件和目录生成摘要，再匹配语义上相近的目录。
- 对匹配目录中的文件进行函数级相似度分析。
- 从数据结构、函数签名、算法逻辑、命名习惯、协议一致性等维度给出综合评分。
- 支持 Markdown 和 JSON 报告输出。
- 支持 OpenAI-compatible API，可接入 OpenAI、OneAPI、NewAPI、LiteLLM、vLLM、Ollama 等兼容网关。
- 支持断点缓存和中间产物输出，便于长任务恢复和结果排查。

## 安装

需要 Python 3.11 或更高版本。

```bash
git clone <this-repo-url>
cd SCD
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装完成后可以使用 `scd` 命令：

```bash
scd --help
scd compare --help
```

## 配置

SCD 会读取项目根目录下的 `.scd.env`。可以从示例文件复制一份：

```bash
cp .scd.env.example .scd.env
```

最小配置：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

如果使用兼容网关，将 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 改为对应服务的地址和模型名即可。

可选配置：

```env
RPS=3.0
MATCH_BATCH_SIZE=40
DIR_CONFIDENCE=high
USE_JSON_MODE=false
PARALLEL_TOOL_CALLS=false
```

其中：

- `RPS`：限制每秒请求数，避免触发 API 限流。
- `MATCH_BATCH_SIZE`：目录匹配阶段每批传给模型的目录数量。
- `DIR_CONFIDENCE`：只有置信度 `>=` 该等级（`high|medium|low`）的目录匹配会进入函数比较阶段，默认 `high`；低于该等级的目录对仍会出现在报告的目录匹配部分。
- `USE_JSON_MODE`：是否强制使用 `response_format=json_object`。
- `PARALLEL_TOOL_CALLS`：是否允许并行 tool calls。

这些能力在不同兼容网关上的支持程度不同；默认保持保守配置，客户端会在部分不支持的场景下自动降级。

## 快速开始

比较两个本地仓库：

```bash
scd compare /path/to/repo-a /path/to/repo-b
```

默认结果分两处写入：最终报告统一放到顶层 `output/` 目录下（按仓库名前缀命名，避免多对比较互相覆盖），中间产物与缓存放到 `<repo_a>_<repo_b>_output/`（目录名取自两个仓库的目录名，例如比较 `foo` 与 `bar` 会得到 `foo_bar_output/`）：

```text
output/
  foo_bar_report.md
  foo_bar_report.json

foo_bar_output/
  dir_summaries.json
  compared_pairs.txt
  pair_cache.json
  .scd_cache/
    repo_a/
      file_summaries.jsonl
      dir_summaries.jsonl
    repo_b/
      file_summaries.jsonl
      dir_summaries.jsonl
```

默认会同时生成 Markdown 报告和 JSON 对比报告，不需要指定输出格式。

指定输出目录：

```bash
scd compare /path/to/repo-a /path/to/repo-b --output-dir reports/run-001
```

只做目录级匹配，不进入函数比较：

```bash
scd compare /path/to/repo-a /path/to/repo-b --shallow
```

只扫描指定语言：

```bash
scd compare /path/to/repo-a /path/to/repo-b --lang py,ts,tsx
```

提高相似度阈值，只保留更相似的结果：

```bash
scd compare /path/to/repo-a /path/to/repo-b --threshold 60
```

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `repo_a` | 必填 | 第一个待比较仓库路径 |
| `repo_b` | 必填 | 第二个待比较仓库路径 |
| `-o, --output` | `output/<repo_a>_<repo_b>_report.md` | 指定 Markdown 报告文件路径，JSON 报告使用同名 `.json` 路径 |
| `--output-dir` | `<repo_a>_<repo_b>_output` | 中间产物与缓存目录（不影响最终报告位置） |
| `-r, --rps` | `3.0` | AI API 每秒最大请求数 |
| `-t, --threshold` | `20` | 最低综合相似度分数，范围 0-100 |
| `-m, --model` | `gpt-4o-mini` | 使用的 LLM 模型名 |
| `--api-key` | 环境变量 | API key，也可通过 `OPENAI_API_KEY` 设置 |
| `--base-url` | 环境变量 | OpenAI-compatible API 地址 |
| `--lang` | 不限制 | 逗号分隔的语言过滤器，例如 `py,ts` |
| `--shallow` | 关闭 | 只做目录匹配，不做函数级比较 |
| `--match-batch-size` | `40` | 目录匹配阶段的批大小 |
| `--max-in-flight` | `8` | 同时进行中的 LLM 请求上限 |
| `--json-mode` / `--no-json-mode` | 自动/关闭 | 是否启用 JSON mode |
| `--parallel-tool-calls` / `--no-parallel-tool-calls` | 自动/关闭 | 是否启用并行 tool calls |
| `-v, --verbose` | 关闭 | 输出详细日志和异常堆栈 |

## 输出说明

Markdown 报告默认为 `output/<repo_a>_<repo_b>_report.md`，主要包含：

- Overview：两个仓库的文件数量、AI 调用次数、相似函数对数量。
- Similarity Distribution：高、中、低、极低相似度的分布。
- Directory Matches：AI 判断相近的目录对及原因。
- Similar Functions：按综合评分排序的相似函数详情，包含文件路径、行号和对应源码片段。

JSON 对比报告默认为 `output/<repo_a>_<repo_b>_report.json`，每个相似函数对一条记录，包含第一个仓库文件路径、第二个仓库文件路径、函数前后 20 行左右对比高亮 HTML、AI 判断理由、相似严重级别和两文件内容 hash。

函数相似度偏向判断“实现是否像”，不是只判断功能是否相同。只有当数据结构、代码形态、变量/命名使用和控制流程都能在代码层面对齐时，才会进入相似函数结果；如果只是高层功能类似、协议类似或解决同一个问题，但字段、变量名、数据流和写法明显不同，会被过滤掉。

函数相似度会按以下维度评分：

- Data Structure：函数里使用的数据结构、字段名、对象/字典键、集合形态是否相近。
- Function Signature：函数名、参数名、参数顺序、类型和返回值是否相近。
- Algorithm Logic：语句顺序、分支循环、关键操作、变量读写和调用链是否能按代码块对齐。
- Naming Convention：变量、常量、辅助函数等具体标识符是否相近，并且是否以相似角色被使用。
- Protocol Conformance：协议、接口或行为约定是否相近；这是辅助证据，不能单独让功能相似的代码被判为相似。

阶段 3 还会对核心维度做本地过滤：Data Structure 和 Algorithm Logic 都必须达到最低门槛；任一项过低时，即使综合分达到阈值，也不会输出为相似函数。

中间产物说明：

- `dir_summaries.json`：两个仓库的目录摘要，便于检查目录匹配依据。
- `.scd_cache/repo_a/` 和 `.scd_cache/repo_b/`：文件摘要和目录摘要缓存，不会写入被比较仓库。
- `compared_pairs.txt`：实际进入函数比较的文件对。
- `pair_cache.json`：函数比较缓存，用于长任务恢复。
- `report.md` / `report.json`：最终报告。

## 工作流程

SCD 的比较流程分为四个阶段：

1. 扫描仓库：收集支持的源代码文件，并应用默认忽略规则。
2. 生成摘要：先为文件生成摘要，再聚合为目录摘要。
3. 匹配目录：用目录摘要判断两个仓库中职责相近的目录，按 `DIR_CONFIDENCE` 过滤后再进入函数比较；未过滤掉的低置信度目录仍会保留在报告里供参考。
4. 比较函数：在匹配目录内构建文件对，进行函数级相似度分析并生成报告。

默认支持的源码类型包括 Python、TypeScript、JavaScript、Go、Java、Rust、C/C++、C#、Ruby、PHP、Swift、Kotlin、Scala、Vue、Svelte 等。

默认会忽略 `.git`、`node_modules`、`vendor`、`build`、`dist`、`target`、虚拟环境、缓存目录、测试目录和常见 lock 文件。

## 项目结构

```text
scd/
  ai/          LLM 客户端和提示词
  pipeline/    扫描后各阶段的编排、摘要、目录匹配和函数比较
  reporter/    Markdown / JSON 报告生成
  scanner/     仓库扫描和忽略规则
  cli.py       命令行入口
  config.py    配置和默认规则
  models.py    核心数据模型
```

## 本地开发

安装开发环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

运行 CLI：

```bash
python -m scd compare /path/to/repo-a /path/to/repo-b -v
```

也可以直接使用安装后的命令：

```bash
scd compare /path/to/repo-a /path/to/repo-b -v
```

## 使用建议

- 首次运行建议使用较低的 `RPS` 和默认 `--max-in-flight`，确认网关稳定后再提高并发。
- 大仓库可以先用 `--shallow` 查看目录匹配是否合理，再运行完整函数比较。
- 使用 `--lang` 缩小扫描范围，可以显著减少 AI 调用量。
- 如果兼容网关不支持 JSON mode 或 parallel tool calls，保持默认关闭即可。

## 注意事项

SCD 的结果依赖 LLM 判断，适合辅助定位相似代码，不应作为唯一结论。对于高相似度结果，建议结合报告中的文件路径、行号、评分维度和分析文本进行人工复核。
