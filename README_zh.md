# ClawLearn

ClawLearn 是一个 Python CLI 工具，用来把真实语料（播客字幕、长文等）
自动转换成 Anki 填空卡（cloze deck，`.apkg`）。

核心能力：

- 从本地 `.txt` / `.md` / `.epub` 文件读取内容；
- 清洗文本、按语境切块；
- 使用 OpenAI-compatible LLM **先抽取 context window + phrase candidates**（结构化 JSON）；
- 再由代码把 phrase span 注入到原句中，生成标准 cloze：`{{cN::<b>...</b>}}(hint)`；
- 使用单独的小模型（small LLM）负责翻译/意译；
- 使用 `edge_tts` 为每张卡生成音频；
- 通过 `genanki` 输出完整的 Anki 牌组。

本文档介绍当前 **V2 CLI** 的行为（含 lingua/textbook 两套 pipeline，以及“先抽取 phrase 再由代码生成 cloze”的两阶段流程）。英文版见 [`README.md`](./README.md)。

---

## 1. 安装与初始化

### 1.1 创建虚拟环境

```bash
python -m venv .venv
# Windows (PowerShell)
. .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 1.2 初始化项目

```bash
python -m clawlearn.cli init
```

`init` 会做几件事：

- 若当前目录存在 `.env.example` 而没有 `.env`，则复制一份；
- 检查以下文件是否存在：
  - `./prompts/cloze_contextual.json`
  - `./prompts/cloze_textbook_examples.json`
  - `./prompts/translate_rewrite.json`
  - `./templates/anki_cloze_default.json`

你可以按需要修改 `.env`（或参考 `ENV_EXAMPLE.md`），填入自己的 LLM 地址、API Key、TTS 配置等。

---

## 2. 配置说明（ENV_EXAMPLE）

所有配置都通过环境变量完成。仓库内的
[`ENV_EXAMPLE.md`](./ENV_EXAMPLE.md) 展示了完整示例。下面只列出核心部分。

### 2.1 主 LLM（生成 cloze）

```env
CLAWLEARN_LLM_PROVIDER=openai_compatible
CLAWLEARN_LLM_BASE_URL=http://127.0.0.1:8000/v1
CLAWLEARN_LLM_API_KEY=YOUR_API_KEY
CLAWLEARN_LLM_MODEL=qwen3-30b
CLAWLEARN_LLM_TIMEOUT_SECONDS=120
CLAWLEARN_LLM_MAX_RETRIES=3
CLAWLEARN_LLM_RETRY_BACKOFF_SECONDS=2.0
# 连续 LLM 请求之间的基础 sleep 秒数；实际会在 [N, 3N] 之间随机，0 表示不主动 sleep
CLAWLEARN_LLM_REQUEST_SLEEP_SECONDS=0
CLAWLEARN_LLM_TEMPERATURE=0.2
```

主 LLM 负责：

- 从 chunk 中选取 2–3 句连续上下文（context window）；
- 抽取 2–6 词长度的 phrase candidates（严格 substring）；
- **不直接输出 cloze 标记**（cloze/hint 由后续代码统一生成）。

要求它是 OpenAI-compatible 的 `/chat/completions` 接口即可。

### 2.2 输入清洗

```env
CLAWLEARN_INGEST_SHORT_LINE_MAX_WORDS=3
```

用于预处理输入文本：

- 会过滤过短、无学习价值的孤立短行；
- 设为 `0` 可关闭该过滤；
- `.md` 会先转纯文本；
- `.epub` 会解包并抽取章节文本；

### 2.3 切块（Chunking）

```env
# 按字符数切块，内部通过句子边界软切。
CLAWLEARN_CHUNK_MAX_CHARS=1800
CLAWLEARN_CHUNK_MIN_CHARS=120
CLAWLEARN_CHUNK_OVERLAP_SENTENCES=1
```

行为：

- 先按段落拆分，再把太短的段落合并到 `CLAWLEARN_CHUNK_MIN_CHARS`；
- 对合并后的段落：
  - 以 `CHUNK_MAX_CHARS` 控制块大小；
  - 从当前句子向后扩展，直到接近上限；
  - 不在句子中间硬截断；
  - 相邻块之间保留 `CHUNK_OVERLAP_SENTENCES` 句的重叠。

### 2.4 Cloze 粒度控制

```env
# 每条 cloze 文本允许的最大句子数（同时写入 prompt 与校验规则）。
CLAWLEARN_CLOZE_MAX_SENTENCES=3

# 每条 cloze 文本的最小字符数（太短的句子会被丢弃，0 表示不限制）。
CLAWLEARN_CLOZE_MIN_CHARS=200

# 难度：beginner | intermediate | advanced
# 命令行 --difficulty 优先级更高。
CLAWLEARN_CLOZE_DIFFICULTY=intermediate

# 每个 chunk 去重后最多保留多少条 cloze 候选（空/0 表示不限制）。
CLAWLEARN_CLOZE_MAX_PER_CHUNK=4

# LLM 一次处理多少个 chunk；1 表示逐块调用，建议 1–4。
CLAWLEARN_LLM_CHUNK_BATCH_SIZE=1
```

解释：

- LLM 决定每个 chunk 实际返回多少条候选（0–N）；
- `CLOZE_MAX_PER_CHUNK` 只是“防爆上限”，在去重之后按 chunk 截断：
  - 设为 4 → 每个 chunk 最多 4 张卡；
  - 设为 0/留空 → 不限制；
- `CLOZE_MIN_CHARS` 可以避免“一句两句太短”的 cloze 通过校验。

当 `content_profile=textbook_examples` 时，如果 `CLOZE_MIN_CHARS > 120`
且未通过 `--cloze-min-chars` 显式覆盖，程序会拒绝执行，避免把教材例句全部过滤掉。

### 2.5 翻译 LLM（small LLM）

```env
# 翻译用的小模型；若不配置则回退到主 LLM。
CLAWLEARN_TRANSLATE_LLM_BASE_URL=
CLAWLEARN_TRANSLATE_LLM_API_KEY=
CLAWLEARN_TRANSLATE_LLM_MODEL=
CLAWLEARN_TRANSLATE_LLM_TEMPERATURE=
```

- 建议主 LLM 用较贵的模型（如 Gemini），翻译 LLM 用便宜模型（如本地 Qwen）；
- 若 `TRANSLATE_LLM_BASE_URL`/`MODEL` 为空，则翻译与 cloze 共用主 LLM。

### 2.6 Prompt、模板与输出

```env
CLAWLEARN_CONTENT_PROFILE=general
CLAWLEARN_PROMPT_CLOZE=./prompts/cloze_contextual.json
CLAWLEARN_PROMPT_CLOZE_TEXTBOOK=./prompts/cloze_textbook_examples.json
CLAWLEARN_PROMPT_TRANSLATE=./prompts/translate_rewrite.json
CLAWLEARN_PROMPT_LANG=zh
CLAWLEARN_ANKI_TEMPLATE=./templates/anki_cloze_default.json

# 中间运行数据（JSONL、media 快照）
CLAWLEARN_OUTPUT_DIR=./runs
# 最终导出的牌组（未显式指定 --output 时）
CLAWLEARN_EXPORT_DIR=./outputs
CLAWLEARN_LOG_DIR=./logs
CLAWLEARN_LOG_LEVEL=INFO
CLAWLEARN_SAVE_INTERMEDIATE=true
CLAWLEARN_DEFAULT_DECK_NAME=ClawLearn Default Deck
```

- `CLAWLEARN_CONTENT_PROFILE=general` 使用 `cloze_contextual.json`。
- `CLAWLEARN_CONTENT_PROFILE=textbook_examples` 使用 `cloze_textbook_examples.json`。
- `CLAWLEARN_PROMPT_LANG` 控制多语言 prompt 使用哪一套文案（`en` 或 `zh`），
  可通过命令行 `--prompt-lang` 覆盖。

### 2.7 TTS（edge_tts）

```env
CLAWLEARN_TTS_PROVIDER=edge_tts
CLAWLEARN_TTS_OUTPUT_FORMAT=mp3
CLAWLEARN_TTS_RATE=+0%
CLAWLEARN_TTS_VOLUME=+0%
CLAWLEARN_TTS_RANDOM_SEED=

CLAWLEARN_TTS_EDGE_EN_VOICES=en-US-AnaNeural,en-US-AndrewNeural,en-GB-SoniaNeural
CLAWLEARN_TTS_EDGE_ZH_VOICES=zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,zh-CN-liaoning-XiaobeiNeural
CLAWLEARN_TTS_EDGE_JA_VOICES=ja-JP-NanamiNeural,ja-JP-KeitaNeural,ja-JP-AoiNeural
```

根据 `source_lang` 在对应 voice 列表中随机挑选一个 voice，为每张卡的
`Original` 合成音频。

---

## 3. Prompt 行为

### 3.1 Cloze Prompt (`prompts/cloze_contextual.json`)

该 prompt 支持多语言文案：

- 旧格式：
  - `"system_prompt": "..."`
- 多语言格式：
  - `"system_prompt": { "en": "...", "zh": "..." }`

运行时会根据 `CLAWLEARN_PROMPT_LANG` / `--prompt-lang` 选择对应语言。

- 输入占位符：
  - `source_lang` / `target_lang`
  - `document_title` / `source_url`
  - `difficulty` / `cloze_max_sentences`
  - 合并后的 `chunk_text`（可能包含多个 `chunk_id=...` 区块）
- 输出（必须是 JSON 数组）：

  ```json
  {
    "chunk_id": "chunk_0001_abcd12",
    "text": "The more {{c1::<b>whimsical explanation</b>}}(异想天开的解释) is that maybe RL training makes the models a little too {{c2::<b>single-minded</b>}}(钻牛角尖) and narrowly focused.",
    "original": "The more whimsical explanation is that maybe RL training makes the models a little too single-minded and narrowly focused.",
    "target_phrases": ["whimsical explanation", "single-minded"],
    "note_hint": "可选简短提示"
  }
  ```

- 规则（通过 prompt + validator 约束）：
  1. `text` 至少包含一个 cloze，语法为 `{{cN::...}}`：
     - N 从 1 开始：`c1`, `c2`, `c3`...
     - cloze 内部使用 `<b>...</b>` 包裹被挖的词组；
     - cloze 后面紧跟括号 `(中文解释)`；
  2. 同一句可以有多个 cloze，例如：

     ```text
     The more {{c1::<b>whimsical explanation</b>}}(异想天开的解释) ... {{c2::<b>single-minded</b>}}(钻牛角尖) ...
     ```

  3. `original` 不含 cloze 标记和 HTML；
  4. 每个 chunk 可以返回 0–4 条高质量候选（受 `CLOZE_MAX_PER_CHUNK` 上限控制）；
  5. 每条候选可以包含 1–3 个 cloze。

- 代码侧：
  - 支持 `{c1::...}` 单大括号形式，并在 validator 中规范化为 `{{c1::...}}`；
  - 若完全没有 cloze 但有 `target_phrases`，尝试从 `target_phrases` 自动注入一个 `{{c1::...}}`；
  - 对句子长度/句子数/target_phrases 做基础校验。

> 未来可能需要对多个 `c1` 做自动重编号（按出现顺序变成 c1/c2/c3）。

### 3.2 教材例句 Prompt (`prompts/cloze_textbook_examples.json`)

该 prompt 用于 `--content-profile textbook_examples`，适合“词条 + 释义 + 例句”结构。
核心策略是：

- 忽略词条标题行；
- 忽略词典式释义行；
- 只从自然例句里抽取可学习表达并挖空。

### 3.3 翻译 Prompt (`prompts/translate_rewrite.json`)

翻译 prompt 同样可以采用多语言格式（参见 cloze prompt 的说明），生效逻辑
由 `CLAWLEARN_PROMPT_LANG` / `--prompt-lang` 控制。

- 输入：`source_lang`、`target_lang`、`document_title`、`source_url`、`text_original`、`chunk_text`；
- 输出：JSON 数组，每项包含一个 `translation` 字段：

  ```json
  {
    "translation": "自然且地道的目标语言翻译"
  }
  ```

- 约束：
  - 不要加 "翻译:" 这类前缀；
  - 可以适度意译，但要自然、口语化；
  - 不允许使用 Markdown `**`（HTML `<b>` 可以）。

---

## 4. CLI 命令

CLI 入口在 `src/clawlearn/cli.py`，使用 Typer 实现。

### 4.1 `init`

```bash
python -m clawlearn.cli init
```

- 创建 `.env`（若不存在）；
- 确保 prompt 与模板存在；
- 可选创建输出目录。

### 4.2 `doctor`

```bash
python -m clawlearn.cli doctor --env-file .env
```

- 检查依赖（edge_tts / genanki / httpx / typer）；
- 校验基础配置（路径、prompt/template）；
- 检查 LLM（主 + translate）配置与连通性；
- 检查 cloze 控制参数（max_sentences / min_chars / difficulty / max_per_chunk / profile）；
- 检查 TTS voice 列表。

### 4.3 `lingua build deck`

```bash
python -m clawlearn.cli lingua build deck INPUT \
  --source-lang en \
  --target-lang zh \
  --material-profile prose_article|transcript_dialogue|textbook_examples \
  --learning-mode lingua_expression|lingua_reading \
  --env-file .env \
  --output deck.apkg \
  --deck-name "My Cloze Deck" \
  --max-chars 1500 \
  --cloze-min-chars 60 \
  --max-notes 200 \
  --temperature 0.2 \
  --difficulty beginner|intermediate|advanced \
  --prompt-lang en|zh \
  --extract-prompt ./prompts/cloze_transcript_advanced.json \
  --explain-prompt ./prompts/translate_rewrite.json \
  --lingua-annotate \
  --save-intermediate \
  --continue-on-error \
  --verbose \
  --debug
```

说明：
- `--content-profile` 仍保留，但已被标记为 deprecated（等价于 `--material-profile`）。
- `material_profile + learning_mode + difficulty` 会共同决定默认使用哪个 extraction prompt。
- 若 extraction prompt 的输出 schema 是 `phrase_candidates_*`，则 LLM 输出不包含 cloze 标记；cloze 由后续代码统一注入生成。

### 4.4 `textbook build deck`

```bash
python -m clawlearn.cli textbook build deck INPUT \
  --source-lang en \
  --target-lang zh \
  --learning-mode textbook_focus|textbook_review \
  --max-chars 1500 \
  --max-notes 200 \
  --save-intermediate \
  --continue-on-error \
  --verbose \
  --debug
```

- `INPUT`：本地文件路径（支持 `.txt` / `.md` / `.epub`）；
- `--content-profile`：切换内容策略（`general` 或 `textbook_examples`）；
- `--difficulty`：覆盖 env 中的 `CLOZE_DIFFICULTY`；
- `--prompt-lang`：覆盖 `CLAWLEARN_PROMPT_LANG`，用于选择多语言 prompt 文案；
- `--max-chars`：覆盖当前 run 的 `CHUNK_MAX_CHARS`；
- `--cloze-min-chars`：覆盖当前 run 的 `CLOZE_MIN_CHARS`；
- `textbook_examples` 模式下，若 env 的 `CLOZE_MIN_CHARS > 120` 且未 CLI 覆盖，会直接拒绝执行；
- `--max-notes`：对整套牌组做全局上限；
- `--save-intermediate`：将中间结果保存到 `OUTPUT_DIR/<run_id>`；
- 未显式提供 `--output` 时，最终 `.apkg` 会写入
  `CLAWLEARN_EXPORT_DIR/<run_id>/output.apkg`。
- `--continue-on-error`：遇到单条失败时跳过、记录错误，而不是直接退出；
- `--debug`：出错时抛出完整 traceback，便于调试。
- 默认牌组名使用输入文件名（不含扩展名）；可用 `--deck-name` 覆盖。

### 4.4 `prompt validate`

```bash
python -m clawlearn.cli prompt validate ./prompts/cloze_contextual.json
python -m clawlearn.cli prompt validate ./prompts/cloze_textbook_examples.json
```

对 prompt 文件做 schema 校验（字段是否齐全、类型是否正确）。

### 4.5 `config show` / `config validate`

```bash
python -m clawlearn.cli config show --env-file .env
python -m clawlearn.cli config validate --env-file .env
```

- `config show`：打印最终解析后的配置 JSON；
- `config validate`：仅做配置校验，不生成牌组。

---

## 5. 输出结构与 Anki 字段

默认模板下，每张卡至少有这些字段：

- **Text**：带 `{{cN::...}}` 的 cloze 句子，可包含 HTML `<b>` 和括号内翻译；
- **Original**：原始句子（不含 cloze 与 HTML）；
- **Translation**：对 `Original` 的翻译/意译；
- **Note**：元信息（标题、URL、chunk_id、target_phrases 等）；
- **Audio**：`Original` 的音频文件，通过 `[sound:xxx.mp3]` 字段引用。

具体字段名和模板结构在 `templates/anki_cloze_default.json` 中定义，可按需定制。

---

## 6. 迁移说明（V1 -> V2）

如果你有旧的脚本/配置，下面是关键变化：

- **命令结构**：
  - 推荐新命令：`python -m clawlearn.cli lingua build deck ...`
  - `python -m clawlearn.cli build deck ...` 仍保留，但属于 **deprecated alias**。

- **Profile 选择**：
  - 推荐使用 `--material-profile` / `CLAWLEARN_MATERIAL_PROFILE`。
  - `--content-profile` / `CLAWLEARN_CONTENT_PROFILE` 为兼容旧配置的 deprecated alias。
  - 旧值 `content_profile=general` 会映射为 `material_profile=prose_article`。

- **learning_mode**：
  - `CLAWLEARN_LEARNING_MODE` 默认是 `lingua_expression`。
  - 所有支持的 mode 以 `src/clawlearn/constants.py` 为准。

- **两阶段抽取（重要）**：
  - 当前默认 prompts 输出 schema 为 `phrase_candidates_*`：LLM 只输出 context + phrase JSON，不包含 cloze。
  - Cloze 标记（`{{cN::<b>...</b>}}(hint)`）由后续代码统一注入生成。
  - 旧 prompts 可能输出 `cloze_cards_*`（由 LLM 直接生成 cloze 文本），该路径仍作为兼容方案保留。

## 7. 典型使用流程

1. 配好 `.env`：LLM、chunking、cloze 控制、TTS。
2. 运行 doctor 确认配置无误：

   ```bash
   python -m clawlearn.cli doctor --env-file .env
   ```

3. 用播客字幕生成牌组：

   ```bash
   python -m clawlearn.cli lingua build deck ./podcast_transcript.md \
     --source-lang en --target-lang zh --env-file .env \
     --material-profile transcript_dialogue --learning-mode lingua_expression \
     --difficulty intermediate --max-chars 1500 \
     --save-intermediate --continue-on-error --verbose
   ```

4. 在 Anki 中导入 `.apkg`，开始复习。
5. 如需调试，查看 `./runs/<run_id>` 下的中间文件：
   - `chunks.jsonl`：切块结果；
   - `text_candidates.raw.jsonl` / `validated.jsonl`：cloze 候选；
   - `translations.jsonl`：翻译文本；
   - `cards.final.jsonl`：最终卡片数据。

---

## 7. 可选 Web 管理界面

除了纯命令行，ClawLearn 还提供一个基于 Gradio 的本地 Web 界面，方便
在浏览器里上传文件、调整参数。该界面是可选的，不会改变 CLI 的行为，
只有在你主动启动时才会运行。

### 7.1 安装

在项目根目录下安装 `web` extra 依赖：

```bash
pip install .[web]
```

### 7.2 启动 Web UI

```bash
clawlearn-web
# 或
python -m clawlearn_web.app
```

默认监听 `127.0.0.1:7860`，在浏览器中访问：
<http://127.0.0.1:7860>

Web 界面包含三个 Tab：

- **Run**：上传 `.txt` / `.md` / `.epub` 文件，选择源语言/目标语言、
  内容 profile（`general` / `textbook_examples`）、难度等级，并按需设置
  本次运行的 override（最大卡片数、input char limit、cloze_min_chars、
  chunk_max_chars、temperature 等）。后端调用与 CLI 相同的
  `run_build_deck` pipeline，将中间数据写入
  `CLAWLEARN_OUTPUT_DIR/<run_id>`，最终牌组写入
  `CLAWLEARN_EXPORT_DIR/<run_id>/output.apkg`。
- **Config**：`.env` 配置编辑器，用于修改常见的
  `CLAWLEARN_*` 变量（例如 LLM 地址/模型、chunk/cloze 默认值、
  prompt 语言、输出/日志目录、默认牌组名称、TTS 等）。点击 Save 时会
  写入新的 `.env`，并通过 `clawlearn.config.validate_base_config` +
  `validate_runtime_config` 做一轮校验，失败会自动回滚；Load defaults
  则从 `ENV_EXAMPLE.md` 载入默认值但不直接写盘。Config 页还提供对主
  LLM 与翻译 LLM 的「列出模型 / 测试 /models 连通性」功能。
- **Prompt**：查看和编辑 prompt JSON 文件（`cloze_contextual.json`、
  `cloze_textbook_examples.json`、`translate_rewrite.json`），支持切换
  文件、查看多语言 prompt 内容、校验 schema、在 JSON 合法时保存，
  并自动生成备份。

对于 OpenClaw 的 skill 或自动化场景，仍然推荐直接使用 CLI，Web UI
主要面向「临时跑一组文件」的人机交互需求。

## 8. 后续可改进点

- cloze 编号：当出现多个 `c1` 时，需要自动重排为 `c1/c2/c3`；
- cloze 格式：进一步收紧到
  `{{cN::<b>phrase</b>}}(translation)` 的统一样式，需要结合实际 LLM 输出
  调整 prompt 与 validator；
- 测试：当前 `main` 上测试目录已移除，未来扩展功能时建议补回针对
  chunking / validator / CLI 的最小测试集。

如需了解英文版说明，请阅读 [`README.md`](./README.md)。
