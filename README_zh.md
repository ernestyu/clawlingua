# ClawLingua

ClawLingua 是一个 Python CLI 工具，用来把真实语料（播客字幕、长文等）
自动转换成 Anki 填空卡（cloze deck，`.apkg`）。

核心能力：

- 从 URL 或本地 `.txt` / `.md` 文件读取内容；
- 清洗文本、按语境切块；
- 使用 OpenAI-compatible LLM 生成带上下文的 cloze 句子；
- 使用单独的小模型（small LLM）负责翻译/意译；
- 使用 `edge_tts` 为每张卡生成音频；
- 通过 `genanki` 输出完整的 Anki 牌组。

本文档介绍 **V1 CLI** 的当前行为。英文版见 [`README.md`](./README.md)。

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
python -m clawlingua.cli init
```

`init` 会做几件事：

- 若当前目录存在 `.env.example` 而没有 `.env`，则复制一份；
- 检查以下文件是否存在：
  - `./prompts/cloze_contextual.json`
  - `./prompts/translate_rewrite.json`
  - `./templates/anki_cloze_default.json`

你可以按需要修改 `.env`（或参考 `ENV_EXAMPLE.md`），填入自己的 LLM 地址、API Key、TTS 配置等。

---

## 2. 配置说明（ENV_EXAMPLE）

所有配置都通过环境变量完成。仓库内的
[`ENV_EXAMPLE.md`](./ENV_EXAMPLE.md) 展示了完整示例。下面只列出核心部分。

### 2.1 主 LLM（生成 cloze）

```env
CLAWLINGUA_LLM_PROVIDER=openai_compatible
CLAWLINGUA_LLM_BASE_URL=http://127.0.0.1:8000/v1
CLAWLINGUA_LLM_API_KEY=YOUR_API_KEY
CLAWLINGUA_LLM_MODEL=qwen3-30b
CLAWLINGUA_LLM_TIMEOUT_SECONDS=120
CLAWLINGUA_LLM_MAX_RETRIES=3
CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS=2.0
# 连续 LLM 请求之间的基础 sleep 秒数；实际会在 [N, 3N] 之间随机，0 表示不主动 sleep
CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS=0
CLAWLINGUA_LLM_TEMPERATURE=0.2
```

主 LLM 负责：

- 从 chunk 中挑选值得学习的表达；
- 生成带 cloze 标记的 `Text` 字段。

要求它是 OpenAI-compatible 的 `/chat/completions` 接口即可。

### 2.2 HTTP / 抓取

```env
CLAWLINGUA_HTTP_TIMEOUT_SECONDS=30
CLAWLINGUA_HTTP_USER_AGENT=ClawLingua/0.1
CLAWLINGUA_HTTP_VERIFY_SSL=true
```

用于 URL 抓取与 doctor 检查。

### 2.3 切块（Chunking）

```env
# 按字符数切块，内部通过句子边界软切。
CLAWLINGUA_CHUNK_MAX_CHARS=1800
CLAWLINGUA_CHUNK_MIN_CHARS=120
CLAWLINGUA_CHUNK_OVERLAP_SENTENCES=1
```

行为：

- 先按段落拆分，再把太短的段落合并到 `CLAWLINGUA_CHUNK_MIN_CHARS`；
- 对合并后的段落：
  - 以 `CHUNK_MAX_CHARS` 控制块大小；
  - 从当前句子向后扩展，直到接近上限；
  - 不在句子中间硬截断；
  - 相邻块之间保留 `CHUNK_OVERLAP_SENTENCES` 句的重叠。

### 2.4 Cloze 粒度控制

```env
# 每条 cloze 文本允许的最大句子数（同时写入 prompt 与校验规则）。
CLAWLINGUA_CLOZE_MAX_SENTENCES=3

# 每条 cloze 文本的最小字符数（太短的句子会被丢弃，0 表示不限制）。
CLAWLINGUA_CLOZE_MIN_CHARS=200

# 难度：beginner | intermediate | advanced
# 命令行 --difficulty 优先级更高。
CLAWLINGUA_CLOZE_DIFFICULTY=intermediate

# 每个 chunk 去重后最多保留多少条 cloze 候选（空/0 表示不限制）。
CLAWLINGUA_CLOZE_MAX_PER_CHUNK=4

# LLM 一次处理多少个 chunk；1 表示逐块调用，建议 1–4。
CLAWLINGUA_LLM_CHUNK_BATCH_SIZE=1
```

解释：

- LLM 决定每个 chunk 实际返回多少条候选（0–N）；
- `CLOZE_MAX_PER_CHUNK` 只是“防爆上限”，在去重之后按 chunk 截断：
  - 设为 4 → 每个 chunk 最多 4 张卡；
  - 设为 0/留空 → 不限制；
- `CLOZE_MIN_CHARS` 可以避免“一句两句太短”的 cloze 通过校验。

### 2.5 翻译 LLM（small LLM）

```env
# 翻译用的小模型；若不配置则回退到主 LLM。
CLAWLINGUA_TRANSLATE_LLM_BASE_URL=
CLAWLINGUA_TRANSLATE_LLM_API_KEY=
CLAWLINGUA_TRANSLATE_LLM_MODEL=
CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE=
```

- 建议主 LLM 用较贵的模型（如 Gemini），翻译 LLM 用便宜模型（如本地 Qwen）；
- 若 `TRANSLATE_LLM_BASE_URL`/`MODEL` 为空，则翻译与 cloze 共用主 LLM。

### 2.6 Prompt、模板与输出

```env
CLAWLINGUA_PROMPT_CLOZE=./prompts/cloze_contextual.json
CLAWLINGUA_PROMPT_TRANSLATE=./prompts/translate_rewrite.json
CLAWLINGUA_ANKI_TEMPLATE=./templates/anki_cloze_default.json

CLAWLINGUA_OUTPUT_DIR=./runs
CLAWLINGUA_LOG_LEVEL=INFO
CLAWLINGUA_SAVE_INTERMEDIATE=true
CLAWLINGUA_DEFAULT_DECK_NAME=ClawLingua Default Deck
```

### 2.7 TTS（edge_tts）

```env
CLAWLINGUA_TTS_PROVIDER=edge_tts
CLAWLINGUA_TTS_OUTPUT_FORMAT=mp3
CLAWLINGUA_TTS_RATE=+0%
CLAWLINGUA_TTS_VOLUME=+0%
CLAWLINGUA_TTS_RANDOM_SEED=

CLAWLINGUA_TTS_EDGE_EN_VOICES=en-US-AnaNeural,en-US-AndrewNeural,en-GB-SoniaNeural
CLAWLINGUA_TTS_EDGE_ZH_VOICES=zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,zh-CN-liaoning-XiaobeiNeural
CLAWLINGUA_TTS_EDGE_JA_VOICES=ja-JP-NanamiNeural,ja-JP-KeitaNeural,ja-JP-AoiNeural
```

根据 `source_lang` 在对应 voice 列表中随机挑选一个 voice，为每张卡的
`Original` 合成音频。

---

## 3. Prompt 行为

### 3.1 Cloze Prompt (`prompts/cloze_contextual.json`)

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

### 3.2 翻译 Prompt (`prompts/translate_rewrite.json`)

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

CLI 入口在 `src/clawlingua/cli.py`，使用 Typer 实现。

### 4.1 `init`

```bash
python -m clawlingua.cli init
```

- 创建 `.env`（若不存在）；
- 确保 prompt 与模板存在；
- 可选创建输出目录。

### 4.2 `doctor`

```bash
python -m clawlingua.cli doctor --env-file .env
```

- 检查依赖（edge_tts / genanki / httpx / typer）；
- 校验基础配置（路径、prompt/template）；
- 检查 LLM（主 + translate）配置与连通性；
- 检查 cloze 控制参数（max_sentences / min_chars / difficulty / max_per_chunk）；
- 检查 TTS voice 列表。

### 4.3 `build deck`

```bash
python -m clawlingua.cli build deck INPUT \
  --input-type auto|url|file \
  --source-lang en \
  --target-lang zh \
  --env-file .env \
  --output deck.apkg \
  --deck-name "My Cloze Deck" \
  --max-chars 1500 \
  --max-notes 200 \
  --temperature 0.2 \
  --difficulty beginner|intermediate|advanced \
  --save-intermediate \
  --continue-on-error \
  --debug
```

- `INPUT`：可以是 URL 或本地文件路径；
- `--difficulty`：覆盖 env 中的 `CLOZE_DIFFICULTY`；
- `--max-chars`：覆盖当前 run 的 `CHUNK_MAX_CHARS`；
- `--max-notes`：对整套牌组做全局上限；
- `--save-intermediate`：将中间结果保存到 `OUTPUT_DIR/runs/<run_id>`；
- `--continue-on-error`：遇到单条失败时跳过、记录错误，而不是直接退出；
- `--debug`：出错时抛出完整 traceback，便于调试。

### 4.4 `prompt validate`

```bash
python -m clawlingua.cli prompt validate ./prompts/cloze_contextual.json
```

对 prompt 文件做 schema 校验（字段是否齐全、类型是否正确）。

### 4.5 `config show` / `config validate`

```bash
python -m clawlingua.cli config show --env-file .env
python -m clawlingua.cli config validate --env-file .env
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

## 6. 典型使用流程

1. 配好 `.env`：LLM、chunking、cloze 控制、TTS。
2. 运行 doctor 确认配置无误：

   ```bash
   python -m clawlingua.cli doctor --env-file .env
   ```

3. 用播客字幕生成牌组：

   ```bash
   python -m clawlingua.cli build deck ./podcast_transcript.md \
     --source-lang en --target-lang zh --env-file .env \
     --difficulty intermediate --max-chars 1500 \
     --save-intermediate --continue-on-error
   ```

4. 在 Anki 中导入 `.apkg`，开始复习。
5. 如需调试，查看 `./runs/<run_id>` 下的中间文件：
   - `chunks.jsonl`：切块结果；
   - `text_candidates.raw.jsonl` / `validated.jsonl`：cloze 候选；
   - `translations.jsonl`：翻译文本；
   - `cards.final.jsonl`：最终卡片数据。

---

## 7. 后续可改进点

- cloze 编号：当出现多个 `c1` 时，需要自动重排为 `c1/c2/c3`；
- cloze 格式：进一步收紧到
  `{{cN::<b>phrase</b>}}(translation)` 的统一样式，需要结合实际 LLM 输出
  调整 prompt 与 validator；
- 测试：当前 `main` 上测试目录已移除，未来扩展功能时建议补回针对
  chunking / validator / CLI 的最小测试集。

如需了解英文版说明，请阅读 [`README.md`](./README.md)。
