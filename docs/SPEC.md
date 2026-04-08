# ClawLingua 项目详细 SPEC（V1）

## 1. 项目定位

`clawlingua` 是一个基于 Python 的命令行工具，用来把互联网上的原始内容加工成**可直接导入 Anki 的语言学习牌组**。

它的输入可以是：

* 网页链接
* 本地文本文件

程序处理流程为：

1. 抓取或读取原始内容
2. 提取正文
3. 清洗和标准化文本
4. 将文本切分为适合大模型处理的块
5. 调用大模型分别完成：

   * 语境挖空卡片生成
   * 翻译与意译生成
6. 将结果整理为 Anki Note
7. 为 Original 字段生成 TTS 音频
8. 导出为一个完整的 `.apkg` Anki 牌组

---

# 2. 项目目标

V1 只做一件事：

**输入原始互联网内容，输出一个可用的 Anki Cloze 牌组。**

不做网页 UI，不做数据库，不做账号系统，不做多种输出格式，不做在线同步。

---

# 3. 输出牌组的固定字段

V1 的最终输出只有一个 Anki 牌组。
牌组中的每张卡片都必须有以下字段：

* `Text`
* `Original`
* `Translation`
* `Note`
* `Audio`

字段定义如下。

## 3.1 Text

带挖空的卡片正面内容。

要求：

* 使用 Anki Cloze 语法
* 内容最多 3 句
* 来自源语言
* 用于正面复习
* 使用提示词文件：`./prompts/cloze_contextual.json`

示例：

```text
When people talk about AI, I have {{c1::<b>stuck to that</b>}}(坚持那个观点) definition forever. It {{c2::<b>turns out</b>}}(事实证明) that this is harder than we thought.
```

---

## 3.2 Original

对应的原始纯文本。

要求：

* 不带任何挖空标记
* 不带 HTML
* 不带格式修饰
* 必须是纯文本
* 用于后续 TTS 音频生成
* 通常与 Text 对应同一段内容，但为无修饰版本

---

## 3.3 Translation

面向语言学习的翻译与意译结果。

要求：

* 使用提示词文件：`./prompts/translate_rewrite.json`
* 是目标语言内容
* 允许补全逻辑、调整语序、适度意译
* 可读性要高
* 与 Text 的学习点尽量对齐
* 可以使用 HTML `<b>...</b>` 做强调
* 不允许 Markdown `**`

---

## 3.4 Note

备用字段。

V1 建议存放以下内容之一或组合：

* 目标短语列表
* 来源标题
* 来源 URL
* 段落编号
* 源语言 / 目标语言
* 处理批次信息

建议默认格式：

```text
phrases: stuck to that | turns out
title: Example Podcast
source: https://example.com/episode
chunk: 12
source_lang: en
target_lang: zh
```

---

## 3.5 Audio

用于存放 Anki 音频字段内容。

要求：

* 音频基于 `Original` 生成
* 默认采用 `edge_tts`
* 存储在导出的 `.apkg` 中
* 字段中保存 Anki 音频引用格式，例如：

```text
[sound:audio_000123.mp3]
```

---

# 4. 多语言支持要求

本项目不是只服务英语，也不是只服务英语到中文。

因此设计必须天然支持：

* 任意源语言 `source_lang`
* 任意目标语言 `target_lang`

要求：

1. 配置项必须显式包含 `source_lang` 和 `target_lang`
2. Prompt 中不得写死 English / Chinese
3. 数据结构中必须保存 `source_lang` 和 `target_lang`
4. TTS 配置必须可按语言切换 voice 列表
5. Anki 导出逻辑不得依赖具体语言

---

# 5. V1 非目标

V1 不做以下内容：

* 图形界面
* Web 服务
* 原始音视频转录
* OCR
* 多种输出文件格式并行支持
* 在线增量更新牌组
* 自动同步到 AnkiConnect
* 向量数据库
* 复杂语义去重
* 多模型路由系统

---

# 6. 一级命令与二级命令设计

一级命令固定为：

```bash
clawlingua
```

V1 二级命令设计如下：

```bash
clawlingua init
clawlingua doctor
clawlingua build deck <input>
clawlingua prompt validate <path>
clawlingua config show
clawlingua config validate
```

其中：

* `build deck` 是核心命令
* 其他命令用于初始化、检查和调试

---

# 7. 命令详细设计

## 7.1 `clawlingua init`

用途：

* 初始化项目配置文件
* 生成 `.env.example`
* 检查 prompts 和 templates 是否存在
* 可选创建输出目录

示例：

```bash
clawlingua init
```

---

## 7.2 `clawlingua doctor`

用途：

* 检查依赖是否齐全
* 检查 `.env` 是否可读
* 检查模型接口是否可连接
* 检查 prompt JSON 是否合法
* 检查模板 JSON 是否合法
* 检查 `edge_tts` 是否可用
* 检查输出目录是否可写

示例：

```bash
clawlingua doctor
clawlingua doctor --env-file .env
```

---

## 7.3 `clawlingua build deck <input>`

用途：

从输入内容构建一个完整的 `.apkg` 牌组。

`<input>` 可以是：

* URL
* 本地 `.txt`
* 本地 `.md`

示例：

```bash
clawlingua build deck ./podcast.md --source-lang en --target-lang zh
clawlingua build deck "https://example.com/article" --input-type url --source-lang en --target-lang zh
```

核心参数：

* `--input-type {auto,url,file}`
* `--source-lang LANG`
* `--target-lang LANG`
* `--env-file PATH`
* `--output PATH`
* `--deck-name TEXT`
* `--max-chars INT`
* `--max-sentences INT`
* `--max-notes INT`
* `--temperature FLOAT`
* `--save-intermediate`
* `--continue-on-error`
* `--verbose`
* `--debug`

---

## 7.4 `clawlingua prompt validate <path>`

用途：

校验 prompt JSON 文件是否合法。

示例：

```bash
clawlingua prompt validate ./prompts/cloze_contextual.json
clawlingua prompt validate ./prompts/translate_rewrite.json
```

---

## 7.5 `clawlingua config show`

用途：

打印当前生效配置，敏感信息脱敏。

---

## 7.6 `clawlingua config validate`

用途：

检查当前配置是否完整可运行。

---

# 8. 帮助信息要求

所有命令都必须提供完整的 `--help`，至少包括：

1. 该命令做什么
2. 输入是什么
3. 输出是什么
4. 常用参数
5. 示例命令
6. 常见失败场景提示

例如：

```bash
clawlingua build deck --help
```

至少应说明：

* 支持 URL 和文件输入
* 最终输出 `.apkg`
* `Text` 用 `./prompts/cloze_contextual.json`
* `Translation` 用 `./prompts/translate_rewrite.json`
* 模板使用 `./templates/anki_cloze_default.json`

---

# 9. 项目目录树

下面是 V1 推荐的项目代码目录树。

```text
clawlingua/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .env.example
├─ prompts/
│  ├─ cloze_contextual.json
│  └─ translate_rewrite.json
├─ templates/
│  └─ anki_cloze_default.json
├─ src/
│  └─ clawlingua/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ exit_codes.py
│     ├─ errors.py
│     ├─ logger.py
│     ├─ constants.py
│     ├─ config.py
│     ├─ helptext.py
│     ├─ runtime.py
│     ├─ models/
│     │  ├─ __init__.py
│     │  ├─ document.py
│     │  ├─ chunk.py
│     │  ├─ card.py
│     │  ├─ prompt_schema.py
│     │  └─ template_schema.py
│     ├─ ingest/
│     │  ├─ __init__.py
│     │  ├─ url_fetcher.py
│     │  ├─ file_reader.py
│     │  ├─ main_content.py
│     │  └─ normalizer.py
│     ├─ chunking/
│     │  ├─ __init__.py
│     │  ├─ splitter.py
│     │  └─ sentence.py
│     ├─ llm/
│     │  ├─ __init__.py
│     │  ├─ client.py
│     │  ├─ prompt_loader.py
│     │  ├─ cloze_generator.py
│     │  ├─ translation_generator.py
│     │  └─ response_parser.py
│     ├─ tts/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ edge_tts_provider.py
│     │  ├─ provider_registry.py
│     │  └─ voice_selector.py
│     ├─ anki/
│     │  ├─ __init__.py
│     │  ├─ template_loader.py
│     │  ├─ note_builder.py
│     │  ├─ media_manager.py
│     │  └─ deck_exporter.py
│     ├─ pipeline/
│     │  ├─ __init__.py
│     │  ├─ build_deck.py
│     │  ├─ validators.py
│     │  └─ dedupe.py
│     └─ utils/
│        ├─ __init__.py
│        ├─ fs.py
│        ├─ jsonx.py
│        ├─ text.py
│        ├─ hash.py
│        └─ time.py
├─ tests/
│  ├─ test_config.py
│  ├─ test_prompt_loader.py
│  ├─ test_template_loader.py
│  ├─ test_chunking.py
│  ├─ test_llm_parser.py
│  ├─ test_edge_tts.py
│  ├─ test_note_builder.py
│  ├─ test_deck_exporter.py
│  └─ test_cli_help.py
└─ runs/
   └─ .gitkeep
```

---

# 10. 必需配置文件

V1 固定要求以下配置文件存在：

## 10.1 Prompt 文件 1

路径：

```text
./prompts/cloze_contextual.json
```

用途：

* 用于生成 `Text` 字段

---

## 10.2 Prompt 文件 2

路径：

```text
./prompts/translate_rewrite.json
```

用途：

* 用于生成 `Translation` 字段

---

## 10.3 Anki 模板文件

路径：

```text
./templates/anki_cloze_default.json
```

用途：

* 定义牌组字段、模板、样式、Note Type、Deck 元数据

---

## 10.4 环境变量文件

路径建议：

```text
./.env
```

用途：

* 保存模型配置
* 保存 TTS 配置
* 保存默认语言
* 保存输出路径等

---

# 11. `.env` 配置规范

示例：

```dotenv
CLAWLINGUA_DEFAULT_SOURCE_LANG=en
CLAWLINGUA_DEFAULT_TARGET_LANG=zh

CLAWLINGUA_LLM_PROVIDER=openai_compatible
CLAWLINGUA_LLM_BASE_URL=http://127.0.0.1:8000/v1
CLAWLINGUA_LLM_API_KEY=YOUR_API_KEY
CLAWLINGUA_LLM_MODEL=qwen3-30b
CLAWLINGUA_LLM_TIMEOUT_SECONDS=120
CLAWLINGUA_LLM_MAX_RETRIES=3
CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS=2.0
CLAWLINGUA_LLM_TEMPERATURE=0.2

CLAWLINGUA_HTTP_TIMEOUT_SECONDS=30
CLAWLINGUA_HTTP_USER_AGENT=ClawLingua/0.1
CLAWLINGUA_HTTP_VERIFY_SSL=true

CLAWLINGUA_CHUNK_MAX_CHARS=1800
CLAWLINGUA_CHUNK_MAX_SENTENCES=8
CLAWLINGUA_CHUNK_MIN_CHARS=120
CLAWLINGUA_CHUNK_OVERLAP_SENTENCES=1

CLAWLINGUA_PROMPT_CLOZE=./prompts/cloze_contextual.json
CLAWLINGUA_PROMPT_TRANSLATE=./prompts/translate_rewrite.json
CLAWLINGUA_ANKI_TEMPLATE=./templates/anki_cloze_default.json

CLAWLINGUA_OUTPUT_DIR=./runs
CLAWLINGUA_LOG_LEVEL=INFO
CLAWLINGUA_SAVE_INTERMEDIATE=true

CLAWLINGUA_TTS_PROVIDER=edge_tts
CLAWLINGUA_TTS_OUTPUT_FORMAT=mp3
CLAWLINGUA_TTS_RATE=+0%
CLAWLINGUA_TTS_VOLUME=+0%

CLAWLINGUA_TTS_EDGE_EN_VOICES=en-US-AnaNeural,en-US-AndrewNeural,en-GB-SoniaNeural
CLAWLINGUA_TTS_EDGE_ZH_VOICES=zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,zh-CN-liaoning-XiaobeiNeural
CLAWLINGUA_TTS_EDGE_JA_VOICES=ja-JP-NanamiNeural,ja-JP-KeitaNeural,ja-JP-AoiNeural
```

规则：

1. CLI 参数优先级高于 `.env`
2. `.env` 高于代码默认值
3. 缺失关键配置时必须报结构化错误
4. `config show` 中 API key 必须脱敏

---

# 12. Prompt JSON 文件规范

你要求 prompt 独立保存为 JSON 文件。V1 固定 schema。

---

## 12.1 `./prompts/cloze_contextual.json`

用途：

* 生成 `Text`

推荐结构：

```json
{
  "name": "cloze_contextual",
  "version": "0.1.0",
  "description": "Generate contextual cloze text for Anki Text field.",
  "mode": "cloze",
  "system_prompt": "这里放系统提示词",
  "user_prompt_template": "这里放用户提示词模板，包含占位符",
  "placeholders": [
    "source_lang",
    "target_lang",
    "document_title",
    "source_url",
    "chunk_text"
  ],
  "output_format": {
    "type": "json",
    "schema_name": "cloze_cards_v1"
  },
  "parser": {
    "strip_code_fences": true,
    "expect_json_array": true
  }
}
```

---

## 12.2 `./prompts/translate_rewrite.json`

用途：

* 生成 `Translation`

推荐结构：

```json
{
  "name": "translate_rewrite",
  "version": "0.1.0",
  "description": "Generate learning-oriented translation for Anki Translation field.",
  "mode": "translate",
  "system_prompt": "这里放系统提示词",
  "user_prompt_template": "这里放用户提示词模板，包含占位符",
  "placeholders": [
    "source_lang",
    "target_lang",
    "document_title",
    "source_url",
    "chunk_text",
    "text_original"
  ],
  "output_format": {
    "type": "json",
    "schema_name": "translation_cards_v1"
  },
  "parser": {
    "strip_code_fences": true,
    "expect_json_array": true
  }
}
```

---

# 13. Anki 模板文件规范

路径：

```text
./templates/anki_cloze_default.json
```

作用：

* 定义 Note Type
* 定义字段顺序
* 定义卡片模板
* 定义 CSS
* 定义 deck 默认信息

推荐结构：

```json
{
  "model_name": "ClawLingua Cloze",
  "deck_name": "ClawLingua Default Deck",
  "fields": [
    "Text",
    "Original",
    "Translation",
    "Note",
    "Audio"
  ],
  "card_templates": [
    {
      "name": "Card 1",
      "qfmt": "{{cloze:Text}}<br>{{Audio}}",
      "afmt": "{{cloze:Text}}<hr id=\"answer\">{{Translation}}<br><br><div class=\"original\">{{Original}}</div><br><div class=\"note\">{{Note}}</div><br>{{Audio}}"
    }
  ],
  "css": ".card { font-family: arial; font-size: 20px; text-align: left; color: black; background-color: white; } .original { color: #666; font-size: 16px; } .note { color: #888; font-size: 14px; }"
}
```

---

# 14. 数据模型规范

建议使用 Pydantic。

---

## 14.1 DocumentRecord

```python
class DocumentRecord(BaseModel):
    run_id: str
    source_type: Literal["url", "file"]
    source_value: str
    source_lang: str
    target_lang: str
    title: str | None
    source_url: str | None
    raw_text: str
    cleaned_text: str
    cleaned_markdown: str | None
    fetched_at: str | None
    metadata: dict = {}
```

---

## 14.2 ChunkRecord

```python
class ChunkRecord(BaseModel):
    run_id: str
    chunk_id: str
    chunk_index: int
    source_text: str
    char_count: int
    sentence_count: int
    metadata: dict = {}
```

---

## 14.3 CardRecord

V1 关键输出模型。

```python
class CardRecord(BaseModel):
    run_id: str
    card_id: str
    chunk_id: str
    source_lang: str
    target_lang: str
    title: str | None
    source_url: str | None
    text: str
    original: str
    translation: str
    note: str
    audio_file: str | None
    audio_field: str | None
    target_phrases: list[str] = []
```

---

# 15. 构建流程总览

`clawlingua build deck <input>` 的内部流程固定如下：

```text
输入
-> ingest
-> normalize
-> chunk
-> 生成 Text
-> 生成 Translation
-> 校验与去重
-> 生成 Audio
-> 组装 Anki Notes
-> 导出 .apkg
```

---

# 16. 输入处理规范

## 16.1 URL 输入

步骤：

1. 校验 URL
2. 下载 HTML
3. 提取正文
4. 标准化文本
5. 写入中间文件

要求：

* 支持超时控制
* 支持 403/404/500 错误识别
* 支持正文为空的检测
* 尽量保留页面标题

---

## 16.2 文件输入

支持：

* `.txt`
* `.md`

步骤：

1. 检查文件是否存在
2. 尝试 UTF-8 读取
3. 失败时尝试编码探测
4. 标准化文本
5. 写入中间文件

---

# 17. 文本清洗与标准化规范

目标：

让后续 LLM 输入尽量稳定，但不破坏原始语言材料。

规则：

1. 合并连续空白行
2. 去除首尾空白
3. 去除明显网页导航噪音
4. 保留原始段落顺序
5. 不随意改写源语言正文
6. 不主动翻译
7. 不主动删除正常口语特征，除非明显损坏结构

---

# 18. 分块规范

默认策略：

1. 先按段落分块
2. 过短段落与相邻段落合并
3. 超长段落按句子切分
4. 邻接块允许 1 句 overlap

默认参数：

* `max_chars = 1800`
* `max_sentences = 8`
* `min_chars = 120`
* `overlap_sentences = 1`

要求：

* 每块必须保留原顺序
* 不产生空块
* 每个块有稳定 `chunk_id`

---

# 19. LLM 调用规范

V1 只支持：

* OpenAI-compatible chat completion API

每个 chunk 要做两类生成：

1. 生成 `Text`
2. 生成 `Translation`

建议流程：

* 第一步先用 `cloze_contextual.json` 对 chunk 生成候选卡片
* 第二步对每个候选卡片的 `Original` 再用 `translate_rewrite.json` 生成 `Translation`

这样比直接对整块一起翻译更稳，因为 Translation 会更精确地对齐最终卡片。

---

# 20. Text 生成规范

`Text` 来自 `./prompts/cloze_contextual.json`

模型输出必须是严格 JSON 数组。每个元素至少包含：

```json
{
  "text": "带挖空的文本，最多3句",
  "original": "对应纯文本原文",
  "target_phrases": ["短语1", "短语2"],
  "note_hint": "可选提示信息"
}
```

校验规则：

1. `text` 非空
2. `original` 非空
3. `text` 中必须包含至少一个 `{{c1::`
4. `original` 不得包含 cloze 标记
5. `text` 最多 3 句
6. `target_phrases` 至少 1 个
7. 不允许模型输出解释性文字

---

# 21. Translation 生成规范

`Translation` 来自 `./prompts/translate_rewrite.json`

输入应是每张卡片的 `original`，不是整块 chunk。

模型输出格式：

```json
{
  "translation": "翻译与意译结果"
}
```

校验规则：

1. 非空
2. 不允许 `翻译：` 这种前缀
3. 可含 `<b>...</b>`
4. 不允许 Markdown `**`

---

# 22. Note 字段生成规范

V1 中 `Note` 可由程序自动拼装，不需要单独调用模型。

默认内容建议：

* 来源标题
* URL
* target phrases
* chunk id

例如：

```text
title: Lex Fridman Podcast
source: https://example.com
phrases: turns out | make sense of
chunk: chunk_0012
```

---

# 23. TTS 设计规范

这是你这次新增的重点，V1 必须从设计起就支持。

---

## 23.1 TTS 总体要求

* 音频内容基于 `Original`
* 默认 provider 为 `edge_tts`
* 设计上必须支持将来扩展其他 TTS provider
* 每张卡片最终都应有对应音频文件
* 音频文件被打包进 `.apkg`

---

## 23.2 TTS Provider 抽象层

必须定义统一接口，例如：

```python
class BaseTTSProvider(Protocol):
    def synthesize(self, text: str, voice: str, output_path: str, lang: str | None = None) -> None:
        ...
```

V1 实现：

* `EdgeTTSProvider`

未来可以扩展：

* `OpenAITTSProvider`
* `FishSpeechProvider`
* `CustomHTTPProvider`

---

## 23.3 Edge TTS 语音选择要求

如果 provider 是 `edge_tts`，必须支持每种语言至少 3 个 voice。

而且 voice 的选择规则必须满足：

* 从该语言对应的 voice 列表中**均匀随机选择**
* 每次生成单条卡片时独立抽样
* 概率尽量均匀，避免长期只听一个声音
* 支持设置随机种子，便于复现

例如英文：

```text
en-US-AnaNeural
en-US-AndrewNeural
en-GB-SoniaNeural
```

例如中文：

```text
zh-CN-XiaoxiaoNeural
zh-CN-YunxiNeural
zh-CN-liaoning-XiaobeiNeural
```

---

## 23.4 Voice 配置规范

Voice 列表放在 `.env` 中，用逗号分隔。

程序运行时根据 `source_lang` 选择 voice 列表。

规则：

1. 若当前 `source_lang` 没有 voice 列表，报结构化错误
2. 若 voice 数量少于 3，允许运行但给 warning
3. 默认采用均匀随机选择
4. 后续可扩展为 round-robin 或 weighted 模式，但 V1 只做 uniform random

---

## 23.5 Audio 字段写入规则

假设音频文件名为：

```text
audio_000123.mp3
```

则卡片 `Audio` 字段写入：

```text
[sound:audio_000123.mp3]
```

---

# 24. Anki 导出规范

V1 最终输出必须是 `.apkg`

不再设计 JSONL/TSV 作为正式输出目标。
中间文件可以保留，但最终用户产物只有 `.apkg`。

要求：

1. 使用 `genanki`
2. 将所有媒体文件加入 media
3. 使用 `./templates/anki_cloze_default.json`
4. note 字段顺序固定为：

* Text
* Original
* Translation
* Note
* Audio

---

# 25. 去重规范

V1 去重只做简单规则。

建议顺序：

1. 按 `original` 归一化后精确去重
2. 再按 `text` 归一化后精确去重

不做语义去重。

---

# 26. 中间产物规范

如果开启 `--save-intermediate`，每次 run 应保存中间文件，便于 Agent 调试。

目录示例：

```text
runs/
└─ 20260407_170000_build_deck/
   ├─ document.json
   ├─ document.md
   ├─ chunks.jsonl
   ├─ text_candidates.raw.jsonl
   ├─ text_candidates.validated.jsonl
   ├─ translations.jsonl
   ├─ cards.final.jsonl
   ├─ media/
   │  ├─ audio_000001.mp3
   │  ├─ audio_000002.mp3
   ├─ output.apkg
   ├─ run_summary.json
   └─ errors.jsonl
```

虽然最终输出只有 `.apkg`，但中间文件对工程调试非常重要。

---

# 27. 错误处理规范

你特别强调了这一点。V1 必须统一实现。

正常用户路径下，不要直接抛 Python traceback。
所有常规错误都必须：

* 非零退出
* 输出结构化导航提示

格式固定：

```text
ERROR | <错误码>
CAUSE | <一句话原因>
DETAIL | <更具体说明>
NEXT | <下一步建议1>
NEXT | <下一步建议2>
```

---

## 27.1 示例：缺少 API key

```text
ERROR | CONFIG_MISSING_API_KEY
CAUSE | LLM API key 缺失。
DETAIL | 未在 .env 或命令行参数中找到 CLAWLINGUA_LLM_API_KEY。
NEXT | 在 .env 中添加 CLAWLINGUA_LLM_API_KEY
NEXT | 或通过 --env-file 指定正确的环境文件
NEXT | 运行 clawlingua config validate 检查配置
```

---

## 27.2 示例：Prompt 文件损坏

```text
ERROR | PROMPT_SCHEMA_INVALID
CAUSE | Prompt JSON 文件不符合要求。
DETAIL | 缺少字段 user_prompt_template。
NEXT | 打开对应 prompt JSON 文件补齐缺失字段
NEXT | 运行 clawlingua prompt validate <path> 再次检查
```

---

## 27.3 示例：模型输出不是 JSON

```text
ERROR | LLM_RESPONSE_PARSE_FAILED
CAUSE | 模型输出无法解析为 JSON。
DETAIL | 返回内容以解释性文本开头，而不是 JSON 数组。
NEXT | 检查 prompt 是否强制要求严格 JSON 输出
NEXT | 尝试降低 temperature，例如 --temperature 0.1
NEXT | 使用 --save-intermediate 检查原始返回内容
```

---

## 27.4 示例：当前语言没有配置 TTS voice

```text
ERROR | TTS_VOICE_NOT_CONFIGURED
CAUSE | 当前源语言没有可用的 TTS voice 配置。
DETAIL | source_lang=de，但 .env 中没有对应的 CLAWLINGUA_TTS_EDGE_DE_VOICES。
NEXT | 在 .env 中为该语言配置至少 3 个 edge_tts voice
NEXT | 或切换为其它已支持 voice 的源语言
NEXT | 运行 clawlingua doctor 检查 TTS 配置
```

---

# 28. 退出码规范

建议定义：

```text
0   成功
2   参数错误
3   配置错误
4   输入读取或抓取失败
5   文本分块失败
6   Prompt 或模板文件非法
7   LLM 请求失败
8   LLM 返回解析失败
9   卡片校验失败
10  TTS 生成失败
11  Anki 导出失败
12  未预期内部错误
```

---

# 29. 日志规范

默认 stdout：

```text
INFO | ingest complete | title="Example Podcast"
INFO | chunking complete | chunks=24
INFO | text generation complete | raw=82 valid=71
INFO | translation generation complete | translated=71
INFO | tts generation complete | audio=71
INFO | deck export complete | file=./runs/.../output.apkg
```

默认 stderr：

* 只输出结构化错误

`--debug` 模式下：

* 允许 traceback
* 允许输出原始响应片段

---

# 30. `requirements.txt`

你要求列出所有需要的包依赖。
下面给出 V1 推荐版本的 `requirements.txt` 内容。

```text
typer>=0.12.0
pydantic>=2.7.0
python-dotenv>=1.0.1
httpx>=0.27.0
readability-lxml>=0.8.1
beautifulsoup4>=4.12.3
markdownify>=0.12.1
orjson>=3.10.0
edge-tts>=6.1.12
genanki>=0.13.1
chardet>=5.2.0
pytest>=8.1.1
rich>=13.7.1
```

说明如下：

* `typer`：CLI
* `pydantic`：数据模型和配置校验
* `python-dotenv`：读取 `.env`
* `httpx`：HTTP 和 LLM 请求
* `readability-lxml`：网页正文提取
* `beautifulsoup4`：HTML 清洗
* `markdownify`：HTML 转 Markdown
* `orjson`：更稳更快的 JSON 读写
* `edge-tts`：TTS 生成
* `genanki`：导出 `.apkg`
* `chardet`：文件编码探测
* `pytest`：测试
* `rich`：更好的 help 与命令行显示

如果你想更保守一点，也可以先不用 `rich`，但我建议保留，因为 help 会更好看。

---

# 31. V1 核心流水线的精确定义

`clawlingua build deck <input>` 的执行顺序必须如下：

## 第 1 步：读取配置

* 读取 CLI 参数
* 读取 `.env`
* 合并配置
* 校验配置完整性

## 第 2 步：加载 prompt 和模板

* 加载 `./prompts/cloze_contextual.json`
* 加载 `./prompts/translate_rewrite.json`
* 加载 `./templates/anki_cloze_default.json`
* 校验 schema

## 第 3 步：输入处理

* 若为 URL，抓取正文
* 若为文件，读取文本
* 输出标准化 `DocumentRecord`

## 第 4 步：分块

* 得到 `ChunkRecord[]`

## 第 5 步：生成 Text 候选

* 对每个 chunk 调用 cloze prompt
* 得到若干候选卡片

## 第 6 步：校验与初步去重

* 校验 `text`
* 校验 `original`
* 过滤无效项
* 去重

## 第 7 步：生成 Translation

* 对每个卡片的 `original` 调用 translate prompt
* 回填 `translation`

## 第 8 步：生成 Note

* 程序自动拼接 note

## 第 9 步：生成 Audio

* 根据 `source_lang` 选择 voice 列表
* 均匀随机选择 voice
* 对 `original` 调用 TTS
* 写入媒体文件
* 回填 `audio_field`

## 第 10 步：导出 Anki

* 生成 notes
* 打包 media
* 导出 `.apkg`

---

# 32. Prompt 工程化要求

你原来那两个 prompt 的方向是对的，但 CLI 批处理里必须改成工程模式。

要求如下：

1. 模型输出必须是严格 JSON
2. 不允许输出说明文字
3. 不允许输出“是否继续下一批”
4. 分批策略由程序控制，不由 prompt 控制
5. prompt 中保留筛选标准、风格要求、语感要求
6. prompt 中移除对话式分页规则

也就是说：

* `./prompts/cloze_contextual.json` 只负责从文本块中提取高质量卡片候选
* `./prompts/translate_rewrite.json` 只负责对单张卡片的 `original` 生成翻译与意译

---

# 33. 测试要求

至少应覆盖以下内容：

1. `.env` 加载
2. prompt 文件校验
3. template 文件校验
4. 文件输入读取
5. URL 正文提取失败提示
6. chunking 稳定性
7. cloze 响应解析
8. translation 响应解析
9. TTS voice 选择均匀性基础测试
10. edge_tts 合成成功
11. genanki 导出成功
12. CLI `--help` 可用

---

# 34. 我对这一版的建议收敛

如果按工程实施角度，我建议你把 V1 范围固定为：

* 只做一个命令：`build deck`
* 只做一个最终输出：`.apkg`
* 只做两个 prompt
* 只做一个默认模板
* TTS 默认只实现 `edge_tts`，但代码接口预留 provider 扩展
* 先不引入复杂 deck 更新和语义去重

这样第一版可以很快做出来，而且结构是对的，不会以后推翻重写。
