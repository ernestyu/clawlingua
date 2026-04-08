[2026/4/8 15:03] Codingcraw: 下面是一个给同事看的“需求+现状”规格，可以直接转发过去当交接说明。

───

ClawLingua 当前需求说明（交接版）

0. 项目目的（大图）

ClawLingua 的目标：

• 输入：一段真实语料（优先是 podcast 字幕/转写文本，英文或中英混合）。
• 处理流程：
  1. 清洗文本 → 按自然语境切成 chunk；
  2. 把每个 chunk 送到「主 LLM」（例如 Gemini），让它从中选出适合学习的表达并生成 cloze 卡片；
  3. 用一个单独的「small LLM」做翻译/意译；
  4. 用 edge_tts 为每张卡片生成音频；
  5. 拼成 Anki .apkg 牌组。
• 输出：一套以 cloze 为主、带上下文、带翻译和音频的 Anki 牌组，适合语言学习用。

───

1. 切 chunk 的行为要求

1.1 切块粒度（Chunking）

• 控制参数（env）：CLAWLINGUA_CHUNK_MAX_CHARS=1500   # 每个 chunk 最大字符数，按字数算
CLAWLINGUA_CHUNK_MIN_CHARS=120    # 段落太短会与后段合并
CLAWLINGUA_CHUNK_OVERLAP_SENTENCES=1
• 行为：
  • 以字符数为主控制 chunk 大小；
  • 按句子边界软切：
    • 不从句子中间硬截断；
    • 对超长段落，从当前位置开始向后扩展句子，直至接近 CHUNK_MAX_CHARS，超出则回退一两句；
    • 相邻 chunk 之间有 CHUNK_OVERLAP_SENTENCES 的句子重叠；
  • 不再有 chunk_max_sentences 这种独立限制，句子条数只是由文本自然决定。

1.2 Chunk 数量 & 批处理

• env：CLAWLINGUA_LLM_CHUNK_BATCH_SIZE=1   # 一次送给主 LLM 处理的 chunk 数；建议 1–4
• 行为：
  • BATCH_SIZE=1 时：逐 chunk 调用 LLM；
  • BATCH_SIZE>1 时：把多个 chunk 组合进一个 prompt 中（带 chunk_id 标识），让 LLM 一次返回多个 chunk 的候选；
  • 输出 JSON 数组中每个元素都带 chunk_id，之后代码按 chunk_id 挂回对应 chunk。

───

2. LLM 配置与调用

2.1 主 LLM（cloze）

• env：CLAWLINGUA_LLM_BASE_URL=...
CLAWLINGUA_LLM_API_KEY=...
CLAWLINGUA_LLM_MODEL=...
CLAWLINGUA_LLM_PROVIDER=openai   # 目前假设 OpenAI-compatible
CLAWLINGUA_LLM_TIMEOUT_SECONDS=120
CLAWLINGUA_LLM_MAX_RETRIES=3
CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS=2.0
CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS=2  # 每次成功请求后 sleep 区间 [2, 6] 秒
CLAWLINGUA_LLM_TEMPERATURE=0.2
• 行为：
  • 使用 OpenAI-compatible API 调 POST /chat/completions；
  • 每次成功请求后，按 sleep（随机 [N, 3N]）做轻量节流，避免整齐节奏触发风控；
  • retry 使用指数 backoff；
  • 主 LLM 主要负责 cloze 选句 + 挖空，不负责翻译。

2.2 small LLM（翻译）

• env：CLAWLINGUA_TRANSLATE_LLM_BASE_URL=
CLAWLINGUA_TRANSLATE_LLM_API_KEY=
CLAWLINGUA_TRANSLATE_LLM_MODEL=
CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE=
• 行为：
  • 若 TRANSLATE_LLM_BASE_URL/MODEL 配齐，则翻译走 small LLM；
  • 若未配置，则 fallback 到主 LLM；
  • 目的：cloze 用贵 LLM，翻译用便宜 LLM。

2.3 doctor 命令

• 命令：python -m clawlingua.cli doctor --env-file ENV_EXAMPLE.md
• doctor 要检查：
  • 主 LLM 配置完整性与连通性；
  • small LLM 配置完整性与连通性；
  • Cloze 控制参数（max_sentences / min_chars / max_per_chunk）；
  • TTS 配置是否合理（edge_tts voices 等）。

───

3. Cloze 卡片生成的具体要求（重点）

3.1 Cloze 控制参数

• env：# 单张卡允许的最大句子数
CLAWLINGUA_CLOZE_MAX_SENTENCES=4   # 建议 3–4

# 单张 cloze 文本的最小字符数（太短丢弃）
CLAWLINGUA_CLOZE_MIN_CHARS=200     # 对应大约 50 个英文词左右

# 难度：beginner / intermediate / advanced
CLAWLINGUA_CLOZE_DIFFICULTY=intermediate

# 每个 chunk 可接受的卡片最多数量（去重后上限，为空/0 表示不限制）
CLAWLINGUA_CLOZE_MAX_PER_CHUNK=4
• 行为：
  • 小于 CLOZE_MIN_CHARS 的 text 候选直接丢弃；
  • 句子数超过 CLOZE_MAX_SENTENCES 的候选丢弃；
  • 每个 chunk 允许生成 0–N 张卡，但最终只保留前 CLOZE_MAX_PER_CHUNK 张（可设置为 0 表示不限制，上限仅用于防爆）。

3.2 Cloze 语法和格式规范

目标格式必须和以下示例一致（这是强约束示例）：

The more {{c1::<b>whimsical explanation</b>}}(异想天开的解释) is that maybe RL training makes the models a little too {{c2::<b>single-minded</b>}}(钻牛角尖) and narrowly focused.

• 结构要求：
  1. cloze 标记：{{cN::...}}，N 从 1 开始（c1, c2, c3，按出现顺序递增）；
  2. cloze 内部必须用 <b>...</b> 包裹被挖空的短语；
  3. cloze 后紧跟括号：(中文解释)；
  4. 同一句可有多个 cloze：c1, c2, c3 等；
  5. original 字段是去掉 cloze 和 HTML 后的纯文本（不含 { / } / <b>）。

3.3 编号规范化 & 自动注入

• 允许 LLM 输出 {c1::...} 单大括号，但 validator 要：
  • 先把 {c1::something} 规范化成 {{c1::something}}；
  • 然后再做后续检查。
• 如果 LLM 返回了多个 c1：
  • 需要编码一段逻辑，把 cloze 按出现顺序重排为 c1, c2, c3；
  • 不能出现同一句中两个 {{c1::...}}。
[2026/4/8 15:03] Codingcraw: • 若完全没有任何 cloze 标记但有 target_phrases：
  • 自动从 target_phrases 中选一个出现在 text 里的短语；
  • 将第一次出现替换为 {{c1::<b>phrase</b>}}(简短解释)（解释来自 LLM 或暂时为空）；
  • 作为兜底策略。

当前代码已有：

• 单大括号 → 双大括号规范化；
• 当没有 cloze 且存在 target_phrases 时自动注入一个 {{c1::...}}。
下一位同事需要补的是「多 c1 → c1/c2/c3 重编号」和 <b>...</b> + 括号解释 的一致性。

───

4. 每个 chunk 生成多少张卡：由 LLM 决定，上限由 env 防爆

• 逻辑：
  • Prompt 只要求「每个 chunk 尝试给出 0–4 个高质量候选」，每一条 JSON 是一张卡；
  • LLM 可以返回 0、1、2、3、4 条；
  • CLAWLINGUA_CLOZE_MAX_PER_CHUNK 在 dedupe 之后按 chunk 进行截断：
    • 如果设为 4，最多保留 4 条；
    • 设置为 0 或留空则不限制。

需求：不要硬编码“每个 chunk 只有一张卡”。
真正的决定权在 LLM，env 只是防止极端情况下爆炸。

───

5. 翻译 & TTS 行为

5.1 翻译（small LLM）

• Prompt：prompts/translate_rewrite.json
• 输出要求：
  • JSON 数组，每项 { "translation": "最终翻译文本" }；
  • 不加「翻译：」前缀；
[2026/4/8 15:03] Codingcraw: • 可以适度意译，要自然。

validator 已有：

• 空翻译 → 丢弃；
• 含 翻译: / 翻译： 前缀 → 丢弃；
• 含 **（Markdown 粗体）→ 丢弃。

5.2 TTS（edge_tts）

• env 里配置各语言的 voice 列表；
• 根据源语言选择 voice，生成 .mp3；
• 每张卡的 Original 都会有对应音频文件和 [sound:xxx.mp3] 字段。

───

6. CLI & doctor

6.1 CLI 子命令（主要是 build deck）

python -m clawlingua.cli build deck <input> \
  --env-file ENV_EXAMPLE.md \
  --difficulty beginner|intermediate|advanced \
  --max-chars 1500 \
  --max-notes 200 \
  --debug \
  --continue-on-error \
  --save-intermediate

• --difficulty 覆盖 env 中的 CLAWLINGUA_CLOZE_DIFFICULTY，优先级：CLI > env。
• --env-file 用于测试环境直接加载示例配置。

6.2 doctor

python -m clawlingua.cli doctor --env-file ENV_EXAMPLE.md

• 对 LLM（主 + translate）、cloze 参数、TTS 做基本检查；
• FAIL 时要给出 NEXT 提示（怎么补 env / 改配置）。

───

7. 总结：下一位同事要重点接着做的几件事

1. Cloze 编号重排
  • 在已经识别 cloze 的基础上，把一条 text 里的 cN 重排为 c1/c2/c3；
  • 避免多个 c1 的情况。
2. Cloze 样式进一步拉拢到目标示例
  • 在 prompt 里用你给的句子做 few-shot 示例；
  • Validator 适度检查 <b>...</b> 和括号解释是否存在（可以先只在 prompt 强约束，validator 不必太死）。
3. 根据实际 LLM 输出调整 prompt
  • 现在 prompt 已经有“目标形态”的描述，但效果还需要根据 Gemini 实际输出调整；
  • 建议通过 --save-intermediate 抓 text_candidates.raw.jsonl，对比实际输出和目标样式，迭代 prompt 文案。

这就是当前 ClawLingua 的需求与实现现状，下一位同事可以从 cloze 编号重排 + 格式收紧这两个点开始继续往下拉整。