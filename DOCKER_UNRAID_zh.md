# ClawLingua Docker + Unraid 部署指南（中文）

> 目标：在本地或 Unraid NAS 上，使用 Docker 镜像运行 ClawLingua 的 Web 界面（可选地也能跑 CLI）。
>
> 适用对象：已经会基本使用 Docker / Unraid，对 `docker build` / `docker run` 不完全陌生。

---

## 1. 仓库结构与 Docker 基线

当前仓库根目录包含：

- `Dockerfile`：构建 Web UI 镜像的定义；
- `.dockerignore`：构建镜像时忽略的文件；
- `docker-compose.yml`：本地开发环境一键启动 Web UI 的示例。

### 1.1 Dockerfile 行为概览

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
    && pip install .[web]

RUN mkdir -p /app/runs /app/outputs /app/logs

ENV CLAWLINGUA_WEB_HOST=0.0.0.0 \
    CLAWLINGUA_WEB_PORT=7860

EXPOSE 7860

CMD ["clawlingua-web"]
```

含义：

- 镜像基于 `python:3.11-slim`；
- 安装 `ffmpeg`；
- 复制整个项目到 `/app`，并执行 `pip install .[web]`；
- 创建默认输出目录：`/app/runs`、`/app/outputs`、`/app/logs`；
- 默认启动命令为 `clawlingua-web`，即 Web UI 模式；
- Web UI 监听 `0.0.0.0:7860`。

### 1.2 docker-compose.yml 概览

```yaml
services:
  clawlingua-web:
    build: .
    container_name: clawlingua-web
    env_file:
      - .env
    ports:
      - "7860:7860"
    volumes:
      - ./runs:/app/runs
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    restart: unless-stopped
```

含义：

- 构建当前目录的 Dockerfile；
- 将宿主机当前目录下的 `runs/`、`outputs/`、`logs/` 映射到容器内对应目录；
- 使用当前目录的 `.env` 作为环境变量来源（主要是 `CLAWLINGUA_*`）；
- 对外暴露 `7860` 端口；
- `restart: unless-stopped` 适合长期运行。

---

## 2. 在开发机上构建并测试镜像

### 2.1 准备 `.env`

在仓库根目录创建或复用一个 `.env`（与 CLI 共用同一份即可），例如：

```env
# LLM 设置
CLAWLINGUA_LLM_BASE_URL=https://api.luznest.com/v1
CLAWLINGUA_LLM_API_KEY=sk-...
CLAWLINGUA_LLM_MODEL=qwen3-30b
CLAWLINGUA_LLM_TIMEOUT_SECONDS=120
CLAWLINGUA_LLM_MAX_RETRIES=3
CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS=2
CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS=0.5
CLAWLINGUA_LLM_TEMPERATURE=0.2

# 多语言 prompt 设置
CLAWLINGUA_PROMPT_LANG=zh

# 其它常用参数按需补充
# CLAWLINGUA_CONTENT_PROFILE=general
# CLAWLINGUA_OUTPUT_DIR=./runs
# CLAWLINGUA_EXPORT_DIR=./outputs
# CLAWLINGUA_LOG_DIR=./logs
```

### 2.2 构建镜像

在开发机终端（WSL / Linux / macOS 等）执行：

```bash
cd /path/to/clawlingua
# 构建镜像，yourname 可替换为 Docker Hub 用户名
docker build -t yourname/clawlingua:latest .
```

构建完成后，用 `docker images` 可看到 `yourname/clawlingua:latest`。

### 2.3 本地用 docker-compose 启动 Web UI

在同一目录下：

```bash
# 首次建议带 --build，之后可以只用 up
docker compose up --build
# 或旧版 Docker：docker-compose up --build
```

启动成功后：

- 打开浏览器访问 <http://127.0.0.1:7860>
- 检查：
  - Run 页是否能上传 `.txt` / `.md` / `.epub` 并触发牌组构建；
  - Config 页是否能从 `.env` 读出配置（LLM、chunk、prompt_lang 等）；
  - Prompt 页是否能查看和编辑 `prompts/*.json`。

确认本地一切正常后，再进行 Unraid 部署。

---

## 3. 镜像如何送到 Unraid

### 路线 A：推送到 Docker Hub（推荐）

适用于：有 Docker Hub 账号，Unraid 能直接访问公网。

1. 登录 Docker Hub：

   ```bash
   docker login
   ```

2. 给镜像打带版本号的 tag（可选但推荐）：

   ```bash
   docker tag yourname/clawlingua:latest yourname/clawlingua:0.1.0
   ```

3. 推送镜像：

   ```bash
   docker push yourname/clawlingua:latest
   docker push yourname/clawlingua:0.1.0
   ```

4. 在 Unraid 上创建容器时，镜像名填：

   - `yourname/clawlingua:latest` 或
   - `yourname/clawlingua:0.1.0`。

### 路线 B：在 Unraid 上直接 build

适用于：只在内网用，不想搭 registry。

1. 在 Unraid 的终端中：

   ```bash
   cd /mnt/user/appdata
   git clone https://github.com/ernestyu/clawlingua.git
   cd clawlingua
   ```

2. 构建本地镜像：

   ```bash
   docker build -t clawlingua-local:latest .
   ```

之后部署时镜像名用 `clawlingua-local:latest` 即可。

---

## 4. 在 Unraid 上部署 Web UI

### 4.1 准备持久化目录

在 Unraid 上建议使用一个持久目录保存配置和输出，例如：

- 根路径：`/mnt/user/appdata/clawlingua`
  - `.env`：配置文件
  - `runs/`：中间 JSONL / 调试输出
  - `outputs/`：最终 `.apkg` 牌组
  - `logs/`：日志

在 Unraid 终端执行：

```bash
mkdir -p /mnt/user/appdata/clawlingua/{runs,outputs,logs}
# 复制或创建 .env
cp /path/to/local/.env /mnt/user/appdata/clawlingua/.env
# 或用 Unraid 编辑器直接在该目录新建 .env
```

### 4.2 方法一：用 Unraid Docker GUI 添加容器

1. 在 Unraid Web UI 的 **Docker** 标签页点击「Add Container」。
2. 基本设置：

   - Name：`clawlingua-web`
   - Repository：
     - 路线 A：`yourname/clawlingua:latest` 或 `yourname/clawlingua:0.1.0`
     - 路线 B：`clawlingua-local:latest`

3. 端口映射（Port Mapping）：

   - Container port：`7860`
   - Host port：`7860`（或其他你想要的外部端口，例如 `17860`）

4. Volume 映射：添加以下路径映射（Path）：

   - Host path：`/mnt/user/appdata/clawlingua/runs` → Container path：`/app/runs`
   - Host path：`/mnt/user/appdata/clawlingua/outputs` → Container path：`/app/outputs`
   - Host path：`/mnt/user/appdata/clawlingua/logs` → Container path：`/app/logs`

5. `.env` 的注入方式（二选一）：

   **方式 A：作为文件挂载（推荐）**

   - Host path：`/mnt/user/appdata/clawlingua/.env`
   - Container path：`/app/.env`
   - 可以设置为只读：`/app/.env:ro`

   ClawLingua 会在容器内读取 `/app/.env`。

   **方式 B：在 GUI 中逐条填写环境变量**

   - 使用「Add another Path, Port, Variable」→ 选择 Variable；
   - 分别添加 `CLAWLINGUA_LLM_BASE_URL`、`CLAWLINGUA_LLM_API_KEY` 等。

6. 其它选项：

   - Restart policy：选择 `unless-stopped`，方便重启后自动恢复。

7. 保存模板并启动容器。
8. 在浏览器访问：

   - `http://<Unraid-IP>:7860`

   即可打开 ClawLingua Web UI。

### 4.3 方法二：在 Unraid 上使用 docker-compose（需要插件）

若你在 Unraid 上安装了类似 "Compose Manager" 的插件，可以直接复用仓库自带的 `docker-compose.yml`。

流程示例：

1. 确保 `/mnt/user/appdata/clawlingua` 下包含：

   - `Dockerfile`
   - `docker-compose.yml`
   - `.env`

2. 切换到该目录：

   ```bash
   cd /mnt/user/appdata/clawlingua
   ```

3. 启动服务：

   ```bash
   docker compose up -d
   ```

   此时 `docker-compose.yml` 中的 `./runs` 等路径会解析为
   `/mnt/user/appdata/clawlingua/runs` 等，并映射到容器内。

4. 同样访问 `http://<Unraid-IP>:7860` 进行使用。

---

## 5. 在 Docker 容器中运行 CLI（可选）

虽然镜像默认启动 Web UI，但也可以临时用来跑 CLI 命令（例如批量构建牌组）：

### 5.1 直接 `docker run` 调用 CLI

示例：

```bash
docker run --rm \
  -v /mnt/user/media/texts:/input \
  -v /mnt/user/appdata/clawlingua/outputs:/app/outputs \
  -v /mnt/user/appdata/clawlingua/runs:/app/runs \
  -v /mnt/user/appdata/clawlingua/.env:/app/.env:ro \
  yourname/clawlingua:latest \
  clawlingua build deck /input/my_text.md \
    --source-lang en \
    --target-lang zh \
    --content-profile general \
    --prompt-lang zh \
    --save-intermediate
```

说明：

- 最后一段 `clawlingua build deck ...` 会覆盖 Dockerfile 里的默认 `CMD`；
- `/input` 是挂载的输入文件目录；
- 输出 `.apkg` 会写到 `/app/outputs`，即宿主机 `/mnt/user/appdata/clawlingua/outputs`。

---

## 6. 常见问题与排查建议

### 6.1 Web UI 打不开 / 端口冲突

- 确认容器正常运行：`docker ps`；
- 确认 Unraid 上没有其他服务占用 7860 端口；必要时在 GUI 中改成 `17860:7860`；
- 浏览器访问 `http://<Unraid-IP>:端口`，而不是 `localhost`（除非在 NAS 本机访问）。

### 6.2 无法连接 LLM 服务

- 在容器内检查 env 是否生效：

  ```bash
  docker exec -it clawlingua-web env | grep CLAWLINGUA_LLM
  ```

- 确认：

  - `CLAWLINGUA_LLM_BASE_URL` 指向可访问的 OpenAI-compatible 端点；
  - `CLAWLINGUA_LLM_API_KEY` 正确；
  - `doctor` 命令可通过（可在容器里临时运行 `clawlingua doctor --env-file /app/.env`）。

### 6.3 性能问题（构建一套牌组耗时较长）

- 大部分时间通常花在远端 LLM 推理上，而不是本地切片：
  - `ingest complete` / `chunking complete` 与第一条 LLM 请求之间的间隔，基本就是模型端的排队+推理时间；
- 可尝试：
  - 使用延迟更低的模型 / 后端；
  - 调整 `CLAWLINGUA_LLM_CHUNK_BATCH_SIZE`（一次处理多个 chunk）；
  - 适当调短 `CLAWLINGUA_LLM_TIMEOUT_SECONDS` 或减少重试次数（视后端稳定性而定）。

---

## 7. 升级与回滚

- 升级镜像：
  - 重新 `git pull` 仓库并 `docker build -t ...`；
  - 或在 Docker Hub 上推送新 tag，然后在 Unraid GUI 中更新镜像版本。
- 回滚：
  - 保留旧 tag（例如 `0.1.0`）；
  - 在 Unraid 中将镜像版本切回旧 tag，并重新部署即可。

持久化数据（`runs/`、`outputs/`、`logs/`、`.env`）都在宿主机上，不会因为重新部署容器而丢失。
