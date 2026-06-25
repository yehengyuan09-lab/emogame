# Python 环境与本地数据采集入门

本文档用于帮助新成员快速准备 Python 开发环境，并运行当前仓库里已经存在的本地数据采集与检查脚本。

## 1. Python 环境要求

本项目推荐使用 Python 3.12 或更新版本。先确认本机版本：

```bash
python3 --version
```

如果输出类似 `Python 3.12.x`，就可以继续。若版本过低，建议通过系统包管理器、pyenv 或 Anaconda 安装新版 Python。

## 2. 创建虚拟环境

建议在项目根目录创建虚拟环境，避免依赖污染系统 Python：

```bash
cd /home/mzhyui/git/emogame
python3 -m venv .venv
source .venv/bin/activate
```

激活后，终端前面通常会出现 `(.venv)`。后续安装依赖和运行脚本都建议在这个环境里执行。

退出虚拟环境：

```bash
deactivate
```

## 3. 安装项目依赖

项目依赖集中在根目录的 `requirements.txt`：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

依赖大致分为几类：

| 类别 | 主要用途 | 示例依赖 |
|------|----------|----------|
| Web 服务 | 后端接口与可视化页面 | fastapi, uvicorn, streamlit |
| 数据采集 | 请求网页、解析页面、异步抓取 | httpx, selenium, beautifulsoup4, aiohttp |
| 数据与机器学习 | 数据处理、建模、调参 | numpy, pandas, scikit-learn, xgboost, optuna |
| 视觉与 VLM | 图片处理、视觉模型调用 | pillow, opencv-python-headless, torch, transformers |
| Agent 与 LLM | 智能体流程和模型接口 | openai, langchain, langgraph |
| 存储与配置 | 数据库、环境变量、日志 | sqlalchemy, alembic, python-dotenv, loguru |

## 4. 基础配置建议

如果后续脚本需要读取 API Key、数据库地址或其他本地配置，建议复制 `.env.example`：

```bash
cp .env.example .env
```

然后按需编辑 `.env`。不要把包含真实密钥的 `.env` 提交到 Git。

常见目录约定：

```text
data/
  wzry_skins/
    skins.sqlite3      # 王者荣耀皮肤元数据 SQLite 数据库
    images/            # 爬取到的皮肤图片
hero-skin-image/       # 本地皮肤图片与英雄 JSON 数据
crawlers/              # 数据采集脚本
vlm/                   # 本地视觉模型测试脚本
```

## 5. 运行王者荣耀皮肤采集脚本

当前可直接运行的采集脚本是：

```text
crawlers/wzry_skin_crawler.py
```

它会从两个王者荣耀官网 JSON 数据源读取数据：

- `herolist.json`：作为英雄与皮肤名称全集基准。
- `heroskinlist.json`：补充皮肤 ID、品质、上线日期、获取方式、详情页、视频和图片 URL。

脚本会把合并后的元数据保存到 SQLite，并按需下载皮肤图片到本地。

### 快速试跑

先只拉取少量数据，确认网络、目录和数据库写入都正常：

```bash
python3 crawlers/wzry_skin_crawler.py --limit 10
```

默认输出位置：

```text
data/wzry_skins/skins.sqlite3
data/wzry_skins/images/
```

脚本结束时会打印类似信息：

```text
heroes=130 skins=10 with_detail=10 assets=30 downloaded_or_existing=10 image_failed=0
db=data/wzry_skins/skins.sqlite3
images=data/wzry_skins/images
```

如果只想检查元数据合并，不下载图片：

```bash
python3 crawlers/wzry_skin_crawler.py --limit 50 --skip-images
```

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--limit` | 只采集前 N 条，`0` 表示全量 | `--limit 50` |
| `--output-dir` | 图片保存目录 | `--output-dir data/wzry_skins/images` |
| `--db` | SQLite 数据库保存路径 | `--db data/wzry_skins/skins.sqlite3` |
| `--sleep` | 每次图片请求之间的等待秒数 | `--sleep 0.1` |
| `--overwrite` | 已存在图片也重新下载 | `--overwrite` |
| `--skip-images` | 只采集元数据和图片 URL，不下载图片 | `--skip-images` |
| `--download-assets` | 下载的资产类型，默认 `skin_primary`，可用 `all` | `--download-assets all` |

全量采集示例：

```bash
python3 crawlers/wzry_skin_crawler.py --sleep 0.05
```

重新下载图片示例：

```bash
python3 crawlers/wzry_skin_crawler.py --overwrite
```

## 6. 查看采集结果

采集完成后，可以用 SQLite 简单检查数据：

```bash
sqlite3 data/wzry_skins/skins.sqlite3 "select count(*) from skins;"
sqlite3 data/wzry_skins/skins.sqlite3 "select hero_name, skin_name, quality, price_text from skins limit 10;"
```

当前数据库会包含：

| 表 | 内容 |
|------|------|
| `heroes` | 英雄 ID、名称、称号、定位、皮肤数量 |
| `skins` | 合并后的皮肤目录、官方增强字段、是否匹配到详情源 |
| `skin_assets` | 每个皮肤的图片 URL、本地路径、下载状态、文件 hash |
| `crawl_runs` | 每次采集的来源 URL、数量统计和失败数量 |

项目内部读取 SQLite 时应优先使用 `data/skin_repository.py`，不要在前端、模型或 Agent 层重复手写 SQL。

采集后可以运行数据质量检查：

```bash
python3 scripts/check_wzry_data.py
```

输出 JSON 方便接入自动化检查：

```bash
python3 scripts/check_wzry_data.py --json
```

## 7. 运行单皮肤情绪溢价评估

采集和查询层跑通后，可以对单个皮肤执行 MVP 规则评分：

```bash
python3 scripts/evaluate_skin.py --search 地狱岩魂
```

如果已经有舆论、营销或销量验证信号，可以通过 JSON 注入：

```bash
python3 scripts/evaluate_skin.py --source-key 105-02 --signals-json market_signals.json --json
```

也可以先导入本地证据库：

```bash
python3 scripts/import_market_signals.py market_signals.json
python3 scripts/evaluate_skin.py --source-key 105-02
```

采集已知 B 站视频 URL/BVID 的舆论证据：

```bash
python3 scripts/fetch_bilibili_evidence.py \
  --source-key 105-02 \
  --video https://www.bilibili.com/video/BVxxxxxxxxxx \
  --aspect-tags visual,feel,craftsmanship
```

按皮肤自动搜索 B 站并导入命中的视频证据：

```bash
python3 scripts/search_bilibili_evidence.py \
  --source-key 105-02 \
  --limit 5 \
  --json
```

导入已经抓取好的微博评论证据：

```bash
python3 scripts/import_weibo_evidence.py \
  data/weibo_comments/wzry_skin_comments_2026-06-23.json \
  --source-key 105-02
```

当前评估系统会输出皮肤维度证据分、官方弱先验、置信度、验证状态和缺失信号提示。没有市场信号时不会输出最终研究分，只会提示 `insufficient_market_evidence`。

生成面向运营/销售的动作报告：

```bash
python3 scripts/generate_sales_report.py --source-key 105-02
python3 scripts/generate_sales_report.py --search 龙胆 --json
```

销售报告不会把分数直接等同销量，而是输出放量决策、购买驱动力、转化阻力、价格动作建议和缺失证据。

也可以查看本地图片数量：

```bash
find data/wzry_skins/images -type f | wc -l
```

## 8. 检查本地 hero-skin-image 数据

仓库中还有一个本地图片数据目录：

```text
hero-skin-image/
```

`test_wzry_skins.py` 会检查其中的 `wzry-heros.json` 结构，并统计本地皮肤图片覆盖情况：

```bash
python3 test_wzry_skins.py
```

这个脚本适合用来确认本地图片集是否完整，以及英雄、皮肤、图片文件名之间是否能够对应。

## 9. 运行本地 VLM 图片评估脚本

如果本机安装并启动了 Ollama，可以使用 `vlm/ollama_vlm_test.py` 对单张皮肤图做视觉模型测试。

先确认 Ollama 可用：

```bash
ollama list
```

如缺少模型，可按脚本提示拉取，例如：

```bash
ollama pull qwen2.5vl:3b
```

运行示例：

```bash
python3 vlm/ollama_vlm_test.py \
  --models qwen2.5vl:3b \
  --image hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg \
  --prompt l2 \
  --timeout 240
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--host` | Ollama 服务地址，默认 `http://127.0.0.1:11434` |
| `--models` | 要测试的模型名，多个模型用英文逗号分隔 |
| `--image` | 输入图片路径 |
| `--prompt` | 使用的评估模板，当前支持 `l1`、`l2` |
| `--timeout` | 单个模型请求超时时间 |

## 10. 常见问题

### ModuleNotFoundError

通常是没有安装依赖，或当前终端没有激活虚拟环境：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 图片下载失败

可能是网络波动或源站限流。可以调大请求间隔后重试：

```bash
python3 crawlers/wzry_skin_crawler.py --sleep 0.2
```

### Ollama 连接失败

确认 Ollama 服务已启动，并且 `--host` 地址正确：

```bash
ollama list
```

如果该命令也失败，先启动或安装 Ollama。

## 11. 推荐的新手运行顺序

```bash
cd /home/mzhyui/git/emogame
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 crawlers/wzry_skin_crawler.py --limit 10
sqlite3 data/wzry_skins/skins.sqlite3 "select count(*) from skins;"
python3 scripts/check_wzry_data.py
python3 scripts/evaluate_skin.py --search 地狱岩魂
python3 test_wzry_skins.py
```

完成以上步骤后，本地 Python 环境、采集脚本、SQLite 数据写入和本地图片数据检查就基本跑通了。
