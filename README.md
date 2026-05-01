# LiteLLM Project

## Giới thiệu

Đây là project LiteLLM phục vụ truy vấn, đánh giá và theo dõi (tracing) cho các bài toán RAG/LLM.
Project có tích hợp Arize Phoenix để quan sát trace và đánh giá kết quả trong quá trình chạy.

## Hướng dẫn chạy

### B1: Chạy Phoenix bằng Docker Compose

Mở terminal và chạy lệnh trong thư mục `phoenix_tools/ci`:

```bash
cd phoenix_tools/ci
docker compose up
```

Sau khi chạy thành công, truy cập: http://localhost:6006/

### B2: Chạy script theo hệ điều hành

- Ubuntu/Linux: dùng script trong thư mục `run_scripts`
- Windows: dùng script trong thư mục `run_scripts_WIN`

Ví dụ:

```bash
# Ubuntu/Linux
./run_scripts/query.sh

# Windows (PowerShell hoặc CMD)
run_scripts_WIN\query.bat
```

## Khởi tạo môi trường (uv)

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync
```

# Configs

Project parameters can be changed via dev.yaml

## Env-driven YAML

Config loader now supports placeholders in YAML:

- `${ENV_NAME}`: read from environment variable, fallback to empty string if missing.
- `${ENV_NAME:default_value}`: read from environment variable, fallback to `default_value`.

You can define an optional `env` block in config file. All values inside `env` will be loaded into runtime environment before app logic starts.

Example:

```yaml
env:
	HUGGINGFACE_API_KEY: ${HUGGINGFACE_API_KEY:API_KEY}
	OPENAI_API_KEY: ${OPENAI_API_KEY:API_KEY}

env_key: HUGGINGFACE_API_KEY
api_key: ${HUGGINGFACE_API_KEY:API_KEY}
```

Switch config file by environment:

- `APP_ENV=dev` -> load `configs/dev.yaml`
- `APP_ENV=staging` -> load `configs/staging.yaml`

## API Key

### Huggingface

Create an API key with “Make Calls” permission to use Inference Providers.

## Retrieve

Configure locations/models for:

- Database path
- Embedding model (`retrieve.type: local`)
- Cloud embedding (`retrieve.type: cloud`) with:
	- `retrieve.cloud.api_key`
	- `retrieve.cloud.base_url`
	- `retrieve.cloud.model_name`

- Reranker model

- top_k

(Manage these in your config as appropriate for your project.)

# Run scripts


## Make scripts executable:

chmod +x run_scripts/\*.sh

source .env

## Run queries:

### Single query and realtime interact on terminal

./run_scripts/query.sh

### Run full dataset query automatic and have csv evaluation

- First create a new dataset on arize local host http://localhost:6006/

- Set the dataset name in huggingface.yaml

- Run ./run_scripts/auto_query.sh

### Run session and realtime interact through gradio

- ./run_scripts/session.sh

# Evaluation metrics

## Run evaluation with available traces and have csv evaluation file

./run_scripts/evaluation.sh

## Available Metrics

Can add or remove metrics in evaluation file

### 1. **Q&A Evaluation** (`"Q&A"` or `"qa"`)

### 2. **Hallucination Detection** (`"hallucination"`)

### 3. **Relevance Evaluation** (`"relevance"`)

### 4. **Toxicity Detection** (`"toxicity"`)

### 5. **Human Evaluation** (`"human_evaluation"`)

### 6. **Custom Evaluation** (`"custom"`)
