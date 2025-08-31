# Khởi tạo môi trường (Sử dụng uv)
pip install uv

B1: uv venv

B2: source .venv/bin/activate

B3: uv sync

# Arize phoenix
## Run Phoenix locally using Docker.

### Steps

- Open terminal

- cd EvaluateLLM/Phoenix/ci

- docker compose up (Windows)

- sudo docker compose up (Linux)

- Then open: http://localhost:6006/

# Configs

Project parameters can be changed via huggingface.yaml

## API Key

### Huggingface
Create an API key with “Make Calls” permission to use Inference Providers.

### OpenAI, Gemini
Paid keys are required for successful runs.

## Retrieve
Configure locations/models for:
- Database path

- Embedding model

- Reranker model

- top_k

(Manage these in your config as appropriate for your project.)

# Run scripts

## Make scripts executable:
chmod +x run_scripts/query.sh run_scripts/auto_query.sh

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