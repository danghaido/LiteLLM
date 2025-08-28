# Moi truong
Neu khong co uv: pip install uv

B1: uv venv
B2: source .venv/bin/activate
B3: uv sync

# API Key

## Huggingface
Nho phai co Make calls to Inference Providers

## OpenAI, Gemini
Key mat phi moi chay duoc

# Arize phoenix
Su dung docker de chay local

cd EvaluateLLM/Phoenix/ci
docker compose up (Windows)
sudo docker compose up (Linux)

# Run scripts
B1: Cap quyen cho scripts
chmod +x run_scripts/query.sh run_scripts/auto_query.sh

B2: Chay scripts

./run_scripts/query.sh

./run_scripts/auto_query.sh

# Run query Scripts tren windows

python -m LiteLLM.scripts.query

python -m LiteLLM.scripts.auto_query

# Evaluation metrics

python -m LiteLLM.scripts.evaluation
## Available Metrics

### 1. **Q&A Evaluation** (`"Q&A"` or `"qa"`)
### 2. **Hallucination Detection** (`"hallucination"`)
### 3. **Relevance Evaluation** (`"relevance"`)
### 4. **Toxicity Detection** (`"toxicity"`)
### 5. **Human Evaluation** (`"human_evaluation"`)
### 6. **Custom Evaluation** (`"custom"`)