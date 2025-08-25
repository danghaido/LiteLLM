# Moi truong
Neu khong co uv: pip install uv

B1: uv venv
B2: source .venv/bin/activate
B3: uv sync

# API Key

## Huggingface
Nho phai co Make calls to Inference Providers

# Run scripts
B1: Cap quyen cho scripts
chmod +x run_scripts/query.sh run_scripts/auto_query.sh

B2: Chay scripts
./run_scripts/query.sh

./run_scripts/auto_query.sh

# Run Scripts tren windows

python -m LiteLLM.scripts.query

python -m LiteLLM.scripts.batch_query