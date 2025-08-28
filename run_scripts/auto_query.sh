#!/bin/bash
# Script để chạy LiteLLM auto_query
export FROM_AUTO_QUERY=true
python -m LiteLLM.scripts.auto_query "$@" && python -m LiteLLM.scripts.evaluation