#!/bin/bash
# Script để chạy LiteLLM auto_query
export FROM_AUTO_QUERY=true
python -m litellm_client.scripts.auto_query "$@" && python -m litellm_client.scripts.evaluation