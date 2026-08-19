#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the selected checkpoint directory}"

swift export \
  --adapters "${ADAPTER_PATH}" \
  --merge_lora true

