#!/usr/bin/env bash
# download.sh — fetch the checkpoint into ./models (or MODELS_DIR).
#
#   turboderp/Qwen3.8-Flash-Next-exl3, 5.05 bpw branch at the pinned revision
#   ~115 GiB, of which the n-gram (PLE) table ngram_embedding.safetensors is
#   ~37 GiB. The MTP drafter is inside the checkpoint.
#
# Put MODELS_DIR on an SSD: the server reads PLE rows from that file while it
# decodes. Budget ~120 GiB of free disk. Uses the hf client from the engine
# image (or the venv, RUNTIME=native), so run ./install.sh first.
# Runs the preflight first, so it stops before downloading if a card reports
# under 60 GiB or the disk is too small.
#
# Already present? Exits 0. Re-fetch:  REFRESH_WEIGHTS=1 ./download.sh
#
# Equivalent to: ./start.sh download
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
