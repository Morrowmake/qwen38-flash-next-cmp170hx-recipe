#!/usr/bin/env bash
# stop.sh — stop the server this checkout started.
#
# Only ever signals the PID in logs/vllm.pid, and only after confirming that
# process is ours. It never searches by process name, so an unrelated vLLM on
# this machine is never touched. Nothing running? Exits 0.
#
# Weights, the venv and the compile caches stay on disk, so a later start is
# a restart rather than a rebuild.
#
# Equivalent to: ./start.sh stop
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" stop
