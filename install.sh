#!/usr/bin/env bash
# install.sh — get the pinned engine.
#
# Container (the default): pulls the engine image pinned in start.sh by digest.
# Native (RUNTIME=native): creates ./venv on Python 3.12, clones
# https://github.com/Morrowmake/vllm-cmp170hx at the commit pinned in .env (or
# start.sh's default), installs it editable with torch 2.13.0+cu130 and the
# runtime extras the engine pins, then verifies every import.
#
# Already there? Exits 0 without touching it.
# Pull or reinstall anyway:  FORCE_INSTALL=1 ./install.sh
# Native only:
#   The sm_80 EXL3 extension is compiled once per pin (needs nvcc and a C++ compiler).
#   Rebuild the venv:        VENV_CLEAR=1 ./install.sh
#   Compile the CUDA extensions instead of using upstream's precompiled ones:
#                            BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh
#
# Equivalent to: ./start.sh install
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" install
