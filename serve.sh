#!/bin/bash
# Launch Qwen3.8-Flash-Next (EXL3 5.05 bpw) on two CMP 170HX cards with whole-expert EP.
#
# Usage: serve.sh [mtp|none|--help]
#   (no argument)  same as "mtp" -- the default.
#   mtp            native MTP drafter, num_speculative_tokens = SPEC_N (default below).
#   none           no speculative decoding.
# Set DRY=1 to print the environment and the command that would run instead
# of running them.
#
# Paths (all relative to this repo unless overridden):
#   VENV            venv from ./start.sh install    (default ./venv)
#   MODEL           checkpoint directory, on an SSD (default ./models/Qwen3.8-Flash-Next-exl3)
#   CUDA_HOME       CUDA toolkit root               (default /usr/local/cuda-13.3)
# API:
#   HOST            address to listen on (default 127.0.0.1: this machine only).
#                   0.0.0.0 listens on every interface -- set API_KEY first.
#   PORT            default below
#   API_KEY         if set, every /v1 request must send "Authorization: Bearer
#                   <key>" (passed to vLLM as VLLM_API_KEY, never on the command line).
#   SERVED_MODEL_NAME  model id clients send (default below)
# Cards: CUDA_VISIBLE_DEVICES selects exactly two distinct physical cards
#   (nvidia-smi numbering; default below).
# Env overrides: MAX_SEQS, MAX_BATCHED, SPEC_N, LOCAL_ARGMAX, GRAPH_MODE,
#   PLE_PLACEMENT, REASONING_PARSER, TOOL_PARSER, EXTRA_ARGS, P2P, BOOT_CHECK.
#   MAX_LEN must stay 262144 and GPU_UTIL 0.95 (refused otherwise);
#   FIRST_LIGHT=1 is the 32K/eager/no-MTP diagnostic, not a serving mode.
#
# ===== sm_80 feature switches ===============================================
# Each has a one-variable kill switch: set it to 0 in the environment and
# restart, no rebuild.
#   VLLM_QWEN4_QSA_TOPK_CANONICAL=0  canonical sparse-attention top-k ties
#   VLLM_CUSTOM_ALLREDUCE_FLAGS=0    flags-in-data two-shot TP2 all-reduce
#   VLLM_EXL3_PLE_OVERLAP=0          PLE row reads overlapping the previous layer
#   VLLM_EXL3_MOE_V2=0               EXL3 MoE kernel v2 for MoE calls of 1-64 rows
#   VLLM_EXL3_DENSE_SPLITK=0         EXL3 split-K dense kernel for small decode launches
#   P2P=auto                         verify peer content before enabling P2P
#   P2P=off                          disable all peer transports
#   P2P=force                        bypass the content check
# PLE placement (PLE_PLACEMENT): ssd (engine default, the shipped target),
#   ram or device (comparison placements, not tested as serving defaults).
# ============================================================================
set -euo pipefail
# generated:serving-defaults:begin
# Serving defaults for Qwen3.8-Flash-Next; recipe copies are generated.
# Constants are deliberately not exported. Launcher knobs retain their existing
# unset-versus-empty policy, diagnostics and deployment checks. An engine switch
# is listed only under its current name and only where its serving value differs
# from the code default (the PLE placement default, ssd, is the engine's own).
QWEN_DEFAULT_FIRST_LIGHT=0
QWEN_DEFAULT_MODE=mtp
QWEN_DEFAULT_DIAGNOSTIC_MODE=none
QWEN_DEFAULT_LAYOUT=tp2
QWEN_DEFAULT_CUDA_VISIBLE_DEVICES=0,1
QWEN_DEFAULT_VLLM_QWEN4_QSA_TOPK_CANONICAL=1
QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS=1
QWEN_DEFAULT_VLLM_EXL3_PLE_OVERLAP=1
QWEN_DEFAULT_VLLM_EXL3_MOE_V2=1
QWEN_DEFAULT_VLLM_EXL3_DENSE_SPLITK=1
QWEN_DEFAULT_VLLM_USE_BREAKABLE_CUDAGRAPH=1
QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1
QWEN_DEFAULT_NCCL_P2P_LEVEL=SYS
QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
QWEN_DEFAULT_MAX_LEN=262144
QWEN_DEFAULT_DIAGNOSTIC_MAX_LEN=32768
QWEN_DEFAULT_GPU_UTIL=0.95
QWEN_DEFAULT_SPEC_N=3
QWEN_DEFAULT_LOCAL_ARGMAX=0
QWEN_DEFAULT_MAX_SEQS=8
QWEN_DEFAULT_MAX_BATCHED=3460
QWEN_DEFAULT_REASONING_PARSER=qwen3
QWEN_DEFAULT_TOOL_PARSER=qwen3_coder
QWEN_DEFAULT_PORT=8000
QWEN_DEFAULT_SERVED_MODEL_NAME=qwen3.8-flash-next
QWEN_DEFAULT_TP=2
QWEN_DEFAULT_PP=1
QWEN_DEFAULT_DTYPE=bfloat16
QWEN_DEFAULT_KV_CACHE_DTYPE=bfloat16
QWEN_DEFAULT_ATTENTION_CONFIG='{"indexer_kv_dtype":"bf16"}'
QWEN_DEFAULT_ENGRAM_CONFIG='{"cpu_offload":false}'
QWEN_DEFAULT_GRAPH_CONFIG='{"cudagraph_mode":"PIECEWISE"}'
QWEN_DEFAULT_DIAGNOSTIC_GRAPH_CONFIG='{"cudagraph_mode":"NONE"}'
QWEN_DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD=0
# generated:serving-defaults:end
FIRST_LIGHT=${FIRST_LIGHT:-$QWEN_DEFAULT_FIRST_LIGHT}
case "$FIRST_LIGHT" in 0|1) ;; *) echo 'FIRST_LIGHT must be 0 or 1' >&2; exit 2;; esac
if [ "$FIRST_LIGHT" = 1 ]; then MODE=${1:-$QWEN_DEFAULT_DIAGNOSTIC_MODE}; else MODE=${1:-$QWEN_DEFAULT_MODE}; fi
case "$MODE" in
  -h|--help) sed -n '2,41p' "$0"; exit 0 ;;
  mtp|none) ;;
  *) echo "serve.sh: unknown argument '$MODE' (expected mtp, none or --help)" >&2; exit 2 ;;
esac
# Container compilers use a private temporary directory on the cache mount.
# This runs for start.sh, Compose and direct launches before Python imports.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    [ "$RECIPE_CACHE_ROOT" = /cache ] || { echo '[boot-check] FAIL: container cache root must be /cache' >&2; exit 1; }
    temp_base="${TMPDIR:-/cache/tmp}"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR must be under /cache' >&2; exit 1 ;; esac
    mkdir -p "$temp_base"
    temp_base="$(realpath "$temp_base")"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR resolves outside /cache' >&2; exit 1 ;; esac
    TMPDIR="$(mktemp -d "$temp_base/run.XXXXXXXX")"
    export TMPDIR
    export TMP="$TMPDIR" TEMP="$TMPDIR"
    unset temp_base
fi
# Cache seeds are opt-in until the image's cold/warm acceptance passes.
if [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ] && [ "${DRY:-0}" = 1 ]; then
    echo "[image-cache] enabled (dry run)"
elif [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ]; then
    [ -f /opt/image-tools/cache_seed.py ] || { echo "[image-cache] enabled but image helper missing" >&2; exit 1; }
    cache_exports="$(/opt/venv/bin/python /opt/image-tools/cache_seed.py seed /opt/cache-seed)"
    eval "$cache_exports"
    unset cache_exports
else
    echo "[image-cache] disabled"
fi
# Explicit compiler destinations also cover launches without a seed.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    export HOME="${HOME:-/cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    export HF_HOME="${HF_HOME:-/cache/huggingface}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
    export VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR="${VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR:-$HOME/.cache/vllm/custom_allreduce_flags}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$HOME/.tilelang/cache/tmp}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/torchinductor}"
    for cache_path in "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
        "$VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR" "$VLLM_CACHE_ROOT" "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR" \
        "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR"; do
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: compiler cache outside /cache: $cache_path" >&2; exit 1 ;; esac
        mkdir -p "$cache_path"
        cache_path="$(realpath "$cache_path")"
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: cache symlink outside /cache: $cache_path" >&2; exit 1 ;; esac
        [ -w "$cache_path" ] || { echo "[boot-check] FAIL: cache is not writable: $cache_path" >&2; exit 1; }
    done
    unset cache_path
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
MODEL="${MODEL:-$REPO_ROOT/models/Qwen3.8-Flash-Next-exl3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
LAYOUT=${LAYOUT:-$QWEN_DEFAULT_LAYOUT}
if [ "$LAYOUT" != tp2 ]; then echo 'EXL3 requires LAYOUT=tp2' >&2; exit 2; fi
TP=$QWEN_DEFAULT_TP PP=$QWEN_DEFAULT_PP
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$QWEN_DEFAULT_CUDA_VISIBLE_DEVICES}
# Card numbers are physical (nvidia-smi) numbers.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] ||
   [ "${CUDA_VISIBLE_DEVICES%,*}" = "${CUDA_VISIBLE_DEVICES#*,}" ]; then
  echo 'CUDA_VISIBLE_DEVICES must select exactly two distinct cards' >&2; exit 2
fi
# Engine switches are exported only where the value differs from the engine default; the
# PLE placement is exported only when PLE_PLACEMENT or VLLM_EXL3_PLE_PLACEMENT asks for one.
PLE_PLACEMENT=${PLE_PLACEMENT:-${VLLM_EXL3_PLE_PLACEMENT:-}}
if [ -n "$PLE_PLACEMENT" ]; then
  case "$PLE_PLACEMENT" in
    ssd|ram|device) ;;
    *) echo 'PLE placement must be ssd, ram or device' >&2; exit 2 ;;
  esac
  export VLLM_EXL3_PLE_PLACEMENT=$PLE_PLACEMENT
fi
# Canonical QSA top-k ties (lowest block index wins), so a request repeats at exact score ties.
export VLLM_QWEN4_QSA_TOPK_CANONICAL=${VLLM_QWEN4_QSA_TOPK_CANONICAL:-$QWEN_DEFAULT_VLLM_QWEN4_QSA_TOPK_CANONICAL}
# Flags-in-data two-shot custom all-reduce kernel for the TP2 all-reduces it covers.
export VLLM_CUSTOM_ALLREDUCE_FLAGS=${VLLM_CUSTOM_ALLREDUCE_FLAGS:-${QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS:-0}}
# SSD PLE row reads overlap the preceding decoder layer instead of stalling the step.
export VLLM_EXL3_PLE_OVERLAP=${VLLM_EXL3_PLE_OVERLAP:-${QWEN_DEFAULT_VLLM_EXL3_PLE_OVERLAP:-0}}
# EXL3 MoE kernel v2 for MoE calls of 1-64 rows (decode and short prefills).
export VLLM_EXL3_MOE_V2=${VLLM_EXL3_MOE_V2:-${QWEN_DEFAULT_VLLM_EXL3_MOE_V2:-0}}
# EXL3 split-K dense kernel for the small-row (tiny and small tile) launches.
export VLLM_EXL3_DENSE_SPLITK=${VLLM_EXL3_DENSE_SPLITK:-${QWEN_DEFAULT_VLLM_EXL3_DENSE_SPLITK:-0}}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-$QWEN_DEFAULT_VLLM_USE_BREAKABLE_CUDAGRAPH}
if [ "$FIRST_LIGHT" = 0 ] && [ "$VLLM_USE_BREAKABLE_CUDAGRAPH" != 1 ]; then
  echo 'Production EXL3 requires VLLM_USE_BREAKABLE_CUDAGRAPH=1' >&2; exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-$QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF}
# Content-verified peer access, shared by both runtimes: sets
# VLLM_CUSTOM_ALLREDUCE_PCIE_P2P and NCCL_P2P_LEVEL only after a pass.
source "$REPO_ROOT/p2p_check.sh"
FL_ARGS=()
# The graph mode defaults come from the serving defaults; GRAPH_MODE overrides them.
if [ "$FIRST_LIGHT" = 1 ]; then DEFAULT_GRAPH_CONFIG=$QWEN_DEFAULT_DIAGNOSTIC_GRAPH_CONFIG; else DEFAULT_GRAPH_CONFIG=$QWEN_DEFAULT_GRAPH_CONFIG; fi
DEFAULT_GRAPH_MODE=${DEFAULT_GRAPH_CONFIG#*\"cudagraph_mode\":\"}
DEFAULT_GRAPH_MODE=${DEFAULT_GRAPH_MODE%%\"*}
GRAPH_MODE=${GRAPH_MODE:-$DEFAULT_GRAPH_MODE}
case "$GRAPH_MODE" in PIECEWISE|FULL|NONE) ;; *) echo 'GRAPH_MODE must be PIECEWISE, FULL or NONE' >&2; exit 2;; esac
if [ "$GRAPH_MODE" = "$DEFAULT_GRAPH_MODE" ]; then
  GRAPH_ARGS=(--compilation-config "$DEFAULT_GRAPH_CONFIG")
else
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$GRAPH_MODE\"}")
fi
if [ "$FIRST_LIGHT" = 1 ]; then
  if [ "$MODE" != none ] || [ "${MAX_LEN-$QWEN_DEFAULT_DIAGNOSTIC_MAX_LEN}" != 32768 ] || [ "$GRAPH_MODE" != NONE ]; then
    echo 'FIRST_LIGHT=1 requires no MTP, MAX_LEN=32768 and GRAPH_MODE=NONE' >&2; exit 2
  fi
  FL_ARGS=(--enforce-eager)
  MAX_LEN=$QWEN_DEFAULT_DIAGNOSTIC_MAX_LEN
else
  if [ "$GRAPH_MODE" = NONE ]; then
    echo 'GRAPH_MODE=NONE requires FIRST_LIGHT=1' >&2; exit 2
  fi
  MAX_LEN=${MAX_LEN-$QWEN_DEFAULT_MAX_LEN}
  if [ "$MAX_LEN" != 262144 ]; then
    echo 'Serving requires MAX_LEN=262144; use FIRST_LIGHT=1 for the 32768 diagnostic' >&2; exit 2
  fi
fi
GPU_UTIL=${GPU_UTIL-$QWEN_DEFAULT_GPU_UTIL}
if [ "$GPU_UTIL" != 0.95 ]; then
  echo 'EXL3 requires GPU_UTIL=0.95, including FIRST_LIGHT=1' >&2; exit 2
fi
SPEC=()
if [ "$MODE" = mtp ]; then
  SPEC_N=${SPEC_N:-$QWEN_DEFAULT_SPEC_N}
  if [[ ! "$SPEC_N" =~ ^[1-9][0-9]*$ ]]; then echo 'SPEC_N must be positive' >&2; exit 2; fi
  # LOCAL_ARGMAX=1: greedy drafts take each rank's local argmax over its vocab shard and gather (value, index)
  # pairs instead of gathering full-vocab draft logits. Same draft tokens.
  LOCAL_ARGMAX=${LOCAL_ARGMAX:-${QWEN_DEFAULT_LOCAL_ARGMAX:-0}}
  case "$LOCAL_ARGMAX" in 0|1) ;; *) echo 'LOCAL_ARGMAX must be 0 or 1' >&2; exit 2;; esac
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N"
  if [ "$LOCAL_ARGMAX" = 1 ]; then SPEC_JSON+=",\"use_local_argmax_reduction\":true"; fi
  SPEC=(--speculative-config "$SPEC_JSON}")
fi
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$QWEN_DEFAULT_SERVED_MODEL_NAME}
PORT=${PORT:-$QWEN_DEFAULT_PORT}
# The key goes to vLLM through the environment rather than --api-key, so it
# does not show up in the process list.
if [ -n "${API_KEY:-}" ]; then export VLLM_API_KEY="$API_KEY"; fi
CMD=("$VENV/bin/vllm" serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "${HOST:-127.0.0.1}"
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" --enable-expert-parallel
  --dtype "$QWEN_DEFAULT_DTYPE" --kv-cache-dtype "$QWEN_DEFAULT_KV_CACHE_DTYPE"
  --load-format exl3 --mm-encoder-tp-mode data
  --attention-config "$QWEN_DEFAULT_ATTENTION_CONFIG"
  --engram-config "$QWEN_DEFAULT_ENGRAM_CONFIG"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "${MAX_SEQS:-$QWEN_DEFAULT_MAX_SEQS}" --max-num-batched-tokens "${MAX_BATCHED:-$QWEN_DEFAULT_MAX_BATCHED}"
  --long-prefill-token-threshold "$QWEN_DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD"
  --reasoning-parser "${REASONING_PARSER:-$QWEN_DEFAULT_REASONING_PARSER}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-$QWEN_DEFAULT_TOOL_PARSER}"
  --port "$PORT"
  ${FL_ARGS[@]+"${FL_ARGS[@]}"} "${GRAPH_ARGS[@]}" ${SPEC[@]+"${SPEC[@]}"})
# Additional arguments are whitespace-separated.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
# FlexibleArgumentParser accepts abbreviations, underscores and dotted JSON keys.
# Config files and model overrides can change several controlled fields at once.
CONTROLLED_ARGS=(
  --config --model --hf-config-path --hf-overrides --model-class-overrides
  --model-impl --runner --convert --load-format --model-loader-extra-config
  --device-ids --distributed-executor-backend --nnodes --node-rank
  --tensor-parallel-size --pipeline-parallel-size
  --decode-context-parallel-size --prefill-context-parallel-size
  --data-parallel-size --data-parallel-size-local --data-parallel-rank
  --data-parallel-start-rank --data-parallel-address --data-parallel-rpc-port
  --data-parallel-backend --data-parallel-hybrid-lb --data-parallel-external-lb
  --data-parallel-multi-port-external-lb --enable-expert-parallel
  --enable-elastic-ep --elastic-ep-max-dp-size
  --dtype --quantization --quantization-config --kv-cache-dtype
  --kv-cache-dtype-skip-layers --mamba-cache-dtype --mamba-ssm-cache-dtype
  --attention-config --compilation-config --optimization-level --performance-mode
  --cudagraph-capture-sizes --max-cudagraph-capture-size
  --enforce-eager --engram-config --speculative-config
  --spec-method --spec-model --spec-tokens --diffusion-config
  --language-model-only --limit-mm-per-prompt --mm-encoder-only
  --mm-encoder-tp-mode --mm-encoder-attn-dtype
  --max-model-len --gpu-memory-utilization --device-memory-utilization --kv-cache-memory-bytes
  --num-gpu-blocks-override
  # No KV or weight offload; the n-gram table placement is PLE_PLACEMENT.
  --cpu-offload-gb --cpu-offload-params --offload-backend --offload-group-size --offload-num-in-group
  --offload-prefetch-step --offload-params --kv-offloading-size --kv-offloading-backend --kv-transfer-config
)
for argument in ${EXTRA[@]+"${EXTRA[@]}"}; do
  blocked=0
  case "$argument" in
    --) blocked=1 ;;
    --*)
      option=${argument%%=*}
      option=${option%%.*}
      option=${option//_/-}
      for controlled in "${CONTROLLED_ARGS[@]}"; do
        if [[ "$controlled" == "$option"* || "--no-${controlled#--}" == "$option"* ]]; then
          blocked=1; break
        fi
      done ;;
    -*)
      option=${argument%%=*}
      option=${option%%.*}
      for controlled in -tp -pp -dp -dcp -pcp -ep -n -r -q -sc -cc -ac -dc -O; do
        if [[ "$option" == "$controlled"* || "$controlled" == "$option"* ]]; then
          blocked=1; break
        fi
      done ;;
  esac
  if [ "$blocked" = 1 ]; then
    echo "EXTRA_ARGS cannot override the EXL3 deployment contract: $argument" >&2; exit 2
  fi
done
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})
echo "serve.sh: EXL3 enabled TP=2 PP=1 whole-expert-EP GPUs=$CUDA_VISIBLE_DEVICES PLE=${VLLM_EXL3_PLE_PLACEMENT:-default} dtype=BF16 KV=BF16 indexer=BF16 vision=on MTP=$MODE FIRST_LIGHT=$FIRST_LIGHT"
echo "serve.sh: Qwen4Exp CUDA graph mode=$GRAPH_MODE (GRAPH_MODE=$GRAPH_MODE)"
BOOT_CHECK=${BOOT_CHECK:-1}
case "$BOOT_CHECK" in 0|1) ;; *) echo "[boot-check] FAIL: BOOT_CHECK must be 0 or 1" >&2; exit 2 ;; esac
if [ "${DRY:-0}" = 1 ]; then
  echo "[boot-check] BOOT_CHECK=$BOOT_CHECK (dry run)"
  echo "serve.sh: DRY=1, layout $LAYOUT (PP=$PP TP=$TP), environment the server would get:"
  env | LC_ALL=C sort | grep -E '^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=|CUDA_VISIBLE_DEVICES=|CUDA_DEVICE_ORDER=)' \
    | grep -v '^VLLM_API_KEY=' | sed 's/^/  /' || true
  if [ -n "${VLLM_API_KEY:-}" ]; then echo "  VLLM_API_KEY=(set, not shown)"; else echo "  (no API key: /v1 is open to anyone who can reach HOST:PORT)"; fi
  echo "serve.sh: DRY=1, command:"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
if [ "$BOOT_CHECK" = 0 ]; then
  echo "[boot-check] disabled (BOOT_CHECK=0)"
elif [ "${RECIPE_BOOT_CHECK_OWNER:-}" = start ]; then
  echo "[boot-check] delegated to start.sh after health"
else
  CLIENT_HOST=${HOST:-127.0.0.1}
  case "$CLIENT_HOST" in 0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;; esac
  boot_args=()
  [ -z "${RECIPE_CACHE_ROOT:-}" ] || boot_args+=(--container)
  exec "$VENV/bin/python" "$REPO_ROOT/boot_check.py" ${boot_args[@]+"${boot_args[@]}"} \
    --base "http://$CLIENT_HOST:$PORT" \
    --model "$SERVED_MODEL_NAME" \
    --ready-timeout "${READY_TIMEOUT:-1800}" --serve "${CMD[@]}"
fi
exec "${CMD[@]}"
