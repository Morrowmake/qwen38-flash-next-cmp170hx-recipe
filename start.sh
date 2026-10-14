#!/usr/bin/env bash
# ============================================================================
# start.sh — Qwen3.8-Flash-Next EXL3 on 2 CUDA GPUs (sm_80 Ampere patches)
# ============================================================================
#
# We serve turboderp/Qwen3.8-Flash-Next-exl3 (5.05 bpw) on two GPUs with a
# patched vLLM: OpenAI API on 127.0.0.1:8000 as "qwen3.8-flash-next" (local
# only unless you set HOST; no API key unless you set API_KEY), native MTP
# speculation at k=3, 262,144-token context, vision on, tensor-parallel 2 with
# whole routed experts per card. Every weight is on the GPUs except the n-gram
# (PLE) table, which is read from the SSD the checkpoint sits on. No FP8, no
# KV quantisation.
#
# The engine runs in a container by default (RUNTIME=container: Docker with the
# NVIDIA Container Toolkit, image pinned by digest), or from a venv built on
# this machine (RUNTIME=native). A checkout that already has a native install
# from an earlier release keeps using it unless RUNTIME says otherwise.
#
# Every step is idempotent and skipped when it is already done, so running
# ./start.sh twice is safe and the second run just launches.
#
# What we do:
#   1. preflight — docker (or uv, python), git, 2 GPUs of >=60 GiB (read only),
#                  link width, disk, port
#   2. install   — the pinned engine image (or venv + vLLM fork), if missing or
#                  the pin moved
#   3. download  — the checkpoint, if missing
#   4. launch    — detached, output to logs/serve.log
#   5. wait      — poll /health up to READY_TIMEOUT, then print KV + model id
#
# Usage:
#   ./start.sh                 preflight, install, download, launch — default
#   ./start.sh install         get the engine only (image, or venv and fork)
#   ./start.sh download        fetch the checkpoint only
#   ./start.sh stop            stop the server this checkout started
#   ./start.sh restart         stop + start
#   ./start.sh status          server + /health + KV cache line
#   ./start.sh logs            follow logs/serve.log
#   ./start.sh update          git pull, reinstall if the pin moved, restart
#   ./start.sh smoke           one chat request and one tool call
#   ./start.sh help            this text
#
# Native installs compile the EXL3 extension (vllm._exl3_C) for sm_80 once per
# engine pin; that needs nvcc at CUDA_HOME and a C++ compiler. The container
# carries the same extension prebuilt.
#
# Config lives in .env, copied from .env.example on first run. A prefix env
# assignment beats .env for every key:
#
#   CUDA_VISIBLE_DEVICES=2,3 ./start.sh restart
#   VLLM_EXL3_PLE_OVERLAP=0 ./start.sh restart
#   IMAGE=<older image> ./start.sh update          # roll back (container), this run only
#   VLLM_COMMIT=<older sha> ./start.sh update      # roll back (native), this run only
#
# To stay rolled back, set IMAGE (or VLLM_COMMIT) in .env; otherwise the next
# plain start or restart goes back to this release's engine.
#
# DRY=1 ./start.sh runs the preflight, says whether it would install or
# download, and prints the server environment and launch command. It installs,
# downloads and launches nothing, and works before the first install. With
# stop, restart or update, DRY=1 also takes the checkout's lock and says what
# it would stop, without stopping, pulling or installing anything.
#
# Lifecycle commands on this checkout are serialised by a flock on
# logs/lifecycle.lock. start/restart/install/download refuse immediately when
# another one holds it; stop waits LOCK_WAIT seconds and then exits 1 WITHOUT
# stopping anything. The lock is per checkout: another clone is not covered.
# A lock left held by a server from an earlier release is recognised and
# cleared, never by signalling anything.
#
# stop only ever signals the PID in logs/vllm.pid (native) or stops the
# container in logs/container.id (container), and only after confirming it is
# the server this checkout launched. It never searches by process or container
# name, so it cannot touch an unrelated vLLM on the same machine.
# ============================================================================
set -euo pipefail

_say() { printf '\033[%sm%s\033[0m %s\n' "$1" "$2" "${*:3}"; }
log()  { _say '1;36' 'qwen38 ·' "$@"; }
warn() { _say '1;33' 'qwen38 !' "$@" >&2; }
die()  { _say '1;31' 'qwen38 ✗' "$@" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

usage() {
    sed -n '3,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '/^set -euo pipefail/d' -e 's/^# \{0,1\}//'
}

# ---------------------------- .env handling --------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || die "missing .env.example"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    log "wrote .env from .env.example — edit it, or override any key inline"
fi
# .env supplies defaults for keys the caller has not already set. It is parsed
# as KEY=VALUE, never sourced, so a config file cannot run code and no shell
# expansion happens inside a value. A key already present in the environment is
# left alone even when it is explicitly empty, which is what makes
# `KEY= ./start.sh` mean "unset this knob" rather than "use the .env value".
#
# Values: a value in matching quotes ("..." or '...') is taken exactly as
# written between them; a note after the closing quote is ignored. An unquoted
# value ends at the first whitespace-then-# (`KEY=v  # note` gives `v`), then a
# trailing note in parentheses after two or more spaces is dropped
# (`KEY=v   (default: x)` gives `v`), then surrounding whitespace is trimmed.
# A # or ( with no whitespace before it (`a#b`, `f(x)`) or after a single
# space (`a (b)`) stays in the value; quote the value to keep anything else.
load_env_defaults() {
    local file="$1" line key val raw
    local re_dq='^"([^"]*)"[[:space:]]*(#.*|\(.*\))?$'
    local re_sq="^'([^']*)'[[:space:]]*(#.*|\\(.*\\))?\$"
    local re_paren='^(.*)[[:space:]][[:space:]]+\([^()]*\)[[:space:]]*$'
    [ -r "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"          # ltrim
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in export\ *) line="${line#export }" ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"; val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"             # rtrim
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -n "${!key+x}" ] && continue                   # caller already set it
        raw="$val"
        val="${val#"${val%%[![:space:]]*}"}"             # ltrim
        if [[ $val =~ $re_dq || $val =~ $re_sq ]]; then  # quoted: verbatim
            val="${BASH_REMATCH[1]}"
        else
            case "$val" in                               # quotes around a value
                \"*\") val="${val#\"}"; val="${val%\"}" ;;   # that holds quotes
                \'*\') val="${val#\'}"; val="${val%\'}" ;;
                *)                                       # unquoted: drop notes
                    val="${raw%%[[:space:]]#*}"          # (KEY= # x is empty)
                    [[ $val =~ $re_paren ]] && val="${BASH_REMATCH[1]}"
                    val="${val#"${val%%[![:space:]]*}"}"
                    val="${val%"${val##*[![:space:]]}"}" ;;      # trim
            esac
        fi
        export "$key=$val"
    done <"$file"
}
load_env_defaults "$SCRIPT_DIR/.env"

# ---------------------------- configuration --------------------------------
VENV="${VENV:-$SCRIPT_DIR/venv}"
VLLM_SRC="${VLLM_SRC:-$SCRIPT_DIR/vllm-src}"
VLLM_REPO="${VLLM_REPO:-https://github.com/Morrowmake/vllm-cmp170hx.git}"
VLLM_BRANCH="${VLLM_BRANCH:-ampere}"
# This release's engine: the fork commit, the upstream nightly wheel whose
# compiled extensions match that commit's upstream base, and the container
# image built from them. All three are filled in at release
# (tools/set-pin.sh moves the commit pin everywhere at once).
RELEASE_VLLM_COMMIT={{PIN}}
RELEASE_WHEEL_COMMIT={{WHEEL_PIN}}
RELEASE_IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:{{IMAGE_DIGEST}}
VLLM_COMMIT="${VLLM_COMMIT:-$RELEASE_VLLM_COMMIT}"
IMAGE="${IMAGE:-$RELEASE_IMAGE}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
# The checkpoint: the 5.05 bpw branch, pinned to the revision the release was
# validated with. The MTP drafter is inside it.
TARGET_REPO="${TARGET_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
TARGET_REVISION="${TARGET_REVISION:-{{CHECKPOINT_REVISION}}}"
MODEL="${MODEL:-$MODELS_DIR/${TARGET_REPO##*/}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
BUILD_FROM_SOURCE="${BUILD_FROM_SOURCE:-0}"
MAX_JOBS="${MAX_JOBS:-16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
# One layout: tp2 (tensor-parallel 2, whole routed experts per card).
LAYOUT="${LAYOUT:-tp2}"
[ "$LAYOUT" = tp2 ] || die "LAYOUT must be tp2, not '$LAYOUT'"
PP=1; TP=2
export LAYOUT
# The two cards, as nvidia-smi numbers them.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
# Native MTP (mtp, the default) or no speculation (none).
SPEC_MODE="${SPEC_MODE:-mtp}"
case "$SPEC_MODE" in mtp|none) ;; *) die "SPEC_MODE must be mtp or none, not '$SPEC_MODE'" ;; esac
# Where the engine runs. An existing native install (from an earlier release)
# keeps running natively unless RUNTIME is set; a fresh checkout uses the
# container.
if [ -z "${RUNTIME:-}" ]; then
    if [ -f "${VENV}/.recipe-stamp" ] || [ -f "${VENV}/.recipe-reqs" ]; then
        RUNTIME=native
    else
        RUNTIME=container
    fi
fi
case "$RUNTIME" in
    container|native) ;;
    *) die "RUNTIME must be container or native, not '$RUNTIME'" ;;
esac
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"
SHM_SIZE="${SHM_SIZE:-16g}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
BOOT_CHECK="${BOOT_CHECK:-1}"
case "$BOOT_CHECK" in 0|1) ;; *) die "BOOT_CHECK must be 0 or 1" ;; esac
LAUNCHED=0
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
MIN_GPUS="${MIN_GPUS:-2}"
MIN_GPU_MIB="${MIN_GPU_MIB:-61440}"      # 60 GiB; the cards we run report 65,536 MiB
NEED_DISK_GB="${NEED_DISK_GB:-120}"      # GiB, as df -BG counts it: checkpoint 115 GiB
LOCK_WAIT="${LOCK_WAIT:-30}"

LOGDIR="$SCRIPT_DIR/logs"
SERVE_LOG="${SERVE_LOG:-$LOGDIR/serve.log}"
PIDFILE="$LOGDIR/vllm.pid"
LOCKFILE="$LOGDIR/lifecycle.lock"
LOCKPID="$LOGDIR/lifecycle.lock.pid"
STAMP="$VENV/.recipe-stamp"
INSTALL_STEPS="$VENV/.recipe-install-steps"
NATIVE_INSTALL_RESUME="${NATIVE_INSTALL_RESUME:-1}"
FLASHINFER_INSTALL_ATTEMPTS="${FLASHINFER_INSTALL_ATTEMPTS:-3}"
FLASHINFER_HTTP_TIMEOUT="${FLASHINFER_HTTP_TIMEOUT:-30}"
FLASHINFER_INSTALL_TIMEOUT="${FLASHINFER_INSTALL_TIMEOUT:-600}"
CIDFILE="$LOGDIR/container.id"
LABEL="qwen38-recipe.checkout"

# Where our own health checks connect. A wildcard bind address is not a
# destination, so talk to the server over loopback in that case.
case "$HOST" in
    0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;;
    *) CLIENT_HOST="$HOST" ;;
esac

export VENV VLLM_SRC MODEL CUDA_HOME HOST PORT SERVED_MODEL_NAME
export SERVE_LOG READY_TIMEOUT BOOT_CHECK
export MODEL_ID="${MODEL_ID:-$SERVED_MODEL_NAME}"

banner() {
    printf '\n  \033[1;36m┌──────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mQwen3.8-Flash-Next \033[0m  \033[2m·  %-14s\033[0m  \033[1;36m│\033[0m\n' "${1:-start.sh}"
    printf '  \033[1;36m└──────────────────────────────────────────────┘\033[0m\n\n'
}

# ------------------------------- locking -----------------------------------
# The flock is the authoritative owner; $LOCKPID is advisory diagnostics only
# and is never signalled.
#
# A server launched by an earlier release inherited the lock's descriptor and
# held the lock for its whole life. When the lock is busy we therefore look at
# who holds it (read only, via /proc, our own processes only). If every holder
# belongs to the server in $PIDFILE (that pid, or its process group/session
# after setsid) and none is a start.sh, the lock is stale: we unlink the file
# and lock a fresh one. The old server keeps its descriptor on the unlinked
# file, which no longer matters. Nothing is ever signalled here; stop, restart
# and update still stop the server through do_stop. The clearing itself is
# serialised by a short-lived second lock, so two commands cannot both clear.

# PIDs (ours to read) with an open descriptor on $LOCKFILE, one per line.
lock_holders() {
    local f p
    for f in /proc/[0-9]*/fd/[0-9]*; do
        [ "$f" -ef "$LOCKFILE" ] || continue
        p="${f#/proc/}"
        echo "${p%%/*}"
    done | sort -u
}

# True when pid $1 is the server pid $2 or runs inside its group/session.
holder_is_server() {
    local h="$1" s="$2" st cl
    cl="$(tr '\0' ' ' <"/proc/$h/cmdline" 2>/dev/null || true)"
    case "$cl" in *start.sh*) return 1 ;; esac
    [ "$h" = "$s" ] && return 0
    read -r st 2>/dev/null <"/proc/$h/stat" || return 1
    st="${st##*) }"                        # fields after "pid (comm) "
    set -- $st                             # state ppid pgrp session ...
    [ "${3:-}" = "$s" ] || [ "${4:-}" = "$s" ]
}

# True when the lock is held, and held only by this checkout's server.
lock_held_by_server_only() {
    local s h holders
    s="$(read_pid)"
    pid_is_ours "$s" || return 1
    holders="$(lock_holders)"
    [ -n "$holders" ] || return 1
    for h in $holders; do
        holder_is_server "$h" "$s" || return 1
    done
    return 0
}

# True when fd 9 is still the file at $LOCKFILE (not one unlinked meanwhile).
lock_is_current() {
    [ -e "/proc/$$/fd/9" ] || return 0
    [ "$LOCKFILE" -ef "/proc/$$/fd/9" ]
}

# Lock $LOCKFILE on fd 9. Returns 0 when locked, 1 when a real command holds
# it. Clears a stale lock (see above) on the way.
acquire_lock() {
    local tries=0
    mkdir -p "$LOGDIR"
    while [ "$tries" -lt 5 ]; do
        tries=$((tries + 1))
        exec 9>"$LOCKFILE"
        if flock -n 9; then
            # Make sure the path still names the file we locked; if another
            # command replaced it meanwhile, go round again.
            lock_is_current && return 0
            exec 9>&-
            continue
        fi
        exec 9>&-
        exec 8>"$LOCKFILE.gate"
        flock -w 10 8 || { exec 8>&-; return 1; }
        exec 9>"$LOCKFILE"
        if flock -n 9; then                  # freed or cleared meanwhile
            exec 8>&-
            lock_is_current && return 0
            exec 9>&-
            continue
        fi
        exec 9>&-
        if lock_held_by_server_only; then
            log "an earlier server was holding the lock; continuing"
            rm -f "$LOCKFILE"
            exec 8>&-
            continue
        fi
        exec 8>&-
        return 1
    done
    return 1
}

take_lock() {
    if ! acquire_lock; then
        local holder; holder="$(tr -d '[:space:]' 2>/dev/null <"$LOCKPID" || true)"
        die "this checkout is busy${holder:+ with pid $holder}. Wait for that command to finish, then try again."
    fi
    echo $$ >"$LOCKPID" 2>/dev/null || true
}
take_lock_for_stop() {
    if ! acquire_lock; then
        exec 9>"$LOCKFILE"
        if ! flock -w "$LOCK_WAIT" 9 || ! lock_is_current; then
            local holder; holder="$(tr -d '[:space:]' 2>/dev/null <"$LOCKPID" || true)"
            warn "gave up waiting ${LOCK_WAIT}s for this checkout${holder:+ (pid $holder has it)}."
            die "The server was left running. Try stop again once that command finishes."
        fi
    fi
    echo $$ >"$LOCKPID" 2>/dev/null || true
}

# ------------------------------ preflight ----------------------------------
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 not found — $2"
}

# preflight [--with-port]   -- the port is only checked when we are about to
# bind it, so `install` and `download` do not fail on a busy port.
preflight() {
    local check_port=0
    [ "${1:-}" = "--with-port" ] && check_port=1
    log "preflight (runtime: $RUNTIME, layout: $LAYOUT)"
    need_cmd git  "install it: apt-get install -y git"
    need_cmd curl "install it: apt-get install -y curl"
    [ "$BOOT_CHECK" = 0 ] || need_cmd python3 "install Python 3 for the boot check"
    if [ "$RUNTIME" = container ]; then
        preflight_docker
    else
        need_cmd uv   "install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
        command -v python3 >/dev/null 2>&1 || die "python3 not found — install Python ${PYTHON_VERSION}"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        # Read only. We never set power, persistence, clocks or fan state.
        local mems n small
        mems="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -z "$mems" ]; then
            warn "nvidia-smi present but returned no GPUs — check the driver"
        else
            n="$(printf '%s\n' "$mems" | grep -c .)"
            small="$(printf '%s\n' "$mems" | awk -v m="$MIN_GPU_MIB" '$1+0 < m' | grep -c . || true)"
            [ "$n" -ge "$MIN_GPUS" ] || die "found $n GPU(s), need $MIN_GPUS"
            if [ "${small:-0}" -gt 0 ]; then
                die "$small GPU(s) report under $((MIN_GPU_MIB/1024)) GiB. This recipe needs about 64 GB on each card (the W4A16 weights take ~45 GiB per card at TP=4, plus the KV cache). Nothing was downloaded. Set MIN_GPU_MIB to override this check."
            fi
            log "  GPUs: $n visible, smallest $(printf '%s\n' "$mems" | sort -n | head -1) MiB"
        fi
        # Read only. Tensor parallelism wants x16 links.
        local widths narrow
        widths="$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -n "$widths" ]; then
            log "  PCIe link width: $(printf '%s' "$widths" | tr -d ' ' | paste -sd, -)"
            narrow="$(printf '%s\n' "$widths" | awk '$1+0 > 0 && $1+0 < 16' | grep -c . || true)"
            if [ "${narrow:-0}" -gt 0 ] && [ "$TP" -gt 1 ]; then
                warn "  $narrow card(s) are below x16 and TP=$TP. Every decode step runs small"
                warn "  collectives between the two cards, so expect less than the published numbers"
                warn "  on narrow links (they were measured on x16). Continuing anyway."
            fi
        fi
    else
        warn "nvidia-smi not found — skipping the GPU check"
    fi

    if [ "$RUNTIME" = native ]; then
        [ -d "$CUDA_HOME" ] || warn "no CUDA toolkit at $CUDA_HOME — Triton and TileLang want one at runtime (set CUDA_HOME)"
    fi

    mkdir -p "$MODELS_DIR"
    local avail_gb; avail_gb="$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
    if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -gt 0 ]; then
        if [ "$avail_gb" -lt "$NEED_DISK_GB" ] && ! models_present; then
            die "only ${avail_gb} GiB free at $MODELS_DIR, need ~${NEED_DISK_GB} GiB for the checkpoints — free space or set MODELS_DIR"
        fi
        log "  disk: ${avail_gb} GiB free at $MODELS_DIR"
    fi

    if [ "$check_port" = 1 ]; then
        if ! port_busy; then
            log "  port $PORT: free"
        elif server_is_ours; then
            log "  port $PORT: in use by our server"
        elif [ -n "${DRY:-}" ]; then
            warn "  port $PORT is in use by something else (not fatal under DRY)"
        else
            die "port $PORT is already taken by another process. Set PORT to a free one, or stop whatever holds it."
        fi
    fi
}

# Docker must be installed and usable by this user, with the NVIDIA Container
# Toolkit for --gpus. Under DRY a missing or unusable docker is only a warning.
preflight_docker() {
    local how="install Docker Engine and the NVIDIA Container Toolkit (see \"What you need\" in the README), or set RUNTIME=native"
    if ! command -v docker >/dev/null 2>&1; then
        [ -n "${DRY:-}" ] && { warn "docker not found — $how"; return 0; }
        die "docker not found — $how"
    fi
    if ! docker info >/dev/null 2>&1; then
        [ -n "${DRY:-}" ] && { warn "docker is installed but this user cannot reach the Docker daemon"; return 0; }
        die "docker is installed but this user cannot reach the Docker daemon (permission denied, or the daemon is not running). Fix that, or set RUNTIME=native."
    fi
    if docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia; then
        log "  docker: ok, NVIDIA runtime present"
    else
        warn "  docker: no 'nvidia' runtime registered — without the NVIDIA Container Toolkit, --gpus fails"
    fi
}

port_busy() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$PORT" 2>/dev/null | tail -n +2 | grep -q .
    else
        curl -fsS -m 2 "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1
    fi
}

# ------------------------------- install -----------------------------------
# Container: the image is the whole engine, so "installed" means the image
# named by IMAGE (a digest) is present locally. A release that moves the pin
# names a new image, which is pulled on the next start or update.
container_image_present() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

# Native: $STAMP holds the fork commit the venv was built at (first line) and a
# hash of the engine's requirements files (second line, "reqs <hash>").
install_done() {
    if [ "$RUNTIME" = container ]; then container_image_present; return; fi
    [ -x "$VENV/bin/vllm" ] || return 1
    [ -f "$STAMP" ] || return 1
    [ -d "$VLLM_SRC/.git" ] || return 1
    local want have
    want="$(git -C "$VLLM_SRC" rev-parse "$VLLM_COMMIT" 2>/dev/null || echo "?")"
    have="$(head -n1 "$STAMP" 2>/dev/null || echo "??")"
    [ "$want" = "$have" ] || return 1
    [ "$(git -C "$VLLM_SRC" rev-parse HEAD 2>/dev/null || echo '???')" = "$want" ] || return 1
    [ "$(reqs_hash)" = "$(sed -n 's/^reqs //p' "$STAMP")" ] || return 1
    [ "$(native_install_key)" = "$(sed -n 's/^native //p' "$STAMP")" ] || return 1
    return 0
}

# The engine's own dependency pins (FlashInfer, TileLang, the quantisation
# kernels, ...) live in its requirements files, so a pin that moves to a new
# upstream base can change them. Their hash decides whether the venv is reused.
reqs_hash() {
    cat "$VLLM_SRC/requirements/common.txt" "$VLLM_SRC/requirements/cuda.txt" 2>/dev/null \
        | sha256sum | cut -c1-16
}

# The EXL3 extension the checkpoint needs, built once per pin and toolchain.
install_exl3() {
    local builder="$VLLM_SRC/csrc/exl3/build_standalone.py"
    local binary="$VLLM_SRC/vllm/_exl3_C.abi3.so" stamp="$VENV/.exl3-stamp"
    [ -f "$builder" ] || die "this engine pin has no EXL3 builder (csrc/exl3/build_standalone.py); select a compatible VLLM_COMMIT."
    [ -x "$CUDA_HOME/bin/nvcc" ] || die "the EXL3 extension needs nvcc at $CUDA_HOME/bin/nvcc; set CUDA_HOME to a CUDA toolkit."
    command -v "${CXX:-c++}" >/dev/null || die "the EXL3 extension needs a C++ compiler (CXX)."
    rm -f "$VLLM_SRC/vllm/_exl3_C.so" "$VLLM_SRC/vllm"/_exl3_C.cpython-*.so
    local key digest have
    key="$({
        git -C "$VLLM_SRC" rev-parse HEAD &&
        git -C "$VLLM_SRC" diff HEAD -- csrc cmake CMakeLists.txt &&
        reqs_hash &&
        CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -c 'import sysconfig, torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI, sysconfig.get_config_var("SOABI"))' &&
        "$CUDA_HOME/bin/nvcc" --version &&
        "${CXX:-c++}" --version
    } | sha256sum | cut -d' ' -f1)" || die "cannot fingerprint the EXL3 build dependencies"
    digest="$(sha256sum "$binary" 2>/dev/null | cut -d' ' -f1 || true)"
    have="$(cat "$stamp" 2>/dev/null || true)"
    if [ -n "$digest" ] && [ "$have" = "$key $digest" ]; then
        log "  EXL3: matching sm_80 extension already installed"
        return 0
    fi
    # Invalidate before compiling: a failed rebuild cannot leave a usable stale library.
    rm -f "$binary" "$stamp"
    log "  EXL3: building the sm_80 extension"
    ( cd "$VLLM_SRC" && CUDA_VISIBLE_DEVICES= \
        CUDA_HOME="$CUDA_HOME" MAX_JOBS="$MAX_JOBS" PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" \
        "$VENV/bin/python" "$builder" --out "$VLLM_SRC/vllm" \
        --build-dir "$VENV/.exl3-build/$key" ) \
        || { rm -f "$binary" "$stamp"; die "EXL3 build failed; fix the compiler/toolkit error and rerun ./start.sh install"; }
    [ -s "$binary" ] || die "the EXL3 builder did not produce $binary"
    digest="$(sha256sum "$binary" | cut -d' ' -f1)"
    printf '%s %s\n' "$key" "$digest" >"$stamp"
}

# FlashInfer is restored after the editable install at the release lock's pins.
flashinfer_requirements() {
    local package pin
    for package in flashinfer-python flashinfer-cubin; do
        pin="$(awk -v p="$package" '$0 ~ "^" p "==" {print}' "$SCRIPT_DIR/docker/CONSTRAINTS")" \
            || return 1
        [[ "$pin" =~ ^$package==[0-9][A-Za-z0-9.+_-]*$ ]] \
            || { warn "missing or ambiguous $package pin in docker/CONSTRAINTS"; return 1; }
        printf '%s\n' "$pin"
    done
}

native_install_key() {
    local pins
    pins="$(flashinfer_requirements)" || return 1
    { printf '%s\n' native-install-v1 "$pins" "$BUILD_FROM_SOURCE" \
        "${VLLM_PRECOMPILED_WHEEL_COMMIT:-$RELEASE_WHEEL_COMMIT}" \
        "${TORCH_INDEX_URL:-}" "$CUDA_HOME"; git -C "$VLLM_SRC" rev-parse HEAD; reqs_hash; } \
        | sha256sum | cut -d' ' -f1
}

install_step_done() {
    [ "$NATIVE_INSTALL_RESUME" = 1 ] && [ "${FORCE_INSTALL:-0}" != 1 ] \
        && [ "$(cat "$INSTALL_STEPS/$1" 2>/dev/null || true)" = "$2" ]
}

write_install_step() {
    mkdir -p "$INSTALL_STEPS"
    printf '%s\n' "$2" >"$INSTALL_STEPS/$1.tmp"
    mv "$INSTALL_STEPS/$1.tmp" "$INSTALL_STEPS/$1"
}

install_flashinfer() {
    local pins attempt
    pins="$(flashinfer_requirements)" || die "cannot read FlashInfer pins from docker/CONSTRAINTS"
    local packages=()
    mapfile -t packages <<<"$pins"
    command -v timeout >/dev/null 2>&1 || die "FlashInfer install needs the timeout command (coreutils)"
    log "  FlashInfer: ${packages[*]} from https://flashinfer.ai/whl/"
    for ((attempt=1; attempt<=FLASHINFER_INSTALL_ATTEMPTS; attempt++)); do
        if timeout --kill-after=5s "${FLASHINFER_INSTALL_TIMEOUT}s" \
            env UV_HTTP_TIMEOUT="$FLASHINFER_HTTP_TIMEOUT" UV_HTTP_RETRIES=0 \
            uv pip install --no-deps --extra-index-url https://flashinfer.ai/whl/ "${packages[@]}"; then
            return 0
        fi
        warn "FlashInfer install attempt $attempt/$FLASHINFER_INSTALL_ATTEMPTS failed (index https://flashinfer.ai/whl/)"
        [ "$attempt" -eq "$FLASHINFER_INSTALL_ATTEMPTS" ] || sleep "$((attempt * 2))"
    done
    die "FlashInfer install failed from https://flashinfer.ai/whl/ after $FLASHINFER_INSTALL_ATTEMPTS attempts (HTTP timeout ${FLASHINFER_HTTP_TIMEOUT}s, attempt timeout ${FLASHINFER_INSTALL_TIMEOUT}s). Rerun ./start.sh install to resume completed steps."
}

do_install() {
    if [ "$RUNTIME" = container ]; then
        container_install; return
    fi
    if install_done && [ "${FORCE_INSTALL:-0}" != "1" ] && [ "${VENV_CLEAR:-0}" != "1" ]; then
        log "install: already at $(head -n1 "$STAMP" | cut -c1-10) — skipping (FORCE_INSTALL=1 to redo)"
        install_exl3
        return 0
    fi
    case "$NATIVE_INSTALL_RESUME" in 0|1) ;; *) die "NATIVE_INSTALL_RESUME must be 0 or 1" ;; esac
    local setting
    for setting in FLASHINFER_INSTALL_ATTEMPTS FLASHINFER_HTTP_TIMEOUT FLASHINFER_INSTALL_TIMEOUT; do
        [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || die "$setting must be a positive integer"
    done
    log "[native-install] resume=$NATIVE_INSTALL_RESUME (NATIVE_INSTALL_RESUME=0 to repeat pending steps)"
    log "install: venv + vLLM fork @ $VLLM_COMMIT"
    export PATH="$CUDA_HOME/bin:$PATH"

    if [ ! -d "$VLLM_SRC/.git" ]; then
        # Blobless, but NOT shallow: vLLM's precompiled-wheel resolution runs
        # `git merge-base` against upstream main, which needs real history.
        git clone --filter=blob:none --branch "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_SRC"
    else
        git -C "$VLLM_SRC" fetch origin "$VLLM_BRANCH"
    fi
    # When the fork's branch moves to a new upstream base, older pins are no
    # longer on it; every pin a release has used is kept reachable by a
    # qwen38-recipe-<version> tag on the fork.
    if ! git -C "$VLLM_SRC" rev-parse --verify -q "$VLLM_COMMIT^{commit}" >/dev/null; then
        log "  $VLLM_COMMIT is not on $VLLM_BRANCH — fetching the release tags"
        git -C "$VLLM_SRC" fetch origin 'refs/tags/qwen38-recipe-*:refs/tags/qwen38-recipe-*' || true
        git -C "$VLLM_SRC" rev-parse --verify -q "$VLLM_COMMIT^{commit}" >/dev/null \
            || die "commit $VLLM_COMMIT is neither on $VLLM_BRANCH nor under a qwen38-recipe-* tag of $VLLM_REPO"
    fi
    # Stay on a named branch rather than a detached HEAD, for the same reason
    # (`git branch --show-current` has to return something).
    git -C "$VLLM_SRC" checkout -B "$VLLM_BRANCH" "$VLLM_COMMIT"
    git -C "$VLLM_SRC" submodule update --init --recursive --depth 1
    log "  HEAD: $(git -C "$VLLM_SRC" log --oneline -1)"

    local reqs old_reqs key
    reqs="$(reqs_hash)"
    old_reqs="$(cat "$VENV/.recipe-reqs" 2>/dev/null || sed -n 's/^reqs //p' "$STAMP" 2>/dev/null || true)"
    if [ -x "$VENV/bin/python" ] && [ "${VENV_CLEAR:-0}" != "1" ] && [ "$reqs" = "$old_reqs" ]; then
        log "  reusing venv at $VENV (VENV_CLEAR=1 to rebuild)"
    else
        if [ -x "$VENV/bin/python" ] && [ "${VENV_CLEAR:-0}" != "1" ]; then
            log "  the engine's requirements differ from the venv's — rebuilding the venv"
        fi
        uv venv --clear --python "$PYTHON_VERSION" "$VENV"
    fi
    printf '%s\n' "$reqs" >"$VENV/.recipe-reqs.tmp"
    mv "$VENV/.recipe-reqs.tmp" "$VENV/.recipe-reqs"
    key="$(native_install_key)" || die "cannot fingerprint native install"
    rm -f "$STAMP"
    export VIRTUAL_ENV="$VENV"

    local pip_args=()
    [ -n "${TORCH_INDEX_URL:-}" ] && pip_args+=(--extra-index-url "$TORCH_INDEX_URL")

    # torch 2.13.0 from PyPI is the cu130 build on linux-x86_64 (torch.version.cuda
    # == '13.0'). Set TORCH_INDEX_URL if your platform resolves differently.
    if ! install_step_done torch "$key"; then
        rm -f "$INSTALL_STEPS/torch" "$INSTALL_STEPS/fork" "$INSTALL_STEPS/extras" "$INSTALL_STEPS/flashinfer"
        log "  torch 2.13.0 (cu130)"
        uv pip install "${pip_args[@]}" "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
        write_install_step torch "$key"
    else
        log "  torch: completed — skipping"
    fi

    if ! install_step_done fork "$key"; then
        rm -f "$INSTALL_STEPS/fork" "$INSTALL_STEPS/extras" "$INSTALL_STEPS/flashinfer"
        rm -f "$VLLM_SRC/vllm/_exl3_C.so" "$VLLM_SRC/vllm/_exl3_C.abi3.so" \
            "$VLLM_SRC/vllm"/_exl3_C.cpython-*.so "$VENV/.exl3-stamp"
        # Keep the base engine precompiled; the EXL3 extension is built separately below.
        if [ "$BUILD_FROM_SOURCE" = "1" ]; then
            log "  vLLM (compiling extensions, MAX_JOBS=$MAX_JOBS)"
            [ -d "$CUDA_HOME" ] || die "BUILD_FROM_SOURCE=1 needs a CUDA toolkit at $CUDA_HOME"
            env -u VLLM_USE_PRECOMPILED \
                TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
                CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 \
                VIRTUAL_ENV="$VENV" CUDA_HOME="$CUDA_HOME" \
                uv pip install "${pip_args[@]}" --no-build-isolation -e "$VLLM_SRC"
        else
            # The extensions come from upstream's nightly wheel for the fork's
            # upstream base, found through `git merge-base`. When the base has no
            # wheel of its own, the release names the wheel to use instead
            # (RELEASE_WHEEL_COMMIT, above); VLLM_PRECOMPILED_WHEEL_COMMIT
            # overrides it for any pin.
            local wheel_commit="${VLLM_PRECOMPILED_WHEEL_COMMIT:-}"
            if [ -z "$wheel_commit" ] && [ "$(git -C "$VLLM_SRC" rev-parse HEAD)" = \
                 "$(git -C "$VLLM_SRC" rev-parse --verify -q "$RELEASE_VLLM_COMMIT^{commit}" || true)" ]; then
                wheel_commit="$RELEASE_WHEEL_COMMIT"
            fi
            log "  vLLM (upstream precompiled extensions${wheel_commit:+ from wheel ${wheel_commit:0:10}})"
            env ${wheel_commit:+VLLM_PRECOMPILED_WHEEL_COMMIT="$wheel_commit"} \
                VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="$VENV" CUDA_HOME="$CUDA_HOME" \
                uv pip install "${pip_args[@]}" -e "$VLLM_SRC"
        fi

        write_install_step fork "$key"
    else
        log "  vLLM editable install: completed — skipping"
    fi

    if ! install_step_done extras "$key"; then
        rm -f "$INSTALL_STEPS/extras" "$INSTALL_STEPS/flashinfer"
        # Preserve relative includes while separating FlashInfer and its index.
        local runtime_reqs="$VENV/.recipe-runtime-requirements"
        mkdir -p "$runtime_reqs"
        cp -a "$VLLM_SRC/requirements/." "$runtime_reqs/"
        awk '!/^[[:space:]]*flashinfer-(python|cubin)([[:space:]=<>!~;]|$)/ &&
             !/^[[:space:]]*--extra-index-url[[:space:]]+https:\/\/flashinfer\.ai\/whl\/?([[:space:]]|$)/' \
            "$VLLM_SRC/requirements/cuda.txt" >"$runtime_reqs/cuda.txt"
        log "  runtime extras (requirements/cuda.txt, FlashInfer installed separately)"
        uv pip install "${pip_args[@]}" -r "$runtime_reqs/cuda.txt" \
            "ninja" "huggingface_hub[hf_xet]>=1.0"
        write_install_step extras "$key"
    else
        log "  runtime extras: completed — skipping"
    fi

    if ! install_step_done flashinfer "$key"; then
        rm -f "$INSTALL_STEPS/flashinfer"
        install_flashinfer
        write_install_step flashinfer "$key"
    else
        log "  FlashInfer: completed — skipping"
    fi

    log "  verifying"
    install_exl3
    "$VENV/bin/python" - <<'PY'
import importlib, sys, torch
print(f"    python      {sys.version.split()[0]}")
print(f"    torch       {torch.__version__} (cuda {torch.version.cuda})")
import vllm
print(f"    vllm        {vllm.__version__}")
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops  # noqa: F401
print("    vllm C ext  ok")
for mod in ("triton", "tilelang", "flashinfer"):
    m = importlib.import_module(mod)
    print(f"    {mod:<11} {getattr(m, '__version__', 'ok')}")
import vllm._exl3_C                                                     # noqa: F401
from vllm.sm80 import envs as sm80_envs                                 # noqa: F401
from vllm.sm80.kernels import exl3                                      # noqa: F401
print("    sm_80 patch modules ok")
PY
    { git -C "$VLLM_SRC" rev-parse HEAD; echo "reqs $reqs"; echo "native $key"; } >"$STAMP.tmp"
    mv "$STAMP.tmp" "$STAMP"
    log "install: done"
}

container_install() {
    if container_image_present && [ "${FORCE_INSTALL:-0}" != "1" ]; then
        log "install: image present — skipping (FORCE_INSTALL=1 to pull again)"
        log "  $IMAGE"
        return 0
    fi
    log "install: docker pull (the engine image, several GB)"
    log "  $IMAGE"
    docker pull "$IMAGE"
    log "install: done"
}

# ------------------------------ download -----------------------------------
model_present() { [ -f "$1/config.json" ]; }
models_present() { model_present "$MODEL"; }

# hf_download <repo> <revision> <local dir>: with the venv's hf CLI (native) or the one
# in the engine image (container, run as this user so the files are yours).
hf_download() {
    local repo="$1" dir="$2"
    mkdir -p "$dir"
    if [ "$RUNTIME" = container ]; then
        container_image_present || die "the engine image is not here yet — run ./start.sh install first"
        docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf \
            ${HF_TOKEN:+-e HF_TOKEN} -v "$dir:/download" \
            --entrypoint /opt/venv/bin/hf "$IMAGE" download "$repo" --local-dir /download
    else
        local hf="$VENV/bin/hf"
        [ -x "$hf" ] || hf="$(command -v hf || true)"
        [ -n "$hf" ] || die "no 'hf' CLI — run ./start.sh install first"
        "$hf" download "$repo" --local-dir "$dir"
    fi
}

do_download() {
    mkdir -p "$MODELS_DIR"

    if model_present "$MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "download: $(basename "$MODEL") already present — skipping"
    else
        log "download: $TARGET_REPO @ $TARGET_REVISION (~115 GiB)"
        hf_download "$TARGET_REPO" "$TARGET_REVISION" "$MODEL"
    fi
    log "download: done"
}

# ------------------------------- process -----------------------------------
read_pid() { [ -f "$PIDFILE" ] && tr -dc '0-9' <"$PIDFILE" || true; }

# True only when $1 is alive AND is the server this checkout launched. This is
# the whole safety story for stop: we never match by process name, so another
# vLLM on this machine is invisible to us.
pid_is_ours() {
    local pid="${1:-}"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    local cl; cl="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    [ -n "$cl" ] || return 1
    case "$cl" in
        *"$MODEL"*) : ;;
        *) return 1 ;;
    esac
    case "$cl" in
        *vllm*) return 0 ;;
        *) return 1 ;;
    esac
}

server_is_ours() {
    if [ "$RUNTIME" = container ]; then container_is_ours; else pid_is_ours "$(read_pid)"; fi
}

# ------------------------------ container ----------------------------------
# The container this checkout started is recorded by id in $CIDFILE and carries
# the label $LABEL=<this checkout>. Only a container with both is ever stopped
# or removed; nothing is matched by name alone.
read_cid() { [ -f "$CIDFILE" ] && tr -dc '0-9a-f' <"$CIDFILE" || true; }

# container_state <id>: "<running> <label>" (true/false), or nothing.
container_state() {
    docker container inspect -f "{{.State.Running}} {{index .Config.Labels \"$LABEL\"}}" "$1" 2>/dev/null || true
}
container_is_ours() {
    local id; id="$(read_cid)"
    [ -n "$id" ] || return 1
    [ "$(container_state "$id")" = "true $SCRIPT_DIR" ]
}

# Address:port on the host the API is published on (IPv6 in brackets).
publish_addr() {
    case "$HOST" in
        \[*) printf '%s:%s' "$HOST" "$PORT" ;;
        *:*) printf '[%s]:%s' "$HOST" "$PORT" ;;
        *) printf '%s:%s' "$HOST" "$PORT" ;;
    esac
}

# The environment the container's serve.sh gets: the in-container paths, then
# every setting of yours that serve.sh reads. serve.sh (mounted from this
# checkout) turns them into the same flags as a native start.
CONTAINER_KEYS="P2P P2P_PREBUILT P2P_CHECK_CACHE LAYOUT CUDA_VISIBLE_DEVICES MAX_LEN MAX_SEQS MAX_BATCHED GPU_UTIL
SPEC_N LOCAL_ARGMAX GRAPH_MODE PLE_PLACEMENT FIRST_LIGHT REASONING_PARSER TOOL_PARSER EXTRA_ARGS
SERVED_MODEL_NAME API_KEY"
container_env() {
    local k
    echo "VENV=/opt/venv"
    echo "MODEL=/models/$(basename "$MODEL")"
    echo "CUDA_HOME=/usr/local/cuda"
    echo "HOST=0.0.0.0"
    echo "PORT=8000"
    echo "BOOT_CHECK=$BOOT_CHECK"
    echo "RECIPE_BOOT_CHECK_OWNER=start"
    # The server runs as you, not root: every cache it writes (Triton,
    # FlashInfer, TileLang, torch, vLLM, Hugging Face) goes under the mounted
    # cache directory, owned by you. USER and LOGNAME give Python a user name
    # for a uid the image does not know.
    echo "PYTHONDONTWRITEBYTECODE=1"
    echo "RECIPE_CACHE_ROOT=/cache"
    echo "TMPDIR=/cache/tmp"
    echo "HOME=/cache"
    echo "USER=$(id -un)"
    echo "LOGNAME=$(id -un)"
    echo "XDG_CACHE_HOME=/cache/.cache"
    echo "HF_HOME=/cache/huggingface"
    echo "FLASHINFER_WORKSPACE_BASE=/cache"
    echo "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor"
    echo "TRITON_CACHE_DIR=/cache/.triton/cache"
    echo "TORCH_EXTENSIONS_DIR=/cache/.cache/torch_extensions"
    echo "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR=/cache/.cache/vllm/custom_all_reduce_flags"
    echo "VLLM_CACHE_ROOT=/cache/.cache/vllm"
    echo "TILELANG_CACHE_DIR=/cache/.tilelang/cache"
    echo "TILELANG_TMP_DIR=/cache/.tilelang/cache/tmp"
    echo "CUDA_CACHE_PATH=/cache/.nv/ComputeCache"
    echo "VLLM_IMAGE_CACHE_SEED=${VLLM_IMAGE_CACHE_SEED:-0}"
    for k in $CONTAINER_KEYS; do
        [ -n "${!k+x}" ] && printf '%s=%s\n' "$k" "${!k}"
    done
    env | LC_ALL=C sort | grep -E '^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=)' \
        | grep -vE '^VLLM_(COMMIT|REPO|BRANCH|SRC|USE_PRECOMPILED|PRECOMPILED_[A-Z_]*|API_KEY)=' || true
}

# The docker run arguments, into the array CRUN.
container_args() {
    local envfile="$1"
    CRUN=(docker run -d --name "$CONTAINER_NAME" --label "$LABEL=$SCRIPT_DIR"
        --user "$(id -u):$(id -g)" --workdir /cache
        --gpus all --shm-size "$SHM_SIZE"
        -p "$(publish_addr):8000"
        --env-file "$envfile"
        -v "$SCRIPT_DIR/serve.sh:/recipe/serve.sh:ro"
        -v "$SCRIPT_DIR/boot_check.py:/recipe/boot_check.py:ro"
        -v "$SCRIPT_DIR/p2p_check.py:/recipe/p2p_check.py:ro"
        -v "$SCRIPT_DIR/p2p_check.sh:/recipe/p2p_check.sh:ro"
        -v "$SCRIPT_DIR/p2p_probe.cu:/recipe/p2p_probe.cu:ro"
        -v "$MODEL:/models/$(basename "$MODEL"):ro")
    CRUN+=(-v "$CACHE_DIR:/cache"
        --entrypoint /bin/bash "$IMAGE" /recipe/serve.sh "$SPEC_MODE")
}

container_launch() {
    if [ -n "${DRY:-}" ]; then
        container_args "<env file>"
        log "launch: DRY=1, a real start would run:"
        printf '  '; printf '%q ' "${CRUN[@]}"; printf '\n'
        log "  with this environment in the container:"
        container_env | sed -e 's/^\(API_KEY\)=.*/\1=(set, not shown)/' -e 's/^/    /'
        log "  and serve.sh inside it would set:"
        env $(container_env | grep -E '^(VENV|MODEL|CUDA_HOME|HOST|PORT)=') \
            DRY=1 "$SCRIPT_DIR/serve.sh" "$SPEC_MODE" | sed 's/^/    /'
        return 0
    fi
    if container_is_ours; then
        log "launch: already running (container $(read_cid | cut -c1-12)) — skipping"
        return 0
    fi
    # A stopped container of ours is removed; any other container with the
    # same name is left alone and we stop.
    local id st
    id="$(read_cid)"
    if [ -n "$id" ]; then
        st="$(container_state "$id")"
        [ "${st#* }" = "$SCRIPT_DIR" ] && docker rm -f "$id" >/dev/null 2>&1 || true
        rm -f "$CIDFILE"
    fi
    st="$(container_state "$CONTAINER_NAME")"
    if [ -n "$st" ]; then
        if [ "$st" = "false $SCRIPT_DIR" ]; then
            docker rm "$CONTAINER_NAME" >/dev/null
        elif [ "$st" = "true $SCRIPT_DIR" ]; then
            docker container inspect -f '{{.Id}}' "$CONTAINER_NAME" >"$CIDFILE"
            log "launch: already running (container $CONTAINER_NAME) — skipping"
            return 0
        else
            die "a container named $CONTAINER_NAME exists and was not started from this checkout. Set CONTAINER_NAME to another name."
        fi
    fi
    mkdir -p "$LOGDIR" "$CACHE_DIR"
    [ -w "$CACHE_DIR" ] || die "$CACHE_DIR is not writable by $(id -un); the server runs as you and keeps its compile caches there. Fix its ownership or set CACHE_DIR."
    log "launch: container, $SPEC_MODE, layout $LAYOUT (TP=$TP PP=$PP), ctx ${MAX_LEN:-262144}, $(publish_addr)"
    log "  image: $IMAGE"
    log "  log: $SERVE_LOG"
    BOOT_LOG_OFFSET=0
    [ ! -f "$SERVE_LOG" ] || BOOT_LOG_OFFSET="$(wc -c < "$SERVE_LOG")"
    local envfile="$LOGDIR/container.env.run"
    ( umask 077; container_env >"$envfile" )
    container_args "$envfile"
    id="$("${CRUN[@]}")" || { rm -f "$envfile"; die "docker run failed"; }
    rm -f "$envfile"
    echo "$id" >"$CIDFILE"
    LAUNCHED=1
    log "  container: ${id:0:12} ($CONTAINER_NAME)"
    # Copy the container's output into $SERVE_LOG, as a native start writes it.
    # The follower ends when the container does. 9>&-: see launch().
    setsid nohup docker logs -f "$id" 9>&- >>"$SERVE_LOG" 2>&1 </dev/null &
}

container_stop() {
    local id st; id="$(read_cid)"
    if [ -z "$id" ]; then
        log "stop: no container started from this checkout"
        return 0
    fi
    st="$(container_state "$id")"
    if [ -z "$st" ]; then
        log "stop: container ${id:0:12} is gone — clearing the id file"
        rm -f "$CIDFILE"; return 0
    fi
    if [ "${st#* }" != "$SCRIPT_DIR" ]; then
        warn "stop: container ${id:0:12} was not started from this checkout — leaving it alone and clearing the id file"
        rm -f "$CIDFILE"; return 0
    fi
    if [ "${st%% *}" = true ]; then
        log "stop: docker stop ${id:0:12} (up to ${STOP_TIMEOUT}s)"
        docker stop -t "$STOP_TIMEOUT" "$id" >/dev/null || true
    fi
    docker rm "$id" >/dev/null 2>&1 || true
    rm -f "$CIDFILE"
    log "stop: done"
}

health_ok() { curl -fsS -m "${1:-5}" "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1; }

# First "id" in /v1/models. A plain greedy sed would pick up the permission id
# further down the same line, so match the field and take the first one.
# /v1 needs the key when API_KEY is set; /health never does.
served_id() {
    local auth=()
    [ -n "${API_KEY:-}" ] && auth=(-H "Authorization: Bearer $API_KEY")
    curl -fsS -m 5 ${auth[@]+"${auth[@]}"} "http://$CLIENT_HOST:$PORT/v1/models" 2>/dev/null \
        | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4
}

kv_line() {
    [ -r "$SERVE_LOG" ] || return 1
    local l; l="$(grep -h 'GPU KV cache size' "$SERVE_LOG" 2>/dev/null | tail -1 || true)"
    [ -n "$l" ] || return 1
    printf '%s\n' "${l#*] }"
}

# -------------------------------- launch -----------------------------------
launch() {
    if [ "$RUNTIME" = container ]; then container_launch; return; fi
    if [ -n "${DRY:-}" ]; then
        DRY=1 "$SCRIPT_DIR/serve.sh" "$SPEC_MODE"
        return 0
    fi
    if server_is_ours; then
        log "launch: already running (pid $(read_pid)) — skipping"
        return 0
    fi
    if [ -f "$PIDFILE" ]; then
        log "launch: clearing a stale pid file"
        rm -f "$PIDFILE"
    fi
    mkdir -p "$LOGDIR"
    log "launch: native, $SPEC_MODE, layout $LAYOUT (TP=$TP PP=$PP), ctx ${MAX_LEN:-262144}, :$PORT"
    log "  log: $SERVE_LOG"
    # 9>&- : the server must not inherit the lifecycle lock (fd 9). serve.sh
    # execs vllm, so an inherited fd would hold the lock for the server's whole
    # life and block every later stop, restart and update.
    BOOT_LOG_OFFSET=0
    [ ! -f "$SERVE_LOG" ] || BOOT_LOG_OFFSET="$(wc -c < "$SERVE_LOG")"
    RECIPE_BOOT_CHECK_OWNER=start setsid nohup "$SCRIPT_DIR/serve.sh" "$SPEC_MODE" 9>&- >>"$SERVE_LOG" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" >"$PIDFILE"
    LAUNCHED=1
    log "  pid: $pid"
}

# server_alive <pid>: the native process, or this checkout's container, is up.
server_alive() {
    if [ "$RUNTIME" = container ]; then container_is_ours; else kill -0 "$1" 2>/dev/null; fi
}

wait_ready() {
    [ -z "${DRY:-}" ] || return 0
    local url="http://$CLIENT_HOST:$PORT/health" elapsed=0 pid started=$SECONDS checker status
    pid="$(read_pid)"
    log "waiting for $url (weight load and graph capture on a 320B MoE are slow; timeout ${READY_TIMEOUT}s)"
    local log_args=(--log "$SERVE_LOG" --log-offset "${BOOT_LOG_OFFSET:-0}")
    [ "$RUNTIME" != container ] || log_args+=(--container)
    while true; do
        elapsed=$((SECONDS - started))
        [ "$elapsed" -lt "$READY_TIMEOUT" ] || break
        if [ "$LAUNCHED" = 1 ] && [ "$BOOT_CHECK" != 0 ] && [ -f "$SERVE_LOG" ]; then
            if ! python3 "$SCRIPT_DIR/boot_check.py" "${log_args[@]}" --log-only; then
                warn "[boot-check] FAIL: stopping this checkout's failed launch"
                do_stop
                return 1
            fi
        fi
        if health_ok "$((READY_TIMEOUT - elapsed < 5 ? READY_TIMEOUT - elapsed : 5))"; then
            log "healthy after ${elapsed}s"
            if [ "$LAUNCHED" = 1 ]; then
                if [ "$BOOT_CHECK" = 0 ]; then
                    log "[boot-check] disabled (BOOT_CHECK=0)"
                else
                    # Health can briefly precede the metrics/API frontend. Keep
                    # connection readiness inside the original launch budget.
                    python3 "$SCRIPT_DIR/boot_check.py" --wait-ready \
                        --ready-timeout "$((READY_TIMEOUT - (SECONDS - started)))" \
                        --base "http://$CLIENT_HOST:$PORT" --model "$SERVED_MODEL_NAME" "${log_args[@]}" &
                    checker=$!
                    trap 'kill "$checker" 2>/dev/null || true; wait "$checker" 2>/dev/null || true; do_stop; exit 1' INT TERM
                    while kill -0 "$checker" 2>/dev/null; do
                        if ! server_alive "$pid"; then
                            kill "$checker" 2>/dev/null || true
                            break
                        fi
                        sleep 1
                    done
                    status=0
                    wait "$checker" || status=$?
                    trap - INT TERM
                    if [ "$status" != 0 ]; then
                        warn "[boot-check] FAIL: stopping this checkout's failed launch"
                        do_stop
                        return 1
                    fi
                fi
            else
                log "[boot-check] existing server: no new requests"
            fi
            local kv; kv="$(kv_line || true)"
            [ -n "$kv" ] && log "  $kv"
            log "  model: $(served_id)"
            log "  API:   http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
            return 0
        fi
        if ! server_alive "$pid"; then
            warn "the server exited after ${elapsed}s — last 40 lines:"
            sleep 1
            tail -40 "$SERVE_LOG" >&2 || true
            [ "$RUNTIME" = container ] || rm -f "$PIDFILE"
            return 1
        fi
        sleep 5; elapsed=$((SECONDS - started))
    done
    warn "not healthy after ${READY_TIMEOUT}s; it may still be loading — ./start.sh logs"
    return 1
}

# --------------------------------- stop ------------------------------------
# Stops whatever this checkout started, native or container, whatever RUNTIME
# is set to now, so switching RUNTIME and restarting stops the old server.
do_stop() {
    local did=0
    if [ -f "$CIDFILE" ]; then container_stop; did=1; fi
    if [ -f "$PIDFILE" ]; then native_stop; did=1; fi
    [ "$did" = 1 ] || log "stop: nothing started from this checkout is running"
}

native_stop() {
    local pid; pid="$(read_pid)"
    if [ -z "$pid" ]; then
        log "stop: nothing started from this checkout is running"
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        log "stop: pid $pid is gone — clearing a stale pid file"
        rm -f "$PIDFILE"
        return 0
    fi
    if ! pid_is_ours "$pid"; then
        warn "stop: pid $pid is alive but is not our server — leaving it alone and clearing the pid file"
        rm -f "$PIDFILE"
        return 0
    fi
    log "stop: SIGINT to pid $pid (process group)"
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$STOP_TIMEOUT" ]; do
        sleep 2; waited=$((waited + 2))
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "stop: still alive after ${STOP_TIMEOUT}s — SIGKILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PIDFILE"
    log "stop: done (${waited}s)"
}

# -------------------------------- status -----------------------------------
do_status() {
    local pid; pid="$(read_pid)"
    log "runtime: $RUNTIME, layout $LAYOUT (TP=$TP PP=$PP)"
    if [ "$RUNTIME" = container ]; then
        local id st; id="$(read_cid)"
        if [ -z "$id" ]; then
            log "process: no container started from this checkout"
        else
            st="$(container_state "$id")"
            case "$st" in
                "true $SCRIPT_DIR") log "process: container running, ${id:0:12} ($CONTAINER_NAME)" ;;
                "false $SCRIPT_DIR") log "process: container ${id:0:12} has stopped — ./start.sh logs" ;;
                "") log "process: container ${id:0:12} is gone (stale id file)" ;;
                *) log "process: container ${id:0:12} is not ours (stale id file)" ;;
            esac
        fi
    elif [ -z "$pid" ]; then
        log "process: no pid file — not started from this checkout"
    elif pid_is_ours "$pid"; then
        log "process: running, pid $pid"
    elif kill -0 "$pid" 2>/dev/null; then
        log "process: pid $pid is alive but is not our server (stale pid file)"
    else
        log "process: pid $pid is gone (stale pid file)"
    fi
    if health_ok; then
        log "API:     healthy — http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
        log "model:   $(served_id)"
    else
        log "API:     not responding on :$PORT"
    fi
    local kv; kv="$(kv_line || true)"
    if [ -n "$kv" ]; then log "KV:      $kv"; else log "KV:      (no KV line in $SERVE_LOG)"; fi
    if [ "$RUNTIME" = container ]; then
        log "install: $(install_done && echo "image present" || echo 'image missing or pin moved — ./start.sh install'), $IMAGE"
    else
        log "install: $(install_done && echo "ok, $VLLM_COMMIT" || echo 'missing or pin moved — ./start.sh install')"
    fi
    log "models:  $(models_present && echo ok || echo 'missing — ./start.sh download')"
}

# --------------------------------- logs ------------------------------------
do_logs() {
    [ -f "$SERVE_LOG" ] || die "no log at $SERVE_LOG — nothing has been launched yet"
    log "following $SERVE_LOG"
    trap '' INT
    tail -n 100 -f "$SERVE_LOG" || true
    trap - INT
}

# -------------------------------- update -----------------------------------
do_update() {
    if [ -d "$SCRIPT_DIR/.git" ]; then
        local before after
        before="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo none)"
        log "update: git pull"
        git -C "$SCRIPT_DIR" pull --ff-only
        after="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
        if [ "$before" = "$after" ]; then
            log "  already up to date"
        else
            log "  ${before:0:10} -> ${after:0:10}"
            # The pull may have moved the pin, which this shell read before the
            # pull. Re-exec the new start.sh so we act on the new value. exec
            # keeps our PID, so fd 9 keeps holding the lifecycle lock.
            if [ -z "${_QWEN38_REEXEC:-}" ]; then
                log "  re-executing the updated start.sh"
                export _QWEN38_REEXEC=1
                exec "$SCRIPT_DIR/start.sh" update
            fi
        fi
    else
        warn "update: not a git checkout — skipping the pull"
    fi
    local pin="$VLLM_COMMIT"
    [ "$RUNTIME" = container ] && pin="$IMAGE"
    if install_done; then
        log "update: engine unchanged at $pin — no reinstall"
        [ "$RUNTIME" != native ] || install_exl3
    else
        log "update: engine moved to $pin — installing it"
        do_install
    fi
    do_stop
    start_unlocked
}

# --------------------------------- start -----------------------------------
# DRY: report what a real start would do, then print the server environment
# and command. Nothing is installed, downloaded or launched.
dry_start() {
    preflight --with-port
    if [ "$RUNTIME" = container ]; then
        if command -v docker >/dev/null 2>&1 && install_done; then
            log "install: image present — a real start would skip it"
        else
            log "install: a real start would pull $IMAGE"
        fi
    elif install_done; then
        log "install: already at $VLLM_COMMIT — a real start would skip it"
    else
        log "install: a real start would build $VENV and the vLLM fork @ $VLLM_COMMIT"
    fi
    if models_present; then
        log "download: checkpoint present — a real start would skip it"
    else
        log "download: a real start would fetch $TARGET_REPO @ $TARGET_REVISION into $MODELS_DIR"
    fi
    launch
}

# DRY stop/update: the lock is taken for real (that is what they check), and
# nothing is signalled, pulled or installed.
dry_stop() {
    local id; id="$(read_cid)"
    [ -z "$id" ] || log "stop: DRY=1, a real stop would docker stop container ${id:0:12} if it is ours"
    local pid; pid="$(read_pid)"
    [ -n "$pid" ] || [ -z "$id" ] || return 0
    if [ -z "$pid" ]; then
        log "stop: DRY=1, nothing started from this checkout is running"
    elif pid_is_ours "$pid"; then
        log "stop: DRY=1, a real stop would send SIGINT to pid $pid (process group)"
    else
        log "stop: DRY=1, pid $pid is not our server; a real stop would only clear the pid file"
    fi
}
dry_update() {
    log "update: DRY=1, a real update would git pull --ff-only, reinstall if the pin moved, then restart"
    dry_stop
    dry_start
}

start_unlocked() {
    if [ -n "${DRY:-}" ]; then dry_start; return 0; fi
    preflight --with-port
    do_install
    if models_present; then
        log "download: checkpoint present — skipping"
    else
        do_download
    fi
    launch
    [ -n "${DRY:-}" ] || wait_ready
}

# ---------------------------------- main -----------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        -h|--help|help) usage; exit 0 ;;
    esac
    banner "$cmd"
    case "$cmd" in
        start)    if [ -n "${DRY:-}" ]; then dry_start; else take_lock; start_unlocked; fi ;;
        install)  take_lock; preflight; do_install ;;
        download) take_lock; preflight; do_download ;;
        stop)     take_lock_for_stop; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi ;;
        restart)  take_lock; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi; start_unlocked ;;
        update)   take_lock; if [ -n "${DRY:-}" ]; then dry_update; else do_update; fi ;;
        status)   do_status ;;
        logs)     do_logs ;;
        smoke)    HOST="$CLIENT_HOST" exec "$SCRIPT_DIR/smoke.sh" ;;
        *) warn "unknown command: $cmd"; usage; exit 1 ;;
    esac
}

main "$@"
