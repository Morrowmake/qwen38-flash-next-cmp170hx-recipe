#!/usr/bin/env bash
# prepare.sh: storage and checkpoint for the server, run inside the engine
# image (this repository mounts it at /recipe/prepare.sh).
#
#   prepare.sh prepare           Docker Compose's prepare service. Started as
#                                root: makes the cache and model directories
#                                writable by the serving user, then, as that
#                                user, downloads the checkpoint unless it is
#                                complete. Hugging Face resumes a partial
#                                download; a complete checkpoint is skipped
#                                without contacting the Hub.
#   prepare.sh fetch REPO DIR [REVISION]
#                                download one checkpoint into DIR as the
#                                current user (./start.sh download).
#   prepare.sh serve ARGS...     Docker Compose's server: runs serve.sh ARGS as
#                                the owner of /cache, the serving user chosen by
#                                prepare, never as root.
#   prepare.sh check DIR         exit 0 when DIR holds a complete checkpoint;
#                                runs on the host too (./start.sh).
#
# The serving user is the owner of the checkout (mounted at /checkout), or
# HOST_UID and HOST_GID when set, so `docker compose up` needs no settings for
# any user. ./start.sh runs the server as the user who runs it.
#
# Ownership: Docker creates a missing bind-mount directory as root. The cache
# is this recipe's own, so everything in it is given to the serving user. In
# the model directory only directories that are owned by root and hold nothing
# else (what Docker or this script just created) change owner; existing
# checkpoints and other files are never changed or removed.
set -euo pipefail

# The 5.05 bpw branch at the revision the release was validated with; the MTP
# drafter is inside the checkpoint.
TARGET_REPO="${TARGET_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
TARGET_REVISION="${TARGET_REVISION:-{{CHECKPOINT_REVISION}}}"
NEED_DISK_GB="${NEED_DISK_GB:-120}"
MODELS=/models
CACHE=/cache
CHECKOUT=/checkout
HF=/opt/venv/bin/hf

say() { printf 'prepare: %s\n' "$*"; }
fail() { printf 'prepare: %s\n' "$*" >&2; exit 1; }

# complete <dir>: the config, at least one weight file, every shard the
# index names, and no partial download left by an interrupted fetch.
complete() {
    local dir="$1" index shard
    [ -f "$dir/config.json" ] || return 1
    compgen -G "$dir/*.safetensors" >/dev/null || return 1
    if [ -d "$dir/.cache/huggingface/download" ] \
        && [ -n "$(find "$dir/.cache/huggingface/download" -name '*.incomplete' -print -quit)" ]; then
        return 1
    fi
    for index in "$dir"/*.safetensors.index.json; do
        [ -f "$index" ] || continue
        while IFS= read -r shard; do
            [ -f "$dir/$shard" ] || return 1
        done < <(grep -o '"[^"]*\.safetensors"' "$index" | tr -d '"' | sort -u)
    done
    return 0
}

fetch() {
    local repo="$1" dir="$2" revision="${3:-}"
    [ -x "$HF" ] || fail "no Hugging Face client at $HF"
    say "downloading $repo${revision:+ @ $revision} (resumes a partial download)"
    "$HF" download "$repo" ${revision:+--revision "$revision"} --local-dir "$dir"
    complete "$dir" || fail "$repo: the download ended but $dir is still incomplete; run the same command again to resume"
    say "$repo: complete"
}

# claim <dir> [name...]: give <dir> to the serving user when root owns it and
# it holds nothing but the listed names.
claim() {
    local dir="$1" entry base; shift
    [ "$(stat -c %u "$dir")" = 0 ] || return 0
    for entry in "$dir"/* "$dir"/.[!.]* "$dir"/..?*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue
        base="${entry##*/}"
        case " $* " in *" $base "*) ;; *) return 0 ;; esac
    done
    chown "$SERVE_UID:$SERVE_GID" "$dir"
    say "$dir: now owned by $SERVE_UID:$SERVE_GID"
}

serving_ids() {
    if [ -n "${HOST_UID:-}" ]; then
        SERVE_UID="$HOST_UID"
        SERVE_GID="${HOST_GID:-$HOST_UID}"
    else
        [ -d "$CHECKOUT" ] || fail "mount the checkout at $CHECKOUT, or set HOST_UID and HOST_GID"
        SERVE_UID="$(stat -c %u "$CHECKOUT")"
        SERVE_GID="$(stat -c %g "$CHECKOUT")"
    fi
    [[ "$SERVE_UID" =~ ^[0-9]+$ && "$SERVE_GID" =~ ^[0-9]+$ ]] \
        || fail "HOST_UID and HOST_GID must be numeric ids"
}

# as_serving_user CMD...: from root, run CMD as the serving user in place of
# this process. Returns when the serving user is root itself.
as_serving_user() {
    [ "$SERVE_UID" != 0 ] || return 0
    exec setpriv --reuid="$SERVE_UID" --regid="$SERVE_GID" --clear-groups -- "$@"
}

prepare() {
    local target="$MODELS/${TARGET_REPO##*/}"
    if [ "$(id -u)" = 0 ]; then
        serving_ids
        say "serving user: $SERVE_UID:$SERVE_GID"
        mkdir -p "$CACHE" "$target"
        find "$CACHE" -xdev \( ! -uid "$SERVE_UID" -o ! -gid "$SERVE_GID" \) \
            -exec chown -h "$SERVE_UID:$SERVE_GID" {} +
        claim "$target"
        claim "$MODELS" "${target##*/}"
        as_serving_user /bin/bash "$0" prepare
    fi
    local used avail
    if complete "$target"; then
        say "$target: complete, skipping"
        say "checkpoint ready"
        return 0
    fi
    [ -w "$target" ] || fail "$target is not writable by uid $(id -u); give it to that user, or set HOST_UID and HOST_GID to its owner"
    used="$(du -s -BG "$target" | awk '{s += $1} END {print s + 0}')"
    avail="$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc '0-9')"
    if [ -n "$avail" ] && [ "$((avail + used))" -lt "$NEED_DISK_GB" ]; then
        fail "only ${avail} GiB free under the model directory and ${used} GiB already downloaded; the checkpoint needs ~${NEED_DISK_GB} GiB"
    fi
    fetch "$TARGET_REPO" "$target" "$TARGET_REVISION"
    say "checkpoint ready"
}

serve() {
    if [ "$(id -u)" = 0 ]; then
        SERVE_UID="$(stat -c %u "$CACHE")"
        SERVE_GID="$(stat -c %g "$CACHE")"
        as_serving_user /bin/bash /recipe/serve.sh "$@"
    fi
    exec /bin/bash /recipe/serve.sh "$@"
}

case "${1:-}" in
    prepare) prepare ;;
    fetch) [ "$#" = 3 ] || [ "$#" = 4 ] || fail "usage: prepare.sh fetch REPO DIR [REVISION]"; fetch "$2" "$3" "${4:-}" ;;
    serve) shift; serve "$@" ;;
    check) [ "$#" = 2 ] || fail "usage: prepare.sh check DIR"; complete "$2" ;;
    *) fail "usage: prepare.sh prepare | fetch REPO DIR [REVISION] | serve ARGS... | check DIR" ;;
esac
