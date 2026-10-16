# Changelog

## 0.1.0 — {{RELEASE_DATE}}

First release: Qwen3.8-Flash-Next (EXL3 5.05 bpw) on two CMP 170HX cards.

- Engine: Morrowmake/vllm-cmp170hx at `{{PIN}}`; image
  `ghcr.io/morrowmake/vllm-cmp170hx@sha256:{{IMAGE_DIGEST}}` (seed: {{SEED_YES_NO}}).
- Checkpoint: `turboderp/Qwen3.8-Flash-Next-exl3`, 5.05 bpw branch at revision
  `{{CHECKPOINT_REVISION}}`; the MTP drafter is inside it.
- Tensor-parallel 2 with whole routed experts per card; every weight on the
  GPUs except the n-gram table, read from the SSD.
- 262,144-token context, BF16 KV and indexer caches, native MTP with 3 draft
  tokens, vision, `qwen3` reasoning and `qwen3_coder` tool parsers.
- On by default, each with a kill switch: canonical sparse-attention top-k
  ties (`VLLM_QWEN4_QSA_TOPK_CANONICAL`), the flags-in-data TP2 all-reduce
  (`VLLM_CUSTOM_ALLREDUCE_FLAGS`), n-gram reads overlapped with the previous
  layer (`VLLM_EXL3_PLE_OVERLAP`), the EXL3 MoE kernel v2 for MoE calls of 1
  to 64 rows (`VLLM_EXL3_MOE_V2`) and the EXL3 split-K dense kernel for the
  small decode launches (`VLLM_EXL3_DENSE_SPLITK`).
- The engine keeps the failed-token mask alive until its copy to the host has
  run. Before, a step could report healthy requests as failed (seen with
  prompt log-probabilities on chunked prompts) or miss a real n-gram read
  failure.
- **One command from a fresh clone: `./start.sh` or `docker compose up`.**
  Compose runs a `prepare` service first: it makes `./cache` and the model
  directory writable by the serving user and downloads the checkpoint; then
  the server starts as the owner of the checkout, whatever its UID, never as
  root. No `HOST_UID`/`HOST_GID`, `mkdir`, `./install.sh` or `./download.sh`
  step is needed (`HOST_UID` and `HOST_GID` still override the user), and
  `docker compose ps` shows the server healthy once `/health` answers.
- **The recipe ships the container runtime only**: there is no native mode
  (no `RUNTIME`, no venv build); `./start.sh` and Compose run the same pinned
  image.
- **Rolling back to an earlier release is not a supported path.**
  `./start.sh update` moves forward only. Older releases are git tags; their
  images may be retired.
- **Docker Compose restarts the server after a crash** (`restart: on-failure:3`):
  a non-zero exit brings it back, up to 3 times; `docker compose stop` and
  `down` stop it for good. `./start.sh` does not restart a crashed server; run
  `./start.sh restart`.
- **An interrupted checkpoint download is resumed** instead of being taken as
  complete: a checkpoint counts as present only with its config, every shard
  its index names and no partial download left behind; `./start.sh` and
  Compose use the same check.
- `.env` settings: all new; nothing to delete.
- Measured numbers: see the README (filled from this release's gate legs).
