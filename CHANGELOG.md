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
  layer (`VLLM_EXL3_PLE_OVERLAP`).
- `.env` settings: all new; nothing to delete.
- Measured numbers: see the README (filled from this release's gate legs).
