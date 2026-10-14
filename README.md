# Qwen3.8-Flash-Next on 2× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%20{{PIN}}-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/{{PIN}})

Serve [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(EXL3, 5.05 bpw) on two NVIDIA CMP 170HX cards with a patched vLLM: an
OpenAI-compatible API with the full 262,144-token context, native MTP
speculative decoding, vision, reasoning and tool calls. One command installs,
downloads and starts it:

```bash
git clone https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe
cd qwen38-flash-next-cmp170hx-recipe
./start.sh
```

The engine runs in a container pinned by digest (`RUNTIME=container`, the
default) or from a venv built on your machine (`RUNTIME=native`).

> **Which setups this release is for.** Two cards in tensor-parallel 2,
> measured on **PCIe x16 links** (cards with the x16 capacitor
> modification). Every decode step exchanges activations between the two
> cards; performance on cards limited to x4 links has not been measured.
> `./start.sh` reports each card's link width and warns if one is narrower
> than x16.

## Hardware

| | |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX (sm_80), 65,536 MiB each, power limit 180 W |
| Links | PCIe Gen2 x16 per card; peer-to-peer verified at start |
| Driver | 580 or newer (measured on 610.57.04) |
| Disk | ~120 GiB free **on an SSD** for the checkpoint (115 GiB, of which the n-gram table is 37 GiB) |
| Software | Docker with the NVIDIA Container Toolkit, or `uv`, Python 3.12, CUDA 13.3 toolkit and a C++ compiler (native) |

Every weight sits on the two GPUs except the compressed n-gram (PLE) table,
which the server reads from the checkpoint file on the SSD while it decodes.
Two cards only: tensor-parallel 2 with whole routed experts on each card. Pick
the cards with `CUDA_VISIBLE_DEVICES` (two distinct nvidia-smi numbers).

## Engine

| | |
|---|---|
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) at `{{PIN}}` |
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:{{IMAGE_DIGEST}}` (seed: TBD-at-release) |
| Checkpoint | `turboderp/Qwen3.8-Flash-Next-exl3`, branch 5.05 bpw, revision `{{CHECKPOINT_REVISION}}` |
| Precision | EXL3 5.05 bpw weights, BF16 activations, BF16 KV cache, BF16 indexer cache |
| Speculation | native MTP, 3 draft tokens |

## Speed

Measured on 2× CMP 170HX (74 SMs, 180 W, PCIe x16, driver 610.57.04) with
P2P on and the recipe defaults: 400 greedy tokens per request, output tokens
counted as one plus the accepted drafts per step.

**Draft: every number marked ‡ is re-measured on the release tree before
publication and replaced by that measurement.**

| Users | Output tok/s (sum) | Step |
|---|---|---|
| 1 | 69.3 ‡ | 33.36 ms ‡ |
| 2 | 112.5 ‡ | 43.6 ms ‡ |
| 4 | 166.0 ‡ | 56.2 ms ‡ |
| 8 | 242.2 ‡ | 80.6 ms ‡ |

Two EXL3 kernels written for this card are on by default: a MoE kernel for
calls of 1 to 64 rows and a split-K dense kernel for the small decode
launches (see Engine switches).

| Other measurements | Value |
|---|---|
| KV cache | 1,040,074 tokens ‡ |
| 262,144-token context | 262,143 prompt + 1 output token accepted; one more token refused (HTTP 400) ‡ |
| Quality | HumanEval pass@1 0.9817 (161/164); GSM8K 0.9780 (1,290/1,319) ‡ |
| Vision | image and video requests pass ‡ |

## Using it

`./start.sh` runs preflight (read-only GPU, link, disk and port checks),
installs the engine, downloads the checkpoint, launches the server detached
(output in `logs/serve.log`) and waits for `/health`, then runs a two-request
startup check. Commands:

| Command | What it does |
|---|---|
| `./start.sh` | preflight, install, download, launch (default) |
| `./start.sh install` | the engine only (image, or venv and fork) |
| `./start.sh download` | the checkpoint only |
| `./start.sh stop` / `restart` | stop or restart the server this checkout started |
| `./start.sh status` | server, `/health`, KV cache line |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh update` | `git pull`, reinstall if the pin moved, restart |
| `./start.sh smoke` | one chat request and one tool call |
| `DRY=1 ./start.sh` | print what would happen, change nothing |

The API listens on `127.0.0.1:8000` as model `qwen3.8-flash-next`, with no API
key. Examples: [`examples/`](examples/). Docker Compose:
[`docker-compose.yml`](docker-compose.yml). The image by hand and how to build
it: [`docker/README.md`](docker/README.md).

## Options

Settings go in `.env` (copied from [`.env.example`](.env.example) on the first
run); a prefix assignment beats `.env` for one run
(`CUDA_VISIBLE_DEVICES=2,3 ./start.sh restart`). Every option:

| Setting | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | the two cards (exactly two, distinct) |
| `HOST` | `127.0.0.1` | listen address; `0.0.0.0` for every interface (set `API_KEY` first) |
| `PORT` | `8000` | API port |
| `API_KEY` | unset | require `Authorization: Bearer <key>` |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | model id clients send |
| `MODELS_DIR` | `./models` | checkpoint location; must be on an SSD |
| `RUNTIME` | `container` | `container` or `native` |
| `P2P` | `auto` | `auto` (content-checked), `off`, `force` |
| `SPEC_MODE` | `mtp` | `mtp` or `none` (no speculation; set `BOOT_CHECK=0`, the startup check expects drafting) |
| `SPEC_N` | `3` | MTP draft tokens per step |
| `LOCAL_ARGMAX` | `0` | `1`: per-card draft argmax, same drafts |
| `MAX_SEQS` | `8` | concurrent sequences |
| `MAX_BATCHED` | `3460` | tokens per scheduler step |
| `PLE_PLACEMENT` | `ssd` | n-gram table placement: `ssd`; `ram` and `device` are comparison placements, not the tested default |
| `REASONING_PARSER` / `TOOL_PARSER` | `qwen3` / `qwen3_coder` | output parsers |
| `EXTRA_ARGS` | empty | extra vLLM arguments; options that change the deployment are refused |
| `BOOT_CHECK` | `1` | two-request startup check after `/health` |
| `READY_TIMEOUT` / `STOP_TIMEOUT` | `1800` / `120` | seconds |
| `IMAGE` / `VLLM_COMMIT` | this release | engine pin; set an older one to roll back |

The context (262,144) and GPU memory fraction (0.95) are fixed: `serve.sh`
refuses other values. `FIRST_LIGHT=1` is a 32K/eager/no-MTP diagnostic, not a
serving mode. Install, path and preflight settings are in
[`.env.advanced.example`](.env.advanced.example).

### Engine switches

Each is on by default and has a one-variable kill switch: set it to `0` and
restart, no rebuild.

| Switch | What it does |
|---|---|
| `VLLM_QWEN4_QSA_TOPK_CANONICAL` | canonical sparse-attention top-k ties, so a request repeats at exact score ties |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS` | flags-in-data two-shot custom all-reduce for the TP2 all-reduces |
| `VLLM_EXL3_PLE_OVERLAP` | n-gram row reads overlap the previous decoder layer |
| `VLLM_EXL3_MOE_V2` | EXL3 MoE kernel v2 for MoE calls of 1 to 64 rows (decode and short prefills) |
| `VLLM_EXL3_DENSE_SPLITK` | EXL3 split-K dense kernel for the small-row decode launches |
| `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` | custom all-reduce over PCIe peer access (only after the P2P check passes) |

## Checking a release

`tools/check-env-match.py` compares the launcher's CPU dry run with a
validated server's environment (`--effective <file> --layout tp2`) or a
validation JSON; `tools/check-release.py` checks commit identity and wording;
`tools/set-pin.sh <commit>` moves the engine pin everywhere at once.

## Licence

The recipe is MIT ([`LICENSE`](LICENSE)). The engine is Apache-2.0; the
checkpoint has its own licence (see its model card).
