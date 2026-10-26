# Qwen3.8-Flash-Next on 2× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%2018305948ac-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8)
![release](https://img.shields.io/badge/release-1.0.0-2ea44f)

**Qwen3.8-Flash-Next on two CMP 170HX cards with the full 262,144-token
context: 152.8 tok/s for one user, 674.7 tok/s across eight, 2,651 tok/s cold
prefill and a 1,044,325-token KV pool. EXL3 5.05 bpw weights, BF16 KV cache,
every weight on the GPUs except the n-gram table.**

Also on these cards: [DeepSeek-V4.1-Flash on four CMP 170HX](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe).

Serve [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(EXL3, 5.05 bpw) on two NVIDIA CMP 170HX cards with a patched vLLM: an
OpenAI-compatible API with the full 262,144-token context, native MTP
speculative decoding, vision, reasoning and tool calls. The engine runs only
in a container pinned by digest. One command from a fresh clone, after the
[one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe
cd qwen38-flash-next-cmp170hx-recipe
./start.sh          # pull the engine image, fetch the model, start, wait for /health
```

`./start.sh` skips anything already done, so running it twice is safe; an
interrupted download resumes. The checkpoint is downloaded at the exact
Hugging Face revision this release was validated with.

**Prefer Docker Compose?** The same container, also one command from a fresh
clone:

```bash
docker compose up              # or in the background: docker compose up -d
```

It downloads the checkpoint, then starts the server as the owner of the
checkout; `docker compose ps` shows it healthy once `/health` answers. Compose
restarts a crashed server up to 3 times; `./start.sh` does not (run
`./start.sh restart`). Use one or the other, not both.

> **Which setups this release is for.** Two cards in tensor-parallel 2,
> measured on **PCIe x16 links** (cards with the x16 capacitor
> modification). Every decode step exchanges activations between the two
> cards; performance on cards limited to x4 links has not been measured.
> `./start.sh` reports each card's link width and warns if one is narrower
> than x16.

## One-time host setup

- **2× NVIDIA CMP 170HX, each exposing 64 GiB**; PCIe x16 links
- Linux with NVIDIA driver **580 or newer**
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  usable by your user: `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists the cards
- optional: a P2P-capable driver build for verified peer-to-peer; without it
  the server runs with P2P disabled (`P2P=auto` checks at every start)
- for `./start.sh`: `git`, `curl`, `flock`, `setsid`, Python 3 for the boot
  check, and `jq` for the smoke test
- about 120 GiB of free disk **on an SSD** for the checkpoint, plus the engine
  image and the kernel compile caches

## Hardware

| | |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX (sm_80), 65,536 MiB each, power limit 180 W |
| Links | PCIe Gen2 x16 per card; peer-to-peer verified at start |
| Driver | 580 or newer (measured on 610.57.04) |
| Disk | ~120 GiB free **on an SSD** for the checkpoint (115 GiB, of which the n-gram table is 37 GiB) |
| Software | Docker with the NVIDIA Container Toolkit |

Every weight sits on the two GPUs except the compressed n-gram (PLE) table,
which the server reads from the checkpoint file on the SSD while it decodes.
Two cards only: tensor-parallel 2 with whole routed experts on each card. Pick
the cards with `CUDA_VISIBLE_DEVICES` (two distinct nvidia-smi numbers).

## Engine

| | |
|---|---|
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8) at `18305948acba3f4c4d912cdd56f2202d2ebd94e8` |
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79` (seed: no). Starts compile on first boot (cold), so the first start is slower; later starts reuse `./cache` |
| Checkpoint | `turboderp/Qwen3.8-Flash-Next-exl3`, branch 5.05 bpw, revision `7cef615f7fd3681295b68848018876dfabc336c7` |
| Precision | EXL3 5.05 bpw weights, BF16 activations, BF16 KV cache, BF16 indexer cache |
| Speculation | native MTP, 3 draft tokens |

## Headline numbers

Measured 2026-10-15 on 2× CMP 170HX, 74 SMs per card, P2P on, recipe defaults,
262,144-token context, 180 W per card and PCIe x16 links. Decode uses the
MiaAI-Lab protocol: 400 tokens, temperature 0, thinking off; one user is the
median of ten runs (five on each of two boots), two, four and eight users the
median aggregate rate of three runs on one boot.

| | Tensor-parallel 2<br>+ peer-to-peer (default after verification) |
|---|---:|
| **1 user** · structured | **152.8** |
| **1 user** · code | **152.6** |
| **1 user** · prose | **118.3** |
| **2 users, total** · structured | 236.5¹ |
| **2 users, total** · code | 265.0 |
| **2 users, total** · prose | 204.6 |
| **4 users, total** · structured | 357.1¹ |
| **4 users, total** · code | 458.6 |
| **4 users, total** · prose | 327.2 |
| **8 users, total** · structured | 588.2¹ |
| **8 users, total** · code | 674.7 |
| **8 users, total** · prose | 489.9 |
| **Prompt reading** (cold prefill) | 2,651 |
| **KV pool** (tokens) | 1,044,325 |

All rates are up-to figures. One user is streaming speed per request; two, four
and eight users are the combined rate of that many simultaneous requests, in tok/s. Prompt reading is
the median of nine cold prefills.

¹ Some structured replies finish early: the median is 346, 273 and 279 tokens
per reply at two, four and eight users, against 400 for code and prose. The
total counts the tokens actually produced, so shorter replies lower it.
Per-request decode (structured / code / prose) is 147.3 / 150.4 / 111.6 tok/s
at two users, 137.3 / 135.2 / 91.6 at four and 104.6 / 101.2 / 67.8 at eight.

| Other measurements | Value |
|---|---|
| Quality | HumanEval 157/164; GSM8K 1,289/1,319 |
| 262,144-token context | every boundary case passes: the full window is accepted, one token more is refused |
| Long context | needle retrieval 9/9 at 8,192, 32,768 and 131,072 tokens; verbatim copy from deep context passes |
| Vision | 31/31 image and video checks pass |
| Endurance | 3,257 requests over 20 minutes of sustained multi-turn load, passed with no GPU errors |

EXL3 kernels written for this card are on by default: MoE kernels for decode
and prefill calls, a split-K dense kernel with tensor-core blocks for the small
decode launches and a tensor-core dense kernel for prefill (see Engine
switches). With these six off (`VLLM_EXL3_DENSE_MMA`,
`VLLM_EXL3_DENSE_PREFILL_TC`, `VLLM_EXL3_MOE_PREFILL_TC`,
`VLLM_EXL3_PLE_NATIVE_READ`, `VLLM_THIN_GEMM`, `VLLM_THIN_GEMM_GLUON`), the
same build takes 18 % longer per decode step for one user and 43–56 % longer
at two to eight users.

## Using it

`./start.sh` runs preflight (read-only GPU, link, disk and port checks),
installs the engine, downloads the checkpoint, launches the server detached
(output in `logs/serve.log`) and waits for `/health`, then runs a two-request
startup check. Commands:

| Command | What it does |
|---|---|
| `./start.sh` | preflight, install, download, launch (default) |
| `./start.sh install` | the engine image only |
| `./start.sh download` | the checkpoint only |
| `./start.sh stop` / `restart` | stop or restart the server this checkout started |
| `./start.sh status` | server, `/health`, KV cache line |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh update` | `git pull`, pull the image if the pin moved, restart |
| `./start.sh smoke` | one chat request and one tool call |
| `DRY=1 ./start.sh` | print what would happen, change nothing |

The API listens on `127.0.0.1:8000` as model `qwen3.8-flash-next`, with no API
key. Reasoning comes back in the message's `reasoning` field and the answer in
`content`; size `max_tokens` for both. Examples: [`examples/`](examples/).
Docker Compose: [`docker-compose.yml`](docker-compose.yml) (`up`, `up -d`,
`ps`, `logs -f`, `down`). The image by hand and how to build it:
[`docker/README.md`](docker/README.md).

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
| `P2P` | `auto` | `auto` (content-checked), `off`, `force` |
| `SPEC_MODE` | `mtp` | `mtp` or `none` (no speculation; set `BOOT_CHECK=0`, the startup check expects drafting) |
| `SPEC_N` | `3` | MTP draft tokens per step |
| `LOCAL_ARGMAX` | `0` | `1`: per-card draft argmax, same drafts |
| `SPEC_SCHEDULE` | empty | draft count by batch size, e.g. `[[1,2,4],[3,8,3]]` with `SPEC_N=4`; not a tested serving default |
| `MAX_SEQS` | `8` | concurrent sequences |
| `MAX_BATCHED` | `3460` | tokens per scheduler step |
| `PLE_PLACEMENT` | `ssd` | n-gram table placement: `ssd`; `ram` and `device` are comparison placements, not the tested default |
| `REASONING_PARSER` / `TOOL_PARSER` | `qwen3` / `qwen3_coder` | output parsers |
| `EXTRA_ARGS` | empty | extra vLLM arguments; options that change the deployment are refused |
| `BOOT_CHECK` | `1` | two-request startup check after `/health` |
| `READY_TIMEOUT` / `STOP_TIMEOUT` | `1800` / `120` | seconds |

The context (262,144) and GPU memory fraction (0.95) are fixed: `serve.sh`
refuses other values. `FIRST_LIGHT=1` is a 32K/eager/no-MTP diagnostic, not a
serving mode. Path and preflight settings are in
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
| `VLLM_EXL3_DENSE_MMA` | tensor-core blocks for the split-K kernel's dense launches of 1 to 32 rows (needs `VLLM_EXL3_DENSE_SPLITK`) |
| `VLLM_EXL3_DENSE_PREFILL_TC` | EXL3 tensor-core dense kernel for prefill launches of more than 64 rows |
| `VLLM_EXL3_MOE_PREFILL_TC` | EXL3 tensor-core MoE kernel for prefill MoE calls of 65 to 3460 rows |
| `VLLM_EXL3_PLE_NATIVE_READ` | n-gram rows of a step read in one native call instead of a thread pool |
| `VLLM_THIN_GEMM` | thin-M BF16 GEMM for the small-row calls of the unquantized projections |
| `VLLM_THIN_GEMM_GLUON` | Gluon thin GEMM for the decode shapes captured in CUDA graphs (needs `VLLM_THIN_GEMM`) |
| `VLLM_FLA_AUTOTUNE_SKIP_SPILLING` | FLA autotune skips configs that spill local memory, so a first boot on a cold kernel cache keeps its KV cache space |
| `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` | custom all-reduce over PCIe peer access (only after the P2P check passes) |

## Update

- `./start.sh update` pulls the new release and engine image and restarts; your
  `.env` is preserved.
- With Compose: `git pull`, then `docker compose up -d`.
- Older releases are git tags; their images may be retired.

## Checking a release

`tools/check-env-match.py` compares the launcher's CPU dry run with a
validated server's environment (`--effective <file> --layout tp2`) or a
validation JSON; `tools/check-release.py` checks commit identity and wording;
`tools/set-pin.sh <commit>` moves the engine pin everywhere at once.

## Licence

The recipe is MIT ([`LICENSE`](LICENSE)). The engine is Apache-2.0; the
checkpoint has its own licence (see its model card). The benchmark prompts
come from MiaAI-Lab's repository, used as data with attribution.

## Credits

If this recipe helps you, please credit Morrowmake and link this repository.
Citation metadata: [`CITATION.cff`](CITATION.cff).
