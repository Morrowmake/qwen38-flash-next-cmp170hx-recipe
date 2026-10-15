# Container image
The engine in a container: the only way this recipe runs. `./start.sh` and
`docker compose up` pull the image, start it and manage it for you (see the
[README](../README.md)); this page describes the image, how to build it, and
how to run it by hand with `docker run`.
## What the image contains
- **Engine:** the [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx)
  pinned at `{{PIN}}` (filled in at release). The builders install the fork and
  its pinned runtime dependencies in `/opt/venv`; the resulting package list is
  recorded in `/opt/venv/requirements.lock.txt`.
- **EXL3 extension:** the fork's compiled sm_80 EXL3 library (`vllm._exl3_C`),
  built once by `native_build.py` and packed by hash, next to the fork's other
  compiled extensions.
- **Base:** `nvidia/cuda:13.3.1-devel-ubuntu24.04`, pinned by digest. It is the
  devel image because Triton, FlashInfer and the flags-in-data all-reduce
  compile kernels at run time and need `nvcc` and `g++`.
- **Entry point:** the `vllm` CLI. The recipe runs it with this repository's
  [`serve.sh`](../serve.sh) mounted and used as the entry point instead, so the
  cards and every setting are chosen at run time. No launch arguments or
  settings are baked in.
- **Peer check:** a prebuilt sm_80 content probe, selected only when its source
  hash matches the mounted recipe. `P2P_PREBUILT=0` uses runtime compilation.
- **Seed mount:** `/opt/cache-seed` exists even when no seed is included.
- **No weights.** The checkpoint is mounted from the host.

Engine image tag: `ghcr.io/morrowmake/vllm-cmp170hx:engine-{{PIN}}`. Image tags
identify engine source, independently of recipe release numbers. Publication
requires release validation and acceptance of the exact image digest.
## Build it
**The fork's compiled extensions** (EXL3 among them) are built once, before the
image, by `docker/native_build.py` inside the pinned CUDA base image (rootless,
with bubblewrap; fixed paths, so two builds of one commit give the same bytes).
Native validation runs those exact files, and both builders pack them by hash;
neither compiles them. `RELEASE_NATIVE_MANIFEST_SHA256` in `build.sh` (and
`NATIVE_MANIFEST_SHA256` in the `Dockerfile`) pins the validated
`manifest.json`; a different build, commit or toolchain fails the image build.
```bash
crane pull --platform linux/amd64 --format oci \
  nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 /path/to/base-oci
python3 docker/native_build.py build --repo /path/to/fork-clone --commit <full-engine-commit> \
  --base-oci /path/to/base-oci --work /path/to/new-scratch --out /path/to/native --uv "$(command -v uv)" \
  [--fetchcontent /path/to/fetchcontent-sources]   # only when the commit builds EXL3 through CMake
python3 docker/native_build.py install --native /path/to/native --tree /path/to/fork-worktree   # native use
NATIVE_DIR=/path/to/native VLLM_COMMIT=<full-engine-commit> CLONE_FROM=/path/to/source \
  CLONE_BRANCH=<source-branch> STAGE=/path/to/new-owned-staging-directory ./docker/build.sh
```
`manifest.json` records the source commit and input trees, the base image and
compiler digests, torch and its library hashes, the flags and each file's
sha256; it names no host path. For the Docker path, copy the output directory
to `docker/native/`. `build.sh` ends by checking that the assembled OCI image
holds exactly the manifest's files (`native-check.json`).

The builder replaces `STAGE`: choose a new dedicated path. It produces a local
OCI image in `docker/out/` without publishing it. Both builders remove all Git
metadata before the application payload enters any runtime layer; the
read-only `/opt/vllm-src/provenance.json` and the OCI revision label record the
engine commit. Layer assembly rejects retained Git stores and forbidden build
paths. Set `IMAGE_FORBIDDEN_STRINGS` to a JSON array of additional identifying
strings for the rootless build; never pass private identifier lists through
Docker build arguments or labels.

Build the release source with Docker instead:
```bash
bash docker/build-docker.sh vllm-cmp170hx:engine-<full-engine-commit>
```
Both build paths print the total filesystem layer count and fail above 123.
The build needs no GPU but needs network access. See the
[release procedure](RELEASE.md) for the dependency lock, cache-seed gates and
publication. A rebuilt image is a separate artifact: validate its provenance,
layers and runtime before distributing it.
## Run it by hand
You need two of the cards, NVIDIA driver **580 or newer**, Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and the checkpoint on an SSD on the host (`./download.sh` puts it in
`./models`). From the root of this repository:
```bash
IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:{{IMAGE_DIGEST}}
MODELS=$PWD/models                  # holds Qwen3.8-Flash-Next-exl3
CACHE=$PWD/cache                    # kernel compile caches, kept between starts
mkdir -p "$CACHE"
docker run -d --name qwen38-flash-next --user "$(id -u):$(id -g)" --workdir /cache \
  --gpus all --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  --env-file docker/container.env \
  -v "$PWD/serve.sh:/recipe/serve.sh:ro" \
  -v "$PWD/boot_check.py:/recipe/boot_check.py:ro" \
  -v "$PWD/p2p_check.py:/recipe/p2p_check.py:ro" \
  -v "$PWD/p2p_check.sh:/recipe/p2p_check.sh:ro" \
  -v "$PWD/p2p_probe.cu:/recipe/p2p_probe.cu:ro" \
  -v "$MODELS/Qwen3.8-Flash-Next-exl3:/models/Qwen3.8-Flash-Next-exl3:ro" \
  -v "$CACHE:/cache" \
  --entrypoint /bin/bash "$IMAGE" /recipe/serve.sh
docker logs -f qwen38-flash-next    # wait for "Application startup complete"
curl http://127.0.0.1:8000/health
```
The same launch, with the checkpoint download and storage ownership handled
for you, is [`docker-compose.yml`](../docker-compose.yml).

**Cards.** [`container.env`](container.env) sets `CUDA_VISIBLE_DEVICES=0,1`
(nvidia-smi numbers; the container sees every card). Any setting from
[`.env.example`](../.env.example) or [`.env.advanced.example`](../.env.advanced.example)
added to `container.env` becomes the same flag as `./start.sh`'s container
gets, so the server is the same either way.

The server listens on every interface inside the container;
`-p 127.0.0.1:8000:8000` keeps it local, as `./start.sh` does. It has **no API
key**: to serve other machines, publish the port on another address and add
`-e API_KEY=<a long random string>`. The server runs as you (`--user`): every
compile cache lands under the cache mount. Stop it with
`docker stop -t 120 qwen38-flash-next`.

**PCIe peer-to-peer is automatic** (`P2P=auto`, content-checked at start).
`P2P=off` disables all peer transports; `P2P=force` bypasses verification.
## Component licences
The OCI Apache-2.0 label describes the vLLM engine, not a blanket licence for
every file in the image. CUDA/NVIDIA components, Python dependencies and
upstream third-party code retain their respective licences and notices. The
image keeps the recipe's MIT grant at `/opt/image-tools/LICENSE` and
`/opt/image-tools/THIRD_PARTY_NOTICES`; image smoke checks the complete bytes.
Model weights are downloaded separately and have their own licence (see the
checkpoint's model card).
