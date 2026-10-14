#!/usr/bin/env python3
"""Compare the launcher's CPU dry run with a validated environment.

  check-env-match.py REFERENCE
      REFERENCE is a validation JSON: {"tp2": {KEY: value, ...}} (keys starting
      with "_" are metadata). Every key must equal the dry run's export, or the
      resolved launcher input of the same name (MAX_BATCHED, MAX_LEN, ...).
  check-env-match.py --effective EFFECTIVE --layout tp2 [--serve-log LOG]
      EFFECTIVE is the full environment a validated server ran with. Every
      engine-scope key (VLLM_*, NCCL_*, PYTORCH_CUDA_ALLOC_CONF,
      CUDA_DEVICE_ORDER) must match the dry run in both directions, as must
      every launcher input it records. The first serve.sh banner of LOG
      (default: serve.log beside EFFECTIVE) must equal the dry run's.

The dry run uses P2P=force: validated servers run with peer access on after
the content check, which a CPU run cannot perform.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ("tp2",)
ENGINE_SCOPE = re.compile(r"^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF$|CUDA_DEVICE_ORDER$)")
# Keys a release may legitimately differ in, with the reason.
ALLOWED = {
    "CUDA_VISIBLE_DEVICES": "card choice: the release checks two distinct cards, not which",
    "VLLM_API_KEY": "secret, never compared",
}
# Launcher inputs and the command option that carries each resolved value.
LAUNCHER_INPUTS = {
    "MAX_BATCHED": "--max-num-batched-tokens",
    "MAX_SEQS": "--max-num-seqs",
    "MAX_LEN": "--max-model-len",
    "GPU_UTIL": "--gpu-memory-utilization",
}


def dry_run(layout, overrides=None):
    with tempfile.TemporaryDirectory(prefix="env-match-") as scratch:
        environment = {
            "PATH": os.defpath, "HOME": scratch, "DRY": "1", "P2P": "force",
            "LAYOUT": layout, "VENV": scratch, "MODEL": scratch,
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
        environment.update(overrides or {})
        result = subprocess.run(["bash", str(ROOT / "serve.sh")], cwd=ROOT,
                                env=environment, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(f"{layout}: dry run failed ({result.returncode}): {result.stderr.strip()}")
    exports = dict(re.findall(r"^  ([A-Z][A-Z0-9_]*)=(.*)$", result.stdout, re.M))
    command = shlex.split(result.stdout.split("DRY=1, command:\n", 1)[1])
    banner = next(line for line in result.stdout.splitlines() if line.startswith("serve.sh: EXL3 enabled"))
    return exports, command, banner


def resolved_inputs(command):
    values = {key: command[command.index(option) + 1] for key, option in LAUNCHER_INPUTS.items()}
    if "--speculative-config" in command:
        spec = json.loads(command[command.index("--speculative-config") + 1])
        values["SPEC_N"] = str(spec["num_speculative_tokens"])
        values["LOCAL_ARGMAX"] = "1" if spec.get("use_local_argmax_reduction") else "0"
    graph = json.loads(command[command.index("--compilation-config") + 1])
    values["GRAPH_MODE"] = graph["cudagraph_mode"]
    values["FIRST_LIGHT"] = "1" if "--enforce-eager" in command else "0"
    return values


def without_cards(banner):
    return re.sub(r" GPUs=\S+", "", banner)


def check(reference):
    failures = []
    metadata = sorted(key for key in reference if key.startswith("_"))
    print("Ignored metadata: " + (", ".join(metadata) or "none"))
    layouts = sorted(key for key in reference if not key.startswith("_"))
    if layouts != list(LAYOUTS):
        print(f"Environment match: FAIL (reference layouts {layouts}, expected {list(LAYOUTS)})")
        return 1
    for layout in LAYOUTS:
        exports, command, _ = dry_run(layout)
        actual = {**exports, **resolved_inputs(command)}
        compared = 0
        for key, value in sorted(reference[layout].items()):
            if key in ALLOWED:
                print(f"{layout}: ignored {key} ({ALLOWED[key]})")
            elif actual.get(key) != str(value):
                failures.append(f"{layout}: {key}: expected {value!r}, got {actual.get(key)!r}")
                compared += 1
            else:
                compared += 1
        extras = sorted(key for key in exports if key not in reference[layout] and key not in ALLOWED)
        for key in extras:
            failures.append(f"{layout}: {key}={exports[key]!r} exported but not in the validated environment")
        print(f"{layout}: compared {compared} settings")
    if failures:
        print("Environment match: FAIL\n" + "\n".join(failures))
        return 1
    print("Environment match: PASS (TP2)")
    return 0


def check_effective(path, layout, serve_log=None):
    effective = json.loads(path.read_text())
    overrides = {key: str(effective[key]) for key in ("FIRST_LIGHT",) if key in effective}
    exports, command, banner = dry_run(layout, overrides)
    failures = []
    expected = {k: str(v) for k, v in effective.items() if ENGINE_SCOPE.match(k) and k not in ALLOWED}
    actual = {k: v for k, v in exports.items() if ENGINE_SCOPE.match(k) and k not in ALLOWED}
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            failures.append(f"environment: {key}: validated {expected.get(key)!r}, recipe {actual.get(key)!r}")
    resolved = resolved_inputs(command)
    inputs = 0
    for key, value in sorted(resolved.items()):
        if key in effective:
            inputs += 1
            if str(effective[key]) != value:
                failures.append(f"launcher input: {key}: validated {effective[key]!r}, recipe {value!r}")
    if "LAYOUT" in effective and effective["LAYOUT"] != layout:
        failures.append(f"launcher input: LAYOUT: validated {effective['LAYOUT']!r}, checked {layout!r}")
    log = serve_log or path.with_name("serve.log")
    if log.exists():
        logged = next((line.strip() for line in log.read_text(errors="replace").splitlines()
                       if line.startswith("serve.sh: EXL3 enabled")), None)
        if logged is None:
            failures.append(f"banner: no serve.sh banner in {log.name}")
        elif without_cards(logged) != without_cards(banner):
            failures.append(f"banner: validated {logged!r}, recipe {banner!r}")
        else:
            print("banner: equal apart from the card numbers")
    else:
        print(f"banner: {log.name} not found; banner not compared")
    for key, reason in sorted(ALLOWED.items()):
        print(f"{layout}: allowed difference {key} ({reason})")
    print(f"Full environment: layout={layout}; DRY=1; P2P=force; {len(set(expected) | set(actual))} engine-scope "
          f"keys, {inputs} launcher inputs")
    for failure in failures:
        print(failure)
    print(f"Full environment match: {'FAIL' if failures else 'PASS'} ({layout}; {len(failures)} mismatches)")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, nargs="?", help="validation JSON path")
    parser.add_argument("--effective", type=Path, help="full effective environment JSON")
    parser.add_argument("--layout", choices=LAYOUTS)
    parser.add_argument("--serve-log", type=Path, help="default: serve.log beside the effective JSON")
    args = parser.parse_args()
    try:
        if args.effective:
            if args.reference or not args.layout:
                parser.error("--effective requires --layout and cannot take reference")
            return check_effective(args.effective, args.layout, args.serve_log)
        if not args.reference or args.layout or args.serve_log:
            parser.error("supply reference or --effective with --layout")
        return check(json.loads(args.reference.read_text()))
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(f"Environment match: FAIL ({exc})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
