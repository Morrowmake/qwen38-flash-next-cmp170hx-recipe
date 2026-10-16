import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/check-env-match.py"
VALIDATED = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_P2P_LEVEL": "SYS",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "VLLM_CUSTOM_ALLREDUCE_FLAGS": "1",
    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": "1",
    "VLLM_EXL3_DENSE_MMA": "1",
    "VLLM_EXL3_DENSE_PREFILL_TC": "1",
    "VLLM_EXL3_DENSE_SPLITK": "1",
    "VLLM_EXL3_MOE_PREFILL_TC": "1",
    "VLLM_EXL3_MOE_V2": "1",
    "VLLM_EXL3_PLE_NATIVE_READ": "1",
    "VLLM_EXL3_PLE_OVERLAP": "1",
    "VLLM_FLA_AUTOTUNE_SKIP_SPILLING": "1",
    "VLLM_QWEN4_QSA_TOPK_CANONICAL": "1",
    "VLLM_THIN_GEMM": "1",
    "VLLM_THIN_GEMM_GLUON": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
}
BANNER = ("serve.sh: EXL3 enabled TP=2 PP=1 whole-expert-EP GPUs=2,3 PLE=default dtype=BF16 KV=BF16 "
          "indexer=BF16 vision=on MTP=mtp FIRST_LIGHT=0\n")
# The unfilled release-pin slot, spelled so tools/set-pin.sh never rewrites it.
PIN_SLOT = "{" * 2 + "PIN" + "}" * 2


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], text=True, capture_output=True)


class ReferenceMode(unittest.TestCase):
    def check(self, reference):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "new_env.json"
            path.write_text(json.dumps(reference))
            return run(path)

    def test_validated_defaults_pass(self):
        result = self.check({"_note": "x", "tp2": {**VALIDATED, "MAX_BATCHED": "3460", "SPEC_N": "3",
                                                   "CUDA_VISIBLE_DEVICES": "2,3"}})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Environment match: PASS (TP2)", result.stdout)
        self.assertIn("ignored CUDA_VISIBLE_DEVICES", result.stdout)

    def test_changed_value_fails(self):
        result = self.check({"tp2": {**VALIDATED, "VLLM_EXL3_PLE_OVERLAP": "0"}})
        self.assertEqual(result.returncode, 1)
        self.assertIn("VLLM_EXL3_PLE_OVERLAP: expected '0', got '1'", result.stdout)

    def test_unvalidated_export_fails(self):
        reference = dict(VALIDATED)
        del reference["VLLM_CUSTOM_ALLREDUCE_FLAGS"]
        result = self.check({"tp2": reference})
        self.assertEqual(result.returncode, 1)
        self.assertIn("VLLM_CUSTOM_ALLREDUCE_FLAGS='1' exported but not in the validated", result.stdout)

    def test_launcher_input_mismatch_fails(self):
        result = self.check({"tp2": {**VALIDATED, "MAX_BATCHED": "1800"}})
        self.assertEqual(result.returncode, 1)
        self.assertIn("MAX_BATCHED: expected '1800', got '3460'", result.stdout)

    def test_wrong_layouts_fail(self):
        result = self.check({"tp4": VALIDATED})
        self.assertEqual(result.returncode, 1)


class EffectiveMode(unittest.TestCase):
    def check(self, effective, banner=BANNER):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "effective_env_private.json"
            path.write_text(json.dumps(effective))
            if banner is not None:
                (Path(scratch) / "serve.log").write_text(banner)
            return run("--effective", path, "--layout", "tp2")

    def test_validated_server_passes(self):
        result = self.check({**VALIDATED, "CUDA_VISIBLE_DEVICES": "2,3", "MAX_BATCHED": "3460",
                             "MAX_LEN": "262144", "GPU_UTIL": "0.95", "FIRST_LIGHT": "0", "LAYOUT": "tp2",
                             "HOME": "/elsewhere", "PORT": "8013"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("banner: equal apart from the card numbers", result.stdout)

    def test_engine_switch_missing_from_recipe_fails(self):
        result = self.check({**VALIDATED, "VLLM_QWEN4_ROUTER_FP32": "1"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("VLLM_QWEN4_ROUTER_FP32: validated '1', recipe None", result.stdout)

    def test_recipe_export_missing_from_validation_fails(self):
        effective = dict(VALIDATED)
        del effective["VLLM_EXL3_PLE_OVERLAP"]
        result = self.check(effective)
        self.assertEqual(result.returncode, 1)
        self.assertIn("VLLM_EXL3_PLE_OVERLAP: validated None, recipe '1'", result.stdout)

    def test_ports_off_validation_fails(self):
        for name in ("VLLM_EXL3_MOE_V2", "VLLM_EXL3_DENSE_SPLITK", "VLLM_EXL3_DENSE_MMA",
                     "VLLM_EXL3_DENSE_PREFILL_TC", "VLLM_EXL3_MOE_PREFILL_TC", "VLLM_EXL3_PLE_NATIVE_READ",
                     "VLLM_THIN_GEMM", "VLLM_THIN_GEMM_GLUON", "VLLM_FLA_AUTOTUNE_SKIP_SPILLING"):
            with self.subTest(name=name):
                effective = dict(VALIDATED)
                del effective[name]
                result = self.check(effective)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{name}: validated None, recipe '1'", result.stdout)

    def test_launcher_input_mismatch_fails(self):
        result = self.check({**VALIDATED, "MAX_BATCHED": "1800"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("MAX_BATCHED: validated '1800', recipe '3460'", result.stdout)

    def test_banner_mismatch_fails(self):
        result = self.check(VALIDATED, BANNER.replace("PLE=default", "PLE=ram"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("banner: validated", result.stdout)

    def test_missing_log_is_reported(self):
        result = self.check(VALIDATED, None)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("banner not compared", result.stdout)


class LauncherRefusals(unittest.TestCase):
    def dry(self, **env):
        base = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "DRY": "1"}
        return subprocess.run(["bash", str(ROOT / "serve.sh")], cwd=ROOT, env={**base, **env},
                              text=True, capture_output=True)

    def test_refusals(self):
        for env in ({"CUDA_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "1,1"}, {"LAYOUT": "tp4"},
                    {"MAX_LEN": "131072"}, {"GPU_UTIL": "0.9"}, {"EXTRA_ARGS": "--max-model-len 1000"},
                    {"P2P": "maybe"}, {"PLE_PLACEMENT": "nfs"},
                    {"EXTRA_ARGS": "--device-memory-utilization 0.97"}, {"EXTRA_ARGS": "--cpu-offload-gb 8"},
                    {"EXTRA_ARGS": "--offload_backend prefetch"}, {"EXTRA_ARGS": "--kv-offloading-size=64"},
                    {"EXTRA_ARGS": '--kv-transfer-config {"kv_connector":"OffloadingConnector"}'},
                    {"GRAPH_MODE": "FULL"}, {"GRAPH_MODE": "NONE"}, {"SPEC_SCHEDULE": "[[1, 8, 3]]"},
                    {"SPEC_SCHEDULE": "[[1,8]]"}):
            with self.subTest(env=env):
                self.assertEqual(self.dry(**env).returncode, 2)

    def test_p2p_auto_dry_disables_peers(self):
        result = self.dry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("  VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0", result.stdout)
        self.assertIn("  NCCL_P2P_DISABLE=1", result.stdout)

    def test_paths_are_repository_relative(self):
        result = self.dry(P2P="force")
        self.assertIn(str(ROOT / "venv/bin/vllm"), result.stdout)
        self.assertIn(str(ROOT / "models/Qwen3.8-Flash-Next-exl3"), result.stdout)

    def test_validated_server_options(self):
        for schedule in ("", "[[1,2,4],[3,8,3]]"):
            with self.subTest(schedule=schedule):
                result = self.dry(P2P="force", SPEC_N="4", SPEC_SCHEDULE=schedule)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(result.stdout.split("DRY=1, command:\n", 1)[1])
                self.assertEqual(command[command.index("--prefix-match-unit") + 1], "64")
                spec = json.loads(command[command.index("--speculative-config") + 1])
                self.assertIs(spec["disable_eagle_block_drop"], True)
                self.assertEqual(spec.get("num_speculative_tokens_per_batch_size"),
                                 json.loads(schedule) if schedule else None)



class ForkPin(unittest.TestCase):
    def recipe_with_pin(self, scratch, pin):
        """A copy of the launcher files whose start.sh carries pin instead of the current release pin."""
        recipe = Path(scratch) / "recipe"
        (recipe / "tools").mkdir(parents=True)
        for name in ("serve.sh", "p2p_check.sh", "tools/check-env-match.py"):
            (recipe / name).write_text((ROOT / name).read_text())
        start = (ROOT / "start.sh").read_text()
        current = re.search(r"^RELEASE_VLLM_COMMIT=(\S+)$", start, re.M).group(1)
        start = start.replace("RELEASE_VLLM_COMMIT=" + current, "RELEASE_VLLM_COMMIT=" + pin)
        (recipe / "start.sh").write_text(start)
        return recipe

    def fork_at(self, scratch):
        fork = Path(scratch) / "fork"
        subprocess.run(["git", "init", "-q", str(fork)], check=True)
        subprocess.run(["git", "-C", str(fork), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "--allow-empty", "-m", "x"], check=True)
        return fork, subprocess.run(["git", "-C", str(fork), "rev-parse", "HEAD"], text=True,
                                    capture_output=True, check=True).stdout.strip()

    def check(self, recipe, fork):
        effective = Path(fork).parent / "effective_env_private.json"
        effective.write_text(json.dumps(VALIDATED))
        return subprocess.run([sys.executable, str(recipe / "tools/check-env-match.py"), "--effective",
                               str(effective), "--layout", "tp2", "--fork", str(fork)],
                              text=True, capture_output=True)

    def test_pin_placeholder_fails_and_matching_pin_passes(self):
        with tempfile.TemporaryDirectory() as scratch:
            fork, head = self.fork_at(scratch)
            result = self.check(self.recipe_with_pin(scratch + "/slot", PIN_SLOT), fork)
            self.assertEqual(result.returncode, 1)
            self.assertIn("release pin not set", result.stdout)
            result = self.check(self.recipe_with_pin(scratch, head), fork)
            self.assertEqual(result.returncode, 0, result.stdout)
            result = self.check(self.recipe_with_pin(scratch + "/other", "f" * 40), fork)
            self.assertEqual(result.returncode, 1)
            self.assertIn("is not a commit of fork", result.stdout)

if __name__ == "__main__":
    unittest.main()
