import json
from pathlib import Path
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
    "VLLM_EXL3_PLE_OVERLAP": "1",
    "VLLM_QWEN4_QSA_TOPK_CANONICAL": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
}
BANNER = ("serve.sh: EXL3 enabled TP=2 PP=1 whole-expert-EP GPUs=2,3 PLE=default dtype=BF16 KV=BF16 "
          "indexer=BF16 vision=on MTP=mtp FIRST_LIGHT=0\n")


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
        result = self.check({**VALIDATED, "VLLM_EXL3_MOE_V2": "1"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("VLLM_EXL3_MOE_V2: validated '1', recipe None", result.stdout)

    def test_recipe_export_missing_from_validation_fails(self):
        effective = dict(VALIDATED)
        del effective["VLLM_EXL3_PLE_OVERLAP"]
        result = self.check(effective)
        self.assertEqual(result.returncode, 1)
        self.assertIn("VLLM_EXL3_PLE_OVERLAP: validated None, recipe '1'", result.stdout)

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
                    {"P2P": "maybe"}, {"PLE_PLACEMENT": "nfs"}):
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


if __name__ == "__main__":
    unittest.main()
