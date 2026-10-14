from pathlib import Path
ROOT = Path(__file__).parents[2]
def test_default_samples_use_content_gate_not_hard_off():
    assert "P2P=auto" in (ROOT / "docker/container.env").read_text()
    assert "\nVLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0\n" not in (ROOT / "docker/container.env").read_text()
    assert "# P2P=auto" in (ROOT / ".env.example").read_text()
    serve = (ROOT / "serve.sh").read_text()
    assert "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:-0" not in serve
    assert "QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False" in serve
    assert "PYTORCH_CUDA_ALLOC_CONF-$QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF" in serve
