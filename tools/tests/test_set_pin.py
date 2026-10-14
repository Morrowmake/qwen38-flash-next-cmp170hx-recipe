from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CURRENT_PIN = re.search(r"^RELEASE_VLLM_COMMIT=(.+)$",
                        (ROOT / "start.sh").read_text(), re.M)[1].strip("\"' ")
SLOT = chr(123) * 2 + "PIN" + chr(125) * 2

@pytest.fixture
def scratch(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ["start.sh", "docker/build.sh", "docker/Dockerfile",
                 ".env.advanced.example", "README.md", "CHANGELOG.md", "tools/set-pin.sh"]:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    (tmp_path / "extra.md").write_text("another current reference " + CURRENT_PIN)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    return tmp_path

def run(root, pin):
    return subprocess.run(["bash", str(root / "tools/set-pin.sh"), pin],
                          cwd=root, text=True, capture_output=True)

@pytest.mark.parametrize("bad", ["ampere", "next-release", "1234567890", "g" * 40, "a" * 41])
def test_invalid_pin_changes_nothing(scratch, bad):
    before = {p: p.read_bytes() for p in scratch.rglob("*") if p.is_file()}
    assert run(scratch, bad).returncode != 0
    assert all(p.read_bytes() == data for p, data in before.items())

def history_of(text):
    """Everything from the second release heading on: earlier entries keep their pins."""
    sections = re.split(r"(?m)^(?=## )", text)
    return "".join(sections[2:])

def test_moves_every_reference_and_preserves_history(scratch):
    history = history_of((scratch / "CHANGELOG.md").read_text())
    for pin in ["A" * 40, "b" * 40]:
        result = run(scratch, pin)
        assert result.returncode == 0, result.stderr
        pin = pin.lower()
        for name in ["start.sh", "docker/build.sh", "docker/Dockerfile",
                     ".env.advanced.example", "README.md", "extra.md"]:
            text = (scratch / name).read_text()
            assert SLOT not in text and pin in text
            assert name + ":" in result.stdout
        assert history_of((scratch / "CHANGELOG.md").read_text()) == history
    assert run(scratch, "b" * 40).returncode == 0

def test_mismatched_defaults_change_nothing(scratch):
    path = scratch / "docker/Dockerfile"
    path.write_text(path.read_text().replace(CURRENT_PIN, "c" * 40))
    before = (scratch / "start.sh").read_bytes()
    result = run(scratch, "d" * 40)
    assert result.returncode != 0 and "disagree" in result.stderr
    assert (scratch / "start.sh").read_bytes() == before
