"""One command from a fresh clone: Compose, checkpoint completeness, serving user."""
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / 'docker' / 'prepare.sh'
CHECKPOINT = 'Qwen3.8-Flash-Next-exl3'


def release_image():
    return re.search(r'^RELEASE_IMAGE=(.+)$', (ROOT / 'start.sh').read_text(), re.M)[1]


def test_every_default_image_is_the_start_pin():
    image = release_image()
    compose = (ROOT / 'docker-compose.yml').read_text()
    assert re.findall(r'\$\{IMAGE:-([^}]+\}\}|[^}]+)\}', compose) == [image]
    # Release history may name earlier images; nothing else may default to one.
    history = {ROOT / 'CHANGELOG.md', ROOT / 'docker' / 'RELEASE.md'}
    files = subprocess.run(['git', '-C', str(ROOT), 'ls-files'], capture_output=True,
                           text=True, check=True).stdout.split()
    digest = r'ghcr\.io/morrowmake/vllm-cmp170hx@sha256:(?:[0-9a-f]{64}|\{\{IMAGE_DIGEST\}\})'
    for path in (ROOT / name for name in files):
        if path in history or 'tests' in path.parts or not path.is_file():
            continue
        for found in re.findall(digest, path.read_text(errors='replace')):
            assert found == image, path


def test_compose_runs_the_start_container_after_prepare():
    yaml = pytest.importorskip('yaml')
    spec = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    prepare, server = spec['services']['prepare'], spec['services']['qwen38-flash-next']
    assert server['depends_on'] == {'prepare': {'condition': 'service_completed_successfully'}}
    assert prepare['image'] == server['image'] == '${IMAGE:-' + release_image() + '}'
    assert prepare['user'] == server['user'] == '0:0'
    assert prepare['entrypoint'] == ['/bin/bash', '/recipe/prepare.sh']
    assert prepare['command'] == ['prepare']
    assert server['entrypoint'] == ['/bin/bash', '/recipe/prepare.sh', 'serve']
    assert server['command'] == ['mtp'] and server['shm_size'] == '16g'
    assert server['env_file'] == ['docker/container.env']
    assert server['healthcheck']['start_period'] == '1800s'
    start = (ROOT / 'start.sh').read_text()
    assert 'READY_TIMEOUT="${READY_TIMEOUT:-1800}"' in start
    for name in ('serve.sh', 'boot_check.py', 'p2p_check.py', 'p2p_check.sh', 'p2p_probe.cu'):
        assert f'./{name}:/recipe/{name}:ro' in server['volumes']
        assert f'-v "$SCRIPT_DIR/{name}:/recipe/{name}:ro"' in start
    assert f'${{MODELS_DIR:-./models}}/{CHECKPOINT}:/models/{CHECKPOINT}:ro' in server['volumes']
    assert './docker/prepare.sh:/recipe/prepare.sh:ro' in prepare['volumes']
    assert './docker/prepare.sh:/recipe/prepare.sh:ro' in server['volumes']
    assert '.:/checkout:ro' in prepare['volumes']
    assert './cache:/cache' in prepare['volumes'] and './cache:/cache' in server['volumes']
    assert '${MODELS_DIR:-./models}:/models' in prepare['volumes']
    assert 'HOST_UID' not in str(server)


def test_prepare_fetches_the_start_checkpoint():
    start = (ROOT / 'start.sh').read_text()
    prepare = PREPARE.read_text()
    for key in ('TARGET_REPO', 'TARGET_REVISION', 'NEED_DISK_GB'):
        default = re.compile(r'^' + key + r'="\$\{' + key + r':-(.+)\}"', re.M)
        assert default.search(prepare)[1] == default.search(start)[1], key


def write_checkpoint(path, shards=('a.safetensors', 'b.safetensors')):
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    weight_map = ', '.join(f'"layer.{i}.weight": "{name}"' for i, name in enumerate(shards))
    (path / 'model.safetensors.index.json').write_text('{"weight_map": {' + weight_map + '}}')
    for name in shards:
        (path / name).write_bytes(b'x')
    return path


def check(path):
    return subprocess.run(['bash', str(PREPARE), 'check', str(path)]).returncode


def test_check_accepts_only_complete_checkpoints(tmp_path):
    assert check(write_checkpoint(tmp_path / 'complete')) == 0
    single = tmp_path / 'single'
    single.mkdir()
    (single / 'config.json').write_text('{}')
    assert check(single) == 1
    (single / 'model.safetensors').write_bytes(b'x')
    assert check(single) == 0
    assert check(tmp_path / 'absent') == 1
    shard = write_checkpoint(tmp_path / 'shard')
    (shard / 'b.safetensors').unlink()
    assert check(shard) == 1
    partial = write_checkpoint(tmp_path / 'partial')
    download = partial / '.cache' / 'huggingface' / 'download' / 'sub'
    download.mkdir(parents=True)
    (download / 'c.safetensors.incomplete').write_bytes(b'')
    assert check(partial) == 1
    noconfig = write_checkpoint(tmp_path / 'noconfig')
    (noconfig / 'config.json').unlink()
    assert check(noconfig) == 1


def prepare_functions(body, **env):
    """Run prepare.sh's functions with stat and chown replaced by recorders."""
    source = PREPARE.read_text().split('case "${1:-}" in', 1)[0]
    script = source + '''
stat() { case "$3" in *root*|*checkout*) echo "${FAKE_OWNER:-0}" ;; *) echo 1000 ;; esac; }
chown() { echo "chown $*"; }
''' + body
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                          env=dict(os.environ, **env))


def test_fetch_passes_the_pinned_revision(tmp_path):
    hf = tmp_path / 'hf'
    hf.write_text('#!/bin/sh\necho "hf $*"\n')
    hf.chmod(0o755)
    target = write_checkpoint(tmp_path / 'target')
    run = prepare_functions(f'HF="{hf}"\nfetch org/repo "{target}" rev1\nfetch org/repo "{target}"\n')
    assert run.returncode == 0, run.stderr
    lines = [line for line in run.stdout.splitlines() if line.startswith('hf ')]
    assert lines == [f'hf download org/repo --revision rev1 --local-dir {target}',
                     f'hf download org/repo --local-dir {target}']


def test_claim_changes_only_directories_docker_created(tmp_path):
    fresh = tmp_path / 'root-fresh'
    fresh.mkdir()
    (fresh / CHECKPOINT).mkdir()
    other = tmp_path / 'root-other'
    other.mkdir()
    (other / 'lost+found').mkdir()
    user = tmp_path / 'user-owned'
    user.mkdir()
    run = prepare_functions(f'''SERVE_UID=1234; SERVE_GID=4321
claim "{fresh}" {CHECKPOINT}
claim "{other}" {CHECKPOINT}
claim "{user}"
''')
    assert run.returncode == 0, run.stderr
    assert f'chown 1234:4321 {fresh}' in run.stdout
    assert str(other) not in run.stdout and str(user) not in run.stdout


@pytest.mark.parametrize('env,expected', [({'FAKE_OWNER': '1001'}, '1001 1001'),
                                          ({'HOST_UID': '1500', 'HOST_GID': '1600'}, '1500 1600'),
                                          ({'HOST_UID': '1500'}, '1500 1500'),
                                          ({'FAKE_OWNER': '0'}, '0 0')])
def test_serving_user_is_the_checkout_owner(tmp_path, env, expected):
    (tmp_path / 'checkout').mkdir()
    run = prepare_functions(f'''CHECKOUT="{tmp_path}/checkout"
serving_ids; echo "$SERVE_UID $SERVE_GID"
as_serving_user false; echo returned
''', **{'HOST_UID': '', 'HOST_GID': '', **env})
    assert run.stdout.splitlines()[0] == expected, run.stderr
    assert ('returned' in run.stdout) == expected.startswith('0 ')


def test_rejects_non_numeric_override():
    run = prepare_functions('serving_ids', HOST_UID='someone', HOST_GID='')
    assert run.returncode == 1 and 'numeric' in run.stderr


def test_dry_start_needs_no_engine_on_the_host(tmp_path):
    """The DRY launch shows serve.sh's whole result without the image's venv."""
    import shutil
    checkout = tmp_path / 'checkout'
    shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns(
        '.git', '.env', 'models', 'cache', 'logs', 'venv', 'vllm-src', '__pycache__'))
    shim = tmp_path / 'shim'
    shim.mkdir()
    for name in ('nvidia-smi', 'docker'):
        (shim / name).write_text('#!/bin/sh\nexit 1\n')
        (shim / name).chmod(0o755)
    run = subprocess.run(['bash', str(checkout / 'start.sh')], capture_output=True, text=True,
                         env={'HOME': str(tmp_path), 'PATH': f'{shim}:/usr/bin:/bin',
                              'DRY': '1', 'NEED_DISK_GB': '0'}, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'No such file' not in run.stdout + run.stderr
    assert 'serve.sh: DRY=1, command:' in run.stdout
