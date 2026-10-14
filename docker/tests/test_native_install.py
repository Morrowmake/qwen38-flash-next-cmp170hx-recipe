import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).parents[2]
PIN = 'a' * 40

MOCK = r'''
import json, os, pathlib, shutil, sys, time
root = pathlib.Path(os.environ['MOCK_ROOT'])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
def record(stage, **extra):
    with (root / 'calls').open('a') as out:
        out.write(json.dumps(dict(stage=stage, args=args, **extra)) + '\n')
if name == 'git':
    args = args[2:]
    if args[0] == 'rev-parse':
        print((root / 'head').read_text() if args[-1] == 'HEAD' else os.environ['VLLM_COMMIT'])
    elif args[0] == 'checkout':
        (root / 'head').write_text(args[-1])
    elif args[0] == 'log':
        print('pinned fork')
    sys.exit(0)
if name == 'timeout':
    record('timeout')
    os.execv(os.environ['REAL_TIMEOUT'], [os.environ['REAL_TIMEOUT']] + args)
if name == 'sleep':
    record('sleep')
    sys.exit(0)
if name == 'python':
    if args and args[0] == '-c':
        print('torch-mock')
        sys.exit(0)
    if args and args[0].endswith('build_standalone.py'):
        out = pathlib.Path(args[args.index('--out') + 1])
        (out / '_exl3_C.abi3.so').write_bytes(b'exl3')
        with (root / 'exl3_builds').open('a') as log:
            log.write('built\n')
        sys.exit(0)
    sys.stdin.read()
    record('verify')
    sys.exit(1 if os.environ.get('MOCK_VERIFY_FAIL') == '1' or not (root / 'flashinfer').exists() else 0)
if args[0] == 'venv':
    record('venv')
    dest = pathlib.Path(args[-1])
    shutil.rmtree(dest, ignore_errors=True)
    (dest / 'bin').mkdir(parents=True)
    shutil.copy2(root / 'bin/python', dest / 'bin/python')
    (dest / 'bin/vllm').write_text('#!/bin/sh\nexit 0\n')
    (dest / 'bin/vllm').chmod(0o755)
    (root / 'flashinfer').unlink(missing_ok=True)
    sys.exit(0)
if '-e' in args:
    stage = 'fork'
elif '-r' in args:
    stage = 'extras'
elif any(arg.startswith('flashinfer-python==') for arg in args):
    stage = 'flashinfer'
else:
    stage = 'torch'
extra = dict(http_timeout=os.environ.get('UV_HTTP_TIMEOUT'), http_retries=os.environ.get('UV_HTTP_RETRIES'),
             precompiled=os.environ.get('VLLM_USE_PRECOMPILED'),
             wheel=os.environ.get('VLLM_PRECOMPILED_WHEEL_COMMIT'))
if stage == 'extras':
    req = pathlib.Path(args[args.index('-r') + 1])
    extra.update(requirements=req.read_text(), common=(req.parent / 'common.txt').read_text())
record(stage, **extra)
if stage == 'fork':
    (root / 'flashinfer').unlink(missing_ok=True)
if os.environ.get('MOCK_FAIL_STAGE') == stage:
    sys.exit(1)
if stage == 'flashinfer':
    count_file = root / 'attempts'
    count = int(count_file.read_text()) + 1 if count_file.exists() else 1
    count_file.write_text(str(count))
    if os.environ.get('MOCK_FI_STALL') == '1':
        time.sleep(60)
    if count <= int(os.environ.get('MOCK_FI_FAILS', '0')):
        sys.exit(1)
    (root / 'flashinfer').touch()
'''


@pytest.fixture
def install(tmp_path):
    (tmp_path / '.env.example').write_text('')
    (tmp_path / 'docker').mkdir()
    shutil.copy2(ROOT / 'docker/CONSTRAINTS', tmp_path / 'docker/CONSTRAINTS')
    src = tmp_path / 'fork'
    (src / '.git').mkdir(parents=True)
    (src / 'vllm').mkdir()
    (src / 'requirements').mkdir()
    (src / 'requirements/common.txt').write_text('some-package==1.0\n')
    (src / 'requirements/cuda.txt').write_text(
        '-r common.txt\n--extra-index-url https://flashinfer.ai/whl/\n'
        'flashinfer-python==9.9.9\nflashinfer-cubin == 9.9.9 # upstream\ntilelang==0.1.12\n')
    (tmp_path / 'head').write_text(PIN)
    (tmp_path / 'toolkit').mkdir()
    (src / 'csrc/exl3').mkdir(parents=True)
    (src / 'csrc/exl3/build_standalone.py').write_text('')
    (tmp_path / 'toolkit/bin').mkdir()
    (tmp_path / 'toolkit/bin/nvcc').write_text('#!/bin/sh\necho nvcc-mock\n')
    (tmp_path / 'toolkit/bin/nvcc').chmod(0o755)
    (tmp_path / 'bin').mkdir()
    for name in ('uv', 'git', 'timeout', 'sleep', 'python'):
        path = tmp_path / 'bin' / name
        path.write_text('#!' + sys.executable + '\n' + MOCK)
        path.chmod(0o755)
    script = tmp_path / 'start.sh'
    script.write_text((ROOT / 'start.sh').read_text().rsplit('main "$@"', 1)[0] + '\ndo_install\n')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('VLLM_', 'FLASHINFER_', 'UV_', 'MOCK_'))}
    env.update(CUDA_VISIBLE_DEVICES='', RUNTIME='native', MOCK_ROOT=str(tmp_path),
               VENV=str(tmp_path / 'venv'), VLLM_SRC=str(src), VLLM_COMMIT=PIN,
               CUDA_HOME=str(tmp_path / 'toolkit'), BUILD_FROM_SOURCE='0',
               FORCE_INSTALL='0', VENV_CLEAR='0', NATIVE_INSTALL_RESUME='1',
               FLASHINFER_INSTALL_ATTEMPTS='3', FLASHINFER_HTTP_TIMEOUT='30',
               FLASHINFER_INSTALL_TIMEOUT='600', REAL_TIMEOUT=shutil.which('timeout'),
               PATH=str(tmp_path / 'bin') + ':/usr/bin:/bin')

    class Install:
        root = tmp_path
        steps = tmp_path / 'venv/.recipe-install-steps'
        stamp = tmp_path / 'venv/.recipe-stamp'

        def run(self, **overrides):
            return subprocess.run(['bash', str(script)], env=dict(env, **overrides),
                                  text=True, capture_output=True, timeout=10)

        def calls(self, stage):
            path = tmp_path / 'calls'
            return [json.loads(line) for line in path.read_text().splitlines()
                    if json.loads(line)['stage'] == stage] if path.exists() else []

    return Install()


@pytest.mark.parametrize('source', ['0', '1'])
def test_order_and_release_pins(install, source):
    result = install.run(BUILD_FROM_SOURCE=source)
    assert result.returncode == 0, result.stderr
    stages = [json.loads(line)['stage'] for line in (install.root / 'calls').read_text().splitlines()]
    assert stages == ['venv', 'torch', 'fork', 'extras', 'timeout', 'flashinfer', 'verify']
    call = install.calls('flashinfer')[0]
    assert 'flashinfer-python==0.7.0' in call['args']
    assert 'flashinfer-cubin==0.7.0' in call['args']
    assert call['http_timeout'] == '30' and call['http_retries'] == '0'
    assert install.calls('timeout')[0]['args'][:2] == ['--kill-after=5s', '600s']
    for stage in ('torch', 'fork', 'extras'):
        assert 'https://flashinfer.ai/whl/' not in install.calls(stage)[0]['args']
    extras = install.calls('extras')[0]
    assert extras['requirements'] == '-r common.txt\ntilelang==0.1.12\n'
    assert extras['common'] == 'some-package==1.0\n'
    assert install.calls('fork')[0]['precompiled'] == ('1' if source == '0' else None)
    assert '[native-install] resume=1' in result.stdout
    assert install.stamp.exists()
    assert {p.name for p in install.steps.iterdir()} == {'torch', 'fork', 'extras', 'flashinfer'}
    result = install.run(BUILD_FROM_SOURCE=source)
    assert result.returncode == 0 and 'already at' in result.stdout
    assert len(install.calls('fork')) == len(install.calls('flashinfer')) == 1


def test_index_exhaustion_resumes_without_uninstall(install):
    result = install.run(MOCK_FI_FAILS='3')
    assert result.returncode != 0
    assert 'https://flashinfer.ai/whl/ after 3 attempts' in result.stderr
    assert len(install.calls('flashinfer')) == 3
    assert [c['args'] for c in install.calls('sleep')] == [['2'], ['4']]
    assert not install.stamp.exists() and not (install.steps / 'flashinfer').exists()
    assert not install.calls('verify')
    result = install.run()
    assert result.returncode == 0, result.stderr
    assert len(install.calls('venv')) == len(install.calls('fork')) == len(install.calls('extras')) == 1
    assert len(install.calls('flashinfer')) == 4
    assert install.stamp.exists()


def test_retry_recovers_in_same_run(install):
    result = install.run(MOCK_FI_FAILS='2')
    assert result.returncode == 0, result.stderr
    assert len(install.calls('flashinfer')) == 3 and install.stamp.exists()


def test_verification_failure_only_repeats_verification(install):
    result = install.run(MOCK_VERIFY_FAIL='1')
    assert result.returncode != 0 and not install.stamp.exists()
    assert (install.steps / 'flashinfer').exists()
    assert install.run().returncode == 0
    assert len(install.calls('verify')) == 2
    for stage in ('venv', 'torch', 'fork', 'extras', 'flashinfer'):
        assert len(install.calls(stage)) == 1


@pytest.mark.parametrize('stage', ['torch', 'fork', 'extras'])
def test_stage_failure_resumes_at_failed_step(install, stage):
    assert install.run(MOCK_FAIL_STAGE=stage).returncode != 0
    assert not (install.steps / stage).exists() and not install.stamp.exists()
    assert install.run().returncode == 0
    assert len(install.calls(stage)) == 2
    stages = ['torch', 'fork', 'extras', 'flashinfer']
    for previous in stages[:stages.index(stage)]:
        assert len(install.calls(previous)) == 1
    assert len(install.calls('venv')) == 1


@pytest.mark.parametrize('setting', ['FORCE_INSTALL', 'VENV_CLEAR'])
def test_explicit_reinstall(install, setting):
    assert install.run().returncode == 0
    assert install.run(**{setting: '1'}).returncode == 0
    for stage in ('torch', 'fork', 'extras', 'flashinfer'):
        assert len(install.calls(stage)) == 2
    assert len(install.calls('venv')) == (2 if setting == 'VENV_CLEAR' else 1)


@pytest.mark.parametrize('change', ['commit', 'requirements', 'wheel', 'source', 'lock', 'torch_index'])
def test_changed_inputs_invalidate_steps(install, change):
    assert install.run(MOCK_FI_FAILS='3').returncode != 0
    overrides = {}
    if change == 'commit':
        overrides['VLLM_COMMIT'] = 'b' * 40
    elif change == 'requirements':
        with (install.root / 'fork/requirements/common.txt').open('a') as out:
            out.write('other-package==2.0\n')
    elif change == 'wheel':
        overrides['VLLM_PRECOMPILED_WHEEL_COMMIT'] = 'c' * 40
    elif change == 'source':
        overrides['BUILD_FROM_SOURCE'] = '1'
    elif change == 'lock':
        lock = install.root / 'docker/CONSTRAINTS'
        lock.write_text(lock.read_text().replace('flashinfer-python==0.7.0', 'flashinfer-python==0.7.1'))
    else:
        overrides['TORCH_INDEX_URL'] = 'https://torch.example/whl/'
    result = install.run(**overrides)
    assert result.returncode == 0, result.stderr
    assert len(install.calls('fork')) == len(install.calls('extras')) == 2
    assert len(install.calls('venv')) == (2 if change == 'requirements' else 1)
    if change == 'lock':
        assert 'flashinfer-python==0.7.1' in install.calls('flashinfer')[-1]['args']


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unpinned'])
def test_invalid_lock_fails_before_pip(install, kind):
    lock = install.root / 'docker/CONSTRAINTS'
    value = lock.read_text()
    if kind == 'missing':
        value = value.replace('flashinfer-python==0.7.0\n', '')
    elif kind == 'duplicate':
        value += 'flashinfer-python==0.7.1\n'
    else:
        value = value.replace('flashinfer-python==0.7.0', 'flashinfer-python>=0.7.0')
    lock.write_text(value)
    result = install.run()
    assert result.returncode != 0 and 'pin in docker/CONSTRAINTS' in result.stderr
    assert not install.calls('torch') and not install.stamp.exists()


@pytest.mark.parametrize('setting,value', [('NATIVE_INSTALL_RESUME', '2'),
    ('FLASHINFER_INSTALL_ATTEMPTS', '0'), ('FLASHINFER_INSTALL_ATTEMPTS', '-1'),
    ('FLASHINFER_HTTP_TIMEOUT', 'bad'), ('FLASHINFER_INSTALL_TIMEOUT', '1.5')])
def test_invalid_settings_fail_before_install(install, setting, value):
    result = install.run(**{setting: value})
    assert result.returncode != 0 and setting in result.stderr
    assert not install.calls('venv')


def test_stalled_fetch_has_wall_deadline(install):
    started = time.monotonic()
    result = install.run(MOCK_FI_STALL='1', FLASHINFER_INSTALL_ATTEMPTS='1',
                         FLASHINFER_HTTP_TIMEOUT='1', FLASHINFER_INSTALL_TIMEOUT='1')
    assert result.returncode != 0 and time.monotonic() - started < 5
    assert 'https://flashinfer.ai/whl/ after 1 attempts' in result.stderr
    assert not install.stamp.exists() and not (install.steps / 'flashinfer').exists()


def test_resume_switch_repeats_pending_steps(install):
    assert install.run(MOCK_FI_FAILS='3').returncode != 0
    result = install.run(NATIVE_INSTALL_RESUME='0')
    assert result.returncode == 0 and '[native-install] resume=0' in result.stdout
    assert len(install.calls('fork')) == len(install.calls('extras')) == 2
    assert len(install.calls('venv')) == 1


@pytest.mark.parametrize('stage', ['torch', 'fork', 'extras'])
def test_lost_upstream_stamp_invalidates_downstream(install, stage):
    assert install.run(MOCK_VERIFY_FAIL='1').returncode != 0
    (install.steps / stage).unlink()
    assert install.run().returncode == 0
    stages = ['torch', 'fork', 'extras', 'flashinfer']
    for next_stage in stages[stages.index(stage):]:
        assert len(install.calls(next_stage)) == 2


def test_legacy_stamp_reuses_venv(install):
    assert install.run().returncode == 0
    install.stamp.write_text('\n'.join(install.stamp.read_text().splitlines()[:2]) + '\n')
    shutil.rmtree(install.steps)
    (install.root / 'venv/.recipe-reqs').unlink()
    result = install.run()
    assert result.returncode == 0, result.stderr
    assert len(install.calls('venv')) == 1 and len(install.calls('fork')) == 2
    assert len(install.stamp.read_text().splitlines()) == 3


@pytest.mark.parametrize('stage', ['torch', 'fork', 'extras', 'flashinfer'])
def test_failed_forced_reinstall_cannot_keep_old_step_stamp(install, stage):
    assert install.run().returncode == 0
    assert install.run(FORCE_INSTALL='1', MOCK_FAIL_STAGE=stage).returncode != 0
    assert not install.stamp.exists() and not (install.steps / stage).exists()
    before = len(install.calls(stage))
    assert install.run().returncode == 0
    assert len(install.calls(stage)) == before + 1


def test_partial_native_install_remains_native_without_override(install):
    assert install.run(MOCK_FI_FAILS='3').returncode != 0
    result = install.run(RUNTIME='')
    assert result.returncode == 0, result.stderr
    assert len(install.calls('fork')) == 1 and install.stamp.exists()


def exl3_builds(install):
    path = install.root / 'exl3_builds'
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_exl3_extension_built_before_verify_and_reused(install):
    result = install.run()
    assert result.returncode == 0, result.stderr
    assert (install.root / 'fork/vllm/_exl3_C.abi3.so').read_bytes() == b'exl3'
    assert exl3_builds(install) == 1 and 'EXL3: building the sm_80 extension' in result.stdout
    assert result.stdout.index('EXL3: building') < result.stdout.index('install: done')
    result = install.run()
    assert result.returncode == 0 and 'EXL3: matching sm_80 extension already installed' in result.stdout
    assert exl3_builds(install) == 1
    (install.root / 'fork/vllm/_exl3_C.abi3.so').write_bytes(b'tampered')
    assert install.run().returncode == 0 and exl3_builds(install) == 2


def test_exl3_requires_builder_and_nvcc(install):
    (install.root / 'toolkit/bin/nvcc').unlink()
    result = install.run()
    assert result.returncode != 0 and 'needs nvcc' in result.stderr
    (install.root / 'toolkit/bin/nvcc').write_text('#!/bin/sh\n')
    (install.root / 'fork/csrc/exl3/build_standalone.py').unlink()
    result = install.run()
    assert result.returncode != 0 and 'no EXL3 builder' in result.stderr
