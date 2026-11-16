#!/usr/bin/env python3
"""Synthetic adapter/probe tests. No Docker daemon, GPU, or service is used."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


adapter = module("vllm_adapter", "runtime/vllm/adapters/adapter.py")
probe = module("vllm_probe", "runtime/vllm/capacity_probe.py")
selector = module("runtime_probe", "tools/probe_runtime_capacity.py")


class Backend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.manifest = ('runtime_id="synthetic-runtime"\nruntime_kind="vllm"\n'
                         'allocation_probe="capacity_probe.py"\nimage="synthetic/image:pin"\n'
                         'image_id="sha256:' + '1' * 64 + '"\n')
        (self.runtime / "runtime-manifest.toml").write_text(self.manifest)
        self.profile = ('kind="model-runtime-config"\nmodel_id="synthetic-model"\n'
                        'runtime_id="synthetic-runtime"\n[identity]\n'
                        'source_repository="example/model"\nsource_revision="' + '2' * 40 + '"\n'
                        '[allocation]\npool_tokens=1234\ncache_bytes=5678\n[launch]\nmax-model-len=1024\n')
        (self.root / "model.toml").write_text(self.profile)
        cache = self.root / "cache"
        snapshot = cache / "models--example--model/snapshots" / ('2' * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{}')
        self.env = dict(MODEL_CACHE_ROOT=str(cache), CONTAINER_NAME="synthetic-container",
                        PORT=str(1024 + 7), ROLE="synthetic-role", DGX_MAX_TOTAL_TOKENS="1234",
                        DGX_RUNTIME_CONFIG_DIR=str(self.root / "rendered"), SGLANG_API_KEY="0" * 64)

    def build(self):
        return adapter.build_command(self.runtime, self.root, "synthetic-model", "model",
                                     "model.toml", "synthetic-served", self.env)

    def test_fixed_allocation_labels_and_secret_handling(self):
        args, _, env = self.build()
        config_path = next((self.root / "rendered").glob('*.json'))
        config = json.loads(config_path.read_text())
        self.assertEqual(config['kv-cache-memory-bytes'], 5678)
        self.assertEqual(config['port'], 1031)
        self.assertEqual(config['served-model-name'], 'synthetic-served')
        self.assertNotIn(self.env['SGLANG_API_KEY'], ' '.join(args) + config_path.read_text())
        self.assertEqual(env['VLLM_API_KEY'], self.env['SGLANG_API_KEY'])
        self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(config_path.parent.stat().st_mode & 0o777, 0o700)
        self.assertIn('io.inferencectl.managed=true', args)
        self.assertIn('io.inferencectl.memory_profile=synthetic-model', args)
        self.assertIn(str(Path(self.env['MODEL_CACHE_ROOT'])) + ':/model-cache:ro', args)

    def test_mismatched_or_missing_admission_refuses(self):
        for value in ('0', '1235', '1.5', ''):
            with self.subTest(value=value):
                self.env['DGX_MAX_TOTAL_TOKENS'] = value
                with self.assertRaises(ValueError):
                    self.build()

    def test_runtime_environment_and_ledger_identity(self):
        (self.runtime / 'runtime-manifest.toml').write_text(
            self.manifest + '[environment]\nVLLM_SYNTHETIC_OPTION="synthetic-value"\n')
        ledger = self.root / 'ledger.toml'
        ledger.write_text('# synthetic ledger\n')
        self.env['DGX_MEMORY_LEDGER'] = str(ledger)
        args, _, env = self.build()
        self.assertEqual(env['VLLM_SYNTHETIC_OPTION'], 'synthetic-value')
        self.assertNotIn('synthetic-value', args)
        revision = hashlib.sha256(ledger.read_bytes()).hexdigest()[:16]
        self.assertIn('io.inferencectl.ledger_revision=' + revision, args)
        (self.runtime / 'runtime-manifest.toml').write_text(
            self.manifest + '[environment]\nVLLM_API_KEY="invalid"\n')
        with self.assertRaises(ValueError):
            self.build()

    def test_matching_image_reaches_only_the_managed_docker_command(self):
        argv = ['adapter', 'synthetic-role', str(self.runtime), str(self.root),
                'synthetic-model', 'model', 'model.toml', 'synthetic-served']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, self.env), \
             patch.object(adapter.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'sha256:' + '1' * 64)), \
             patch.object(adapter.os, 'execvpe') as launch:
            self.assertEqual(adapter.main(), 0)
            executable, args, env = launch.call_args.args
            self.assertEqual(executable, 'docker')
            self.assertEqual(args[:3], ['docker', 'run', '--rm'])
            self.assertEqual(env['VLLM_API_KEY'], self.env['SGLANG_API_KEY'])
            self.assertNotIn(self.env['SGLANG_API_KEY'], ' '.join(args))

    def test_managed_launch_fields_cannot_be_overridden(self):
        for key in ('api-key', 'kv-cache-memory-bytes', 'gpu_memory_utilization', 'model', 'port'):
            with self.subTest(key=key):
                (self.root / 'model.toml').write_text(self.profile + f'{key}="invalid"\n')
                with self.assertRaises(ValueError):
                    self.build()

    def test_identity_pins_and_image_drift_refuse(self):
        (self.root / 'model.toml').write_text(self.profile.replace('2' * 40, 'main'))
        with self.assertRaises(ValueError):
            self.build()
        (self.root / 'model.toml').write_text(self.profile)
        argv = ['adapter', 'synthetic-role', str(self.runtime), str(self.root),
                'synthetic-model', 'model', 'model.toml', 'synthetic-served']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, self.env), \
             patch.object(adapter.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'different')), \
             patch.object(adapter.os, 'execvpe') as launch:
            self.assertEqual(adapter.main(), 75)
            launch.assert_not_called()

    def test_capacity_parser_uses_realized_not_context_capacity(self):
        self.assertEqual(probe.capacity('max_model_len=99999\nGPU KV cache size: 1,234 tokens'), 1234)
        self.assertEqual(probe.capacity('GPU KV cache size: 1234 tokens\nGPU KV cache size: 5678 tokens'), 5678)
        for bad in ('', 'max_model_len=1234', 'GPU KV cache size: 0 tokens',
                    'GPU KV cache size: 1,,234 tokens', 'GPU KV cache size: -1 tokens'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.capacity(bad)

    def test_probe_bounds_current_startup_logs_and_never_echoes_them(self):
        replies = [subprocess.CompletedProcess([], 0, '2026-01-01T00:00:00Z\n'),
                   subprocess.CompletedProcess([], 0, '', 'GPU KV cache size: 1234 tokens\n')]
        with patch.object(probe.subprocess, 'run', side_effect=replies) as run:
            self.assertEqual(probe.probe('synthetic-container'), 1234)
            self.assertIn('--since', run.call_args_list[1].args[0])
            self.assertIn('--tail', run.call_args_list[1].args[0])
            self.assertEqual(run.call_args_list[1].kwargs['timeout'], 10)
        with self.assertRaises(ValueError):
            probe.probe('--all')

    def test_probe_selection_validates_paths(self):
        f = self.runtime / 'capacity_probe.py'
        f.write_text('print(1234)\n')
        self.assertEqual(selector.probe_path(self.runtime), f)
        (self.runtime / 'runtime-manifest.toml').write_text('image="synthetic/image"\n')
        self.assertIsNone(selector.probe_path(self.runtime))
        for name in ('../outside.py', '/outside.py', 'missing.py'):
            (self.runtime / 'runtime-manifest.toml').write_text(f'allocation_probe="{name}"\n')
            with self.assertRaises(ValueError):
                selector.probe_path(self.runtime)

    def test_real_admission_verifier_checks_custom_capacity_and_fails_closed(self):
        source = (ROOT / 'src/inferencectl/admission.sh').read_text()
        verifier = source.split('verify_ready() {', 1)[1].split('\nif ! verify_ready;', 1)[0]
        script = '''kill() { return 0; }
sleep() { :; }
log() { :; }
curl() { printf '%s' '{"max_total_num_tokens":25}'; }
''' + 'verify_ready() {' + verifier + '\nverify_ready\n'
        env = dict(os.environ, PROJECT_ROOT=str(ROOT), RUNTIME_ROOT=str(self.runtime),
                   CONTAINER_NAME='synthetic-container', ADAPTER_PID='1', PORT='1031',
                   ADMISSION_READY_TIMEOUT='5', MINTOKENS='20', MAXTOKENS='30')
        env['PATH'] = str(Path(sys.executable).parent) + ':' + env['PATH']
        for output, expected in [('25', 0), ('19', 1), ('31', 1), ('not-capacity', 1)]:
            with self.subTest(output=output):
                (self.runtime / 'capacity_probe.py').write_text(f'print({output!r})\n')
                result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, expected)
        # A failing custom probe must not fall back to SGLang's otherwise-valid response.
        (self.runtime / 'capacity_probe.py').write_text('raise SystemExit(1)\n')
        self.assertNotEqual(subprocess.run(['bash', '-c', script], env=env, capture_output=True,
                                          timeout=10).returncode, 0)
        (self.runtime / 'runtime-manifest.toml').write_text('image="synthetic/image"\n')
        self.assertEqual(subprocess.run(['bash', '-c', script], env=env, capture_output=True,
                                       timeout=10).returncode, 0)


if __name__ == '__main__':
    unittest.main()
