#!/usr/bin/env python3
"""Exercise the watchdog with synthetic memory and fake commands only."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('watchdog', ROOT / 'tools/memory_watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class Guard(unittest.TestCase):
    def check(self, *, state='active', label='synthetic-model', available=1,
              inspect_rc=0, stop_error=False):
        self.commands = []
        def run(args, **kwargs):
            self.commands.append((args, kwargs))
            self.assertEqual(kwargs['timeout'], 2.0)
            if args[:2] == ['systemctl', 'is-active']:
                return subprocess.CompletedProcess(args, 0, state, '')
            if args[0] == 'docker':
                return subprocess.CompletedProcess(args, inspect_rc, label, '')
            if stop_error:
                raise subprocess.CalledProcessError(1, args, stderr='suppressed')
            return subprocess.CompletedProcess(args, 0, '', '')
        return watchdog.check_once('synthetic.service', 'synthetic-container', 'synthetic-model',
                                   2.0, 2.0, read_memory=lambda: f'MemAvailable: {available * (1 << 20)} kB\n',
                                   run=run)

    def test_low_memory_stops_exactly_the_selected_service(self):
        self.assertEqual(self.check(), 'stopped')
        self.assertEqual(self.commands[-1][0], ['systemctl', '--no-block', 'stop', 'synthetic.service'])
        self.assertTrue(self.commands[-1][1]['check'])
        self.assertEqual(len(self.commands), 3)

    def test_floor_and_higher_are_healthy(self):
        for value in (2, 3):
            with self.subTest(value=value):
                self.assertEqual(self.check(available=value), 'healthy')
                self.assertEqual(len(self.commands), 2)

    def test_inactive_service_never_inspects_or_stops_container(self):
        for state in ('inactive', 'failed', 'deactivating', 'unknown'):
            self.assertEqual(self.check(state=state), 'inactive')
            self.assertEqual(len(self.commands), 1)

    def test_startup_and_reload_are_monitored(self):
        for state in ('activating', 'reloading'):
            self.assertEqual(self.check(state=state), 'stopped')

    def test_unrelated_or_absent_container_is_not_stopped(self):
        self.assertEqual(self.check(label='other-model'), 'waiting')
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.check(inspect_rc=1), 'waiting')
        self.assertEqual(len(self.commands), 2)

    def test_memavailable_requires_one_valid_kernel_field(self):
        self.assertEqual(watchdog.available_gib('MemAvailable: 1048576 kB\n'), 1.0)
        for text in ('', 'MemFree: 1048576 kB', 'MemAvailable: -1 kB',
                     'MemAvailable: 1 MB', 'MemAvailable: 1 kB\nMemAvailable: 2 kB'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                watchdog.available_gib(text)

    def test_bad_targets_and_thresholds_cannot_execute_commands(self):
        for service, container, floor in [('--all', 'synthetic', 1), ('synthetic.service', '--all', 1),
                                           ('synthetic.service', 'synthetic', float('nan')),
                                           ('synthetic.service', 'synthetic', 0)]:
            with patch.object(watchdog.subprocess, 'run') as run, self.assertRaises(ValueError):
                watchdog.check_once(service, container, 'synthetic-model', floor, 2)
            run.assert_not_called()

    def test_stop_failure_and_inspection_timeout_are_not_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.check(stop_error=True)
        with patch.object(watchdog.subprocess, 'run', side_effect=subprocess.TimeoutExpired('synthetic', 2)):
            with self.assertRaises(subprocess.TimeoutExpired):
                watchdog.check_once('synthetic.service', 'synthetic', 'synthetic-model', 1, 2)

    def test_main_exits_after_one_stop_without_restart(self):
        with patch.object(watchdog, 'check_once', return_value='stopped') as check, \
             patch.object(watchdog.time, 'sleep') as sleep:
            self.assertEqual(watchdog.main(['--service', 'synthetic.service', '--container', 'synthetic',
                                           '--memory-profile', 'synthetic-model', '--minimum-available-gib', '2']), 0)
            check.assert_called_once()
            sleep.assert_not_called()

    def test_optional_template_is_not_activated_by_installer(self):
        template = (ROOT / 'deploy/systemd/inference-memory-watchdog.service.in').read_text()
        self.assertIn('BindsTo=__TARGET_SERVICE__', template)
        self.assertIn('Restart=no', template)
        installer = (ROOT / 'deploy/install.sh').read_text()
        self.assertNotIn('inference-memory-watchdog.service', installer)


if __name__ == '__main__':
    unittest.main()
