#!/usr/bin/env python3
"""Version reporting works in checkouts and installed trees without live state."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = Path('src/inferencectl/inference-cli.sh')


class ProjectVersion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config'
        self.config.mkdir()
        # Reporting version must never source even an existing operator file.
        (self.config / 'inference.env').write_text('exit 97\n')

    def report(self, root):
        return subprocess.run(
            ['bash', str(root / CLI), '--config-root', str(self.config), '--version'],
            cwd=self.root, env=dict(os.environ, PROJECT_ROOT='/unused'),
            text=True, capture_output=True, timeout=10,
        )

    def test_source_is_one_semver_without_tag_prefix(self):
        text = (ROOT / 'VERSION').read_text()
        number = r'(?:0|[1-9][0-9]*)'
        identifier = rf'(?:{number}|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)'
        semver = rf'{number}\.{number}\.{number}(?:-{identifier}(?:\.{identifier})*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?'
        self.assertIsNotNone(re.fullmatch(semver + '\n', text))

    def test_checkout_reports_without_loading_operator_state(self):
        result = self.report(ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'dgx-spark-inference ' + (ROOT / 'VERSION').read_text())

    def test_real_program_install_copies_authoritative_version(self):
        installed = self.root / 'installed'
        installer = (ROOT / 'deploy/install.sh').read_text()
        function = installer.split('install_program_tree() {', 1)[1].split('\n}\n', 1)[0]
        # Exercise only program-file installation in a temporary directory.
        # Ownership changes are skipped; no sudo, units, or operator state run.
        script = '''set -e
srun() { if [ "$1" = chown ]; then return 0; fi; "$@"; }
install_program_tree() {''' + function + '\n}\ninstall_program_tree\n'
        result = subprocess.run(
            ['bash', '-c', script],
            env=dict(os.environ, SOURCE_DIR=str(ROOT), INSTALL_ROOT=str(installed)),
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((installed / '.git').exists())
        self.assertEqual((installed / 'VERSION').read_bytes(), (ROOT / 'VERSION').read_bytes())
        self.assertEqual(self.report(installed).stdout, self.report(ROOT).stdout)
        # Synthetic replacement proves reporting uses this artifact, not Git,
        # an operator override, or a second hard-coded version string.
        (installed / 'VERSION').write_text('9.8.7-test\n')
        self.assertEqual(self.report(installed).stdout, 'dgx-spark-inference 9.8.7-test\n')
        (installed / 'VERSION').unlink()
        self.assertNotEqual(self.report(installed).returncode, 0)


if __name__ == '__main__':
    unittest.main()
