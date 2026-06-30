#!/usr/bin/env python3
"""Contextual secret/path scanner for dgx-spark-inference.

NOT a blanket ban (image digests, model revisions, and documented loopback/bind
addresses would false-positive). This scanner rejects only things that should
never be in a public reference repo:

  REJECT (contextual):
    - credential assignments:  key = "...", api_key = "...", Authorization: Bearer ...
    - PEM private-key headers
    - user-specific absolute paths:  /home/<user>, /Users/<user>
    - private hostnames / private IPs (RFC1918, link-local, loopback excluded by
      exception; loopback and 0.0.0.0 are allowed as documented bind)

  ALLOW (explicit exceptions):
    - image digests:  sha256:<64 hex>
    - git/HF revisions:  40-hex sha
    - documented bind addresses: 127.0.0.1, 0.0.0.0
    - the dummy placeholder REDACTED-PLACEHOLDER emitted by --dry-run

Exit 0 if clean, 1 if any finding. Findings go to stdout; a summary to stderr.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ---- things that are fine (do not flag even if they match a reject pattern) --
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SHA40 = re.compile(r"\b[0-9a-f]{40}\b")
ALLOWED_HOSTS = {"127.0.0.1", "0.0.0.0", "localhost"}
PLACEHOLDER = "REDACTED-PLACEHOLDER"

# ---- reject patterns ---------------------------------------------------------
# credential assignment:  name(secret-looking-value)  with an actual token value.
# The value must be a concrete token (>=8 chars, not an ellipsis/placeholder/
# variable-ref like "...", "<...>", {FAKE_KEY}, or the dummy placeholder).
CRED_ASSIGN = re.compile(
    r"""(?ix)
    \b(?:api[_-]?key|secret|token|password|passwd|bearer|authorization|
          sglang[_-]?api[_-]?key)\b\s*[:=]\s*
    (?:Bearer\s+)?["']?([A-Za-z0-9_\-+/=]+)
    """,
)
# tokens that are examples/placeholders, not real secrets -> do not flag
_PLACEHOLDER_VALUE = re.compile(r"^(?:\.\.\.+|<[^>]+>|\{[A-Za-z0-9_]+\}|REDACTED[-_]?PLACEHOLDER|example|your[_-]?key|xxxx+)$", re.I)
PEM_HEADER = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----")
USER_PATH = re.compile(r"(?i)(?:/home|/Users)/[A-Za-z0-9._-]+")
# private IPv4 (RFC1918 + link-local); loopback handled by ALLOWED_HOSTS.
PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|169\.254\.\d{1,3}\.\d{1,3})\b"
)
# local-link hostnames that leak the deployment (operator host). Allowlist of
# neutral service names so we don't flag generic words.
HOSTNAME_LEAK = re.compile(r"\b[a-z0-9-]+\.local\b", re.I)

REJECTORS = [
    ("PEM private key header", PEM_HEADER),
    ("credential assignment", CRED_ASSIGN),
    ("user-specific path (/home|/Users)", USER_PATH),
    ("private IPv4", PRIVATE_IP),
    ("local hostname (.local)", HOSTNAME_LEAK),
]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
SKIP_SUFFIXES = {".pyc"}


def line_allows(text: str) -> bool:
    # whole-line allow exceptions
    if PLACEHOLDER in text:
        return True
    return False


def finding_is_allowed(match_text: str, full_line: str) -> bool:
    # digest / 40-hex revision spans the matched secret-ish token
    if DIGEST.search(match_text) or SHA40.search(match_text):
        return True
    # the matched IP/host is a documented bind
    for h in ALLOWED_HOSTS:
        if h in match_text:
            return True
    return False


def credential_value_is_placeholder(value: str) -> bool:
    """A credential-assignment value that is an example/placeholder, not a real
    secret (e.g. '...', '<token>', '{FAKE_KEY}', 'REDACTED-PLACEHOLDER')."""
    if _PLACEHOLDER_VALUE.match(value):
        return True
    # require a concrete token of meaningful length; very short example values
    # (like 'x' or 'abc') are treated as placeholders too.
    return len(value) < 8


def scan_path(root: Path) -> list[tuple[Path, int, str]]:
    """Return (path, lineno, finding_type) tuples. The matched text is used
    internally for allow/placeholder decisions but is NEVER included in the
    output, so a real secret can never leak into scanner/CI logs."""
    findings: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_dir() or path.suffix in SKIP_SUFFIXES:
            continue
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            continue  # binary
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat in REJECTORS:
                for m in pat.finditer(line):
                    matched = m.group(0)
                    if finding_is_allowed(matched, line):
                        continue
                    # credential assignment: only flag if the captured value is a
                    # concrete token, not a documented placeholder/example.
                    if name == "credential assignment":
                        val = m.group(1) if m.groups() else ""
                        if credential_value_is_placeholder(val):
                            continue
                    findings.append((path, lineno, name))
    return findings


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    if not root.exists():
        print(f"scan_secrets: path not found: {root}", file=sys.stderr)
        return 2
    findings = scan_path(root)
    if not findings:
        print(f"scan_secrets: CLEAN ({root})", file=sys.stderr)
        return 0
    for path, lineno, name in findings:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        # Deliberately do NOT print the matched text (could be a real secret).
        print(f"{rel}:{lineno}: {name}")
    print(f"scan_secrets: {len(findings)} finding(s) in {root}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
