#!/usr/bin/env python3
"""measure_model_budget.py — parse a sglang startup log and emit a budget-ledger
``[[profiles]]`` entry for the memory planner.

The memory planner (``tools/memory_planner/resolve_memory_plan.py``) turns a
per-model budget ledger + a residency plan into the two launch knobs SGLang
needs (``mem_fraction_static`` + ``max_total_tokens``). This tool MEASURES the
budget fields from an actual sglang startup log, so a new model/profile can be
enrolled without hand-reading log lines.

Design: **parse, don't launch.** Feed it a log file (or pipe sglang's stdout).
Each extracted value is printed to stderr with its source line so the operator
can verify/adjust before appending to the ledger.

Stdlib-only (no GPU, no third-party deps). See ``config/schemas/memory-plan.md``
for the field definitions and ``docs/measure-model-budget.md`` for the walkthrough.

Usage::

    # capture a launch log (run sglang once with a known mem_fraction)
    python3 -m sglang.launch_server ... 2>&1 | tee /tmp/model.log

    # measure
    python3 tools/memory_planner/measure_model_budget.py \\
        --log /tmp/model.log \\
        --model-id qwen36-27b-nvfp4-kv-fp8-e4m3-mtp \\
        --mem-fraction 0.60 \\
        > /tmp/new_profile.toml

    # review the stderr trace, then append
    cat /tmp/new_profile.toml >> tools/memory_planner/budget_ledger.toml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

GIB = 1 << 30  # binary GiB — matches the resolver's unit convention


# ---------------------------------------------------------------------------
# Parsing helpers — each returns (value, source_line) or (None, None)
# ---------------------------------------------------------------------------

def _find(pattern: str, lines: list[str]) -> Optional[str]:
    """Return the first line matching the regex pattern, or None."""
    rx = re.compile(pattern)
    for line in lines:
        if rx.search(line):
            return line
    return None


def _find_all(pattern: str, lines: list[str]) -> list[str]:
    """Return all lines matching the regex pattern."""
    rx = re.compile(pattern)
    return [line for line in lines if rx.search(line)]


def _extract_float(line: str, key: str) -> Optional[float]:
    """Extract a 'key=NN.NN' or 'key: NN.NN' float value from a log line."""
    m = re.search(rf"{re.escape(key)}\s*[=:]\s*([0-9.]+)", line)
    return float(m.group(1)) if m else None


def _extract_int(line: str, key: str) -> Optional[int]:
    """Extract a 'key=NN' or 'key: NN' int value from a log line."""
    m = re.search(rf"{re.escape(key)}\s*[=:]\s*([0-9]+)", line)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

class Measurement:
    """Holds extracted values + their derivation trace (source lines)."""

    def __init__(self) -> None:
        self.trace: list[tuple[str, str, object]] = []
        # (label, value, source_or_note)

    def record(self, label: str, value: object, source: str) -> None:
        self.trace.append((label, value, source))

    def warn(self, msg: str) -> None:
        self.trace.append(("WARNING", msg, ""))

    # -- weights_gib: sum of all "Load weight end" mem_usage lines -----------

    def measure_weights(self, lines: list[str]) -> float:
        load_lines = _find_all(r"Load weight end\.", lines)
        if not load_lines:
            self.warn("no 'Load weight end.' lines found — weights_gib will be 0")
            self.record("weights_gib", 0.0, "(no load lines)")
            return 0.0
        total = 0.0
        parts: list[str] = []
        for line in load_lines:
            mu = _extract_float(line, "mem usage")
            if mu is not None:
                total += mu
                # extract the model type for the trace
                type_m = re.search(r"type=(\S+)", line)
                type_str = type_m.group(1) if type_m else "?"
                parts.append(f"{mu:.2f} ({type_str})")
            else:
                self.warn(f"'Load weight end.' line missing 'mem usage': {line.strip()[:100]}")
        self.record("weights_gib", round(total, 2), "; ".join(parts))
        return total

    # -- target_kv_tokens + kv_bytes_per_token: from "KV Cache is allocated" --

    def measure_kv_pool(self, lines: list[str]) -> tuple[int, float]:
        kv_lines = _find_all(r"KV Cache is allocated\.", lines)
        if not kv_lines:
            self.warn("no 'KV Cache is allocated.' lines — pool fields will be 0")
            self.record("target_kv_tokens", 0, "(no KV cache line)")
            self.record("kv_bytes_per_token", 0.0, "(no KV cache line)")
            return 0, 0.0
        # Take the pool with the most tokens (primary > draft for MTP)
        best_tokens = 0
        best_k = 0.0
        best_v = 0.0
        best_line = ""
        for line in kv_lines:
            tokens = _extract_int(line, "#tokens") or _extract_int(line, "tokens") or 0
            k_size = _extract_float(line, "K size") or 0.0
            v_size = _extract_float(line, "V size") or 0.0
            if tokens > best_tokens:
                best_tokens = tokens
                best_k = k_size
                best_v = v_size
                best_line = line.strip()
        if best_tokens > 0 and (best_k + best_v) > 0:
            kv_bytes = (best_k + best_v) * GIB / best_tokens
        else:
            kv_bytes = 0.0
            self.warn("KV cache line found but could not derive kv_bytes_per_token")
        if len(kv_lines) > 1:
            self.warn(f"multiple KV pools found ({len(kv_lines)}); using the largest ({best_tokens} tokens)")
        self.record("target_kv_tokens", best_tokens, best_line[:120])
        self.record("kv_bytes_per_token", round(kv_bytes, 1), f"({best_k}+{best_v}) GiB / {best_tokens} tokens")
        return best_tokens, kv_bytes

    # -- mamba_cache_gib: from "Mamba Cache is allocated" (hybrid models) ---

    def measure_mamba_cache(self, lines: list[str]) -> float:
        mamba_line = _find(r"Mamba Cache is allocated\.", lines)
        if not mamba_line:
            self.record("mamba_cache_gib", 0.0, "(no Mamba cache — dense attention model)")
            return 0.0
        # Sum all '*size*GB' values on the line
        sizes = re.findall(r"([\d.]+)\s*GB", mamba_line)
        total = sum(float(s) for s in sizes) if sizes else 0.0
        self.record("mamba_cache_gib", round(total, 2), mamba_line.strip()[:120])
        return total

    # -- cuda_graph_peak_gib: sum of "Capture ... end" mem usage deltas ------

    def measure_graph_peak(self, lines: list[str]) -> float:
        end_lines = _find_all(r"Capture .*(CUDA graph|cuda graph).*end\.", lines)
        if not end_lines:
            self.record("cuda_graph_peak_gib", 0.0, "(no CUDA graph capture lines)")
            return 0.0
        total = 0.0
        parts: list[str] = []
        for line in end_lines:
            mu = _extract_float(line, "mem usage")
            if mu is not None and mu > 0:
                total += mu
                # extract the graph type for the trace
                type_m = re.search(r"Capture (\S+ \S+ \S+)", line)
                type_str = type_m.group(1) if type_m else "?"
                parts.append(f"{mu:.2f} ({type_str})")
        self.record("cuda_graph_peak_gib", round(total, 2), "; ".join(parts) if parts else "(all deltas <= 0)")
        return total

    # -- A_preload: first "Load weight begin. avail mem" --------------------

    def measure_a_preload(self, lines: list[str]) -> float:
        begin_line = _find(r"Load weight begin\.", lines)
        if not begin_line:
            self.warn("no 'Load weight begin.' line — A_preload unknown; static_overhead will be 0")
            self.record("A_preload_gib", 0.0, "(no load-begin line)")
            return 0.0
        avail = _extract_float(begin_line, "avail mem")
        if avail is None:
            self.warn("'Load weight begin.' line missing 'avail mem'")
            self.record("A_preload_gib", 0.0, begin_line.strip()[:100])
            return 0.0
        self.record("A_preload_gib", round(avail, 2), begin_line.strip()[:100])
        return avail

    # -- static_overhead_gib: the measured gap ------------------------------

    def compute_static_overhead(
        self, mem_fraction: float, a_preload: float,
        weights: float, kv_tokens: int, kv_bytes: float,
        static_pad: float, mamba_cache: float,
    ) -> float:
        if a_preload <= 0 or kv_tokens <= 0:
            self.record("static_overhead_gib", 0.0, "(cannot compute: missing A_preload or pool)")
            return 0.0
        kv_gib = (kv_tokens * kv_bytes) / GIB
        budget = mem_fraction * a_preload  # what the fraction reserved
        components = weights + kv_gib + static_pad
        overhead = budget - components
        if overhead < 0:
            self.warn(
                f"static_overhead computed NEGATIVE ({overhead:.2f} GiB): "
                f"weights+kv+pad ({components:.2f}) exceeds fraction*budget ({budget:.2f}). "
                f"The mem_fraction may be too low, or this profile is pool-dominated "
                f"(set static_overhead_gib = 0)."
            )
        # Include Mamba cache in the overhead (it's reserved into the static budget
        # for hybrid models, not a separate line item in the ledger schema).
        overhead_incl_mamba = overhead  # overhead already reflects total reservation
        self.record(
            "static_overhead_gib",
            round(max(overhead_incl_mamba, 0.0), 2),
            f"({mem_fraction} × {a_preload:.2f}) - ({weights:.2f} + {kv_gib:.2f} + {static_pad})"
            + (f" [includes ~{mamba_cache:.2f} GiB Mamba state]" if mamba_cache > 0 else ""),
        )
        return max(overhead_incl_mamba, 0.0)

    # -- available_gpu_mem: from the max_total_num_tokens line --------------

    def measure_available_gpu_mem(self, lines: list[str]) -> float:
        line = _find(r"max_total_num_tokens=", lines)
        if not line:
            self.record("available_gpu_mem_post_capture", 0.0, "(no max_total_num_tokens line)")
            return 0.0
        avail = _extract_float(line, "available_gpu_mem")
        if avail is None:
            self.record("available_gpu_mem_post_capture", 0.0, line.strip()[:100])
            return 0.0
        self.record("available_gpu_mem_post_capture", round(avail, 2), line.strip()[:120])
        return avail

    # -- context_length: from the max_total_num_tokens line -----------------

    def measure_context_length(self, lines: list[str]) -> int:
        line = _find(r"max_total_num_tokens=", lines)
        if not line:
            return 0
        ctx = _extract_int(line, "context_len")
        return ctx or 0


# ---------------------------------------------------------------------------
# TOML emission
# ---------------------------------------------------------------------------

def emit_profile(
    model_id: str,
    weights: float,
    kv_tokens: int,
    kv_bytes: float,
    static_overhead: float,
    cuda_graph_peak: float,
    static_pad: float,
    request_workspace: float,
    gpu_headroom: float,
    minimum_pool: int,
    context_length: int,
) -> str:
    """Emit a [[profiles]] TOML block ready to append to budget_ledger.toml."""
    lines = [
        "",
        "[[profiles]]",
        f'model_id = "{model_id}"',
    ]
    if context_length:
        lines.append(f"# context_length={context_length}, measured pool={kv_tokens}")
    lines.append("[profiles.budget]")
    lines.append(f"weights_gib            = {weights:.2f}")
    lines.append(f"target_kv_tokens       = {kv_tokens}")
    if minimum_pool > 0:
        lines.append(f"minimum_admissible_pool_tokens = {minimum_pool}   # role contract floor")
    lines.append(f"kv_bytes_per_token     = {kv_bytes:.1f}")
    lines.append(f"static_pad_gib         = {static_pad}")
    if static_overhead > 0:
        lines.append(f"static_overhead_gib    = {static_overhead:.2f}")
    lines.append(f"cuda_graph_peak_gib    = {cuda_graph_peak:.2f}")
    lines.append(f"request_workspace_gib  = {request_workspace}")
    lines.append(f"gpu_headroom_gib       = {gpu_headroom}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse a sglang startup log and emit a budget-ledger profile entry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log", type=Path, default=None,
                    help="sglang startup log file (default: read stdin)")
    ap.add_argument("--model-id", required=True,
                    help="model_id for the profile entry (e.g. qwen36-27b-nvfp4-kv-fp8-e4m3-mtp)")
    ap.add_argument("--mem-fraction", type=float, required=True,
                    help="the mem_fraction_static used during this launch (e.g. 0.60)")
    ap.add_argument("--context-length", type=int, default=None,
                    help="context_length (for the entry comment; auto-detected if omitted)")
    ap.add_argument("--minimum-pool", type=int, default=0,
                    help="minimum admissible pool tokens (role contract floor; 0 = unenforced)")
    ap.add_argument("--static-pad", type=float, default=0.5,
                    help="static_pad_gib (default: 0.5)")
    ap.add_argument("--request-workspace", type=float, default=2.0,
                    help="request_workspace_gib (default: 2.0)")
    ap.add_argument("--gpu-headroom", type=float, default=1.0,
                    help="gpu_headroom_gib (default: 1.0)")
    args = ap.parse_args()

    # Read the log
    if args.log:
        if not args.log.is_file():
            print(f"REFUSE: log file not found: {args.log}", file=sys.stderr)
            return 75
        text = args.log.read_text(errors="replace")
    else:
        text = sys.stdin.read()
    lines = text.splitlines()

    m = Measurement()

    # Extract all fields
    weights = m.measure_weights(lines)
    kv_tokens, kv_bytes = m.measure_kv_pool(lines)
    mamba_cache = m.measure_mamba_cache(lines)
    graph_peak = m.measure_graph_peak(lines)
    a_preload = m.measure_a_preload(lines)
    avail_post = m.measure_available_gpu_mem(lines)
    ctx = args.context_length or m.measure_context_length(lines)
    static_overhead = m.compute_static_overhead(
        args.mem_fraction, a_preload, weights, kv_tokens, kv_bytes,
        args.static_pad, mamba_cache,
    )

    # Print the derivation trace to stderr (transparency)
    print("=" * 60, file=sys.stderr)
    print(f"DERIVATION TRACE for {args.model_id}", file=sys.stderr)
    print(f"(mem_fraction_static={args.mem_fraction})", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for label, value, source in m.trace:
        if label == "WARNING":
            print(f"  ⚠ {value}", file=sys.stderr)
        else:
            print(f"  {label:30s} = {value}", file=sys.stderr)
            if source:
                print(f"    ← {source}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    if avail_post > 0:
        print(f"  (post-capture VRAM still free: {avail_post:.2f} GiB — if large, the", file=sys.stderr)
        print(f"   planner may allocate a larger pool at launch)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Emit the TOML profile to stdout
    toml = emit_profile(
        model_id=args.model_id,
        weights=weights,
        kv_tokens=kv_tokens,
        kv_bytes=kv_bytes,
        static_overhead=static_overhead,
        cuda_graph_peak=graph_peak,
        static_pad=args.static_pad,
        request_workspace=args.request_workspace,
        gpu_headroom=args.gpu_headroom,
        minimum_pool=args.minimum_pool,
        context_length=ctx,
    )
    print(toml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
