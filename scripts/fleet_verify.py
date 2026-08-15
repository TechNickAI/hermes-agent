#!/usr/bin/env python3
"""Verify the fleet patch is actually LIVE in each running gateway process.

Why this exists: on an editable git install the checkout is the deployed code,
so `git rev-parse HEAD` tells you what is ON DISK. It does NOT tell you what a
long-running gateway process actually IMPORTED — a process started before the
deploy keeps serving the old module from memory. Disk state and process state
diverge silently, and the symptom (bubbles still piling up) looks exactly like
"the patch didn't work" rather than "the gateway wasn't restarted".

So this checks three independent layers and reports them separately:

  1. DISK      — is the checkout at the expected commit, and does the patched
                 source actually contain the marker symbol?
  2. PROCESS   — is each gateway PID younger than the deploy commit's file
                 mtime? An older PID cannot have imported the new code.
  3. BEHAVIOUR — is cleanup_progress actually enabled for the platform? The
                 patch is inert without it, so a "deployed" host with the flag
                 off will still strand commentary.

Exit codes: 0 all green, 1 precondition/error, 2 at least one layer failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MARKER = "_notify_transient_message"
MARKER_FILE = "gateway/stream_consumer.py"


def sh(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - report, never raise
        return 1, f"{type(exc).__name__}: {exc}"


def check_disk(checkout: Path, expect_sha: str | None, base_tag: str | None = None) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    rc, head = sh(["git", "rev-parse", "HEAD"], cwd=str(checkout))
    if rc != 0:
        return [("disk.checkout", False, f"not a git checkout: {head}")]
    if expect_sha:
        ok = head.startswith(expect_sha) or expect_sha.startswith(head[:10])
        out.append(("disk.commit", ok, f"HEAD={head[:10]} expected={expect_sha[:10]}"))
    else:
        out.append(("disk.commit", True, f"HEAD={head[:10]}"))

    # The commit sha alone is not proof the patch is present: a rebase, a bad
    # cherry-pick, or a partial checkout can leave the sha right and the code
    # wrong. Assert the actual symbol exists in the actual file.
    src = checkout / MARKER_FILE
    if not src.exists():
        out.append(("disk.marker", False, f"missing {MARKER_FILE}"))
    else:
        present = MARKER in src.read_text(errors="replace")
        out.append(("disk.marker", present,
                    f"{MARKER} {'present' if present else 'ABSENT'} in {MARKER_FILE}"))

    rc, dirty = sh(["git", "status", "--porcelain"], cwd=str(checkout))
    n = len([x for x in dirty.splitlines() if x.strip()]) if rc == 0 else -1
    out.append(("disk.clean", n == 0, f"{n} modified/untracked file(s)"))

    # Prove the base is the RELEASE TAG, not upstream main or a stray commit.
    # Without this the branch name is the only evidence of what we are based on,
    # and a branch name is not a fact. `--is-ancestor` is a real lineage check.
    if base_tag:
        rc, _ = sh(["git", "merge-base", "--is-ancestor", base_tag, "HEAD"], cwd=str(checkout))
        if rc == 0:
            rc2, extra = sh(["git", "log", "--oneline", f"{base_tag}..HEAD"], cwd=str(checkout))
            n_patches = len([x for x in extra.splitlines() if x.strip()]) if rc2 == 0 else -1
            out.append(("disk.base", True, f"descends from {base_tag} + {n_patches} local patch(es)"))
        else:
            out.append(("disk.base", False,
                        f"HEAD does NOT descend from {base_tag} (tag missing locally, or wrong base)"))
    return out


def gateway_pids() -> list[tuple[int, str, float]]:
    """Return (pid, profile, start_epoch) for each running hermes gateway."""
    rc, out = sh(["ps", "-eo", "pid,lstart,command"])
    if rc != 0:
        return []
    found: list[tuple[int, str, float]] = []
    for line in out.splitlines():
        if "hermes_cli.main" not in line or "gateway" not in line:
            continue
        if "grep" in line:
            continue
        parts = line.split(None, 1)
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        profile = "default"
        toks = line.split()
        for flag in ("--profile", "-p"):
            if flag in toks:
                i = toks.index(flag)
                if i + 1 < len(toks):
                    profile = toks[i + 1]
        start = 0.0
        try:
            # ps lstart is a fixed 5-field date starting at token 1
            start = time.mktime(time.strptime(" ".join(toks[1:6])))
        except Exception:  # noqa: BLE001
            start = 0.0
        found.append((pid, profile, start))
    return found


def check_processes(checkout: Path) -> list[tuple[str, bool, str]]:
    src = checkout / MARKER_FILE
    if not src.exists():
        return [("proc.source", False, f"missing {MARKER_FILE}")]
    code_mtime = src.stat().st_mtime

    pids = gateway_pids()
    if not pids:
        return [("proc.running", False, "no hermes gateway processes found")]

    results: list[tuple[str, bool, str]] = []
    for pid, profile, start in pids:
        if start == 0.0:
            results.append((f"proc.{profile}", False,
                            f"pid={pid} could not determine start time"))
            continue
        fresh = start >= code_mtime
        age_min = (code_mtime - start) / 60.0
        results.append((
            f"proc.{profile}", fresh,
            f"pid={pid} started {'AFTER' if fresh else 'BEFORE'} code mtime"
            + ("" if fresh else f" by {age_min:.0f} min - RESTART REQUIRED"),
        ))
    return results


def check_behaviour(hermes_home: Path, platform: str) -> list[tuple[str, bool, str]]:
    """cleanup_progress must be ON or the patch does nothing observable."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return [("cfg.yaml", False, "pyyaml unavailable; cannot read config")]

    out: list[tuple[str, bool, str]] = []
    configs = sorted(hermes_home.glob("profiles/*/config.yaml")) + [hermes_home / "config.yaml"]
    for cfg_path in configs:
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            out.append((f"cfg.{cfg_path.parent.name}", False, f"unreadable: {exc}"))
            continue
        disp = cfg.get("display") or {}
        plats = disp.get("platforms") or {}
        per = (plats.get(platform) or {}) if isinstance(plats, dict) else {}
        val = per.get("cleanup_progress", disp.get("cleanup_progress"))
        interim = per.get("interim_assistant_messages",
                          disp.get("interim_assistant_messages"))
        enabled = str(val).lower() in ("true", "1", "yes", "on")
        out.append((
            f"cfg.{cfg_path.parent.name}", enabled,
            f"cleanup_progress={val} interim_assistant_messages={interim}",
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", default=os.path.expanduser("~/.hermes/hermes-agent"))
    ap.add_argument("--hermes-home", default=os.path.expanduser("~/.hermes"))
    ap.add_argument("--expect-sha", default=None)
    ap.add_argument("--base-tag", default="v2026.8.13",
                    help="release tag the fleet branch must descend from")
    ap.add_argument("--platform", default="telegram")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    checkout = Path(args.checkout)
    if not checkout.exists():
        print(f"FAIL: no checkout at {checkout} (host may use a non-git install)")
        return 1

    rows: list[tuple[str, bool, str]] = []
    rows += check_disk(checkout, args.expect_sha, args.base_tag)
    rows += check_processes(checkout)
    rows += check_behaviour(Path(args.hermes_home), args.platform)

    if args.json:
        print(json.dumps([{"check": c, "ok": o, "detail": d} for c, o, d in rows], indent=2))
    else:
        host = os.uname().nodename
        print(f"=== {host} ===")
        for check, ok, detail in rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {check:24} {detail}")

    return 0 if all(o for _, o, _ in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
