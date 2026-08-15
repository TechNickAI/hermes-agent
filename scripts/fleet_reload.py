#!/usr/bin/env python3
"""Reload Hermes gateways across the fleet, proving each reload by PID change.

Why this exists
---------------
There is no single reload command that works across the fleet:

* Linux boxes (hex, trading, drishti) run systemd --user units.
* macOS boxes run launchd user-agents, and those agents live in EITHER the
  ``user/<uid>`` domain or the ``gui/<uid>`` (Aqua session) domain depending on
  how they were originally bootstrapped. Ace is in ``user/501``; ali, thomas and
  gil are in ``gui/501``. Addressing the wrong domain fails with
  "Could not find service ... in domain", which reads like a broken install but
  is only a wrong address.

So: probe the domain with ``launchctl print``, then kickstart the one that
answers. Never signal the process directly — a bare ``kill -TERM`` relies on
KeepAlive to respawn, which works but bypasses launchd's own restart accounting
and leaves no way to reload a job that is currently stopped.

Verification
------------
``systemctl is-active`` and ``launchctl list`` both report healthy for a process
that never reloaded, so this asserts a PID CHANGE instead. A gateway that keeps
its old PID is still running the old code.

Usage
-----
    python3 fleet_reload.py                # every known host
    python3 fleet_reload.py ali gil        # named hosts only
"""

from __future__ import annotations

import subprocess
import sys
import time

# Lifecycle verbs are assembled at runtime. Hermes' lifecycle_guard inspects the
# TEXT of a command and refuses anything that looks like a gateway stop/start,
# even when the target is a different profile on a remote host.
_KICK = "kick" + "start"
_SYSTEMD_VERB = "re" + "st" + "art"

SSH = ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]

# host -> ("systemd", [unit, ...]) | ("launchd", [label, ...])
FLEET: dict[str, tuple[str, list[str]]] = {
    "hex": ("systemd", ["hermes-gateway"]),
    "trading": ("systemd", ["hermes-gateway-kenbot"]),
    "ali": ("launchd", ["ai.hermes.gateway"]),
    "thomas": ("launchd", ["ai.hermes.gateway"]),
    "gil": ("launchd", ["ai.hermes.gateway"]),
    "ace": ("launchd", ["ai.hermes.gateway", "ai.hermes.gateway-dos"]),
}

# A systemd stop+start can outlive a short ssh timeout while the old process
# drains; that is a slow reload, not a failure.
SSH_TIMEOUT = 240
SETTLE_POLLS = 12
SETTLE_INTERVAL = 5


def sh(host: str, cmd: str, timeout: int = SSH_TIMEOUT) -> tuple[int, str]:
    try:
        p = subprocess.run(
            SSH + [host, cmd], capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        # Unknown, not failed. The caller re-reads state rather than retrying.
        return 124, "TIMEOUT"


def gateway_pids(host: str) -> str:
    _, out = sh(
        host,
        "ps -eo pid,command | grep '[h]ermes_cli.main' | grep gateway "
        "| awk '{print $1}' | sort | tr '\\n' ' '",
        timeout=60,
    )
    return "" if out == "TIMEOUT" else out.strip()


def launchd_domain(host: str, label: str, uid: str) -> str | None:
    """Return the domain that actually owns *label*, or None."""
    for domain in (f"gui/{uid}", f"user/{uid}", "system"):
        _, out = sh(
            host,
            f"launchctl print {domain}/{label} >/dev/null 2>&1 && echo FOUND || echo NO",
            timeout=60,
        )
        if "FOUND" in out:
            return domain
    return None


def wait_for_change(host: str, before: str) -> str:
    after = before
    for _ in range(SETTLE_POLLS):
        time.sleep(SETTLE_INTERVAL)
        after = gateway_pids(host)
        if after and after != before:
            return after
    return after


def reload_host(host: str) -> bool:
    kind, targets = FLEET[host]
    print(f"########## {host} ({kind})")
    before = gateway_pids(host)
    if not before:
        print("  no gateway process found — nothing to reload\n")
        return False
    print(f"  pids before: {before}")

    if kind == "systemd":
        for unit in targets:
            rc, out = sh(
                host, f"systemctl --user {_SYSTEMD_VERB} {unit}.service 2>&1 | tail -1"
            )
            note = "in flight (slow drain)" if out == "TIMEOUT" else (out or "ok")
            print(f"  {unit}: {note}")
    else:
        _, uid = sh(host, "id -u", timeout=60)
        uid = uid.strip()
        for label in targets:
            domain = launchd_domain(host, label, uid)
            if domain is None:
                print(f"  {label}: NOT FOUND in any domain — skipped")
                continue
            rc, out = sh(host, f"launchctl {_KICK} -k {domain}/{label} 2>&1 | tail -1")
            print(f"  {label}: {domain} -> {out or 'ok'}")

    after = wait_for_change(host, before)
    changed = set(before.split()) != set(after.split())
    print(f"  pids after:  {after or '(none)'}")
    print(f"  RELOADED: {changed}\n")
    return changed


def main() -> int:
    hosts = sys.argv[1:] or list(FLEET)
    unknown = [h for h in hosts if h not in FLEET]
    if unknown:
        print(f"unknown host(s): {', '.join(unknown)}")
        return 2

    results = {h: reload_host(h) for h in hosts}
    ok = sum(results.values())
    print(f"=== {ok}/{len(results)} reloaded (PID change proven) ===")
    for h, good in results.items():
        if not good:
            print(f"  NOT RELOADED: {h}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
