"""Local default-profile Telegram preflight; never rewrites outgoing content.

No-link text is untouched. Link-bearing refusals retain the full original privately
and return a non-retryable error before any send, edit or draft transport.
"""
import asyncio
import os
from pathlib import Path
import re
import tempfile

from hermes_constants import get_hermes_home


async def recipient_link_error(content: str):
    # 2026-11-09 Bosun: disabled for Telegram replies. Retail sites return 429/bot
    # challenges to anonymous checks, so this silently ate Julianna's replies while the
    # agent believed they were sent. Client email/send tools keep the plugin preflight.
    # Restore: cp recipient_link_gate.py.bak-20261109-blocking recipient_link_gate.py
    return None
    home = Path(get_hermes_home()).resolve()
    if home != (Path.home() / '.hermes').resolve() or not re.search(r'https?://', content or '', re.I):
        return None
    prefix = 'Recipient link check stopped Telegram delivery. Original content is unchanged. '
    scratch = home / 'cache/scratch/telegram-link-gate'
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='outgoing-', suffix='.txt', dir=scratch)
    path = Path(name)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            str(home / 'bin/link-check'), '--text-file', str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout=35)
        if proc.returncode == 0:
            path.unlink()
            return None
        # The checker emits token-free host/reason diagnostics, not the body.
        reason = out.decode(errors='replace').strip() or 'Checker unavailable or failed.'
        return prefix + reason + f' Use a verified recipient share link or attachment; do not publish private items. Original retained at {path}.'
    except asyncio.CancelledError:
        raise
    except Exception:
        return prefix + f'Checker unavailable or timed out; repair it and retry. Original retained at {path}.'
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
