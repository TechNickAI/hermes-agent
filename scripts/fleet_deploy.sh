#!/usr/bin/env bash
# Deploy the fleet's pinned hermes-agent release+patch branch to one host.
#
# Model: every fleet host runs an EDITABLE git checkout at ~/.hermes/hermes-agent,
# so the checkout IS the deployed code. Deploying = fetching the pinned branch,
# hard-resetting to it, and restarting the gateway(s) so the new code is imported.
#
# The branch is built on a RELEASE TAG (not upstream main) plus a small set of
# reviewed local patches. See docs/fleet-pinned-release.md.
#
# Usage:
#   fleet_deploy.sh --branch fleet/v2026.8.13 [--check] [--rollback <sha>]
#
#   --check     verify only; make no changes, exit 0 if already correct
#   --rollback  reset to an explicit sha instead of the branch head
#
# Exit codes: 0 ok, 1 precondition failed, 2 verification failed after deploy.

set -euo pipefail

FORK_URL="https://github.com/TechNickAI/hermes-agent.git"
REMOTE_NAME="fleet"
CHECKOUT="${HERMES_CHECKOUT:-$HOME/.hermes/hermes-agent}"
BRANCH=""
CHECK_ONLY=0
ROLLBACK_SHA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --branch)   BRANCH="$2"; shift 2 ;;
    --check)    CHECK_ONLY=1; shift ;;
    --rollback) ROLLBACK_SHA="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[ -n "$BRANCH" ] || { echo "FAIL: --branch is required" >&2; exit 1; }

log() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# ---------------------------------------------------------------- preconditions
[ -d "$CHECKOUT/.git" ] || { echo "FAIL: no git checkout at $CHECKOUT" >&2; exit 1; }
cd "$CHECKOUT"

# A dirty tree means someone hand-edited production. Never silently discard it:
# a hard reset would erase a fix a human made under pressure at 3am.
DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
if [ "$DIRTY" != "0" ]; then
  echo "FAIL: working tree has $DIRTY modified/untracked file(s)." >&2
  echo "      Refusing to reset. Inspect and stash/commit first:" >&2
  git status --short >&2
  exit 1
fi

# Local commits that exist nowhere else are the failure mode that started this
# whole effort. Refuse rather than destroy them.
if git rev-parse --verify -q "$REMOTE_NAME/$BRANCH" >/dev/null 2>&1; then
  UNPUSHED="$(git rev-list --count "$REMOTE_NAME/$BRANCH..HEAD" 2>/dev/null || echo 0)"
  if [ "${UNPUSHED:-0}" != "0" ] && [ "$UNPUSHED" != "" ]; then
    CURRENT_BRANCH="$(git branch --show-current || echo detached)"
    if [ "$CURRENT_BRANCH" = "$BRANCH" ]; then
      echo "FAIL: $UNPUSHED local commit(s) not on $REMOTE_NAME/$BRANCH." >&2
      git log --oneline "$REMOTE_NAME/$BRANCH..HEAD" >&2
      exit 1
    fi
  fi
fi

# -------------------------------------------------------------------- fetch
if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  log "adding remote $REMOTE_NAME -> $FORK_URL"
  [ "$CHECK_ONLY" = "1" ] || git remote add "$REMOTE_NAME" "$FORK_URL"
fi

if [ "$CHECK_ONLY" = "0" ]; then
  log "fetching $REMOTE_NAME/$BRANCH"
  git fetch --quiet "$REMOTE_NAME" "$BRANCH"
fi

TARGET="${ROLLBACK_SHA:-$(git rev-parse "$REMOTE_NAME/$BRANCH" 2>/dev/null || echo '')}"
[ -n "$TARGET" ] || { echo "FAIL: cannot resolve $REMOTE_NAME/$BRANCH" >&2; exit 1; }

CURRENT="$(git rev-parse HEAD)"
log "current=${CURRENT:0:10}  target=${TARGET:0:10}"

if [ "$CHECK_ONLY" = "1" ]; then
  if [ "$CURRENT" = "$TARGET" ]; then log "OK: at target"; exit 0; fi
  log "DRIFT: not at target"; exit 2
fi

# ------------------------------------------------------------------- deploy
if [ "$CURRENT" != "$TARGET" ]; then
  log "checking out $TARGET"
  git checkout --quiet -B "$BRANCH" "$TARGET"
else
  log "already at target; continuing to dependency + verify steps"
fi

# An editable install still needs re-sync when dependency metadata changes
# between releases. Cheap when it is a no-op, and skipping it produces the
# classic "code deployed, import fails" outage.
if [ -x "$CHECKOUT/venv/bin/python" ]; then
  if ! git diff --quiet "${CURRENT}" "${TARGET}" -- pyproject.toml setup.py 2>/dev/null; then
    log "dependency metadata changed; reinstalling editable package"
    "$CHECKOUT/venv/bin/python" -m pip install -q -e "$CHECKOUT" 2>&1 | tail -3 || {
      echo "FAIL: editable reinstall failed" >&2; exit 2; }
  fi
fi

log "deployed. RESTART REQUIRED for running gateways to import new code."
exit 0
