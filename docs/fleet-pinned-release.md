# Fleet pinned-release model for hermes-agent

How the fleet runs hermes-agent: a **release tag** plus a small set of
**reviewed local patches**, on a branch in our own fork. Not upstream `main`,
not an unpinned checkout, not a vendored third-party plugin.

---

## Why this exists

Two failures pushed us here.

**Drift with no owner.** On 2026-08-07 three fixes were committed directly to
`main` in the live checkout at `~/.hermes/hermes-agent` and never pushed
anywhere. They existed in exactly one place on earth, on a box that also *ran*
them, so four gateways were serving patched code that no one could review, no
one could reproduce, and a single `rm -rf` would have erased. Committing to
`main` on a tracking checkout also guarantees permanent divergence from
upstream.

**Chasing `main`.** The checkout was 21,966 commits behind `origin/main` while
being only *one release* behind the latest tag. Tracking `main` means rebasing
against a repo that merges ~1,400 PRs per release. Tracking releases means a
scheduled, bounded upgrade.

The model below fixes both: every patch is pushed, reviewed, and has a defined
removal condition; the base moves on our schedule, not upstream's.

---

## The model

```
NousResearch/hermes-agent  tag v2026.8.13  (release v0.20.1)
              |
              +-- TechNickAI/hermes-agent  branch fleet/v2026.8.13
                      |
                      +-- fix(gateway): track interim commentary bubbles
                      |
                      +-- (future reviewed patches)
                                |
                                v
                     12 gateways / 6 hosts, editable git installs
```

**Key property:** fleet hosts run an *editable* install — the checkout IS the
deployed code. There is no build step and no copy step. Deploying is
`git reset` to the pinned branch; the code only becomes live when the gateway
process **restarts and re-imports it**.

That last sentence is the source of nearly every silent failure in this system.
Disk state and process state diverge quietly, and the symptom looks exactly
like "the patch didn't work."

---

## What is on the branch

| Patch | Why it is local | Removal condition |
|---|---|---|
| `fix(gateway): track interim commentary bubbles for cleanup_progress` | Upstream never registered commentary message ids, so `cleanup_progress` stranded every interim message. Open upstream issues #4882 / #47858; PR #22613 contains a fix but has been conflicted and unreviewed since 2026-07-19. | Upstream merges an equivalent fix — then drop the patch and rebase onto the next release. |

**Every patch on this branch must have a removal condition.** A patch with no
exit path is how a fork becomes a permanent maintenance tax.

---

## Deploy

```bash
# on each host
scripts/fleet_deploy.sh --branch fleet/v2026.8.13
# then restart the gateway (see host-specific restart below)
scripts/fleet_verify.py --expect-sha <sha>
```

`fleet_deploy.sh` **refuses** to run when:

- the working tree is dirty (someone hand-edited production — inspect, do not
  discard)
- there are local commits not present on the pinned branch (the exact failure
  that created this document)

Both are hard failures, not warnings. Destroying a human's emergency 3am fix to
satisfy a deploy script is worse than a failed deploy.

---

## Verify — three independent layers

`fleet_verify.py` checks three things separately, because passing one proves
nothing about the others:

**1. DISK** — is the checkout at the expected commit, *and does the source
actually contain the marker symbol*? The sha alone is insufficient: a bad
cherry-pick or partial checkout can leave the sha right and the code wrong.

**2. PROCESS** — is each gateway PID younger than the patched file's mtime? A
process started before the deploy is still serving the old module from memory.
**This is the check that catches "deployed but not live."**

**3. BEHAVIOUR** — is `cleanup_progress` actually enabled for the platform? The
patch is inert without it. A host can be perfectly deployed and still strand
commentary because the flag is off.

Exit codes: `0` all green, `1` precondition failure, `2` at least one check
failed.

The verifier was validated in **both directions** before use: it returns exit 2
against an unpatched checkout (correctly reporting `disk.marker ABSENT`) and
passes against the patched tree. A check that cannot fail is decoration.

---

## Rollback

```bash
scripts/fleet_deploy.sh --branch fleet/v2026.8.13 --rollback <previous-sha>
# restart gateway, then verify
```

Rollback target for the current branch is the stock release commit
`f80f453ae0679347e38abc917c7f94f717bf96c5` (v0.20.1 unmodified). Rolling back
to stock is always safe: the patch is purely additive and gated behind
`cleanup_progress`, so stock code simply stops deleting commentary bubbles.

---

## Upgrading to the next release

1. Sync fork `main` from upstream (`gh api repos/TechNickAI/hermes-agent/merge-upstream -f branch=main`).
2. Create `fleet/<new-tag>` from the **new release tag**, not from `main`.
3. Cherry-pick each still-needed patch. **Check each removal condition first** —
   if upstream merged it, drop it instead of carrying it.
4. Run the patch's tests plus the surrounding suites.
5. Multi-review if any patch changed materially during the rebase.
6. Stage the rollout (below).

`gh repo sync` silently no-ops when the fork has diverged; use the
`merge-upstream` API call and **verify the resulting sha changed**.

---

## Staged rollout order

Blast radius first, not convenience. This changes message *deletion* behaviour
in chats belonging to real people.

| Stage | Targets | Gate before proceeding |
|---|---|---|
| 1 | `bosun` only (operator's own agent) | A real multi-tool turn shows commentary bubbles removed and the final answer intact. Live observation, not a test run. |
| 2 | Remaining operator-owned profiles on the Mac Studio (`cora`, `sterling`, `argus`) | 24h with no lost-message reports and no gateway errors. |
| 3 | Operator-controlled remote hosts | Same-day verify pass on each. |
| 4 | Client-owned agents | Explicit owner notice first. These are other people's chats. |

Hosts without a git checkout cannot take a source patch at all and must be
reported as **excluded-by-install-method**, not as pending.

---

## Known trap: this cannot be a plugin

Worth recording so it is not re-litigated. The commentary message id exists only
inside `_send_commentary`'s local scope and is discarded immediately. No hook
event carries it (`gateway/hooks.py` exposes lifecycle events only —
`agent:start`, `agent:step`, `agent:end`, `session:*`, `command:*`). Reaching it
from a plugin requires monkeypatching gateway internals, which is what
`hermes-progress-tail` does across 13 patch families and 16k lines. That was
evaluated and rejected: unlicensed (no LICENSE file, so no legal right to
vendor), requires Python 3.12 (most fleet hosts are 3.11), and couples us to
`gateway/run.py` internals that change every release.

The fix belongs in core, which is why it is also going upstream.
