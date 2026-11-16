# Handover — issue #266

`vibe-handover version=1`

An earlier run working this issue was interrupted before it finished. The worker
wrote this note — not the agent — so any host and any tooling can pick the work
up from this branch. It carries nothing tied to one host, one conversation or
one agent provider.

## This attempt

- 2026-11-16T21:11:25Z — execute hit the Claude subscription usage limit after
  291s; 0 uncommitted file(s) preserved; 26 commit(s) added to the branch
- Branch: `issue-266-experiment-benchmark-budget-aware-routing-of-local-redo-2`
- Wind-down notice: not delivered — the interruption arrived without warning

## What was done

Commits this run added to the branch, newest first:

- wip: handover note for the interrupted run on issue #266 Vibe-Coder-Run-Id:
  vibe-mwjqn4yr-b1e2ce
- Record this run's work and the reviewers' verdicts in the #266 PR summary
- Correct the claim that later commits leave the bench code unchanged (#266)
- Document landscape coverage and the compute ledger (#266)
- Regenerate landscape evidence with coverage at cf2ff11 (#266)
- Record landscape failures and coverage instead of stopping (#266)
- Document the saturated AMM reference result (#266)
- Run the AMM reference on saturated-hard-tanh (#266)
- Rewrite the #266 PR summary for the re-landed work and the landscape cost fix
- Document the landscape preparation count (#266)
- Regenerate landscape evidence with the preparation count (#266)
- Count the selection scan in the landscape command's evaluations (#266)
- Record the landscape work in the #266 PR summary and tighten its plateau test
- Commit landscape evidence and saturated hard-tanh arm runs (#266)
- Add a saturated hard-tanh family and a two-sided landscape diagnosis (#266)
- Rewrite the PR summary for the re-landed #266 follow-up
- Scope the "later commits" note to the benchmark code (#266)
- Pin router-era AMM row compatibility with a fixture test (#266)
- Merge the closed PR #275 weight bench follow-up onto Develop (#266)
- Add the PR summary for the #266 follow-up
- WIP checkpoint: periodic agent progress snapshot
- Weight bench: regenerate evidence at 0.1.49 and record its reproduction (#266)
- Bump neat_ai_lamarck to 0.1.49 for the weight bench changes (#266)
- Weight bench: regenerate evidence from head and pin it with tests (#266)
- WIP checkpoint: periodic agent progress snapshot
- WIP checkpoint: periodic agent progress snapshot

The working tree was clean at the interruption — the work above is already
committed on this branch.

## What remains

The run was interrupted after 291s, so it never reported completion: whatever
the issue still asks for beyond the changes above is outstanding.

Diff `issue-266-experiment-benchmark-budget-aware-routing-of-local-redo-2`
against its base branch to see the 26 commit(s) and 0 preserved file(s) named
above, continue from them, and do not revert them unless they are wrong.

The closing deliverables are outstanding too unless the list above names them:
completion reads `docs/archive/pr-summaries/pr-summary-266.md` — with its
`## Acceptance Criteria` closure block when the issue states criteria — and a
run that finishes without it fails at the gate however complete the code is.

## Known blockers

None were recorded. The run was stopped by the interruption named above, not by
a blocker it reported.

## Previous attempts

Earlier runs on this issue were interrupted too:

- 2026-11-16T21:11:20Z — execute hit the Claude subscription usage limit after
  286s; 0 uncommitted file(s) preserved; 25 commit(s) added to the branch
