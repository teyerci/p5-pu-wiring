# Working rules for this repository

## Every commit is reviewed first (Fable / Codex, alternating)

Before each commit, the staged diff is reviewed by an independent reviewer. The reviewers alternate:

- **Assigned reviewer:** the one *not* assigned at the previous commit (Fable first). `python3 scripts/review_gate.py status` prints who is next.
- **Backup:** if the assigned reviewer cannot run (unavailable, error, timeout), the other reviews instead and the approval is recorded with `--backup`.
- **Fable** runs as a subagent with model `fable`. **Codex** runs through the `codex:codex-rescue` agent (default model). Both review read-only.

Procedure:

1. Stage the changes in a command of their own (`git add …`).
2. The assigned reviewer reviews `git diff --cached` (skip notebook output blobs) for correctness, leakage, and claims not supported by the numbers.
3. Fix or discuss blocking findings and restage. Restaging changes the diff, so the review applies to the final staged diff.
4. Record the pass: `python3 scripts/review_gate.py approve <reviewer> [--backup] --note "<one-line verdict>"`.
5. Commit on its own: `[cd DIR &&] git [-C DIR] commit [message options]`. Do not use `-a`, `-i`, `-o`, `-p` or pathspecs, and do not chain other commands. For a multi-line message use `-F - <<'EOF'` with the message as the heredoc body; the `-m "$(cat <<'EOF' …)"` form is refused.

A PreToolUse hook (`.claude/settings.json`) blocks any `git commit` in this repository whose staged diff does not match the approved hash. The review log is kept in `.git/review/log.tsv`, outside version control.

## Data and conventions

Follow the course's Week 2 Data Conventions: `config.py` and `eda_shared/` are shared files and are never edited here. Project deviations are stated in the README and the methods section.
