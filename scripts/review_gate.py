#!/usr/bin/env python3
"""Pre-commit review gate: every commit's staged diff is reviewed first.

Reviewers alternate between Fable and Codex, once per commit. A re-review of
the same commit (fix, restage, review again) stays with the reviewer assigned
to it. If the assigned reviewer cannot run (unavailable, error, timeout), the
other one reviews as backup and the log records it. State lives in
.git/review/ (never committed):

    approved   sha256 of the staged diff that passed review
    log.tsv    date, staged-diff hash, assigned, reviewer, role, verdict, note, HEAD at approval

Usage:
    review_gate.py status                  next assigned reviewer, staged hash, approval
    review_gate.py approve REVIEWER [--backup] [--note TEXT]
                                           record a passed review of the current staged diff
    review_gate.py hook                    Claude Code PreToolUse hook (reads JSON on stdin)

Stage everything first, in its own command; review; approve; then commit on
its own as `[cd DIR &&] git [-C DIR] commit [message options]`, so the diff
that was reviewed is exactly the diff that is committed. The hook refuses any
other shape (pathspecs, -a, -i, -o, -p, chained commands) inside this repository.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REVIEWERS = ("fable", "codex")
FIRST = "fable"


def repo_root(start: Path) -> Path | None:
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return Path(out)
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None


def staged_hash(root: Path) -> str:
    diff = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--binary"],
                          capture_output=True, check=True).stdout
    return hashlib.sha256(diff).hexdigest() if diff else ""


def state_dir(root: Path) -> Path:
    git_dir = subprocess.run(["git", "-C", str(root), "rev-parse", "--absolute-git-dir"],
                             capture_output=True, text=True, check=True).stdout.strip()
    d = Path(git_dir) / "review"
    d.mkdir(exist_ok=True)
    return d


def head_commit(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "none"


def log_rows(root: Path) -> list[list[str]]:
    log = state_dir(root) / "log.tsv"
    if not log.exists():
        return []
    return [l.split("\t") for l in log.read_text().splitlines() if l.strip()]


def next_assigned(root: Path) -> str:
    """Alternate per commit: while HEAD has not moved since the last approval,
    the same reviewer stays assigned."""
    rows = log_rows(root)
    if not rows:
        return FIRST
    last = rows[-1]
    last_assigned, last_head = last[2], (last[7] if len(last) > 7 else "")
    if last_head == head_commit(root):
        return last_assigned
    return REVIEWERS[1 - REVIEWERS.index(last_assigned)]


def cmd_status(root: Path) -> int:
    h = staged_hash(root)
    f = state_dir(root) / "approved"
    approved = f.read_text().strip() if f.exists() else ""
    nxt = next_assigned(root)
    print(f"next assigned reviewer: {nxt} (backup: {REVIEWERS[1 - REVIEWERS.index(nxt)]})")
    print(f"staged diff: {h[:12] or '(nothing staged)'}")
    print(f"approved: {'yes' if h and h == approved else 'no'}")
    return 0


def cmd_approve(root: Path, args: list[str]) -> int:
    if not args or args[0] not in REVIEWERS:
        print(f"approve needs a reviewer: one of {REVIEWERS}", file=sys.stderr)
        return 1
    reviewer, backup = args[0], "--backup" in args
    note = ""
    if "--note" in args:
        i = args.index("--note")
        if i + 1 >= len(args):
            print("--note needs a value", file=sys.stderr)
            return 1
        note = args[i + 1]
    h = staged_hash(root)
    if not h:
        print("nothing staged", file=sys.stderr)
        return 1
    assigned = next_assigned(root)
    if reviewer != assigned and not backup:
        print(f"{assigned} is the assigned reviewer; pass --backup if {reviewer} reviewed because {assigned} could not",
              file=sys.stderr)
        return 1
    d = state_dir(root)
    (d / "approved").write_text(h + "\n")
    row = [dt.datetime.now().isoformat(timespec="seconds"), h[:12], assigned, reviewer,
           "backup" if backup else "primary", "approved", note.replace("\t", " ").replace("\n", " "),
           head_commit(root)]
    with open(d / "log.tsv", "a") as f:
        f.write("\t".join(row) + "\n")
    print(f"approved {h[:12]} (reviewer {reviewer}, {'backup' if backup else 'primary'})")
    return 0


# Any mention of a git commit: path-qualified git, global options, inside quotes or eval.
# Deliberately broad; a false positive only asks for the simple form below.
GIT_COMMIT = re.compile(r"(?:^|[;&|(\s\"'`=])(?:\S*/)?git(?:\s+-[cC]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+commit(?![\w-])")

# The only commit shape let through: `[cd DIR &&] git [-C DIR] commit [OPTIONS]`,
# where OPTIONS never change which content is committed.
VALUE_OPTS = {"-m", "--message", "-F", "--file", "--author", "--date", "-C", "--reuse-message",
              "-c", "--reedit-message", "--fixup", "--squash", "-t", "--template", "--cleanup"}
FLAG_OPTS = {"-q", "--quiet", "-v", "--verbose", "-s", "--signoff", "-n", "--no-verify",
             "--allow-empty", "--allow-empty-message", "--amend", "--no-edit", "-e", "--edit",
             "--no-gpg-sign", "--status", "--no-status"}
VALUE_PREFIXES = ("--message=", "--file=", "--author=", "--date=", "--reuse-message=",
                  "--reedit-message=", "--fixup=", "--squash=", "--template=", "--cleanup=")


def split_command(command: str) -> list[str] | None:
    """Tokens of the first line. A heredoc is allowed only as the commit message:
    `... <<'DELIM'` must end line 1, and nothing but blank lines may follow the
    closing DELIM line. Anything else returns None (not the simple form)."""
    lines = command.strip("\n").split("\n")
    try:
        lex = shlex.shlex(lines[0], posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return None
    if "<<" not in tokens:
        return tokens if len(lines) == 1 else None
    k = tokens.index("<<")
    rest = tokens[k + 1:]
    if len(rest) != 1:
        return None                                   # something follows the heredoc delimiter on line 1
    delim = rest[0][1:] if rest[0].startswith("-") and "<<-" in lines[0] else rest[0]
    body = lines[1:]
    close = next((j for j, l in enumerate(body) if l.strip() == delim), None)
    if close is None or any(l.strip() for l in body[close + 1:]):
        return None                                   # unterminated, or commands after the heredoc
    return tokens[:k]


def mentions_commit(command: str) -> bool:
    """GIT_COMMIT on the raw text and on each line with shell quoting removed,
    so `git 'com'mit` or `git com\\mit` are caught too."""
    if GIT_COMMIT.search(command):
        return True
    for line in command.split("\n"):
        try:
            if GIT_COMMIT.search(" ".join(shlex.split(line))):
                return True
        except ValueError:
            continue
    return False


def simple_commit(tokens: list[str], cwd: Path) -> tuple[Path, str] | None:
    """(directory, problem) for `[cd DIR &&] git [-C DIR] commit [OPTIONS]`; None for any other shape.
    problem is "" when the options are all allowed."""
    d, i = cwd, 0
    if tokens[:1] == ["cd"]:
        if len(tokens) < 3 or tokens[2] != "&&":
            return None
        p = Path(os.path.expandvars(tokens[1])).expanduser()
        d = p if p.is_absolute() else cwd / p
        i = 3
    if tokens[i:i + 1] != ["git"]:
        return None
    i += 1
    if tokens[i:i + 1] == ["-C"] and i + 1 < len(tokens):
        p = Path(os.path.expandvars(tokens[i + 1])).expanduser()
        d = p if p.is_absolute() else d / p
        i += 2
    if tokens[i:i + 1] != ["commit"]:
        return None
    i += 1
    while i < len(tokens):
        t = tokens[i]
        if t in VALUE_OPTS or re.fullmatch(r"-[qvsne]*m", t):     # value follows, e.g. `-m x`, `-sm x`
            i += 2
            continue
        if t in FLAG_OPTS or t.startswith(VALUE_PREFIXES) or re.fullmatch(r"-[qvsn]*m.+", t) \
                or re.fullmatch(r"-[qvsne]+", t):
            i += 1
            continue
        return d, f"`{t}` changes or extends what is committed beyond the reviewed index"
    return d, ""


def deny(reason: str) -> int:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    return 0


def cmd_hook() -> int:
    data = json.load(sys.stdin)
    command = (data.get("tool_input") or {}).get("command", "")
    if not mentions_commit(command):
        return 0
    cwd = Path(data.get("cwd") or ".")
    tokens = split_command(command)
    parsed = simple_commit(tokens, cwd) if tokens else None
    target = parsed[0] if parsed else cwd
    root = repo_root(target) or repo_root(cwd)    # unresolvable target: fall back to cwd, never allow blindly
    if root is None or not (root / "scripts" / "review_gate.py").exists():
        return 0                                   # not a repository under this policy
    nxt = next_assigned(root)
    how = (f"Assigned reviewer: {nxt} (backup: {REVIEWERS[1 - REVIEWERS.index(nxt)]}). Stage in a separate "
           "command, have the reviewer review the staged diff, record it with "
           f"`python3 scripts/review_gate.py approve {nxt}` (or `approve <other> --backup`), then commit on its "
           "own as `[cd DIR &&] git [-C DIR] commit [message options]`. For a multi-line message use "
           "`-F - <<'EOF'` with the message as the heredoc body (not `-m \"$(cat <<'EOF' ...)\"`).")
    if parsed is None:
        return deny("Review gate: in this repository a commit must run on its own, so the reviewed index is "
                    "exactly what is committed. " + how)
    if parsed[1]:
        return deny(f"Review gate: {parsed[1]}. " + how)
    h = staged_hash(root)
    f = state_dir(root) / "approved"
    approved = f.read_text().strip() if f.exists() else ""
    if h and h == approved:
        return 0
    return deny(f"Review gate: the staged diff ({h[:12] or 'nothing staged'}) has not been reviewed. " + how)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "hook":
        return cmd_hook()
    root = repo_root(Path.cwd())
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return 1
    if sys.argv[1] == "status":
        return cmd_status(root)
    if sys.argv[1] == "approve":
        return cmd_approve(root, sys.argv[2:])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
