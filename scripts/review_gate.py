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
that was reviewed is exactly the diff that is committed. Where the hook is
registered it works as an allowlist: any other command that mentions git or
commit must be plain and use git only in ways that cannot commit (see the
threat-model comment below and CLAUDE.md).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REVIEWERS = ("fable", "codex")
FIRST = "fable"


GIT_TIMEOUT = 15                                       # seconds; the hook itself is limited to 20


def repo_root(start: Path) -> Path | None:
    """Top level of the repository containing `start`, read as raw bytes. None only when git says
    `start` is not in a repository, or `start` does not exist; any other failure raises, so the
    hook refuses rather than treating the target as outside the policy."""
    r = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                       capture_output=True, timeout=GIT_TIMEOUT)
    if r.returncode == 0:
        return Path(os.fsdecode(r.stdout[:-1] if r.stdout.endswith(b"\n") else r.stdout))
    if b"not a git repository" in r.stderr or not start.exists():
        return None
    raise OSError(f"git rev-parse failed for {start}: {r.stderr[-300:]!r}")


def staged_hash(root: Path) -> str:
    diff = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--binary"],
                          capture_output=True, check=True, timeout=GIT_TIMEOUT).stdout
    return hashlib.sha256(diff).hexdigest() if diff else ""


def state_dir(root: Path) -> Path:
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "--absolute-git-dir"],
                         capture_output=True, check=True, timeout=GIT_TIMEOUT).stdout
    d = Path(os.fsdecode(out[:-1] if out.endswith(b"\n") else out)) / "review"
    d.mkdir(exist_ok=True)
    return d


def head_commit(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD"],
                       capture_output=True, text=True, errors="replace", timeout=GIT_TIMEOUT)
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


# Threat model: this gate keeps an agent from committing an unreviewed index by accident or through
# a careless command shape. It is not a sandbox: a program that runs git itself (a script, a tool)
# can always commit without a Bash command line the hook could read.
#
# Scope: the hook runs only in sessions whose project settings register it. There it works as an
# allowlist, so that what it lets through is read exactly rather than guessed:
# - A command that does not mention git or commit (after removing quotes and backslashes) runs.
# - A commit is only `[cd /ABS &&] git [-C DIR] commit [message options]` (simple_commit). It names
#   its repository exactly; in a repository carrying scripts/review_gate.py it needs the approved
#   staged diff.
# - Any other command that mentions git or commit must be plain (plain_commands): no expansion,
#   substitution, glob, heredoc or process substitution, so its words are exactly bash's. Each git
#   call must start its command with a real, non-commit git subcommand (judge_plain), with only the
#   global options in GIT_GLOBALS, no `VAR=value` prefix and no config writes, since config and
#   environment can name a pager, editor or diff program; git or commit may appear as arguments only
#   to read-only commands (INERT) or read-only git subcommands, and only those or git may read a pipe.
# Out of scope: deliberately obfuscated spellings of git (`$'\x67it'`, a name built from pieces in
# variables), and programs, scripts, Makefiles, shell functions or aliases that run git themselves.
GIT_WORDS = re.compile(r"(?<![A-Za-z0-9])(?:git|commit)(?![A-Za-z0-9])", re.I)   # also GIT_PAGER=…
# Global options that only choose the repository or switch features off; `-c`, `-p`, `--config-env`,
# `--exec-path` and the rest are refused, since they can name a program git runs.
GIT_GLOBALS = {"--no-pager", "-P", "--bare", "--no-optional-locks", "--literal-pathspecs",
               "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs", "--no-replace-objects"}
GIT_GLOBALS_WITH_VALUE = {"-C", "--git-dir", "--work-tree"}
# Commands that never run or evaluate their arguments: git or commit may appear in their arguments.
# Not printf or test: `printf -v 'a[$(…)]'` and `test -v 'a[$(…)]'` evaluate a subscript.
# Not uniq (writes an output file) or cd (into .git, then a relative write); none of these writes a file.
INERT = {"echo", "grep", "egrep", "fgrep", "cat", "ls", "head", "tail", "wc", "cut", "tr",
         "cmp", "true", "false", "pwd", "stat", "basename", "dirname", "realpath", "readlink"}
# Git subcommands that run no other program from their arguments: git or commit may appear there.
READ_ONLY_GIT = {"log", "show", "diff", "status", "blame", "ls-files", "rev-parse", "describe", "shortlog"}

# The only commit shape let through: `[cd DIR &&] git [-C DIR] commit [OPTIONS]`,
# where OPTIONS never change which content is committed.
VALUE_OPTS = {"-m", "--message", "-F", "--file", "--author", "--date", "-C", "--reuse-message",
              "-c", "--reedit-message", "--fixup", "--squash", "-t", "--template", "--cleanup"}
FLAG_OPTS = {"-q", "--quiet", "-v", "--verbose", "-s", "--signoff", "-n", "--no-verify",
             "--allow-empty", "--allow-empty-message", "--amend", "--no-edit", "-e", "--edit",
             "--no-gpg-sign", "--status", "--no-status"}
VALUE_PREFIXES = ("--message=", "--file=", "--author=", "--date=", "--reuse-message=",
                  "--reedit-message=", "--fixup=", "--squash=", "--template=", "--cleanup=")


# The commit line is read with a small allowlist grammar, not a shell parser: a word is a run of
# plain characters, 'single-quoted' text, or "double-quoted" text without $, backticks or
# backslashes. Nothing in such a line can expand: no $, backticks, tildes, globs, braces,
# comments, escapes or redirections, so the words below are exactly the arguments bash passes to git.
WORD_PART = re.compile(r"[A-Za-z0-9_./:@%+,=^-]+|'[^']*'|\"[^\"$`\\\\]*\"")
# A heredoc is allowed only with a single-quoted delimiter, so bash reads its body literally.
HEREDOC = re.compile(r"[ \t]<<(-?)'([A-Za-z_][A-Za-z0-9_]*)'[ \t]*$")


def split_words(line: str) -> list[str] | None:
    """Words of `line` under the grammar above (quotes removed), `&&` as its own word; None otherwise."""
    words, i, n = [], 0, len(line)
    while True:
        while i < n and line[i] in " \t":
            i += 1
        if i >= n:
            return words
        if line.startswith("&&", i):
            words.append("&&")
            i += 2
            continue
        word = ""
        while i < n and line[i] not in " \t":
            m = WORD_PART.match(line, i)
            if not m:
                return None
            part = m.group()
            word += part[1:-1] if part[0] in "'\"" else part
            i = m.end()
        words.append(word)


def split_command(command: str) -> list[str] | None:
    """Words of the commit line, or None if the command is not that line alone. A heredoc
    message must end line 1 as `<<'WORD'` (or `<<-'WORD'`); its closing line is matched the
    way bash matches it, and only lines of spaces and tabs may follow (bash would run a line
    holding, say, a carriage return as a command)."""
    lines = command.strip("\n").split("\n")
    m = HEREDOC.search(lines[0])
    if m is None:
        return split_words(lines[0]) if len(lines) == 1 else None
    strip_tabs, delim = m.group(1) == "-", m.group(2)
    body = lines[1:]
    close = next((j for j, l in enumerate(body) if (l.lstrip("\t") if strip_tabs else l) == delim), None)
    if close is None or any(l.strip(" \t") for l in body[close + 1:]):
        return None                                   # unterminated, or commands after the heredoc
    return split_words(lines[0][:m.start()])


# A plain command: words as in the commit line (WORD_PART), joined by `&&`, `||`, `;`, `|`, newlines
# or parentheses, with redirections to a plain target (`> out`, `2>/dev/null`, `2>&1`, `< in`).
PLAIN_OPERATOR = re.compile(r"&&|\|\||[;|()\n]")
PLAIN_REDIRECT = re.compile(r"[0-9]*(?:>>|>&|>|<)")
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*", re.S)


def plain_commands(command: str) -> list[tuple[list[str], bool, list[str]]] | None:
    """The simple commands of `command` as (words, reads a pipe, redirection targets), quotes removed,
    or None if anything in it is not plain: `$`, backticks, backslashes, globs, braces, tildes,
    comments, `&`, heredocs, here-strings, process substitution. A group `( … )` that reads a pipe
    passes it to every command inside. A newline right after `|`, `&&`, `||` or `(` continues the
    command, as in bash. Unbalanced parentheses are not plain."""
    cmds, words, targets, target, piped, groups, i, n = [], [], [], False, False, [], 0, len(command)
    last_op = None
    while i < n:
        if command[i] in " \t":
            i += 1
            continue
        m = PLAIN_OPERATOR.match(command, i)
        if m:
            if target:
                return None
            op = m.group()
            if op in ("|", "&&", "||", ";") and (last_op in ("|", "&&", "||", ";", "(") or (not cmds and not words
                                                                                          and not targets)):
                return None                            # an operator with no command before it
            if op == "\n" and last_op in ("|", "&&", "||", "("):
                i = m.end()                            # continuation: the pending pipe or list goes on
                continue
            if words or targets:
                cmds.append((words, piped or any(groups), targets))
                words, targets = [], []
            if op == "(":
                groups.append(piped)
            elif op == ")":
                if not groups:
                    return None
                groups.pop()
            piped = op == "|" or (op == "(" and piped)
            last_op = op
            i = m.end()
            continue
        last_op = None
        m = PLAIN_REDIRECT.match(command, i)
        if m:
            if target:
                return None
            target, i = True, m.end()
            continue
        word = ""
        while i < n and command[i] not in " \t<>" and not PLAIN_OPERATOR.match(command, i):
            m = WORD_PART.match(command, i)
            if not m:
                return None
            part = m.group()
            word += part[1:-1] if part[0] in "'\"" else part
            i = m.end()
        if target:
            target = False                             # a redirection target is a file name, not run
            targets.append(word)
        else:
            words.append(word)
    if target or groups:
        return None
    if words or targets:
        cmds.append((words, piped or any(groups), targets))
    return cmds


def config_read(rest: list[str]) -> bool:
    """`git config` arguments of a plain read: KEY, --get KEY, --get-all KEY, --list or -l."""
    if rest in (["--list"], ["-l"]):
        return True
    if len(rest) == 2 and rest[0] in ("--get", "--get-all"):
        rest = rest[1:]
    return len(rest) == 1 and re.fullmatch(r"[A-Za-z][\w.-]*", rest[0]) is not None


def judge_plain(cmds: list[tuple[list[str], bool, list[str]]], commands: set[str]) -> str:
    """Why a plain command that mentions git or commit is refused, or "" if it may run."""
    for words, piped, targets in cmds:
        if any(GIT_WORDS.search(t) for t in targets):
            return f"a redirection to or from `{next(t for t in targets if GIT_WORDS.search(t))}`, which mentions git"
        k = 0
        while k < len(words) and ASSIGNMENT.fullmatch(words[k]):
            if GIT_WORDS.search(words[k]):
                return f"the assignment `{words[k]}` mentions git or commit"
            k += 1
        if k == len(words):
            continue
        name, args = words[k].rsplit("/", 1)[-1], words[k + 1:]
        if name == "git":
            if k > 0:
                return "git is run with a `VAR=value` prefix, which can name a program for git to run"
            j = 0
            while j < len(args) and args[j].startswith("-"):
                opt = args[j].split("=", 1)[0]
                if args[j] in GIT_GLOBALS or (opt in GIT_GLOBALS_WITH_VALUE and "=" in args[j] and opt != "-C"):
                    j += 1
                elif args[j] in GIT_GLOBALS_WITH_VALUE and j + 1 < len(args):
                    j += 2
                else:
                    return f"the git option `{args[j]}` is refused (only {sorted(GIT_GLOBALS | GIT_GLOBALS_WITH_VALUE)})"
            if j == len(args):
                return "git is run without a subcommand"
            sub, rest = args[j], args[j + 1:]
            if sub == "commit":
                return "a commit must be the simple form, on its own"
            if sub not in commands:
                return f"`git {sub}` is not a git command (aliases are refused)"
            if sub == "config" and not config_read(rest):
                return "`git config` may only read (KEY, --get KEY, --get-all KEY, --list)"
            options = rest[:rest.index("--")] if "--" in rest else rest     # operands after `--` are paths
            if any((a == "--output" or a.startswith("--output=")) and sub in READ_ONLY_GIT for a in options):
                return f"`git {sub}` is given an output file option"
            if sub not in READ_ONLY_GIT and any(GIT_WORDS.search(a) for a in rest):
                return f"`git {sub}` is given an argument mentioning git or commit"
        elif name not in INERT:
            if piped:
                return f"`{name}` reads a pipe; in a command that mentions git or commit only git and read-only commands may"
            if any(GIT_WORDS.search(w) for w in words[k:]):
                return f"`{name}` is given git or commit as a word; only read-only commands may be"
    return ""


def git_commands() -> set[str]:
    """Names of real git commands (built-in and installed); anything else is an alias or a typo."""
    try:
        r = subprocess.run(["git", "--list-cmds=main,others,nohelpers"], capture_output=True, text=True,
                           errors="replace", timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):      # no usable git: every subcommand counts as unknown
        return set()
    return set(r.stdout.split())


def simple_commit(tokens: list[str], cwd: Path) -> tuple[Path, str] | None:
    """(directory, problem) for `[cd DIR &&] git [-C DIR] commit [OPTIONS]`; None for any other shape.
    problem is "" when the options are all allowed."""
    d, i = cwd, 0
    if tokens[:1] == ["cd"]:
        # Only an absolute path without `.` or `..` parts: `cd -` (OLDPWD) and relative paths (CDPATH)
        # can land elsewhere, and bash resolves `..` logically or, failing that, physically.
        if len(tokens) < 3 or tokens[2] != "&&" or not Path(tokens[1]).is_absolute() \
                or {".", ".."} & set(tokens[1].split("/")):
            return None
        d = Path(tokens[1])
        i = 3
    if tokens[i:i + 1] != ["git"]:
        return None
    i += 1
    if tokens[i:i + 1] == ["-C"] and i + 1 < len(tokens):
        p = Path(tokens[i + 1])
        d = p if p.is_absolute() else d / p
        i += 2
    if tokens[i:i + 1] != ["commit"]:
        return None
    i += 1
    while i < len(tokens):
        t = tokens[i]
        fixup = t[len("--fixup="):] if t.startswith("--fixup=") else \
            tokens[i + 1] if t == "--fixup" and i + 1 < len(tokens) else ""
        if fixup.startswith(("amend:", "reword:")):     # reword: implies --only, which ignores the index
            return d, f"`--fixup={fixup}` can commit something other than the reviewed index"
        if t in VALUE_OPTS or re.fullmatch(r"-[qvsne]*m", t):     # value follows, e.g. `-m x`, `-sm x`
            if i + 1 >= len(tokens):
                return d, f"`{t}` needs a value"
            i += 2
            continue
        if t in FLAG_OPTS or t.startswith(VALUE_PREFIXES) or re.fullmatch(r"-[qvsn]*m.+", t) \
                or re.fullmatch(r"-[qvsne]+", t):
            i += 1
            continue
        return d, f"`{t}` changes or extends what is committed beyond the reviewed index"
    return d, ""


def deny(reason: str) -> int:
    """Refuse the tool call: JSON on stdout (exit 0); if that cannot be written, the reason on stderr
    and exit 2, which Claude Code also treats as a blocking refusal."""
    try:
        sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                            "permissionDecision": "deny",
                                                            "permissionDecisionReason": reason}}) + "\n")
        sys.stdout.flush()
        return 0
    except BaseException:
        try:
            sys.stderr.write(reason + "\n")
            sys.stderr.flush()
        except BaseException:
            pass
        return 2


def mentions_git(text: str) -> bool:
    """git or commit in `text`, in any letter case, after removing quotes and backslashes."""
    return GIT_WORDS.search(re.sub(r"[\"'\\\\]", "", text)) is not None


def payload_mentions_git(raw: str) -> bool:
    """mentions_git on every string of the decoded JSON payload (keys too, walked without recursion),
    and on the raw text with JSON escapes turned into what they stand for (`\\uXXXX` decoded, `\\n`,
    `\\t` and the like as spaces, so word boundaries survive). Any failure counts as True."""
    try:
        text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), raw)
        if mentions_git(raw) or mentions_git(re.sub(r"\\[nrtbf]", " ", text)):
            return True
        try:
            stack = [json.loads(raw)]
        except ValueError:
            return False                               # not JSON: the text views above are all there is
        while stack:
            x = stack.pop()
            if isinstance(x, str) and mentions_git(x):
                return True
            if isinstance(x, dict):
                stack.extend(x.keys())
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
        return False
    except BaseException:                              # e.g. RecursionError on deep nesting: fail closed
        return True


def cmd_hook(raw: str) -> int:
    try:
        data = json.loads(raw)
        command = data["tool_input"]["command"]
        if not isinstance(command, str) or ("cwd" in data and not (isinstance(data["cwd"], str) and data["cwd"])):
            raise TypeError                        # a malformed cwd must not fall back to the hook's own directory
        cwd = Path(data.get("cwd", "."))
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
        # Unreadable payload: refuse what might touch git, let everything else run.
        return deny("Review gate: could not read the hook payload.") if payload_mentions_git(raw) else 0
    if not mentions_git(command):
        return 0                                   # mentions neither git nor commit
    how = ("Commit on its own as `[cd /ABSOLUTE/DIR &&] git [-C DIR] commit [message options]`; for a "
           "multi-line message use `-F - <<'EOF'` with the message as the heredoc body. Any other command "
           "that mentions git or commit must be plain: words of plain characters, 'single-quoted' or "
           "\"double-quoted\" text without $, backticks or backslashes, joined by &&, ||, ;, |, newlines or "
           "parentheses, with simple redirections; git first in its command with a real subcommand; git or "
           "commit as arguments only to read-only commands (echo, grep, cat, ls, head, ...). Write other "
           "such text with the file tools.")
    tokens = split_command(command)
    parsed = simple_commit(tokens, cwd) if tokens else None
    if parsed is None:
        cmds = plain_commands(command)
        if cmds is None:
            return deny("Review gate: this command mentions git or commit but is not plain. " + how)
        reason = judge_plain(cmds, git_commands())
        return deny(f"Review gate: {reason}. " + how) if reason else 0
    if parsed[1]:                                  # before the exemption below: a chained or extended
        return deny(f"Review gate: {parsed[1]}. " + how)   # commit is refused whatever it targets first
    try:
        root = repo_root(parsed[0])
        policy = root is not None and (root / "scripts" / "review_gate.py").exists()
    except (OSError, ValueError, subprocess.SubprocessError) as e:   # e.g. E2BIG, NUL, timeout: refuse, never exempt
        return deny(f"Review gate: the target repository cannot be resolved ({e}). " + how)
    if not policy:
        return 0                                   # the simple form targets a repository outside this policy
    try:                                           # unreadable state: refuse, never crash or allow
        nxt = next_assigned(root)
        h = staged_hash(root)
        f = state_dir(root) / "approved"
        approved = f.read_text().strip() if f.exists() else ""
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as e:
        return deny(f"Review gate: the approval state of {root} cannot be read ({e}). " + how)
    how = (f"Assigned reviewer: {nxt} (backup: {REVIEWERS[1 - REVIEWERS.index(nxt)]}). Stage in a separate "
           "command, have the reviewer review the staged diff, record it with "
           f"`python3 scripts/review_gate.py approve {nxt}` (or `approve <other> --backup`). " + how)
    if h and h == approved:
        return 0
    return deny(f"Review gate: the staged diff ({h[:12] or 'nothing staged'}) has not been reviewed. " + how)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "hook":
        raw = ""
        try:
            raw = sys.stdin.buffer.read().decode("utf-8", "replace")
            return cmd_hook(raw)
        except BaseException as e:                     # any failure, even an interrupt: refuse what touches git
            touches = not raw or payload_mentions_git(raw)
            if touches:
                return deny(f"Review gate: the hook failed ({type(e).__name__}); refusing to be safe.")
            return 0
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
    code = main()
    try:
        sys.stdout.flush()
    except BaseException:                              # output lost (e.g. closed pipe): a refusal must still block
        code = 2 if code == 0 and len(sys.argv) > 1 and sys.argv[1] == "hook" else code
    try:
        sys.stderr.flush()
    except BaseException:
        pass
    os._exit(code)
