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
5. Commit on its own: `[cd /ABSOLUTE/DIR &&] git [-C DIR] commit [message options]`. Do not use `-a`, `-i`, `-o`, `-p` or pathspecs, and do not chain other commands. For a multi-line message use `-F - <<'EOF'` with the message as the heredoc body; the `-m "$(cat <<'EOF' …)"` form is refused. `cd` takes an absolute path without `.` or `..` parts. Every word on the commit line must be plain (letters, digits, `_./:@%+,=^-`), `'single-quoted'`, or `"double-quoted"` without `$`, backticks or backslashes, so nothing can expand or run before git does; only a single-quoted heredoc delimiter is accepted.

A PreToolUse hook (`.claude/settings.json`, matcher `Bash|Write|Edit|MultiEdit|NotebookEdit`) enforces this in sessions where it is registered (this repository and the course folder). It is an allowlist, so what it lets through is read exactly rather than guessed:

- A Bash command that does not mention `git` or `commit` (in any letter case, after removing quotes and backslashes) runs untouched.
- A commit runs only in the form of step 5. It names its repository exactly; in a repository carrying `scripts/review_gate.py` it needs the approved staged diff. The commit form may carry any text in its message heredoc.
- Any other command that mentions `git` or `commit` must be **plain**: words as in step 5, joined by `&&`, `||`, `;`, `|`, newlines or parentheses, with redirections to a plain file name (`> out`, `2>&1`). No `$`, backticks, backslashes, globs, braces, `~`, comments, `&`, heredocs, here-strings or `<(…)`. Each git call starts its command with a real git subcommand other than `commit` (aliases are refused), with no `VAR=value` prefix and only the global options `-C DIR`, `--git-dir`, `--work-tree`, `--no-pager` and a few switches (no `-c`, `-p`, `--config-env`); `git config` may only read (`KEY`, `--get KEY`, `--get-all KEY`, `--list`), and read-only git subcommands may not write an `--output` file. A redirection to or from a path mentioning git or commit is refused (`> .git/x` and `< .git/config` alike), and `cd` may not enter such a path. `git` or `commit` may appear as arguments only to read-only commands (`echo`, `grep`, `cat`, `ls`, `head`, `tail`, `wc`, …; not `printf` or `test`) or to read-only git subcommands (`log`, `show`, `diff`, `status`, …), never after a wrapper (`env`, `timeout`, `sudo`, `xargs`) or a shell. Only git and read-only commands may read a pipe.

So `timeout 10 git status`, `"$(command -v git)" status`, `git -c user.name=x status`, `git config user.name x`, `git log | python3 x.py`, `cat > f <<'EOF'` with git in the text, or `git status # note` are refused; write the git part plainly, set config by editing files, or write such text with the file tools.

Assumption: the approval state in `.git/review/` is written only by `review_gate.py approve`. The hook enforces part of this: the file tools (Write, Edit, MultiEdit, NotebookEdit) may not write inside any `.git` directory, judged on the path as given and with symlinks resolved, and Bash redirections to paths that mention git are refused. A Bash write through a relative path or symlink, or a program, is not detected. Never write under `.git/` by any means. If the approval state cannot be read, the target repository cannot be resolved, or the hook itself fails, a command that mentions git or commit is refused; if the refusal cannot be written to stdout, the hook exits with code 2, which also blocks the call.

Out of scope: deliberately obfuscated spellings of git (`$'\x67it'`, a name assembled in variables), and programs, scripts, Makefiles, shell functions or shell aliases that run git themselves. The hook guards against accidental or careless commits by an agent; it is not a sandbox. The review log is kept in `.git/review/log.tsv`, outside version control.

## Data and conventions

Follow the course's Week 2 Data Conventions: `config.py` and `eda_shared/` are shared files and are never edited here. Project deviations are stated in the README and the methods section.
