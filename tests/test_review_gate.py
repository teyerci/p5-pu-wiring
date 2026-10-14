"""Tests for scripts/review_gate.py on a throwaway repository."""
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "scripts" / "review_gate.py"


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(GATE, r / "scripts" / "review_gate.py")
    run(["git", "init", "-q", "-b", "main"], r)
    run(["git", "config", "user.email", "t@example.com"], r)
    run(["git", "config", "user.name", "t"], r)
    (r / "a.txt").write_text("a\n")
    run(["git", "add", "-A"], r)
    return r


def hook(command, cwd):
    payload = json.dumps({"tool_name": "Bash", "cwd": str(cwd), "tool_input": {"command": command}})
    r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr                 # a crash must fail the test, not read as "allow"
    if not r.stdout.strip():
        return "allow"
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    return "deny"


def approve(repo, *args):
    return subprocess.run([sys.executable, "scripts/review_gate.py", "approve", *args],
                          cwd=repo, capture_output=True, text=True)


COMMIT = "git commit"          # assembled here so test commands read naturally


def test_unreviewed_commit_is_denied(repo):
    assert hook(f"{COMMIT} -q -m x", repo) == "deny"


def test_reviewed_simple_commit_is_allowed(repo):
    assert approve(repo, "fable").returncode == 0
    assert hook(f"{COMMIT} -q -m x", repo) == "allow"
    assert hook(f"cd {repo} && {COMMIT} -q -F - <<'EOF'\nmessage\nEOF", "/") == "allow"
    assert hook(f"git -C {repo} {COMMIT[4:]} -m x", "/") == "allow"
    assert hook(f"{COMMIT} -sm x", repo) == "allow"
    assert hook(f"{COMMIT} -q -F - <<'EOF'\nline 1\n\nline 2\nEOF\n", repo) == "allow"
    assert hook(f"{COMMIT} -F - <<'EOF'\nquoted body keeps $(this) literal\nEOF", repo) == "allow"
    assert hook(f"{COMMIT} -F - <<-'EOF'\n\tmsg\n\tEOF", repo) == "allow"
    assert hook(f"{COMMIT} -F - <<'EOF'\nsubject\n\ngit foo is refused; {COMMIT} -a too\nEOF", repo) == "allow"
    assert hook(f"{COMMIT} -m 'literal $(x) and `y`' -m \"it's fine\" --author='A B <a@b.c>'", repo) == "allow"


@pytest.mark.parametrize("cmd", [
    f"{COMMIT} a.txt -m x", f"{COMMIT} -o -m x -- a.txt", f"{COMMIT} -i a.txt -m x", f"{COMMIT} -p",
    f"{COMMIT} -a -m x", f"{COMMIT} --all -m x",
    f"git stage a.txt && {COMMIT} -m x", f"git rm --cached a.txt && {COMMIT} -m x",
    f"git restore --staged a.txt; {COMMIT} -m x", f"git stash pop && {COMMIT} -m x",
    f"bash -c '{COMMIT} -m x'", f"eval '{COMMIT} -m x'", f"/usr/bin/{COMMIT} -m x",
    f"git --no-pager {COMMIT[4:]} -m x",
    f"{COMMIT} -F - <<'EOF' && git add z && {COMMIT} -a -m y\nmsg\nEOF",
    f"{COMMIT} -F - <<'EOF'\nmsg\nEOF\ngit add z && {COMMIT} -a -m y",
    f"{COMMIT} -F - <<'EOF'\nmsg without end",
    "git 'com'mit -a -m x", 'git "commit" -a -m x', "git com\\mit -a -m x",
    f'{COMMIT} -m "$(git add -A; echo msg)"', f'{COMMIT} -m "`git add -A`msg"',
    f'{COMMIT} --message="$(git add -A)"', f'{COMMIT} -sm"$(git add -A)"', f'{COMMIT} --author "$(git add -A)"',
    f"{COMMIT} -F - <<EOF\n$(git add -A)\nEOF", f"{COMMIT} -F - <<EOF\n`git add -A`\nEOF",
    # found in review: anything that can expand is refused, not only $( and backticks
    f"{COMMIT} -F - <<EOF # <<'\n$(git add -A)\nmessage\nEOF",
    f"{COMMIT} -F - <<EOF\n$\\\n(git add -A)\nmessage\nEOF",
    f"{COMMIT} -m \"msg${{a[$'\\x24\\x28git add -A; echo 0\\x29']}}\"",
    f"{COMMIT} -F - <<`git${{IFS}}add${{IFS}}-A`\nmessage\n `git${{IFS}}add${{IFS}}-A`",
    f"{COMMIT} -m msg${{IFS}}-a", f"{COMMIT} -m *", f"{COMMIT} -m {{x,-a}}", f"{COMMIT} -m x#y -a",
    f"{COMMIT} -m x # -a", f"{COMMIT} -m x\\ -a", f"{COMMIT} -m x > out", f"{COMMIT} -m x &",
    f"{COMMIT} -F - <<'EOF'\nmsg\n EOF\ngit add -A",          # ' EOF' does not close the heredoc in bash
    f"{COMMIT} -F - <<-'EOF'\nmsg\n EOF",                    # <<- strips tabs only
    # second review round
    "git com\\\nmit -a -m x", "git $'com\\x6dit' -a -m x", "git $SUB -a -m x",
    f"cd - && {COMMIT} -m x", f"cd . && {COMMIT} -m x", f"cd '~' && {COMMIT} -m x",
    f"{COMMIT} --fixup=reword:HEAD --no-edit", f"{COMMIT} --fixup reword:HEAD --no-edit",
    f"{COMMIT} --fixup=amend:HEAD",
    f"{COMMIT} -F - <<'EOF'\nmsg\nEOF\n\r\n", f"{COMMIT} -F - <<'EOF'\nmsg\nEOF\n\x0b\n",
    f"{COMMIT} -m ~", f"{COMMIT} -m foo=~:~root",
    "git -c alias.x=commit x -a -m y",
])
def test_bypasses_are_denied_even_when_approved(repo, cmd):
    assert approve(repo, "fable").returncode == 0
    assert hook(cmd, repo) == "deny"


def test_aliases_are_refused_and_prose_is_not(repo):
    run(["git", "config", "alias.ci", "commit"], repo)
    run(["git", "config", "alias.st", "status"], repo)
    assert approve(repo, "fable").returncode == 0
    for cmd in ["git ci -a -m x", "git st", "git -C nested ci -a -m x", "env git ci", "X=1 git ci",
                "git config alias.zz commit && git zz -m x", "git -c alias.x=status x",
                "bash -c 'git ci -a -m x'", "sh -lc 'git st'", "eval 'git ci -a'",
                "env -u UNUSED git -c alias.ci=commit ci -m x", "command -- git -c alias.ci=commit ci -m x",
                "timeout 10 git -c alias.ci=commit ci -m x", "(git status);git -c alias.ci=commit ci -m x",
                "git status\ngit ci -m x", "bash -c 'git status\ngit ci -m x'",
                "2>/dev/null git -c alias.ci=commit ci -m x", "> out git ci -m x",
                'echo "$(git -c alias.ci=commit ci -m x)"', 'X="$(git ci -m x)"', 'echo "`git ci -m x`"',
                "sudo -u bob git ci", "cat <<EOF\n$(git ci -m x)\nEOF"]:
        assert hook(cmd, repo) == "deny", cmd
    assert hook("git config alias.zz status", repo) == "deny"            # config writes are refused
    for cmd in ['echo "fix git hook"', "git config --get user.name", "grep -n 'git foo' README.md",
                "grep -n git README.md", 'git log --grep "git .*"', "git status | grep git | head",
                'echo "notes on git foo.txt"',
                "grep -c 'git foo' README.md", "git status 2>/dev/null | head"]:
        assert hook(cmd, repo) == "allow", cmd
    for cmd in ["git -c alias.st=status status", "git -c user.name=x status"]:   # -c can name a program
        assert hook(cmd, repo) == "deny", cmd


def test_command_outside_the_repo_naming_it_is_gated(repo, tmp_path):
    assert hook(f"cd {repo}; {COMMIT} -m x", tmp_path) == "deny"
    assert hook(f"git -C {repo} add -A && git -C {repo} {COMMIT[4:]} -m x", tmp_path) == "deny"


def test_other_ways_of_naming_the_repo_are_gated(repo, tmp_path):
    for cmd in [f"cd -- {repo} && {COMMIT} -a -m x", f"cd -P {repo} && {COMMIT} -a -m x",
                f"cd {repo}&&{COMMIT} -a -m x", f"git -C {repo.parent} -C {repo.name} {COMMIT[4:]} -a -m x",
                f"git --git-dir={repo}/.git --work-tree={repo} {COMMIT[4:]} -a -m x",
                f"cd {repo.parent} && cd {repo.name} && {COMMIT} -a -m x"]:
        assert hook(cmd, tmp_path) == "deny", cmd


def test_quoted_values_spaces_symlinks_and_nested_repos(repo, tmp_path):
    assert hook(f"git -c user.name='alice smith' {COMMIT[4:]} -a -m x", repo) == "deny"
    spaced = tmp_path / "policy repo"
    eq = tmp_path / "policy=repo"
    for r in (spaced, eq):
        shutil.copytree(repo, r)
        assert hook(f'cd -- "{r}" && {COMMIT} -a -m x', tmp_path) == "deny", r
        assert hook(f'git -C "{r}" {COMMIT[4:]} -a -m x', tmp_path) == "deny", r
    (repo / "src").mkdir()
    link = tmp_path / "shortcut"
    link.symlink_to(repo / "src")
    assert hook(f"git -C {link} {COMMIT[4:]} -a -m x", tmp_path) == "deny"
    nested = repo / "nested"
    shutil.copytree(spaced, nested)
    run(["git", "config", "alias.ci", "record"], nested)
    run(["git", "config", "alias.record", "commit"], nested)
    assert approve(repo, "fable").returncode == 0
    assert hook("git -C nested ci -a -m x", repo) == "deny"


def test_multiline_wrappers_and_cumulative_dirs(repo, tmp_path):
    for cmd in [f"bash -c '{COMMIT} -F - <<EOF\nFix wiring\nEOF\n'", f"sh -c '{COMMIT} -a\n-m x'",
                f"eval '{COMMIT}\n-a -m x'"]:
        assert hook(cmd, repo) == "deny", cmd
    top, mid = repo.parent.parent, repo.parent.name
    assert hook(f"cd {top} && cd {mid} && cd {repo.name} && {COMMIT} -m x", "/") == "deny"
    assert hook(f"git -C {top} -C {mid} -C {repo.name} {COMMIT[4:]} -m x", "/") == "deny"
    assert hook(f"cd -P {top} && cd -- {mid}/{repo.name} && {COMMIT} -m x", "/") == "deny"


def test_nul_in_command_does_not_crash(repo):
    assert hook(f"cd /a\x00b && {COMMIT} -a -m x", repo) == "deny"


def test_compact_operators_do_not_block_ordinary_commands(repo):
    for cmd in ["git status|head", "git status>/dev/null", "git log> /dev/null", "git status&&git log -1",
                "(git status)", "git diff --stat;git log -1"]:
        assert hook(cmd, repo) == "allow", cmd


def test_many_path_arguments_stay_bounded(repo):
    paths = " ".join(f"/abs/p{i} rel{i}" for i in range(1500))
    assert hook(f"git diff -- {paths}", repo) == "allow"


def test_free_form_commits_are_refused_wherever_they_point(repo, tmp_path):
    for cmd in [f"git -C /tmp status; cd {repo.name} && {COMMIT} -m x",
                f"(cd /tmp); cd {repo.name} && {COMMIT} -m x",
                f"pushd /tmp; popd; cd {repo.name} && {COMMIT} -m x",
                f"R={repo.name}; cd \"$R\" && {COMMIT} -m x"]:
        assert hook(cmd, repo.parent) == "deny", cmd
    other = tmp_path / "other2"
    other.mkdir()
    run(["git", "init", "-q"], other)
    assert hook(f"cd {other} && {COMMIT} -m x", repo) == "allow"     # simple form, repository outside the policy
    assert hook(f"cd {other}; {COMMIT} -a -m x", repo) == "deny"     # free form: refused, wherever it points


def test_bad_payloads_do_not_crash(repo):
    for payload in ['{"tool_input": {"command": null}}', "not json", '{"tool_input": null}']:
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout == "", payload
    # refused: mentions a commit; nested too deep to classify (fails closed)
    for payload in [f"{{bad {COMMIT}", "[" * 1500 + "]" * 1500]:
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, text=True)
        assert r.returncode == 0 and '"deny"' in r.stdout, payload[:20]


def test_ordinary_git_commands_pass(repo):
    for cmd in ["git log --grep fix", "git diff --cached | head", 'echo "fix git hook"',
                "git status && git log -1", "ls .git/review"]:
        assert hook(cmd, repo) == "allow", cmd


def test_expanding_cd_path_is_refused(repo):
    assert hook(f'cd "$NO_SUCH_VAR_XYZ/nowhere" && {COMMIT} -m x', repo) == "deny"


def test_chained_commit_is_refused_even_if_it_starts_outside_the_policy(repo, tmp_path):
    other = tmp_path / "plain"
    other.mkdir()
    run(["git", "init", "-q"], other)
    assert hook(f"git -C {other} {COMMIT[4:]} --allow-empty -m a && git -C {repo} {COMMIT[4:]} -a -m b", tmp_path) == "deny"
    assert hook(f"git -C {other} {COMMIT[4:]} -a -m a", tmp_path) == "deny"      # extended form: refused anywhere


def test_cd_through_symlink_and_dotdot_is_refused(repo, tmp_path):
    outside = tmp_path / "outside" / "sub"
    outside.mkdir(parents=True)
    (repo / "out").symlink_to(outside)
    # bash: cd {repo}/out/.. lands in {repo} (logical); a physical resolution would land outside the policy
    assert hook(f"cd {repo}/out/.. && {COMMIT} -m x", tmp_path) == "deny"


def test_cd_with_dot_parts_is_refused(repo):
    assert approve(repo, "fable").returncode == 0
    for cmd in [f"cd /proc/self/cwd/../{repo.name} && {COMMIT} -m x", f"cd {repo}/. && {COMMIT} -m x"]:
        assert hook(cmd, repo) == "deny", cmd


def test_trailing_space_in_repo_name_is_kept(tmp_path, repo):
    spaced = tmp_path / "policy "
    shutil.copytree(repo, spaced)
    assert hook(f"git -C '{spaced}' {COMMIT[4:]} -m x", tmp_path) == "deny"   # unapproved copy, found exactly


def test_heredoc_marker_in_comment_or_quotes_does_not_hide_commands(repo):
    for cmd in ["# message format: <<'EOF'\ngit -c alias.ci=commit ci -m x",
                "echo \"<<'EOF'\"\ngit ci -m x\nEOF", "echo x#; git ci -m x"]:
        assert hook(cmd, repo) == "deny", cmd


def test_unsure_cases_are_refused(repo):
    # Codex round 11: each of these ran an alias commit or a substitution while the hook judged it inert
    for cmd in ["echo '<<' EOF\ngit -c alias.ci=commit ci -m x",
                "cat <<EOF # <<'EOF'\n$(git -c alias.ci=commit ci -m x)\nEOF",
                "bash -co pipefail 'git -c alias.ci=commit ci -m x'",
                "bash -oc pipefail 'git -c alias.ci=commit ci -m x'",
                "env -i bash -c 'git -c alias.ci=commit ci -m x'",
                "bash -o pipefail -c 'git ci -m x'",
                "env --unset UNUSED git -c alias.ci=commit ci -m x", "sudo --user bob git ci",
                "cat <<< \"$(git ci -m x)\"", "bash <<'EOF'\ngit -c alias.ci=commit ci -m x\nEOF",
                # round 12
                "coproc git -c alias.ci=commit ci -m x", "env -S 'git -c alias.ci=commit ci -m x'",
                "\"$(command -v git)\" commit -m x", "$(command -v git) commit -m x",
                "`command -v git` commit -m x", "G=git; $G commit -m x", "\"$(which git)\" ci -m x",
                # round 13
                "G=git; $(command -v \"$G\") commit -m x", "trap 'git -c alias.ci=commit ci -m x' EXIT",
                "source <(printf '%s\\n' 'git ci -m x')", "bash <(printf '%s\\n' 'git ci -m x')",
                # round 14 (allowlist)
                "echo 'git commit -q -m x' | sh", "echo 'git commit -q -m x' | (sh)",
                "printf -v 'a[$(git commit -q -m x)]' '%s' foo", "test -v 'a[$(git commit -q -m x)]'",
                "[ -v 'a[$(git commit -q -m x)]' ]",
                "git -c core.pager=sh -p log -1 --format='tformat:git commit -q -m x'",
                "GIT_PAGER=sh git log -1", "export GIT_PAGER=sh", "git config core.pager sh",
                "git -C/tmp/git-repo status", "git --config-env=core.pager=X log", "sh <<< 'git commit'",
                "git log | python3 x.py", "git config user.name x",
                # round 15
                "echo 'git commit -q -m x' | (true; sh)", "echo 'git commit' | (true && sh)",
                "echo 'git commit' | (false || sh)", "echo 'git commit' | ((sh))", "echo 'git commit' | (\nsh)",
                "git config --unset core.pager", "git config --remove-section core",
                "git config --file --get --edit",
                "git log -1 --format=abc --output=.git/review/approved", "echo abc > .git/review/approved",
                "cd .git/review && echo abc > approved", "uniq in .git/review/approved",
                # round 16
                "echo 'git commit -q -m x' |\nsh", "echo 'git commit -q -m x' |\n(true; sh)",
                "git status; (sh", "git status )"]:
        assert hook(cmd, repo) == "deny", cmd


def test_allowlist_refusals_and_passes(repo):
    # A command mentioning git or commit that is not plain, or passes git words to a command that may run
    # them, is refused; CLAUDE.md lists these. Use the file tools for such text.
    for cmd in ["bash -c 'printf \"%s\\n\" \"$1\"' _ 'git foo'", "timeout 10 grep -n git README.md",
                "cat > notes.txt <<'EOF'\ngit foo is a typo\nEOF", "cat <<'E'O'F'\n$(git ci -m x)\nEOF",
                "git status # check; git hooks", "timeout 10 git status", "env -i git status",
                "nice -n 5 git log -1", '"$(command -v git)" status', 'echo "$(command -v git) available"',
                "ls *.py && git status", "git log -1 --format=%H~1", "git commit-tree HEAD^{tree}",
                "GIT_EDITOR='git commit' git rebase --continue", "git rebase -x 'git log -1' HEAD~2",
                "git -c core.pager='git log' log"]:
        assert hook(cmd, repo) == "deny", cmd
    # nothing mentions git or commit, or only plain, read-only uses
    for cmd in ['"$PY" script.py', "$G log -1", 'env PATH="$PATH" python3 -m pytest', "echo '(' git hooks ')'",
                "ls .git/review && cat .git/review/log.tsv", "git add a.txt && python3 scripts/review_gate.py status",
                "git log --oneline -3 | head -2", "(git status) && echo done", "git diff --cached --stat > out.txt",
                "pip install gitpython", "git show HEAD --stat 2>&1 | tail -1",
                "git --no-pager log -1 12>out", "git --git-dir=.git status", "git -C /tmp status | grep -c x",
                "git config --get user.name", "git config -l", "git log --oneline -3 > log.txt",
                "(git status; git log -1) | head", "git diff 2>&1 | cat",
                "git ls-files -o", "git diff --output-indicator-new=X", "git status &&\ngit log -1"]:
        assert hook(cmd, repo) == "allow", cmd


def test_writes_to_paths_naming_git_are_refused(repo):
    # Spelling only: integrity of .git/review/ is a documented assumption, not enforced by destination.
    for cmd in ["echo abc > .git/review/approved", "echo abc >> .git/review/approved",
                "cat x > ./.git/review/approved", "echo abc >& .git/review/approved"]:
        assert hook(cmd, repo) == "deny", cmd
    assert hook("echo abc > notes.txt 2>&1", repo) == "allow"


def test_operators_without_a_command_are_not_plain(repo):
    for cmd in ["echo 'git commit' |\n&& sh", "| git status", "git status ;; git log", "git status | | cat"]:
        assert hook(cmd, repo) == "deny", cmd


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_approval_state_is_refused_not_a_crash(repo):
    assert approve(repo, "fable").returncode == 0
    assert hook(f"{COMMIT} -m x", repo) == "allow"             # baseline: approved and readable
    review = repo / ".git" / "review"
    review.chmod(0o000)
    try:
        payload = json.dumps({"tool_name": "Bash", "cwd": str(repo), "tool_input": {"command": f"{COMMIT} -m x"}})
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        reason = json.loads(r.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
        assert "cannot be read" in reason
    finally:
        review.chmod(0o755)


def test_unresolvable_target_is_refused_not_a_crash(repo):
    assert hook(f"git -C /{'a' * 140000} {COMMIT[4:]} -m x", repo) == "deny"
    assert hook(f"{COMMIT} -m x", "/" + "a" * 140000) == "deny"


def test_git_without_subcommand_and_output_operands(repo):
    for cmd in ["git", "git --no-pager"]:
        assert hook(cmd, repo) == "deny", cmd
    assert hook("git diff -- --output", repo) == "allow"


def test_value_option_without_operand_is_refused(repo):
    assert approve(repo, "fable").returncode == 0
    assert hook(f"{COMMIT} -m", repo) == "deny"


def test_nul_in_target_path_does_not_crash(repo, tmp_path):
    assert hook(f"git -C '/a\x00b' {COMMIT[4:]} -m x", tmp_path) == "deny"    # unresolvable: refused, no crash


def raw_hook(stdin: bytes, env_extra: dict | None = None) -> str:
    env = {**os.environb, **{k.encode(): v if isinstance(v, bytes) else v.encode()
                             for k, v in (env_extra or {}).items()}}
    r = subprocess.run([sys.executable, str(GATE), "hook"], input=stdin, capture_output=True, env=env)
    assert r.returncode == 0, r.stderr
    if not r.stdout.strip():
        return "allow"
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    return "deny"


def test_malformed_cwd_values_refuse(repo):
    assert approve(repo, "fable").returncode == 0
    for cwd in [0, False, [], {}, "", 1]:
        payload = json.dumps({"cwd": cwd, "tool_input": {"command": f"{COMMIT} -m x"}}).encode()
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, cwd=repo)
        assert r.returncode == 0 and b'"deny"' in r.stdout, cwd


def escaped_payload(command: str, cwd) -> bytes:
    """A valid payload whose command is sent entirely as JSON \\uXXXX escapes."""
    encoded = "".join(f"\\u{ord(ch):04x}" for ch in command)
    return ('{"cwd":' + json.dumps(cwd) + ',"tool_input":{"command":"' + encoded + '"}}').encode()


def short_escape_payload(command: str, cwd) -> bytes:
    """A payload whose command uses newlines and tabs, which JSON sends as `\\n` and `\\t`."""
    return json.dumps({"tool_name": "Bash", "cwd": cwd, "tool_input": {"command": command}}).encode()


TAB_COMMAND = "\n" + COMMIT.replace(" ", "\t") + "\t-m x"
ENCODINGS = [escaped_payload, short_escape_payload]


@pytest.mark.parametrize("encode", ENCODINGS)
def test_malformed_cwd_with_escaped_command_refuses(repo, encode):
    command = TAB_COMMAND if encode is short_escape_payload else f"{COMMIT} -m x"
    for cwd in [0, False, [], {}, "", 1]:
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=encode(command, cwd),
                           capture_output=True, cwd=repo)
        assert r.returncode == 0 and b'"deny"' in r.stdout, cwd
    nested = json.dumps({"x": {"y": [{"z": TAB_COMMAND}]}, "tool_input": {"command": 7}}).encode()
    r = subprocess.run([sys.executable, str(GATE), "hook"], input=nested, capture_output=True, cwd=repo)
    assert r.returncode == 0, r.stderr                            # unexpected keys, nested
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("encode", ENCODINGS)
@pytest.mark.parametrize("fault", [KeyboardInterrupt, SystemExit, subprocess.TimeoutExpired])
def test_failure_fallback_classifies_the_decoded_command(repo, monkeypatch, capsys, fault, encode):
    spec = importlib.util.spec_from_file_location("review_gate_under_test", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    calls = []

    def boom(root):
        calls.append(root)
        raise fault("git", 1) if fault is subprocess.TimeoutExpired else fault()
    monkeypatch.setattr(gate, "next_assigned", boom)
    monkeypatch.setattr(sys, "argv", ["review_gate.py", "hook"])
    command = TAB_COMMAND if encode is short_escape_payload else f"{COMMIT} -m x"
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(encode(command, str(repo)))))
    assert gate.main() == 0
    assert len(calls) == 1                                        # the fault was injected, so the fallback ran
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny" and "hook failed" in out["permissionDecisionReason"]


def file_hook(tool: str, tool_input: dict, cwd) -> str:
    payload = {"tool_name": tool, "tool_input": tool_input}
    if cwd is not None:
        payload["cwd"] = str(cwd)
    r = subprocess.run([sys.executable, str(GATE), "hook"], input=json.dumps(payload),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    if not r.stdout.strip():
        return "allow"
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    return "deny"


def test_file_tools_may_not_write_inside_git(repo, tmp_path):
    (repo / ".git" / "review").mkdir(exist_ok=True)
    (tmp_path / "state-link").symlink_to(repo / ".git" / "review", target_is_directory=True)
    approved = repo / ".git" / "review" / "approved"
    for tool, key in [("Write", "file_path"), ("Edit", "file_path"), ("MultiEdit", "file_path"),
                      ("NotebookEdit", "notebook_path")]:
        assert file_hook(tool, {key: str(approved), "content": "x"}, repo) == "deny", tool
    assert file_hook("Write", {"file_path": ".git/review/approved", "content": "x"}, repo) == "deny"
    assert file_hook("Write", {"file_path": "objects/../review/approved"}, repo / ".git") == "deny"
    assert file_hook("Write", {"file_path": str(tmp_path / "state-link" / "approved")}, tmp_path) == "deny"
    assert file_hook("Edit", {"file_path": str(repo / ".git" / "config")}, repo) == "deny"
    # ordinary files pass, whatever their content says
    assert file_hook("Write", {"file_path": str(repo / "notes.md"), "content": "git commit -a"}, repo) == "allow"
    assert file_hook("Edit", {"file_path": "scripts/review_gate.py", "old_string": "a", "new_string": "b"},
                     repo) == "allow"
    assert file_hook("Write", {"file_path": str(repo / ".gitignore"), "content": "data/"}, repo) == "allow"
    # malformed file payloads refuse
    for tool_input, cwd in [({}, repo), ({"file_path": 7}, repo), ({"file_path": ""}, repo),
                            ({"file_path": "rel.txt"}, None)]:
        assert file_hook("Write", tool_input, cwd) == "deny", tool_input


def test_malformed_file_tool_input_refuses(repo):
    for tool_input in ["oops", ["x"], 7, True]:
        payload = json.dumps({"tool_name": "Write", "cwd": "/tmp", "tool_input": tool_input})
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("review_gate_file_tools", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


@pytest.mark.parametrize("where", ["realpath", "dispatch"])
@pytest.mark.parametrize("fault", [OSError, KeyboardInterrupt, SystemExit])
def test_file_tool_faults_fail_closed(repo, tmp_path, monkeypatch, capsys, fault, where):
    # The path hides .git behind an innocently named symlink, so only realpath could catch it.
    (repo / ".git" / "review").mkdir(exist_ok=True)
    (tmp_path / "shortcut").symlink_to(repo / ".git", target_is_directory=True)
    gate = load_gate_module()
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise fault()
    if where == "realpath":
        monkeypatch.setattr(gate.os.path, "realpath", boom)        # inside judge_file_tool
    else:
        monkeypatch.setattr(gate, "judge_file_tool", boom)         # escapes to main's last resort
    payload = {"tool_name": "Write", "cwd": "/tmp",
               "tool_input": {"file_path": str(tmp_path / "shortcut" / "review" / "approved"), "content": "x"}}
    monkeypatch.setattr(sys, "argv", ["review_gate.py", "hook"])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    assert gate.main() == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("unbuffered", ["", "1"])
def test_refusal_survives_a_closed_stdout(repo, unbuffered):
    payload = json.dumps({"cwd": str(repo), "tool_input": {"command": f"{COMMIT} -m x"}}).encode()
    read_end, write_end = os.pipe()
    os.close(read_end)                                       # nobody reads the JSON
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    if unbuffered:
        env["PYTHONUNBUFFERED"] = unbuffered
    try:
        r = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, stdout=write_end,
                           stderr=subprocess.PIPE, env=env)
    finally:
        os.close(write_end)
    assert r.returncode == 2 and b"not been reviewed" in r.stderr   # exit 2 blocks the tool call


def test_hostile_environment_and_payloads_refuse(repo):
    cmd = json.dumps({"cwd": str(repo), "tool_input": {"command": f"{COMMIT} -m x"}}).encode()
    assert raw_hook(cmd, {"GIT_TRACE": b"invalid\xff"}) == "deny"     # git's warning is not valid UTF-8
    assert raw_hook(cmd, {"PATH": "/usr/share/doc"}) == "deny"
    assert raw_hook(json.dumps({"cwd": 1, "tool_input": {"command": "git 'com'mit -a -m x"}}).encode()) == "deny"
    assert raw_hook(b'{"tool_input": {"command": "git commit \xff"}}', {"PYTHONIOENCODING": "utf-8:strict"}) == "deny"
    assert raw_hook(json.dumps({"cwd": str(repo), "tool_input": {"command": "ls"}}).encode(),
                    {"PATH": "/usr/share/doc"}) == "allow"


def test_shell_script_after_double_dash_is_executed_text(repo):
    assert hook("bash -c -- \"git -c alias.ci='commit' ci -m x\"", repo) == "deny"
    assert hook("bash -c -- 'git ci -m x'", repo) == "deny"


def test_other_commands_and_other_repos_pass(repo, tmp_path):
    assert hook("git status", repo) == "allow"
    assert hook("git commit-tree -m x 4b825dc642cb6eb9a060e54bf8d69288fbee4904", repo) == "allow"
    other = tmp_path / "other"
    other.mkdir()
    run(["git", "init", "-q"], other)
    assert hook(f"{COMMIT} -m x", other) == "allow"


def test_alternation_per_commit(repo):
    assert approve(repo, "codex").returncode == 1           # fable is assigned first
    assert approve(repo, "fable").returncode == 0
    assert approve(repo, "fable").returncode == 0           # re-review before the same commit: still fable
    run(["git", "commit", "-q", "-m", "x"], repo)
    (repo / "b.txt").write_text("b\n")
    run(["git", "add", "-A"], repo)
    assert approve(repo, "fable").returncode == 1           # next commit: codex
    assert approve(repo, "fable", "--backup").returncode == 0
    assert approve(repo, "codex", "--note").returncode == 1  # --note without a value is refused
