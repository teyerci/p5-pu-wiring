"""Tests for scripts/review_gate.py on a throwaway repository."""
import json
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
    out = subprocess.run([sys.executable, str(GATE), "hook"], input=payload, capture_output=True, text=True).stdout
    return "deny" if '"deny"' in out else "allow"


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
])
def test_bypasses_are_denied_even_when_approved(repo, cmd):
    assert approve(repo, "fable").returncode == 0
    assert hook(cmd, repo) == "deny"


def test_unresolvable_cd_falls_back_to_cwd(repo):
    assert hook(f'cd "$NO_SUCH_VAR_XYZ/nowhere" && {COMMIT} -m x', repo) == "deny"


def test_other_commands_and_other_repos_pass(repo, tmp_path):
    assert hook("git status", repo) == "allow"
    assert hook("git commit-tree HEAD^{tree}", repo) == "allow"
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
