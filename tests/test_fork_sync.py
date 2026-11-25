# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the fork's mirror sync against throwaway upstream and fork repositories."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts" / "fork" / "sync.sh"


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Sync Tester",
         "-c", "user.email=sync-test@example.com", "-c", "commit.gpgsign=false",
         "-c", "tag.gpgsign=false", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def commit(repo, message):
    git(repo, "commit", "--quiet", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def test_sync_fast_forwards_main_when_a_fork_tag_shares_an_upstream_tag_name(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "--quiet", "--initial-branch=main")
    commit(upstream, "base")

    fork = tmp_path / "fork"
    git(tmp_path, "clone", "--quiet", "--origin", "upstream", str(upstream), str(fork))
    git(fork, "switch", "--quiet", "-c", "deploy")
    fork_release = commit(fork, "fork release")
    git(fork, "tag", "-a", "v0.2.3", "-m", "fork release", fork_release)

    upstream_tip = commit(upstream, "upstream release")
    git(upstream, "tag", "-a", "v0.2.3", "-m", "upstream release", upstream_tip)

    result = subprocess.run(
        ["bash", str(SYNC)], cwd=fork, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "NO_PUSH": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert git(fork, "rev-parse", "main") == upstream_tip
    assert git(fork, "rev-parse", "v0.2.3^{commit}") == fork_release
    assert "v0.2.3" in result.stderr


def test_sync_reports_fork_and_upstream_release_tags_apart(tmp_path):
    """Upstream `v0.3.2` and fork `v0.3.2-r1`/`v0.3.2-r10` are ranked in their own namespace."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "--quiet", "--initial-branch=main")
    base = commit(upstream, "base")

    fork = tmp_path / "fork"
    git(tmp_path, "clone", "--quiet", "--origin", "upstream", str(upstream), str(fork))
    git(fork, "switch", "--quiet", "-c", "deploy")
    git(fork, "tag", "base/current", base)

    upstream_tip = commit(upstream, "upstream release")
    git(upstream, "tag", "-a", "v0.3.2", "-m", "upstream release", upstream_tip)

    revision_1 = commit(fork, "fork release revision 1")
    git(fork, "tag", "-a", "v0.3.2-r1", "-m", "fork release revision 1", revision_1)
    revision_10 = commit(fork, "fork release revision 10")
    git(fork, "tag", "-a", "v0.3.2-r10", "-m", "fork release revision 10", revision_10)

    result = subprocess.run(
        ["bash", str(SYNC)], cwd=fork, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "NO_PUSH": "1"},
    )

    assert result.returncode == 0, result.stderr
    # A lexical ordering of the tag names would name r1 here, and would rank the
    # fork's own tags against upstream's bare version.
    assert "newest fork release v0.3.2-r10" in result.stdout
    assert "base/current is 1 commit(s) behind newest upstream release tag v0.3.2" in result.stdout
    assert git(fork, "rev-parse", "v0.3.2-r10^{commit}") == revision_10
