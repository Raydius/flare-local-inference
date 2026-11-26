"""Release version forms accepted by the stamping script and by the tag guards.

A release is tagged ``vMAJOR.MINOR.PATCH`` or ``vMAJOR.MINOR.PATCH-rN``; the
version is the tag name without its ``v``. The stamping tests run the script on
three dependency-free crates that carry the release crates' package names.
"""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "set_release_version.py"
CRATES = {
    "flare-cli": "flare-cli",
    "flare-winui/bridge": "flare-winui-bridge",
    "flare-winui/launcher": "flare-launcher",
}
ACCEPTED = ("0.3.2-r2", "0.3.4")
REFUSED = ("0.3.2-r02", "0.3.2-rc1", "0.3.2-r", "0.3.2-r0", "0.03.2", "0.3.2-r2-r3")
GUARD_STEP = "Require a release tag name"
WORKFLOWS = ("container.yml", "release.yml")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    for directory, package in CRATES.items():
        crate = tmp_path / directory
        (crate / "src").mkdir(parents=True)
        (crate / "src" / "lib.rs").write_text("", encoding="utf-8")
        (crate / "Cargo.toml").write_text(
            f'[package]\nname = "{package}"\nversion = "0.0.0"\nedition = "2021"\n', encoding="utf-8"
        )
        (crate / "Cargo.lock").write_text(
            f'version = 4\n\n[[package]]\nname = "{package}"\nversion = "0.0.0"\n', encoding="utf-8"
        )
    return tmp_path


def run(tree: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(tree / "scripts" / SCRIPT.name), *arguments],
        capture_output=True, text=True, check=False,
    )


def versions(tree: Path) -> set[str]:
    found = set()
    for directory in CRATES:
        for name in ("Cargo.toml", "Cargo.lock"):
            text = (tree / directory / name).read_text(encoding="utf-8")
            found.update(line for line in text.splitlines() if line.startswith("version = "))
    return found - {"version = 4"}


@pytest.mark.parametrize("version", ACCEPTED)
def test_accepted_version_is_stamped_into_every_manifest_and_lockfile(tree: Path, version: str) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed; the script updates lockfiles with cargo update")
    result = run(tree, version)
    assert result.returncode == 0, result.stderr
    assert versions(tree) == {f'version = "{version}"'}
    assert run(tree, "--check", version).returncode == 0


@pytest.mark.parametrize("version", REFUSED)
@pytest.mark.parametrize("mode", ((), ("--check",)))
def test_refused_version_exits_nonzero_and_writes_no_manifest(
    tree: Path, version: str, mode: tuple[str, ...]
) -> None:
    result = run(tree, *mode, version)
    assert result.returncode != 0
    assert "MAJOR.MINOR.PATCH-rN" in result.stderr
    assert versions(tree) == {'version = "0.0.0"'}


def guard(workflow: str) -> str:
    document = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))
    steps = [
        step
        for job in document["jobs"].values()
        for step in job["steps"]
        if step.get("name") == GUARD_STEP
    ]
    assert len(steps) == 1, f"{workflow} must have one step named {GUARD_STEP!r}"
    return steps[0]["run"]


def run_guard(workflow: str, tag: str) -> int:
    return subprocess.run(
        ["bash", "-euc", guard(workflow)], env={"GITHUB_REF_NAME": tag, "PATH": "/usr/bin:/bin"},
        capture_output=True, check=False,
    ).returncode


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize("version", ACCEPTED)
def test_tag_guard_accepts_a_release_tag(workflow: str, version: str) -> None:
    assert run_guard(workflow, f"v{version}") == 0


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize("tag", [f"v{version}" for version in REFUSED] + ["0.3.2", "vv0.3.2", "v0.3.2-r2x"])
def test_tag_guard_refuses_any_other_tag(workflow: str, tag: str) -> None:
    assert run_guard(workflow, tag) == 1


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_tag_guard_runs_before_the_version_is_resolved(workflow: str) -> None:
    document = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))
    for job in document["jobs"].values():
        names = [step.get("name", "") for step in job["steps"]]
        if GUARD_STEP in names:
            resolve = next(index for index, name in enumerate(names) if name.startswith("Resolve the "))
            assert names.index(GUARD_STEP) < resolve
            return
    pytest.fail(f"{workflow} has no step named {GUARD_STEP!r}")
