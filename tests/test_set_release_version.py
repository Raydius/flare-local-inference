# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the release version stamping against a scratch tree of the three release crates."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "scripts" / "set_release_version.py"
CRATES = (
    ("flare-cli", "flare-cli"),
    ("flare-winui/bridge", "flare-winui-bridge"),
    ("flare-winui/launcher", "flare-launcher"),
)
ACCEPTED_FORMS = (
    "version must be a stable MAJOR.MINOR.PATCH version "
    "or MAJOR.MINOR.PATCH-r<N> revision"
)
STAMPED_VERSIONS = ("0.3.2-r1", "0.3.4")
REJECTED = ("0.3.2-r01", "0.3.2-1", "0.3.2+R1", "release", "0.3.2-r", "0.3.2-r1-2")


def stamped_files(tree):
    """Every manifest and lockfile the stamping step owns."""
    return [
        path
        for directory, _ in CRATES
        for path in (
            tree / directory / "Cargo.toml",
            tree / directory / "Cargo.lock",
        )
    ]


def scratch_repository(tmp_path):
    """A repository-shaped tree with the release crates stamped at the unstamped version."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(STAMP, tmp_path / "scripts" / "set_release_version.py")
    for directory, package in CRATES:
        crate = tmp_path / directory
        (crate / "src").mkdir(parents=True)
        # Cargo refuses a manifest that names no target, so each crate needs a source file.
        (crate / "src" / "lib.rs").touch()
        (crate / "Cargo.toml").write_text(
            f'[package]\nname = "{package}"\nversion = "0.0.0"\n', encoding="utf-8"
        )
        (crate / "Cargo.lock").write_text(
            f'[[package]]\nname = "{package}"\nversion = "0.0.0"\n', encoding="utf-8"
        )
    return tmp_path


def stamp(tree, *args):
    return subprocess.run(
        [sys.executable, str(tree / "scripts" / "set_release_version.py"), *args],
        cwd=tree,
        capture_output=True,
        text=True,
        env={**os.environ, "CARGO_NET_OFFLINE": "true"},
    )


def contents(files):
    return {path: path.read_bytes() for path in files}


@pytest.mark.parametrize("version", STAMPED_VERSIONS)
def test_an_accepted_version_stamps_every_release_crate(tmp_path, version):
    tree = scratch_repository(tmp_path)

    result = stamp(tree, version)

    assert result.returncode == 0, result.stderr
    for path in stamped_files(tree):
        assert f'version = "{version}"' in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("version", REJECTED)
def test_malformed_version_is_refused_without_writing(tmp_path, version):
    tree = scratch_repository(tmp_path)
    files = stamped_files(tree)
    before = contents(files)

    result = stamp(tree, version)

    assert result.returncode != 0
    assert ACCEPTED_FORMS in result.stderr
    assert contents(files) == before


def test_checking_an_unstamped_tree_succeeds_without_writing(tmp_path):
    tree = scratch_repository(tmp_path)
    files = stamped_files(tree)
    before = contents(files)

    result = stamp(tree, "--check", "0.0.0")

    assert result.returncode == 0, result.stderr
    assert contents(files) == before
