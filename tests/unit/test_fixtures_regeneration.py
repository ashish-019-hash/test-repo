"""Regenerating the fixtures from `telecom_spec.md` must reproduce the committed files
byte-for-byte, so `scripts/make_fixtures.py` stays the single source of truth for them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_fixtures import generate  # noqa: E402


def test_regenerated_fixtures_match_committed(tmp_path: Path, fixtures_dir: Path) -> None:
    outputs = generate(fixtures_dir / "telecom_spec.md", tmp_path)
    for kind, path in outputs.items():
        committed = fixtures_dir / path.name
        assert committed.is_file(), f"missing committed fixture for {kind}: {committed}"
        assert path.read_bytes() == committed.read_bytes(), f"{kind} fixture is not byte-stable"


def test_generate_is_deterministic_across_runs(tmp_path: Path, fixtures_dir: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()
    outputs1 = generate(fixtures_dir / "telecom_spec.md", out1)
    outputs2 = generate(fixtures_dir / "telecom_spec.md", out2)
    for kind in outputs1:
        assert outputs1[kind].read_bytes() == outputs2[kind].read_bytes(), f"{kind} not deterministic"
