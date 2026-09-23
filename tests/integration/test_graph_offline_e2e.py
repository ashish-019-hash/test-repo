"""End-to-end: compiled LangGraph on the MD, PDF and DOCX fixtures with the rules provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_extractor.config.models import AppConfig
from doc_extractor.schemas.state import STAGE_ORDER
from helpers_d import DETERMINISTIC_FILES, EXPECTED_DIR, FORMATS, read_json, run_fixture


def _by_name(items: list[dict], key: str) -> dict[str, dict]:
    return {i[key]: i for i in items}


@pytest.mark.parametrize("fmt", FORMATS)
def test_final_output_matches_golden(
    fmt: str, fixtures_dir: Path, tmp_path: Path, rules_cfg: AppConfig, update_golden: bool
) -> None:
    result = run_fixture(fixtures_dir, fmt, tmp_path / "out", rules_cfg)
    assert result.ok, result.error
    produced = (tmp_path / "out" / "final.json").read_text(encoding="utf-8")

    golden = EXPECTED_DIR / f"final.{fmt}.json"
    if update_golden or not golden.is_file():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(produced, encoding="utf-8")
    assert produced == golden.read_text(encoding="utf-8"), (
        f"final.json for {fmt} differs from golden; run `pytest --update-golden` if intended"
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_pipeline_semantics(fmt: str, fixtures_dir: Path, tmp_path: Path, rules_cfg: AppConfig) -> None:
    out = tmp_path / "out"
    result = run_fixture(fixtures_dir, fmt, out, rules_cfg)
    assert result.ok, result.error

    # every stage succeeded, in order, with a snapshot on disk
    stages = result.metadata.stages
    assert [s for s in stages] == [str(s) for s in STAGE_ORDER]
    assert all(rec["status"] == "succeeded" for rec in stages.values())
    for i, stage in enumerate(STAGE_ORDER, start=1):
        assert (out / "stages" / f"{i:02d}_{stage}.json").is_file()

    for name in DETERMINISTIC_FILES + ("run.json", "checkpoints.sqlite"):
        assert (out / name).exists(), name

    final = read_json(out / "final.json")
    attrs = _by_name(final["attributes"], "attribute_name")
    ents = _by_name(final["entities"], "canonical_name")
    ent_by_id = {e["canonical_id"]: e for e in final["entities"]}
    attr_by_id = {a["attribute_id"]: a for a in final["attributes"]}
    mapped = {
        (attr_by_id[m["attribute_id"]]["attribute_name"], ent_by_id[m["entity_id"]]["canonical_name"])
        for m in final["mappings"]
    }

    # CIR worked example: bound and mapped to canonical EVC (CFS layer)
    assert "cir" in attrs
    assert attrs["cir"]["binding"]["entity_name"] == "EVC"
    assert ("cir", "EVC") in mapped
    assert ents["EVC"]["layer"] == "CFS"

    # UNI: Resource-layer canonical entity with exactly its three table attributes mapped
    assert ents["UNI"]["layer"] == "Resource"
    uni_attrs = {a for a, e in mapped if e == "UNI"}
    assert uni_attrs == {"portSpeed", "physicalMedium", "macAddress"}

    # Order is a lifecycle subject, never an entity; UNI is promoted (not an attribute)
    assert "Order" not in ents
    assert "uni" not in attrs
    discarded = read_json(out / "discarded.json")
    assert any(d["attribute_name"] == "uni" and d["score"]["discard_reason"] == "promoted" for d in discarded)

    # Review queue: mid-band attributes routed to review, all grounded in real attributes
    queue = read_json(out / "review_queue.json")
    kinds = {q["kind"] for q in queue}
    assert "attribute_review" in kinds
    attr_ids = set(attr_by_id)
    for q in queue:
        if q["kind"] == "attribute_review":
            assert set(q["ref_ids"]) <= attr_ids

    # Every evidence string is grounded in its source chunk
    chunks = {c["chunk_id"]: c for c in read_json(out / "stages" / "02_chunking.json")["chunks"]}
    for a in final["attributes"]:
        for ev in a["evidence"]:
            assert ev["text"] in chunks[ev["chunk_id"]]["source_text"]

    # no timestamps / absolute paths leak into deterministic files
    for name in DETERMINISTIC_FILES:
        text = (out / name).read_text(encoding="utf-8")
        assert str(out) not in text, f"absolute path leaked into {name}"
        assert "started_at" not in text and "finished_at" not in text, name


def test_formats_agree_on_entities_and_mappings(
    fixtures_dir: Path, tmp_path: Path, rules_cfg: AppConfig
) -> None:
    summaries = {}
    for fmt in FORMATS:
        out = tmp_path / fmt
        assert run_fixture(fixtures_dir, fmt, out, rules_cfg).ok
        final = read_json(out / "final.json")
        ent_by_id = {e["canonical_id"]: e["canonical_name"] for e in final["entities"]}
        attr_by_id = {a["attribute_id"]: a["attribute_name"] for a in final["attributes"]}
        summaries[fmt] = (
            sorted(ent_by_id.values()),
            sorted(attr_by_id.values()),
            sorted((attr_by_id[m["attribute_id"]], ent_by_id[m["entity_id"]]) for m in final["mappings"]),
        )
    assert summaries["md"] == summaries["pdf"] == summaries["docx"]
