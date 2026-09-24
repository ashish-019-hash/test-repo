from __future__ import annotations

from doc_extractor.schemas import Evidence, SignalHit
from doc_extractor.storage import ids
from doc_extractor.storage.canonical_json import dumps


def test_ids_are_stable_and_prefixed():
    assert ids.document_id(b"abc") == ids.document_id(b"abc")
    assert ids.document_id(b"abc").startswith("doc-") and len(ids.document_id(b"abc")) == 20
    assert ids.block_id("doc-x", 2, 7) == "blk-doc-x-p0002-0007"
    assert ids.chunk_id("doc-x", 12, 1) == "chunk-doc-x-p0012-0001"


def test_chunk_ids_sort_in_document_order_beyond_nine_pages():
    """Regression: unpadded pages made 'p10' sort before 'p2', failing chunking validation."""
    generated = [ids.chunk_id("doc-x", page, n) for page in range(1, 47) for n in range(3)]
    assert generated == sorted(generated)
    assert ids.canonical_id(["b", "a"]) == ids.canonical_id(["a", "b"])
    assert ids.attribute_id("d", "c", "cir", "t") != ids.attribute_id("d", "c", "eir", "t")


def test_dumps_sorts_keys_rounds_floats_and_sorts_record_lists():
    payload = {
        "z": 1.23456789,
        "a": [SignalHit(name="b", weight=0.2, evidence="x"), SignalHit(name="a", weight=0.1, evidence="y")],
        "rows": [["b", "a"], ["a", "b"]],  # positional data must not be reordered
        "ev": [Evidence(chunk_id="c2", text="t"), Evidence(chunk_id="c1", text="t")],
    }
    out = dumps(payload)
    assert out.endswith("\n")
    assert out.index('"a"') < out.index('"z"')
    assert "1.2346" in out
    assert out.index('"name": "a"') < out.index('"name": "b"')
    assert out.index('"chunk_id": "c1"') < out.index('"chunk_id": "c2"')
    assert '["b", "a"]' not in out  # indented form, but order preserved:
    assert out.index('"b",\n      "a"') < out.index('"a",\n      "b"')


def test_dumps_is_idempotent():
    payload = {"k": [{"entity_id": "b"}, {"entity_id": "a"}], "f": 0.1 + 0.2}
    assert dumps(payload) == dumps(payload)
