"""Agent 4: AttributeStorageAgent.

Re-validates `attributes` by round-tripping through JSON and writes
`out/attributes.json` (plus `out/discarded.json` for the discarded bucket) via the
canonical JSON writer. Alters no field: the
reloaded models must equal the input exactly. An empty attribute list is a
legitimate outcome (a document with no detected attributes), so
`validate_input` allows it while still requiring the key to be present and
the predecessor to have succeeded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from doc_extractor.agents.base import BaseAgent
from doc_extractor.exceptions import StageValidationError
from doc_extractor.observability.tracing import TraceCollector
from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.state import PipelineState, StageName
from doc_extractor.storage.canonical_json import dumps, loads
from doc_extractor.storage.json_store import write_models


class AttributeStorageAgent(BaseAgent):
    name: ClassVar[StageName] = StageName.attribute_storage
    requires: ClassVar[tuple[str, ...]] = ("attributes",)
    produces: ClassVar[tuple[str, ...]] = ("attributes_path",)
    allow_empty: ClassVar[frozenset[str]] = frozenset({"attributes"})

    def execute(self, state: PipelineState, trace: TraceCollector) -> dict[str, Any]:
        attributes: list[Attribute] = list(state["attributes"])
        out_dir = Path(state["out_dir"])
        path = write_models(out_dir, "attributes", attributes)
        discarded: list[Attribute] = list(state.get("discarded_attributes") or [])
        write_models(out_dir, "discarded", discarded)
        trace.count("attributes_written", len(attributes))
        trace.count("discarded_written", len(discarded))
        return {"attributes_path": str(path)}

    def validate_output(self, delta: dict[str, Any], state: PipelineState) -> None:
        super().validate_output(delta, state)
        attributes: list[Attribute] = list(state["attributes"])
        path = Path(delta["attributes_path"])
        if not path.is_file():
            raise StageValidationError(str(self.name), f"attributes file not written: {path}")

        raw = loads(path.read_text(encoding="utf-8"))
        reloaded = [Attribute.model_validate(item) for item in raw]
        if len(reloaded) != len(attributes):
            raise StageValidationError(str(self.name), "reloaded attribute count does not match input count")
        by_id = {a.attribute_id: a for a in attributes}
        for model in reloaded:
            original = by_id.get(model.attribute_id)
            if original is None:
                raise StageValidationError(
                    str(self.name), f"reloaded attribute {model.attribute_id} not in input"
                )
            if dumps(model) != dumps(original):
                raise StageValidationError(
                    str(self.name), f"attribute {model.attribute_id} changed during storage"
                )
