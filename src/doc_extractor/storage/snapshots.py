"""Per-stage pipeline state snapshots.

Written to ``out_dir/stages/NN_<stage>.json`` after every successful agent
node so ``--resume-from`` can reload the state as of a given stage without
rerunning earlier stages. Snapshots use the same canonical JSON format as
every other deterministic output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from doc_extractor.schemas.attribute import Attribute
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.document import Document
from doc_extractor.schemas.entity import CanonicalEntity, Entity, NormalizedEntity
from doc_extractor.schemas.mapping import Mapping
from doc_extractor.schemas.output import FinalOutput
from doc_extractor.schemas.review import DuplicateGroup, ReviewDecision
from doc_extractor.schemas.state import HumanReviewItem, PipelineState, StageRecord, TraceRecord
from doc_extractor.storage.canonical_json import dumps, loads

# Keys holding a single model instance.
_SINGLE_MODEL: dict[str, type[BaseModel]] = {
    "document": Document,
    "final_output": FinalOutput,
}

# Keys holding a list of model instances.
_LIST_MODEL: dict[str, type[BaseModel]] = {
    "chunks": Chunk,
    "attributes": Attribute,
    "discarded_attributes": Attribute,
    "entities": Entity,
    "normalized_entities": NormalizedEntity,
    "duplicate_groups": DuplicateGroup,
    "canonical_entities": CanonicalEntity,
    "review_decisions": ReviewDecision,
    "mappings": Mapping,
    "traces": TraceRecord,
    "human_review_queue": HumanReviewItem,
}

# Keys holding a dict[str, model instance].
_DICT_MODEL: dict[str, type[BaseModel]] = {
    "stages": StageRecord,
}


def snapshot_path(out_dir: str | Path, index: int, stage: str) -> Path:
    return Path(out_dir) / "stages" / f"{index:02d}_{stage}.json"


def save_snapshot(out_dir: str | Path, index: int, stage: str, state: PipelineState) -> Path:
    """Write the full pipeline state as of the successful completion of `stage`."""
    path = snapshot_path(out_dir, index, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(dict(state)), encoding="utf-8")
    return path


def load_snapshot(path: str | Path) -> PipelineState:
    """Read a snapshot back, reconstructing typed model instances for known keys."""
    raw: dict[str, Any] = loads(Path(path).read_text(encoding="utf-8"))
    state: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SINGLE_MODEL:
            state[key] = _SINGLE_MODEL[key].model_validate(value)
        elif key in _LIST_MODEL:
            model = _LIST_MODEL[key]
            state[key] = [model.model_validate(v) for v in value]
        elif key in _DICT_MODEL:
            model = _DICT_MODEL[key]
            state[key] = {k: model.model_validate(v) for k, v in value.items()}
        else:
            state[key] = value
    return cast(PipelineState, state)


__all__ = ["load_snapshot", "save_snapshot", "snapshot_path"]
