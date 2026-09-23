"""Agent registry: the ordered mapping from stage name to agent class.

`AGENT_REGISTRY` is the single source of truth for pipeline order. The graph
builder, the CLI `stages` command and the `--resume-from` logic all read it.
"""

from __future__ import annotations

from doc_extractor.agents.attribute_extraction import AttributeExtractionAgent
from doc_extractor.agents.attribute_mapping import AttributeMappingAgent
from doc_extractor.agents.attribute_storage import AttributeStorageAgent
from doc_extractor.agents.base import BaseAgent
from doc_extractor.agents.chunking import ChunkingAgent
from doc_extractor.agents.document_processing import DocumentProcessingAgent
from doc_extractor.agents.entity_generation import EntityGenerationAgent
from doc_extractor.agents.entity_normalization import EntityNormalizationAgent
from doc_extractor.agents.entity_reviewer import EntityReviewerAgent
from doc_extractor.agents.final_output import FinalOutputAgent
from doc_extractor.schemas.state import STAGE_ORDER, StageName

AGENT_REGISTRY: dict[StageName, type[BaseAgent]] = {
    StageName.document_processing: DocumentProcessingAgent,
    StageName.chunking: ChunkingAgent,
    StageName.attribute_extraction: AttributeExtractionAgent,
    StageName.attribute_storage: AttributeStorageAgent,
    StageName.entity_generation: EntityGenerationAgent,
    StageName.entity_normalization: EntityNormalizationAgent,
    StageName.entity_reviewer: EntityReviewerAgent,
    StageName.attribute_mapping: AttributeMappingAgent,
    StageName.final_output: FinalOutputAgent,
}

assert tuple(AGENT_REGISTRY) == STAGE_ORDER, "AGENT_REGISTRY must follow STAGE_ORDER"
for _stage, _cls in AGENT_REGISTRY.items():
    assert _cls.name == _stage, f"{_cls.__name__}.name != {_stage}"

__all__ = [
    "AGENT_REGISTRY",
    "AttributeExtractionAgent",
    "AttributeMappingAgent",
    "AttributeStorageAgent",
    "BaseAgent",
    "ChunkingAgent",
    "DocumentProcessingAgent",
    "EntityGenerationAgent",
    "EntityNormalizationAgent",
    "EntityReviewerAgent",
    "FinalOutputAgent",
]
