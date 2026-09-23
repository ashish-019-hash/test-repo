"""Factory helpers for Group C agent/reviewer tests.

Builds a small, hand-crafted `PipelineState` mirroring `tests/fixtures/telecom_spec.md`:
an EVC spec table (CIR/EIR/EVC ID/Service Type/CoS Name), a UNI section (prose UNI
mention that gets promoted + a UNI attributes table), a Subscriber section
(MSISDN/IMSI via syntactic binding), one dangling attribute, and a Service Provider
mention used for alias/near-duplicate tests.
"""

from __future__ import annotations

from typing import Any

from doc_extractor.llm.base import LLMCallResult, LLMTask
from doc_extractor.schemas.attribute import Attribute, EntityBinding, ScoreBreakdown, SignalHit
from doc_extractor.schemas.chunk import Chunk
from doc_extractor.schemas.common import Evidence, Provenance
from doc_extractor.schemas.document import Block, Document, Page
from doc_extractor.schemas.entity import Entity
from doc_extractor.schemas.state import STAGE_ORDER, StageName, StageRecord

DOCUMENT_ID = "doc-c0ffee00c0ffee00"


class FakeProvider:
    """Minimal LLMProvider stand-in. Group C tests never exercise the azure branch."""

    def __init__(self, name: str = "rules") -> None:
        self.name = name

    def call(self, task: LLMTask, payload: dict[str, Any]) -> LLMCallResult:  # pragma: no cover
        raise AssertionError("FakeProvider.call should not be invoked in rule-based tests")


def make_document() -> Document:
    return Document(
        document_id=DOCUMENT_ID,
        file_name="telecom_spec.md",
        format="md",
        title="Carrier Ethernet Service Specification v2.1",
        metadata={},
        pages=[Page(page_number=1, text="", ocr_required=False, block_ids=[])],
        blocks=[Block(block_id=f"blk-{DOCUMENT_ID}-p1-0001", page=1, kind="paragraph", text="")],
        needs_ocr=False,
        text_coverage=1.0,
    )


def make_chunk(chunk_id: str, *, page: int, heading_path: list[str], source_text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        page_number=page,
        page_end=page,
        source_text=source_text,
        block_ids=[f"blk-{DOCUMENT_ID}-p{page}-0001"],
        heading_path=heading_path,
        token_count=max(1, len(source_text.split())),
        strategy="heading_aware",
    )


CHUNK_INTRO = make_chunk(
    f"chunk-{DOCUMENT_ID}-p0001-0001",
    page=1,
    heading_path=["Carrier Ethernet Service Specification v2.1", "1 Introduction"],
    source_text=(
        "This document specifies the service attributes of an Ethernet Virtual Connection "
        "(EVC) delivered by a Service Provider to a Subscriber. The EVC is an association of "
        "two or more UNIs that limits the exchange of Service Frames."
    ),
)

CHUNK_EVC_TABLE = make_chunk(
    f"chunk-{DOCUMENT_ID}-p0001-0002",
    page=1,
    heading_path=[
        "Carrier Ethernet Service Specification v2.1",
        "4 Service Attributes",
        "4.3 EVC Service Attributes",
    ],
    source_text=(
        "Attribute | Type | Units | M/O/C | Range | Default | Description\n"
        "CIR | Integer | Mbps | M | 10-1000 | 100 | Committed Information Rate for the EVC\n"
        "EIR | Integer | Mbps | O | 0-1000 | 0 | Excess Information Rate for the EVC\n"
        "EVC ID | String | | M | | | Unique identifier for the EVC assigned by the Service Provider\n"
        "Service Type | Enum | | M | Point-to-Point, Multipoint-to-Multipoint | Point-to-Point | "
        "Connectivity type of the EVC\n"
        "CoS Name | String | | O | | Standard | Class of Service name applied to the EVC"
    ),
)

CHUNK_UNI_PROSE = make_chunk(
    f"chunk-{DOCUMENT_ID}-p0001-0003",
    page=1,
    heading_path=["Carrier Ethernet Service Specification v2.1", "5 UNI"],
    source_text=(
        "The UNI has a port speed of 1 Gbps. Each UNI SHALL have a physical medium and a MAC "
        "address. There are 12 such UNIs in the reference deployment."
    ),
)

CHUNK_UNI_TABLE = make_chunk(
    f"chunk-{DOCUMENT_ID}-p0001-0004",
    page=1,
    heading_path=["Carrier Ethernet Service Specification v2.1", "5 UNI", "5.1 UNI Attributes"],
    source_text=(
        "Attribute | Type | Units | M/O/C | Range | Default | Description\n"
        "Port Speed | Integer | Mbps | M | 10, 100, 1000, 10000 | 1000 | Physical port speed of the UNI\n"
        "Physical Medium | Enum | | M | Copper, Fibre | Fibre | Physical medium of the UNI\n"
        "MAC Address | String | | M | | | MAC address of the UNI port"
    ),
)

CHUNK_SUBSCRIBER = make_chunk(
    f"chunk-{DOCUMENT_ID}-p0001-0005",
    page=1,
    heading_path=["Carrier Ethernet Service Specification v2.1", "6 Subscriber"],
    source_text=(
        "Each Subscriber SHALL have an IMSI and an MSISDN. The Subscriber's MSISDN is an E.164 "
        "number such as +447700900123. The Order is created, submitted and terminated by the "
        "operator. Provisioning of the service takes up to 5 working days."
    ),
)

CHUNK_SERVICE_PROVIDER = make_chunk(
    f"chunk-{DOCUMENT_ID}-p0001-0006",
    page=1,
    heading_path=["Carrier Ethernet Service Specification v2.1", "7 Service Provider"],
    source_text=(
        "The Service Provider allocates the EVC ID and maintains the Ethernet Virtual Connection "
        "for the duration of the contract."
    ),
)

ALL_CHUNKS: list[Chunk] = [
    CHUNK_INTRO,
    CHUNK_EVC_TABLE,
    CHUNK_UNI_PROSE,
    CHUNK_UNI_TABLE,
    CHUNK_SUBSCRIBER,
    CHUNK_SERVICE_PROVIDER,
]


def _score(clamped: float, route: str, discard_reason: str | None = None) -> ScoreBreakdown:
    return ScoreBreakdown(
        hits=[SignalHit(name="structural.spec_table_row", weight=0.45, evidence="table row")],
        raw_sum=clamped,
        clamped=clamped,
        short_circuit=clamped >= 1.0,
        route=route,  # type: ignore[arg-type]
        discard_reason=discard_reason,
    )


def make_attribute(
    *,
    attribute_id: str,
    attribute_name: str,
    display_name: str,
    entity: str | None,
    source_chunk: str,
    source_text: str,
    binding_evidence: str | None,
    scope: str = "table_subject",
    confidence: float = 1.0,
    layer: str | None = "CFS",
    route: str = "accept",
    discard_reason: str | None = None,
    attribute_type: str = "string",
) -> Attribute:
    return Attribute(
        attribute_id=attribute_id,
        attribute_name=attribute_name,
        display_name=display_name,
        aliases=[],
        attribute_type=attribute_type,  # type: ignore[arg-type]
        entity=entity,
        binding=EntityBinding(
            entity_name=entity,
            scope=scope,  # type: ignore[arg-type]
            structural_distance=0,
            confidence=1.0 if entity else 0.0,
            evidence=binding_evidence,
        ),
        layer=layer,  # type: ignore[arg-type]
        unit=None,
        cardinality=None,
        optionality="M",
        value_domain=None,
        default=None,
        description=None,
        source_document=DOCUMENT_ID,
        source_chunk=source_chunk,
        source_text=source_text,
        provenance=Provenance(file="telecom_spec.md", page=1, block_id=f"blk-{DOCUMENT_ID}-p1-0001"),
        evidence=[Evidence(chunk_id=source_chunk, text=source_text)],
        confidence=confidence,
        score=_score(confidence, route, discard_reason),
        origin="spec_table",
    )


def make_attributes() -> list[Attribute]:
    return [
        make_attribute(
            attribute_id="attr-cir",
            attribute_name="cir",
            display_name="CIR",
            entity="EVC",
            source_chunk=CHUNK_EVC_TABLE.chunk_id,
            source_text="Committed Information Rate for the EVC",
            binding_evidence="Committed Information Rate for the EVC",
            attribute_type="measure",
        ),
        make_attribute(
            attribute_id="attr-eir",
            attribute_name="eir",
            display_name="EIR",
            entity="EVC",
            source_chunk=CHUNK_EVC_TABLE.chunk_id,
            source_text="Excess Information Rate for the EVC",
            binding_evidence="Excess Information Rate for the EVC",
            attribute_type="measure",
        ),
        make_attribute(
            attribute_id="attr-evcid",
            attribute_name="evcId",
            display_name="EVC ID",
            entity="EVC",
            source_chunk=CHUNK_EVC_TABLE.chunk_id,
            source_text="Unique identifier for the EVC assigned by the Service Provider",
            binding_evidence="Unique identifier for the EVC assigned by the Service Provider",
            attribute_type="identifier",
        ),
        make_attribute(
            attribute_id="attr-servicetype",
            attribute_name="serviceType",
            display_name="Service Type",
            entity="EVC",
            source_chunk=CHUNK_EVC_TABLE.chunk_id,
            source_text="Connectivity type of the EVC",
            binding_evidence="Connectivity type of the EVC",
            attribute_type="enum",
        ),
        make_attribute(
            attribute_id="attr-cosname",
            attribute_name="cosName",
            display_name="CoS Name",
            entity="EVC",
            source_chunk=CHUNK_EVC_TABLE.chunk_id,
            source_text="Class of Service name applied to the EVC",
            binding_evidence="Class of Service name applied to the EVC",
            attribute_type="string",
        ),
        make_attribute(
            attribute_id="attr-portspeed",
            attribute_name="portSpeed",
            display_name="Port Speed",
            entity="UNI",
            source_chunk=CHUNK_UNI_TABLE.chunk_id,
            source_text="Physical port speed of the UNI",
            binding_evidence="Physical port speed of the UNI",
            layer="Resource",
            attribute_type="measure",
        ),
        make_attribute(
            attribute_id="attr-physicalmedium",
            attribute_name="physicalMedium",
            display_name="Physical Medium",
            entity="UNI",
            source_chunk=CHUNK_UNI_TABLE.chunk_id,
            source_text="Physical medium of the UNI",
            binding_evidence="Physical medium of the UNI",
            layer="Resource",
            attribute_type="enum",
        ),
        make_attribute(
            attribute_id="attr-macaddress",
            attribute_name="macAddress",
            display_name="MAC Address",
            entity="UNI",
            source_chunk=CHUNK_UNI_TABLE.chunk_id,
            source_text="MAC address of the UNI port",
            binding_evidence="MAC address of the UNI port",
            layer="Resource",
            attribute_type="identifier",
        ),
        make_attribute(
            attribute_id="attr-msisdn",
            attribute_name="msisdn",
            display_name="MSISDN",
            entity="Subscriber",
            source_chunk=CHUNK_SUBSCRIBER.chunk_id,
            source_text="Each Subscriber SHALL have an IMSI and an MSISDN.",
            binding_evidence="Each Subscriber SHALL have an IMSI and an MSISDN.",
            scope="syntactic",
            layer="Party",
            attribute_type="identifier",
        ),
        make_attribute(
            attribute_id="attr-imsi",
            attribute_name="imsi",
            display_name="IMSI",
            entity="Subscriber",
            source_chunk=CHUNK_SUBSCRIBER.chunk_id,
            source_text="Each Subscriber SHALL have an IMSI and an MSISDN.",
            binding_evidence="Each Subscriber SHALL have an IMSI and an MSISDN.",
            scope="syntactic",
            layer="Party",
            attribute_type="identifier",
        ),
        make_attribute(
            attribute_id="attr-provisioningtime",
            attribute_name="provisioningTime",
            display_name="Provisioning Time",
            entity=None,
            source_chunk=CHUNK_SUBSCRIBER.chunk_id,
            source_text="Provisioning of the service takes up to 5 working days.",
            binding_evidence=None,
            scope="dangling",
            layer=None,
            confidence=0.5,
            route="review",
            attribute_type="measure",
        ),
    ]


def make_promoted_uni_record() -> Attribute:
    return make_attribute(
        attribute_id="attr-uni-promoted",
        attribute_name="uni",
        display_name="UNI",
        entity=None,
        source_chunk=CHUNK_UNI_PROSE.chunk_id,
        source_text="The UNI has a port speed of 1 Gbps.",
        binding_evidence=None,
        scope="dangling",
        layer=None,
        confidence=0.0,
        route="discard",
        discard_reason="promoted",
    )


def stage_records_through(stage: StageName) -> dict[str, StageRecord]:
    """Predecessor StageRecords with status "succeeded" for every stage up to and
    including `stage` (exclusive of stages after it)."""
    records: dict[str, StageRecord] = {}
    for s in STAGE_ORDER:
        records[str(s)] = StageRecord(stage=s, status="succeeded", attempts=1)
        if s == stage:
            break
    return records


def base_state(predecessor: StageName, **overrides: Any) -> dict[str, Any]:
    """A PipelineState dict with `stages` populated through `predecessor` (succeeded)."""
    state: dict[str, Any] = {
        "input_path": "tests/fixtures/telecom_spec.md",
        "out_dir": "out",
        "document": make_document(),
        "chunks": list(ALL_CHUNKS),
        "attributes": make_attributes(),
        "discarded_attributes": [make_promoted_uni_record()],
        "stages": stage_records_through(predecessor),
        "traces": [],
        "human_review_queue": [],
    }
    state.update(overrides)
    return state


def entities_by_binding() -> list[Entity]:
    """Convenience: the entities EntityGenerationAgent should derive from `make_attributes()`."""
    from doc_extractor.agents.entity_generation import EntityGenerationAgent
    from doc_extractor.config import load_config

    cfg = load_config(env={"LLM_PROVIDER": "rules"})
    agent = EntityGenerationAgent(cfg, FakeProvider("rules"))
    state = base_state(StageName.attribute_extraction)
    state["stages"] = stage_records_through(StageName.attribute_storage)
    delta = agent.run(state)
    return delta["entities"]
