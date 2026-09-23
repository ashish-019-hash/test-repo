"""`build_graph`/`export_graph`: render a FinalOutput as a networkx graph.

`FinalOutput` carries a `DocumentSummary` (no chunk list) and `trace` edges built by the
Final Output Agent, so chunk nodes are recovered purely from the trace edges themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import networkx as nx
from networkx.readwrite import json_graph

from doc_extractor.schemas.output import FinalOutput
from doc_extractor.storage.canonical_json import dumps

_PREFIX_KIND: dict[str, str] = {
    "doc-": "document",
    "chunk-": "chunk",
    "attr-": "attribute",
    "ent-": "entity",
    "cent-": "entity",
    "map-": "mapping",
}


def _kind_for_id(node_id: str) -> str:
    for prefix, kind in _PREFIX_KIND.items():
        if node_id.startswith(prefix):
            return kind
    return "unknown"


def build_graph(final: FinalOutput) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_node(final.document.document_id, kind="document")
    for attr in final.attributes:
        graph.add_node(attr.attribute_id, kind="attribute")
    for entity in final.entities:
        graph.add_node(entity.canonical_id, kind="entity")
    for mapping in final.mappings:
        graph.add_node(mapping.mapping_id, kind="mapping")
    for edge in final.trace:
        if edge.from_id not in graph:
            graph.add_node(edge.from_id, kind=_kind_for_id(edge.from_id))
        if edge.to_id not in graph:
            graph.add_node(edge.to_id, kind=_kind_for_id(edge.to_id))
        graph.add_edge(edge.from_id, edge.to_id, kind=edge.kind)
    return graph


def export_graph(final: FinalOutput, path: str | Path, fmt: Literal["json", "graphml"]) -> Path:
    graph = build_graph(final)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "graphml":
        nx.write_graphml(graph, out_path)
    else:
        data = json_graph.node_link_data(graph)
        out_path.write_text(dumps(data), encoding="utf-8")
    return out_path
