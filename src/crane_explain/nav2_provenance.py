"""Build bounded provenance from Nav2 BT transition evidence and an exact retained tree."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

from .models import (
    EpisodeRecord,
    ProvenanceBundle,
    ProvenanceRelationship,
    ProvenanceStrength,
    RuntimeSourceLink,
    SourceAnchor,
    SourceArtifact,
)
from .provenance import ProvenanceValidationError


def _node_identity(node: ET.Element) -> str:
    return node.attrib.get("name") or node.tag


def _locator(node: ET.Element, parents: dict[ET.Element, ET.Element]) -> str:
    parts: list[str] = []
    current: ET.Element | None = node
    while current is not None:
        identity = _node_identity(current)
        parts.append(
            f"{current.tag}[@name='{identity}']" if "name" in current.attrib else current.tag
        )
        current = parents.get(current)
    return "/" + "/".join(reversed(parts))


def _ancestors(node: ET.Element, parents: dict[ET.Element, ET.Element]) -> list[ET.Element]:
    result = [node]
    while result[-1] in parents:
        result.append(parents[result[-1]])
    return result


def _smallest_common_ancestor(
    nodes: list[ET.Element], parents: dict[ET.Element, ET.Element]
) -> ET.Element:
    ancestor_sets = [set(_ancestors(node, parents)) for node in nodes]
    for candidate in _ancestors(nodes[0], parents):
        if all(candidate in values for values in ancestor_sets[1:]):
            return candidate
    raise ProvenanceValidationError("BT nodes do not share a common XML ancestor")


def build_behavior_tree_provenance(
    episode: EpisodeRecord,
    bt_xml: str,
    artifact: SourceArtifact,
) -> ProvenanceBundle:
    """Link named BT transitions to the smallest exact XML subtree containing them."""

    actual_sha256 = hashlib.sha256(bt_xml.encode("utf-8")).hexdigest()
    if actual_sha256 != artifact.content_sha256:
        raise ProvenanceValidationError("BT XML SHA-256 mismatch with retained source artifact")
    try:
        root = ET.fromstring(bt_xml)
    except ET.ParseError as exc:
        raise ProvenanceValidationError(f"invalid BT XML: {exc}") from exc

    runtime_by_name: dict[str, list[str]] = {}
    for evidence in episode.evidence:
        if evidence.kind != "bt_transition" or not isinstance(evidence.value, dict):
            continue
        name = evidence.value.get("node")
        if isinstance(name, str) and name:
            runtime_by_name.setdefault(name, []).append(evidence.id)
    if not runtime_by_name:
        raise ProvenanceValidationError("episode contains no named BT transition evidence")

    parents = {child: parent for parent in root.iter() for child in parent}
    matched: list[ET.Element] = []
    for name in runtime_by_name:
        candidates = [node for node in root.iter() if _node_identity(node) == name]
        if len(candidates) != 1:
            raise ProvenanceValidationError(
                f"BT runtime node {name!r} resolves to {len(candidates)} XML nodes"
            )
        matched.append(candidates[0])

    subtree = _smallest_common_ancestor(matched, parents)
    excerpt = ET.tostring(subtree, encoding="unicode", short_empty_elements=True)
    anchor = SourceAnchor(
        id=f"{artifact.id}-subtree-{_node_identity(subtree)}",
        artifact_id=artifact.id,
        locator=_locator(subtree, parents),
        excerpt=excerpt,
        excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        symbol=_node_identity(subtree),
        configuration_keys=tuple(
            f"{_node_identity(node)}.{key}"
            for node in matched
            for key in sorted(node.attrib)
            if key != "name"
        ),
    )
    link = RuntimeSourceLink(
        id=f"{episode.episode_id}-bt-transitions-to-{anchor.id}",
        runtime_evidence_ids=tuple(
            evidence_id for name in runtime_by_name for evidence_id in runtime_by_name[name]
        ),
        source_anchor_ids=(anchor.id,),
        relationship=ProvenanceRelationship.GOVERNED_BY,
        strength=ProvenanceStrength.EXACT_ARTIFACT,
        rationale=(
            "Each uniquely named runtime BT node resolves inside the retained BT XML whose "
            "content hash was verified before constructing this bounded subtree."
        ),
    )
    return ProvenanceBundle(
        schema_version="crane-explain-provenance/v1",
        artifacts=(artifact,),
        anchors=(anchor,),
        links=(link,),
    )
