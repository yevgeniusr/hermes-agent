"""Pure parsing, auditing, and local route planning for skill topology.

This module deliberately has no profile, CLI, tool-registry, or execution
dependencies.  Callers supply metadata for skills that already passed Hermes'
normal platform, environment, disabled-skill, and permission gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence


TOPOLOGY_LIST_FIELDS = (
    "domains",
    "inputs",
    "outputs",
    "requires",
    "follows",
    "precedes",
    "conflicts",
    "permissions",
)
TOPOLOGY_FIELDS = (*TOPOLOGY_LIST_FIELDS, "lifecycle")
LIFECYCLES = ("experimental", "candidate", "stable", "deprecated")
REFERENCE_FIELDS = ("requires", "follows", "precedes", "conflicts")
ROUTE_ARTIFACT_VERSION = 1
DEFAULT_ROUTE_LIMIT = 5
DEFAULT_ROUTE_BUDGET_CHARS = 24_000

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_values(value: Any) -> tuple[str, ...]:
    """Normalize a scalar/list into stable, lowercase, de-duplicated values."""
    if value is None:
        return ()
    if isinstance(value, (set, frozenset)):
        values = sorted(value, key=lambda item: str(item).casefold())
    else:
        values = value if isinstance(value, (list, tuple)) else (value,)
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None or isinstance(item, Mapping):
            continue
        normalized = str(item).strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


@dataclass(frozen=True)
class SkillTopology:
    """Normalized optional ``metadata.hermes.topology`` manifest."""

    domains: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    follows: tuple[str, ...] = ()
    precedes: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    lifecycle: str | None = None
    declared: bool = False
    invalid_lifecycle: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return only the public V1 schema, with stable keys and list values."""
        return {
            "domains": list(self.domains),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "requires": list(self.requires),
            "follows": list(self.follows),
            "precedes": list(self.precedes),
            "conflicts": list(self.conflicts),
            "permissions": list(self.permissions),
            "lifecycle": self.lifecycle,
        }


def parse_topology(value: Any) -> SkillTopology:
    """Parse optional topology metadata without invalidating the skill.

    Unknown keys and malformed container values are ignored.  An invalid
    lifecycle is retained separately so graph audits can diagnose it while the
    containing SKILL.md remains loadable and backward compatible.
    """
    if isinstance(value, SkillTopology):
        return value
    if not isinstance(value, Mapping):
        return SkillTopology()

    normalized = {
        field: _normalize_values(value.get(field)) for field in TOPOLOGY_LIST_FIELDS
    }
    lifecycle_values = _normalize_values(value.get("lifecycle"))
    lifecycle_value = lifecycle_values[0] if lifecycle_values else None
    lifecycle = lifecycle_value if lifecycle_value in LIFECYCLES else None
    invalid_lifecycle = lifecycle_value if lifecycle_value and lifecycle is None else None
    return SkillTopology(
        **normalized,
        lifecycle=lifecycle,
        declared=True,
        invalid_lifecycle=invalid_lifecycle,
    )


def _record_topology(record: Mapping[str, Any]) -> SkillTopology:
    value = record.get("topology")
    return parse_topology(value)


def _record_name(record: Mapping[str, Any]) -> str:
    return str(record.get("name") or "").strip()


def _name_key(value: str) -> str:
    return value.strip().casefold()


def _index_records(
    skills: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, tuple[Mapping[str, Any], ...]],
]:
    """Index records by canonical name without choosing across collisions."""
    ordered = sorted(
        skills,
        key=lambda item: (
            _name_key(_record_name(item)),
            _record_name(item),
            str(item.get("category") or "").casefold(),
            str(item.get("category") or ""),
        ),
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in ordered:
        key = _name_key(_record_name(record))
        if key:
            grouped.setdefault(key, []).append(record)

    by_name: dict[str, Mapping[str, Any]] = {}
    collisions: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for key in sorted(grouped):
        records = grouped[key]
        if len(records) == 1:
            by_name[key] = records[0]
        else:
            collisions[key] = tuple(records)
    return ordered, by_name, collisions


def _diagnostic(code: str, message: str, **details: Any) -> dict[str, Any]:
    item = {"code": code, "message": message}
    item.update({key: value for key, value in details.items() if value is not None})
    return item


def _collision_diagnostic(
    canonical_name: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    names = sorted(
        (_record_name(record) for record in records),
        key=lambda name: (name.casefold(), name),
    )
    return _diagnostic(
        "canonical_name_collision",
        f"Canonical skill name '{canonical_name}' is ambiguous across "
        f"{len(records)} installed records.",
        canonical_name=canonical_name,
        names=names,
        record_count=len(records),
    )


def _sort_diagnostics(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        marker = repr(sorted(item.items(), key=lambda pair: pair[0]))
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
    return sorted(
        unique,
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("skill", "")),
            str(item.get("reference", "")),
            str(item.get("message", "")),
        ),
    )


def _canonical_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
    """Rotate a closed cycle to a deterministic representation."""
    nodes = list(cycle[:-1])
    if not nodes:
        return tuple(cycle)
    rotations = [nodes[index:] + nodes[:index] for index in range(len(nodes))]
    best = min(rotations, key=lambda values: tuple(_name_key(item) for item in values))
    return tuple([*best, best[0]])


def _dependency_cycles(
    ordered_records: Sequence[Mapping[str, Any]],
    by_name: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    cycles: set[tuple[str, ...]] = set()
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(key: str) -> None:
        if key in active_set:
            index = active.index(key)
            names = [_record_name(by_name[item]) for item in active[index:]]
            names.append(names[0])
            cycles.add(_canonical_cycle(names))
            return
        if key in visited:
            return
        visited.add(key)
        active.append(key)
        active_set.add(key)
        for reference in _record_topology(by_name[key]).requires:
            ref_key = _name_key(reference)
            if ref_key in by_name:
                visit(ref_key)
        active.pop()
        active_set.remove(key)

    for record in ordered_records:
        visit(_name_key(_record_name(record)))
    return [list(cycle) for cycle in sorted(cycles)]


def audit_topology(skills: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit a supplied installed-skill graph without executing any skill."""
    ordered, by_name, collisions = _index_records(skills)
    diagnostics = [
        _collision_diagnostic(key, records)
        for key, records in collisions.items()
    ]

    lifecycle_counts = {value: 0 for value in LIFECYCLES}
    lifecycle_counts.update({"unspecified": 0, "invalid": 0})
    manifests_declared = 0
    conflict_pairs: set[tuple[str, str]] = set()

    for record in ordered:
        name = _record_name(record)
        key = _name_key(name)
        topology = _record_topology(record)
        if topology.declared:
            manifests_declared += 1
        if topology.invalid_lifecycle:
            lifecycle_counts["invalid"] += 1
            diagnostics.append(
                _diagnostic(
                    "invalid_lifecycle",
                    f"Skill '{name}' declares unsupported lifecycle "
                    f"'{topology.invalid_lifecycle}'.",
                    skill=name,
                    value=topology.invalid_lifecycle,
                )
            )
        elif topology.lifecycle:
            lifecycle_counts[topology.lifecycle] += 1
        else:
            lifecycle_counts["unspecified"] += 1

        if key in collisions:
            continue
        for field in REFERENCE_FIELDS:
            for reference in getattr(topology, field):
                ref_key = _name_key(reference)
                if ref_key == key:
                    diagnostics.append(
                        _diagnostic(
                            "self_reference",
                            f"Skill '{name}' references itself in topology.{field}.",
                            skill=name,
                            field=field,
                            reference=reference,
                        )
                    )
                elif ref_key in collisions:
                    continue
                elif ref_key not in by_name:
                    diagnostics.append(
                        _diagnostic(
                            "missing_reference",
                            f"Skill '{name}' references missing skill '{reference}' "
                            f"in topology.{field}.",
                            skill=name,
                            field=field,
                            reference=reference,
                        )
                    )
                elif field == "conflicts":
                    other_name = _record_name(by_name[ref_key])
                    conflict_pairs.add(tuple(sorted((name, other_name), key=_name_key)))

    unique_records = [by_name[key] for key in sorted(by_name)]
    cycles = _dependency_cycles(unique_records, by_name)
    for cycle in cycles:
        diagnostics.append(
            _diagnostic(
                "dependency_cycle",
                f"Dependency cycle detected: {' -> '.join(cycle)}.",
                cycle=cycle,
            )
        )
    conflicts = [
        list(pair)
        for pair in sorted(
            conflict_pairs, key=lambda pair: tuple(map(_name_key, pair))
        )
    ]
    for left, right in conflicts:
        diagnostics.append(
            _diagnostic(
                "conflict",
                f"Skills '{left}' and '{right}' declare a conflict.",
                skills=[left, right],
            )
        )

    count = len(skills)
    coverage = round((manifests_declared / count * 100), 2) if count else 0.0
    sorted_diagnostics = _sort_diagnostics(diagnostics)
    return {
        "version": ROUTE_ARTIFACT_VERSION,
        "status": "issues" if sorted_diagnostics else "ok",
        "summary": {
            "skill_count": count,
            "manifests_declared": manifests_declared,
            "manifest_coverage_percent": coverage,
            "lifecycle_counts": lifecycle_counts,
        },
        "cycles": cycles,
        "conflicts": conflicts,
        "diagnostics": sorted_diagnostics,
    }


def _canonical_phrase(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.casefold()))


def _tokens(value: Any) -> set[str]:
    return set(_WORD_RE.findall(str(value or "").casefold()))


def _score_record(record: Mapping[str, Any], query: str) -> tuple[int, list[str]]:
    query_phrase = _canonical_phrase(query)
    query_tokens = _tokens(query)
    if not query_phrase or not query_tokens:
        return 0, []

    name = _record_name(record)
    topology = _record_topology(record)
    fields: list[tuple[str, Sequence[str], int, int]] = [
        ("tag", _normalize_values(record.get("tags")), 600, 80),
        ("domain", topology.domains, 500, 70),
        ("input", topology.inputs, 350, 60),
        ("output", topology.outputs, 300, 50),
        ("category", _normalize_values(record.get("category")), 200, 30),
    ]
    score = 0
    reasons: list[str] = []
    if _canonical_phrase(name) == query_phrase:
        score += 1000
        reasons.append("matched exact name")
    else:
        overlap = query_tokens & _tokens(name)
        if overlap:
            score += 100 * len(overlap)
            reasons.append("matched name terms")

    for label, values, exact_weight, token_weight in fields:
        exact_values = [
            value for value in values if _canonical_phrase(value) == query_phrase
        ]
        if exact_values:
            score += exact_weight
            reasons.append(f"matched exact {label}")
            continue
        overlap_values = [value for value in values if query_tokens & _tokens(value)]
        if overlap_values:
            overlap_count = len(
                set().union(
                    *(_tokens(value) & query_tokens for value in overlap_values)
                )
            )
            score += token_weight * overlap_count
            reasons.append(f"matched {label} terms")

    description_overlap = query_tokens & _tokens(record.get("description"))
    if description_overlap:
        score += 10 * len(description_overlap)
        reasons.append("matched description terms")
    return score, reasons


def _cost(record: Mapping[str, Any], field: str) -> int:
    try:
        return max(0, int(record.get(field) or 0))
    except (TypeError, ValueError):
        return 0


def _route_conflicts(
    keys: Sequence[str], by_name: Mapping[str, Mapping[str, Any]]
) -> list[list[str]]:
    pairs: set[tuple[str, str]] = set()
    selected_keys = set(keys)
    for key in keys:
        name = _record_name(by_name[key])
        topology = _record_topology(by_name[key])
        for reference in topology.conflicts:
            ref_key = _name_key(reference)
            if ref_key in selected_keys and ref_key != key:
                other = _record_name(by_name[ref_key])
                pairs.add(tuple(sorted((name, other), key=_name_key)))
    return [
        list(pair)
        for pair in sorted(pairs, key=lambda pair: tuple(map(_name_key, pair)))
    ]


def plan_skill_route(
    skills: Sequence[Mapping[str, Any]],
    query: str,
    *,
    max_skills: int,
    budget_chars: int,
) -> dict[str, Any]:
    """Select a deterministic, dependency-ordered local skill neighborhood."""
    digest = hashlib.sha256(str(query).encode("utf-8")).hexdigest()
    diagnostics: list[dict[str, Any]] = []
    if max_skills < 1 or budget_chars < 1:
        if max_skills < 1:
            diagnostics.append(_diagnostic("invalid_limit", "max_skills must be at least 1."))
        if budget_chars < 1:
            diagnostics.append(_diagnostic("invalid_budget", "budget_chars must be at least 1."))
        return {
            "version": ROUTE_ARTIFACT_VERSION,
            "status": "blocked",
            "query_digest": digest,
            "limits": {"max_skills": max_skills, "budget_chars": budget_chars},
            "route": [],
            "total_cost_chars": 0,
            "total_cost_bytes": 0,
            "diagnostics": _sort_diagnostics(diagnostics),
        }

    _, by_name, collisions = _index_records(skills)
    diagnostics.extend(
        _collision_diagnostic(key, records)
        for key, records in collisions.items()
    )

    ranked: list[tuple[int, str, str, Mapping[str, Any], list[str]]] = []
    for key in sorted(by_name):
        record = by_name[key]
        score, reasons = _score_record(record, str(query))
        if score:
            ranked.append(
                (
                    -score,
                    _name_key(_record_name(record)),
                    str(record.get("category") or "").casefold(),
                    record,
                    reasons,
                )
            )
    ranked.sort(key=lambda item: item[:3])

    ambiguous_match = any(
        _score_record(record, str(query))[0]
        for records in collisions.values()
        for record in records
    )
    if ambiguous_match:
        return {
            "version": ROUTE_ARTIFACT_VERSION,
            "status": "blocked",
            "query_digest": digest,
            "limits": {"max_skills": max_skills, "budget_chars": budget_chars},
            "route": [],
            "total_cost_chars": 0,
            "total_cost_bytes": 0,
            "diagnostics": _sort_diagnostics(diagnostics),
        }
    if not ranked:
        diagnostics.append(
            _diagnostic(
                "no_match", "No installed skill metadata matched the query."
            )
        )
        return {
            "version": ROUTE_ARTIFACT_VERSION,
            "status": "no_match",
            "query_digest": digest,
            "limits": {"max_skills": max_skills, "budget_chars": budget_chars},
            "route": [],
            "total_cost_chars": 0,
            "total_cost_bytes": 0,
            "diagnostics": _sort_diagnostics(diagnostics),
        }

    selected: list[str] = []
    selected_tier_score: int | None = None
    roles: dict[str, str] = {}
    scores: dict[str, int] = {}
    reasons_by_name: dict[str, list[str]] = {}

    def dependency_closure(
        root: Mapping[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        closure: list[str] = []
        blockers: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        visiting: list[str] = []
        done: set[str] = set()

        def visit(record: Mapping[str, Any]) -> None:
            name = _record_name(record)
            key = _name_key(name)
            if key in visiting:
                index = visiting.index(key)
                cycle_keys = visiting[index:] + [key]
                cycle = [_record_name(by_name[item]) for item in cycle_keys]
                blockers.append(
                    _diagnostic(
                        "dependency_cycle",
                        f"Dependency cycle blocks skill '{_record_name(root)}': {' -> '.join(cycle)}.",
                        skill=_record_name(root),
                        cycle=cycle,
                    )
                )
                return
            if key in done:
                return
            topology = _record_topology(record)
            if topology.invalid_lifecycle:
                blockers.append(
                    _diagnostic(
                        "invalid_lifecycle",
                        f"Skill '{name}' has invalid lifecycle '{topology.invalid_lifecycle}'.",
                        skill=name,
                        value=topology.invalid_lifecycle,
                    )
                )
                return
            visiting.append(key)
            for field in ("follows", "precedes", "conflicts"):
                for reference in getattr(topology, field):
                    ref_key = _name_key(reference)
                    if ref_key == key:
                        blockers.append(
                            _diagnostic(
                                "self_reference",
                                f"Skill '{name}' references itself in "
                                f"topology.{field}.",
                                skill=name,
                                field=field,
                                reference=reference,
                            )
                        )
                    elif ref_key in collisions:
                        continue
                    elif ref_key not in by_name:
                        findings.append(
                            _diagnostic(
                                "missing_reference",
                                f"Skill '{name}' references missing skill "
                                f"'{reference}' in topology.{field}.",
                                skill=name,
                                field=field,
                                reference=reference,
                            )
                        )
            for reference in topology.requires:
                ref_key = _name_key(reference)
                if ref_key == key:
                    blockers.append(
                        _diagnostic(
                            "self_reference",
                            f"Skill '{name}' requires itself.",
                            skill=name,
                            reference=reference,
                        )
                    )
                    continue
                dependency = by_name.get(ref_key)
                if ref_key in collisions:
                    blockers.append(
                        _collision_diagnostic(ref_key, collisions[ref_key])
                    )
                    continue
                if dependency is None:
                    blockers.append(
                        _diagnostic(
                            "missing_required_skill",
                            f"Skill '{name}' requires missing skill '{reference}'.",
                            skill=name,
                            reference=reference,
                        )
                    )
                    continue
                visit(dependency)
            visiting.pop()
            done.add(key)
            closure.append(key)

        visit(root)
        if blockers:
            return [], blockers, [*findings, *blockers]
        conflicts = _route_conflicts(closure, by_name)
        for left, right in conflicts:
            blockers.append(
                _diagnostic(
                    "route_conflict",
                    f"Route for '{_record_name(root)}' contains conflicting "
                    f"skills '{left}' and '{right}'.",
                    skill=_record_name(root),
                    skills=[left, right],
                )
            )
        return (
            [] if blockers else closure,
            blockers,
            [*findings, *blockers],
        )

    for negative_score, _, _, root, root_reasons in ranked:
        root_score = -negative_score
        if selected_tier_score is not None and root_score != selected_tier_score:
            break
        root_name = _record_name(root)
        root_key = _name_key(root_name)
        if root_key in selected:
            roles[root_key] = "root"
            scores[root_key] = root_score
            reasons_by_name[root_key] = root_reasons
            continue

        closure, blockers, findings = dependency_closure(root)
        diagnostics.extend(findings)
        if blockers:
            continue
        new_keys = [key for key in closure if key not in selected]
        combined_keys = [*selected, *new_keys]
        existing_conflicts = _route_conflicts(combined_keys, by_name)
        if existing_conflicts:
            for left, right in existing_conflicts:
                diagnostics.append(
                    _diagnostic(
                        "route_conflict",
                        f"Omitted '{root_name}' because it conflicts with the selected route.",
                        skill=root_name,
                        skills=[left, right],
                    )
                )
            continue
        if len(combined_keys) > max_skills:
            diagnostics.append(
                _diagnostic(
                    "limit_omission",
                    f"Omitted '{root_name}' because its dependency closure exceeds the skill limit.",
                    skill=root_name,
                    required_skill_count=len(combined_keys),
                )
            )
            continue
        projected_cost = sum(
            _cost(by_name[key], "cost_chars") for key in combined_keys
        )
        if projected_cost > budget_chars:
            diagnostics.append(
                _diagnostic(
                    "budget_omission",
                    f"Omitted '{root_name}' because its dependency closure "
                    "exceeds the character budget.",
                    skill=root_name,
                    required_cost_chars=projected_cost,
                )
            )
            continue

        selected_tier_score = root_score
        selected.extend(new_keys)
        for key in closure:
            if key == root_key:
                roles[key] = "root"
                scores[key] = root_score
                reasons_by_name[key] = root_reasons
            elif key not in roles:
                roles[key] = "required"
                scores[key] = 0
                parents = sorted(
                    (
                        candidate
                        for candidate in closure
                        if key
                        in {
                            _name_key(reference)
                            for reference in _record_topology(
                                by_name[candidate]
                            ).requires
                        }
                    ),
                    key=lambda candidate: (
                        _name_key(_record_name(by_name[candidate])),
                        _record_name(by_name[candidate]),
                    ),
                )
                reasons_by_name[key] = [
                    f"required by {_record_name(by_name[parent])}"
                    for parent in parents
                ]

    route: list[dict[str, Any]] = []
    cumulative_chars = 0
    cumulative_bytes = 0
    for key in selected:
        record = by_name[key]
        name = _record_name(record)
        cost_chars = _cost(record, "cost_chars")
        cost_bytes = _cost(record, "cost_bytes")
        cumulative_chars += cost_chars
        cumulative_bytes += cost_bytes
        topology = _record_topology(record)
        route.append(
            {
                "name": name,
                "category": record.get("category"),
                "graph_role": roles[key],
                "score": scores[key],
                "reasons": reasons_by_name[key],
                "cost_chars": cost_chars,
                "cost_bytes": cost_bytes,
                "cumulative_cost_chars": cumulative_chars,
                "cumulative_cost_bytes": cumulative_bytes,
                "topology": topology.to_dict() if topology.declared else {},
            }
        )

    status = "ok" if route else "blocked"
    return {
        "version": ROUTE_ARTIFACT_VERSION,
        "status": status,
        "query_digest": digest,
        "limits": {"max_skills": max_skills, "budget_chars": budget_chars},
        "route": route,
        "total_cost_chars": cumulative_chars,
        "total_cost_bytes": cumulative_bytes,
        "diagnostics": _sort_diagnostics(diagnostics),
    }


__all__ = [
    "DEFAULT_ROUTE_BUDGET_CHARS",
    "DEFAULT_ROUTE_LIMIT",
    "LIFECYCLES",
    "ROUTE_ARTIFACT_VERSION",
    "SkillTopology",
    "TOPOLOGY_FIELDS",
    "audit_topology",
    "parse_topology",
    "plan_skill_route",
]
