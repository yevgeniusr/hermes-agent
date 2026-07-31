"""Read-only CLI adapter for local skill topology routing and audits."""

from __future__ import annotations

import json
from typing import Any

from agent.skill_topology import audit_topology, plan_skill_route


def _load_skill_records(*, include_disabled: bool) -> list[dict[str, Any]]:
    """Load rich records through the canonical skill scanner."""
    from tools.skills_tool import _find_all_skills

    return _find_all_skills(
        skip_disabled=include_disabled,
        include_topology=True,
        include_ineligible=include_disabled,
    )


def _print_json(artifact: dict[str, Any]) -> None:
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


def _print_diagnostics(diagnostics: list[dict[str, Any]]) -> None:
    if not diagnostics:
        return
    print("Diagnostics:")
    for diagnostic in diagnostics:
        print(f"  - [{diagnostic['code']}] {diagnostic['message']}")


def _print_route(artifact: dict[str, Any]) -> None:
    print(f"Skill route: {artifact['status']}")
    print(f"Query digest: {artifact['query_digest']}")
    print(
        f"Budget: {artifact['total_cost_chars']}/"
        f"{artifact['limits']['budget_chars']} characters; "
        f"{len(artifact['route'])}/{artifact['limits']['max_skills']} skills"
    )
    for index, item in enumerate(artifact["route"], start=1):
        print(
            f"{index}. {item['name']} [{item['graph_role']}] "
            f"score={item['score']} cost={item['cost_chars']} "
            f"cumulative={item['cumulative_cost_chars']}"
        )
        print(f"   Why: {', '.join(item['reasons'])}")
    _print_diagnostics(artifact["diagnostics"])


def _print_audit(artifact: dict[str, Any]) -> None:
    summary = artifact["summary"]
    counts = summary["lifecycle_counts"]
    print(f"Skill topology: {artifact['status']}")
    print(
        "Manifest coverage: "
        f"{summary['manifests_declared']}/{summary['skill_count']} "
        f"({summary['manifest_coverage_percent']:.2f}%)"
    )
    print(
        "Lifecycle: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(
        f"Graph findings: missing/self/invalid diagnostics="
        f"{sum(1 for item in artifact['diagnostics'] if item['code'] not in {'dependency_cycle', 'conflict'})}, "
        f"cycles={len(artifact['cycles'])}, conflicts={len(artifact['conflicts'])}"
    )
    _print_diagnostics(artifact["diagnostics"])


def skills_topology_command(args) -> dict[str, Any]:
    """Dispatch ``hermes skills route|topology`` and return its artifact."""
    action = getattr(args, "skills_action", None)
    if action == "route":
        query = " ".join(args.query)
        records = _load_skill_records(include_disabled=False)
        artifact = plan_skill_route(
            records,
            query,
            max_skills=args.limit,
            budget_chars=args.budget_chars,
        )
        if args.json:
            _print_json(artifact)
        else:
            _print_route(artifact)
        return artifact

    if action == "topology":
        records = _load_skill_records(include_disabled=True)
        artifact = audit_topology(records)
        if args.json:
            _print_json(artifact)
        else:
            _print_audit(artifact)
        return artifact

    raise ValueError(f"Unsupported local skills action: {action}")


__all__ = ["skills_topology_command"]
