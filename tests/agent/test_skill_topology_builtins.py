from pathlib import Path

from agent.skill_topology import (
    DEFAULT_ROUTE_BUDGET_CHARS,
    audit_topology,
    parse_topology,
    plan_skill_route,
)
from agent.skill_utils import parse_frontmatter


REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN = (
    "plan",
    "test-driven-development",
    "systematic-debugging",
    "requesting-code-review",
)


def _load_chain():
    records = []
    for name in CHAIN:
        skill_md = REPO_ROOT / "skills" / "software-development" / name / "SKILL.md"
        raw = skill_md.read_bytes()
        content = raw.decode("utf-8")
        frontmatter, _ = parse_frontmatter(content)
        hermes = frontmatter["metadata"]["hermes"]
        records.append(
            {
                "name": frontmatter["name"],
                "description": frontmatter["description"],
                "category": "software-development",
                "tags": hermes["tags"],
                "topology": parse_topology(hermes.get("topology")),
                "cost_chars": len(content),
                "cost_bytes": len(raw),
            }
        )
    return records


def test_builtin_planning_tdd_debugging_review_topology_is_valid():
    records = _load_chain()

    audit = audit_topology(records)

    assert audit["status"] == "ok"
    assert audit["summary"]["manifests_declared"] == 4


def test_builtin_review_route_loads_tdd_first_without_plan_mode_instructions():
    records = _load_chain()
    route = plan_skill_route(
        records,
        "review",
        max_skills=2,
        budget_chars=DEFAULT_ROUTE_BUDGET_CHARS,
    )

    assert route["status"] == "ok"
    assert [item["name"] for item in route["route"]] == [
        "test-driven-development",
        "requesting-code-review",
    ]
    assert route["total_cost_chars"] <= DEFAULT_ROUTE_BUDGET_CHARS
