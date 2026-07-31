from __future__ import annotations

import hashlib
import json

import pytest

from agent.skill_topology import (
    TOPOLOGY_FIELDS,
    audit_topology,
    parse_topology,
    plan_skill_route,
)


def skill(
    name: str,
    *,
    description: str = "",
    category: str = "software-development",
    tags=(),
    topology=None,
    cost: int = 100,
):
    return {
        "name": name,
        "description": description,
        "category": category,
        "tags": list(tags),
        "topology": topology,
        "cost_chars": cost,
        "cost_bytes": cost,
    }


def diagnostic_codes(artifact):
    return [item["code"] for item in artifact["diagnostics"]]


def test_parse_topology_normalizes_scalars_lists_and_duplicates():
    topology = parse_topology(
        {
            "domains": " Testing ",
            "inputs": ["requirements", " requirements ", None, {"bad": "shape"}],
            "requires": "plan",
            "permissions": ["terminal", 7],
            "lifecycle": " STABLE ",
        }
    )

    assert topology.declared is True
    assert topology.domains == ("testing",)
    assert topology.inputs == ("requirements",)
    assert topology.requires == ("plan",)
    assert topology.permissions == ("terminal", "7")
    assert topology.lifecycle == "stable"
    assert set(topology.to_dict()) == set(TOPOLOGY_FIELDS)


def test_parse_topology_absent_is_valid_and_invalid_lifecycle_is_retained_for_audit():
    absent = parse_topology(None)
    invalid = parse_topology({"lifecycle": "preview"})

    assert absent.declared is False
    assert absent.lifecycle is None
    assert invalid.lifecycle is None
    assert invalid.invalid_lifecycle == "preview"


def test_exact_name_and_taxonomy_matches_beat_description_words():
    skills = [
        skill("testing", description="Loose testing helper"),
        skill("other", topology={"domains": ["testing"], "lifecycle": "stable"}),
        skill("third", tags=["testing"], description="Unrelated"),
    ]

    route = plan_skill_route(skills, "testing", max_skills=3, budget_chars=1000)

    assert [item["name"] for item in route["route"]] == ["testing", "third", "other"]
    assert route["route"][0]["score"] > route["route"][1]["score"]
    assert route["route"][1]["score"] > route["route"][2]["score"]


def test_ranking_ties_are_deterministic_by_name_then_category():
    skills = [
        skill("zeta", tags=["testing"]),
        skill("alpha", tags=["testing"]),
    ]

    first = plan_skill_route(skills, "testing", max_skills=2, budget_chars=1000)
    second = plan_skill_route(list(reversed(skills)), "testing", max_skills=2, budget_chars=1000)

    assert [item["name"] for item in first["route"]] == ["alpha", "zeta"]
    assert first == second


def test_transitive_requirements_are_ordered_before_the_matching_root():
    skills = [
        skill("plan", topology={"lifecycle": "stable"}),
        skill("tdd", topology={"requires": "plan", "lifecycle": "stable"}),
        skill("review", topology={"requires": "tdd", "lifecycle": "stable"}),
    ]

    result = plan_skill_route(skills, "review", max_skills=3, budget_chars=1000)

    assert result["status"] == "ok"
    assert [item["name"] for item in result["route"]] == ["plan", "tdd", "review"]
    assert [item["graph_role"] for item in result["route"]] == ["required", "required", "root"]
    assert result["route"][-1]["cumulative_cost_chars"] == 300


@pytest.mark.parametrize(
    ("topology", "expected_code"),
    [
        ({"requires": "missing"}, "missing_reference"),
        ({"requires": "root"}, "self_reference"),
        ({"lifecycle": "preview"}, "invalid_lifecycle"),
    ],
)
def test_audit_reports_manifest_errors(topology, expected_code):
    audit = audit_topology([skill("root", topology=topology)])

    assert expected_code in diagnostic_codes(audit)
    assert audit["status"] == "issues"


def test_audit_reports_dependency_cycles_and_declared_conflicts():
    skills = [
        skill("a", topology={"requires": "b", "conflicts": "b"}),
        skill("b", topology={"requires": "a"}),
    ]

    audit = audit_topology(skills)

    assert "dependency_cycle" in diagnostic_codes(audit)
    assert "conflict" in diagnostic_codes(audit)
    assert audit["cycles"] == [["a", "b", "a"]]
    assert audit["conflicts"] == [["a", "b"]]


def test_route_is_blocked_when_matching_skill_has_missing_requirement():
    result = plan_skill_route(
        [skill("review", topology={"requires": "missing"})],
        "review",
        max_skills=3,
        budget_chars=1000,
    )

    assert result["status"] == "blocked"
    assert result["route"] == []
    assert "missing_required_skill" in diagnostic_codes(result)


def test_route_reports_non_dependency_reference_faults_without_guessing():
    result = plan_skill_route(
        [
            skill(
                "review",
                topology={"follows": "missing", "precedes": "review"},
            )
        ],
        "review",
        max_skills=2,
        budget_chars=1000,
    )

    assert result["status"] == "blocked"
    assert result["route"] == []
    assert "missing_reference" in diagnostic_codes(result)
    assert "self_reference" in diagnostic_codes(result)


def test_route_is_blocked_by_dependency_cycle_and_conflict():
    cycle = plan_skill_route(
        [
            skill("a", topology={"requires": "b"}),
            skill("b", topology={"requires": "a"}),
        ],
        "a",
        max_skills=3,
        budget_chars=1000,
    )
    conflict = plan_skill_route(
        [
            skill("a", topology={"requires": "b", "conflicts": "b"}),
            skill("b"),
        ],
        "a",
        max_skills=3,
        budget_chars=1000,
    )

    assert cycle["status"] == "blocked"
    assert "dependency_cycle" in diagnostic_codes(cycle)
    assert conflict["status"] == "blocked"
    assert "route_conflict" in diagnostic_codes(conflict)


def test_budget_and_limit_omissions_are_explicit():
    skills = [
        skill("large", tags=["testing"], cost=500),
        skill("small", tags=["testing"], cost=80),
    ]

    budgeted = plan_skill_route(skills, "testing", max_skills=2, budget_chars=100)
    limited = plan_skill_route(
        [skill("root", topology={"requires": "dep"}), skill("dep")],
        "root",
        max_skills=1,
        budget_chars=1000,
    )

    assert [item["name"] for item in budgeted["route"]] == ["small"]
    assert "budget_omission" in diagnostic_codes(budgeted)
    assert limited["status"] == "blocked"
    assert "limit_omission" in diagnostic_codes(limited)


def test_no_match_is_explicit_and_json_artifact_never_contains_raw_query():
    query = "private phrase 8675309"
    result = plan_skill_route(
        [skill("testing", description="Write tests")],
        query,
        max_skills=3,
        budget_chars=1000,
    )

    encoded = json.dumps(result, sort_keys=True)
    assert result["status"] == "no_match"
    assert result["query_digest"] == hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert query not in encoded
    assert "8675309" not in encoded


def test_audit_reports_coverage_and_lifecycle_counts():
    audit = audit_topology(
        [
            skill("stable", topology={"lifecycle": "stable"}),
            skill("old", topology={"lifecycle": "deprecated"}),
            skill("plain"),
        ]
    )

    assert audit["summary"]["skill_count"] == 3
    assert audit["summary"]["manifests_declared"] == 2
    assert audit["summary"]["manifest_coverage_percent"] == pytest.approx(66.67)
    assert audit["summary"]["lifecycle_counts"] == {
        "experimental": 0,
        "candidate": 0,
        "stable": 1,
        "deprecated": 1,
        "unspecified": 1,
        "invalid": 0,
    }
