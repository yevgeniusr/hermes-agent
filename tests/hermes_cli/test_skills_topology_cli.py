from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.skill_topology import parse_topology
from hermes_cli.subcommands.skills import build_skills_parser


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_skills_parser(subparsers, cmd_skills=lambda args: None)
    return parser


def _record(name, *, topology=None, tags=(), description="", cost=100):
    return {
        "name": name,
        "description": description,
        "category": "software-development",
        "tags": list(tags),
        "topology": parse_topology(topology),
        "cost_chars": cost,
        "cost_bytes": cost,
    }


def _write_skill(root: Path, directory: str, name: str) -> None:
    skill_dir = root / directory
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Description for {name}.\n"
        "metadata:\n"
        "  hermes:\n"
        "    topology:\n"
        "      lifecycle: stable\n"
        "---\n\n"
        f"# {name}\n",
        encoding="utf-8",
    )


def test_route_parser_accepts_multiword_query_and_budget_options():
    args = _parser().parse_args(
        [
            "skills",
            "route",
            "review",
            "this",
            "change",
            "--limit",
            "3",
            "--budget-chars",
            "9000",
            "--json",
        ]
    )

    assert args.skills_action == "route"
    assert args.query == ["review", "this", "change"]
    assert args.limit == 3
    assert args.budget_chars == 9000
    assert args.json is True


def test_topology_parser_is_read_only_and_supports_json():
    args = _parser().parse_args(["skills", "topology", "--json"])

    assert args.skills_action == "topology"
    assert args.json is True
    assert not hasattr(args, "deep")


def test_topology_loader_requests_the_full_installed_graph(monkeypatch):
    from hermes_cli import skills_topology
    from tools import skills_tool

    captured = {}

    def fake_find_all_skills(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(skills_tool, "_find_all_skills", fake_find_all_skills)

    skills_topology._load_skill_records(include_disabled=True)

    assert captured == {
        "skip_disabled": True,
        "include_topology": True,
        "include_ineligible": True,
    }


@pytest.mark.parametrize("external_name", ["review", "Review"])
def test_scanned_collision_blocks_cli_route_and_audit_deterministically(
    external_name, tmp_path, capsys
):
    from hermes_cli import skills_topology
    from tools import skills_tool

    local_dir = tmp_path / "local"
    external_a = tmp_path / "external-a"
    external_b = tmp_path / "external-b"
    local_dir.mkdir()
    external_a.mkdir()
    external_b.mkdir()
    _write_skill(local_dir, "local-review", "review")
    _write_skill(external_a, "external-review", external_name)
    _write_skill(external_b, "github-code-review", "github-code-review")
    _write_skill(
        external_b,
        "requesting-code-review",
        "requesting-code-review",
    )
    _write_skill(external_b, "deploy", "deploy")
    route_args = SimpleNamespace(
        skills_action="route",
        query=["review"],
        limit=5,
        budget_chars=10_000,
        json=True,
    )
    deploy_args = SimpleNamespace(
        skills_action="route",
        query=["deploy"],
        limit=5,
        budget_chars=10_000,
        json=True,
    )
    audit_args = SimpleNamespace(skills_action="topology", json=True)

    outputs = []
    artifacts = []
    for external_dirs in ([external_a, external_b], [external_b, external_a]):
        with (
            patch("tools.skills_tool.SKILLS_DIR", local_dir),
            patch(
                "agent.skill_utils.get_external_skills_dirs",
                return_value=external_dirs,
            ),
        ):
            skills_tool._SKILLS_CACHE.clear()
            route = skills_topology.skills_topology_command(route_args)
            route_output = capsys.readouterr().out
            deploy_route = skills_topology.skills_topology_command(deploy_args)
            deploy_output = capsys.readouterr().out
            audit = skills_topology.skills_topology_command(audit_args)
            audit_output = capsys.readouterr().out
        artifacts.append((route, deploy_route, audit))
        outputs.append((route_output, deploy_output, audit_output))

    assert outputs[0] == outputs[1]
    assert artifacts[0] == artifacts[1]
    route, deploy_route, audit = artifacts[0]
    assert route["status"] == "blocked"
    assert route["route"] == []
    assert route["total_cost_chars"] == 0
    assert route["total_cost_bytes"] == 0
    assert [item["code"] for item in route["diagnostics"]] == [
        "canonical_name_collision"
    ]
    assert audit["status"] == "issues"
    assert [item["code"] for item in audit["diagnostics"]] == [
        "canonical_name_collision"
    ]
    assert deploy_route["status"] == "ok"
    assert [item["name"] for item in deploy_route["route"]] == ["deploy"]
    for route_output, deploy_output, audit_output in outputs:
        for root in (local_dir, external_a, external_b):
            assert str(root) not in route_output
            assert str(root) not in deploy_output
            assert str(root) not in audit_output


@pytest.mark.parametrize(
    "argv",
    [
        ["skills", "route", "query", "--limit", "0"],
        ["skills", "route", "query", "--budget-chars", "-1"],
    ],
)
def test_route_parser_rejects_nonpositive_limits(argv):
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)


def test_route_json_is_stable_and_omits_raw_query(monkeypatch, capsys):
    from hermes_cli import skills_topology

    records = [
        _record("plan", topology={"lifecycle": "stable"}),
        _record(
            "review-private-8675309",
            topology={"requires": "plan", "lifecycle": "candidate"},
        ),
    ]
    monkeypatch.setattr(skills_topology, "_load_skill_records", lambda **kwargs: records)
    args = SimpleNamespace(
        skills_action="route",
        query=["review", "private-8675309"],
        limit=2,
        budget_chars=1000,
        json=True,
    )

    first = skills_topology.skills_topology_command(args)
    first_output = capsys.readouterr().out
    second = skills_topology.skills_topology_command(args)
    second_output = capsys.readouterr().out

    payload = json.loads(first_output)
    assert first == second
    assert first_output == second_output
    assert payload["query_digest"] == hashlib.sha256(
        b"review private-8675309"
    ).hexdigest()
    assert [item["name"] for item in payload["route"]] == [
        "plan",
        "review-private-8675309",
    ]
    assert "review private-8675309" not in first_output


def test_route_human_output_shows_order_reasons_and_budget(monkeypatch, capsys):
    from hermes_cli import skills_topology

    records = [
        _record("plan", cost=75),
        _record("review", topology={"requires": "plan"}, cost=125),
    ]
    monkeypatch.setattr(skills_topology, "_load_skill_records", lambda **kwargs: records)
    args = SimpleNamespace(
        skills_action="route",
        query=["review"],
        limit=2,
        budget_chars=500,
        json=False,
    )

    skills_topology.skills_topology_command(args)
    output = capsys.readouterr().out

    assert "1. plan [required]" in output
    assert "2. review [root]" in output
    assert "Why: required by review" in output
    assert "Budget: 200/500 characters" in output


def test_topology_json_reports_coverage_lifecycle_and_faults(monkeypatch, capsys):
    from hermes_cli import skills_topology

    records = [
        _record("a", topology={"requires": "missing", "lifecycle": "stable"}),
        _record("plain"),
    ]
    monkeypatch.setattr(skills_topology, "_load_skill_records", lambda **kwargs: records)
    args = SimpleNamespace(skills_action="topology", json=True)

    artifact = skills_topology.skills_topology_command(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload == artifact
    assert payload["summary"]["manifests_declared"] == 1
    assert payload["summary"]["lifecycle_counts"]["stable"] == 1
    assert payload["diagnostics"][0]["code"] == "missing_reference"


@pytest.mark.parametrize("action", ["route", "topology"])
def test_main_dispatches_local_topology_actions_before_remote_hub(
    action, monkeypatch
):
    import hermes_cli.main as main_module
    from hermes_cli import skills_hub, skills_topology

    calls = []
    monkeypatch.setattr(
        skills_topology,
        "skills_topology_command",
        lambda args: calls.append(("local", args.skills_action)),
    )
    monkeypatch.setattr(
        skills_hub,
        "skills_command",
        lambda args: calls.append(("hub", args.skills_action)),
    )

    main_module.cmd_skills(SimpleNamespace(skills_action=action))

    assert calls == [("local", action)]


@pytest.mark.parametrize(
    ("argv", "expected_status"),
    [
        (["skills", "route", "testing", "--json"], "ok"),
        (["skills", "topology", "--json"], "ok"),
    ],
)
def test_full_hermes_cli_invocation_parses_and_dispatches_topology_actions(
    argv, expected_status, monkeypatch, capsys
):
    import sys

    import hermes_cli.main as main_module
    from hermes_cli import skills_topology
    from hermes_cli import config as config_module

    monkeypatch.setattr(
        skills_topology,
        "_load_skill_records",
        lambda **kwargs: [_record("testing")],
    )
    monkeypatch.setattr(main_module, "_prepare_agent_startup", lambda args: None)
    monkeypatch.setattr(config_module, "get_container_exec_info", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])

    main_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status
