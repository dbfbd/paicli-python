"""Plan 和 Team 共用的计划结构与依赖校验。"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Any


def validate_plan_nodes(data: Any, *, primary_key: str, id_prefix: str) -> list[dict[str, Any]]:
    """在任一执行模式调度前校验整个 DAG，拒绝重复编号、缺失依赖和循环依赖。"""
    if not isinstance(data, dict):
        raise ValueError("plan JSON root must be an object")
    alternate_key = "steps" if primary_key == "tasks" else "tasks"
    nodes = data.get(primary_key) if primary_key in data else data.get(alternate_key)
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("plan must contain a non-empty tasks/steps array")

    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            raise ValueError(f"plan node {index} must be an object")
        node_id = node.get("id", f"{id_prefix}_{index}")
        if not isinstance(node_id, str) or not node_id or node_id != node_id.strip():
            raise ValueError(f"plan node {index} must have a non-empty string id")
        if node_id in ids:
            raise ValueError(f"duplicate plan node id: {node_id}")
        ids.add(node_id)
        dependencies = node.get("dependencies", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(dep, str) or not dep or dep != dep.strip() for dep in dependencies
        ):
            raise ValueError(f"dependencies for {node_id} must be an array of string ids")
        validated.append({**node, "id": node_id, "dependencies": list(dict.fromkeys(dependencies))})

    graph: dict[str, list[str]] = {}
    for node in validated:
        node_id, dependencies = node["id"], node["dependencies"]
        if node_id in dependencies:
            raise ValueError(f"plan node {node_id} cannot depend on itself")
        unknown = set(dependencies) - ids
        if unknown:
            raise ValueError(f"unknown dependencies for {node_id}: {', '.join(sorted(unknown))}")
        graph[node_id] = dependencies
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ValueError("plan contains a cyclic dependency") from exc
    return validated
