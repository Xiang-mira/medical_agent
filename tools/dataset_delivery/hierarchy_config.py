from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OPS = {"OR", "AND", "NOT"}


def _expr_nodes(expr: Any) -> set[str]:
    if isinstance(expr, str):
        return {expr}
    if isinstance(expr, dict):
        op = str(expr.get("op") or "").upper()
        args = expr.get("args")
        if op == "NOT":
            return _expr_nodes(args)
        nodes: set[str] = set()
        for item in args or []:
            nodes |= _expr_nodes(item)
        return nodes
    if isinstance(expr, list):
        nodes = set()
        for item in expr:
            nodes |= _expr_nodes(item)
        return nodes
    return set()


def _validate_expr(expr: Any, known_nodes: set[str], errors: list[dict[str, Any]], path: str) -> None:
    if isinstance(expr, str):
        if expr not in known_nodes:
            errors.append({"type": "unknown_node", "node": expr, "path": path})
        return
    if not isinstance(expr, dict):
        errors.append({"type": "invalid_expression", "path": path, "value": expr})
        return
    op = str(expr.get("op") or "").upper()
    if op not in OPS:
        errors.append({"type": "invalid_operator", "operator": op, "path": path})
    args = expr.get("args")
    if op == "NOT":
        if args is None:
            errors.append({"type": "invalid_expression", "path": path, "reason": "NOT requires args"})
        else:
            _validate_expr(args, known_nodes, errors, path + ".args")
        return
    if not isinstance(args, list) or len(args) < 1:
        errors.append({"type": "invalid_expression", "path": path, "reason": f"{op or 'expression'} requires non-empty args list"})
        return
    for idx, item in enumerate(args):
        _validate_expr(item, known_nodes, errors, f"{path}.args[{idx}]")


def validate_hierarchy_config(path: Path | None, *, known_nodes: set[str], required: bool = False) -> dict[str, Any]:
    if path is None or not path.exists():
        status = "BLOCKED" if required else "DISABLED"
        return {"status": status, "enabled": False, "reason": "hierarchy_config_missing", "path": str(path or "")}
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("nodes") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return {"status": "BLOCKED", "enabled": False, "errors": [{"type": "invalid_schema", "reason": "nodes list required"}], "path": str(path)}
    errors: list[dict[str, Any]] = []
    names = [str(row.get("id") or row.get("node") or "") for row in rows if isinstance(row, dict)]
    duplicates = sorted({name for name in names if name and names.count(name) > 1})
    for name in duplicates:
        errors.append({"type": "duplicate_node", "node": name})
    config_nodes = set(names)
    all_known = set(known_nodes) | config_nodes
    deps: dict[str, set[str]] = {}
    roots = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append({"type": "invalid_node", "node": row})
            continue
        node = str(row.get("id") or row.get("node") or "")
        if not node:
            errors.append({"type": "missing_node_id"})
            continue
        if node not in all_known:
            errors.append({"type": "unknown_node", "node": node})
        expr = row.get("expression")
        if expr is not None:
            _validate_expr(expr, all_known, errors, f"nodes.{node}.expression")
            deps[node] = _expr_nodes(expr) & config_nodes
        else:
            deps[node] = set()
        if bool(row.get("root")):
            roots.append(node)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        if node in visiting:
            errors.append({"type": "cycle_detected", "cycle": [*trail, node]})
            return
        if node in visited:
            return
        visiting.add(node)
        for child in deps.get(node, set()):
            visit(child, [*trail, node])
        visiting.remove(node)
        visited.add(node)

    for node in sorted(config_nodes):
        visit(node, [])
    status = "READY" if not errors else "BLOCKED"
    return {
        "status": status,
        "enabled": status == "READY",
        "path": str(path),
        "node_count": len(config_nodes),
        "root_count": len(roots),
        "roots": roots,
        "errors": errors,
        "operators": sorted(OPS),
    }
