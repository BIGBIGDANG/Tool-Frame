#!/usr/bin/env python3
"""Restricted execution for PartToolGen Python planner programs."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

from ToGo.program.api import API_VERSION, ProgramContext


ENTRYPOINT = "build_program"
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
}
FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
)
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class PlannerProgramError(ValueError):
    """Raised when a Python planner program is unsafe or invalid."""


def extract_python_source(raw_text: str) -> str:
    """Extract Python source from raw text or a fenced code block."""

    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:python|py)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        raise PlannerProgramError("planner program is empty")
    return text


def validate_planner_program_source(source: str) -> ast.Module:
    """Validate a conservative AST subset and require `build_program(ctx)`."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PlannerProgramError(f"planner program is not valid Python: {exc}") from exc

    entrypoints = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == ENTRYPOINT
    ]
    if len(entrypoints) != 1:
        raise PlannerProgramError(f"program must define exactly one {ENTRYPOINT}(ctx)")
    args = entrypoints[0].args
    if len(args.args) != 1 or args.args[0].arg != "ctx":
        raise PlannerProgramError(f"{ENTRYPOINT} must take exactly one argument named ctx")

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise PlannerProgramError(f"forbidden Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise PlannerProgramError("dunder names are not allowed")
            if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_CALLS:
                raise PlannerProgramError(f"forbidden call/name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PlannerProgramError("dunder attributes are not allowed")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                raise PlannerProgramError(f"forbidden call: {func.id}")
    return tree


def make_planner_program_artifact(
    *,
    raw_text: str,
    program_id: str,
    task_instruction: str,
    generated_by: str,
    model: str | None = None,
    prompt_id: str | None = None,
    declared_symbol_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap raw Python source in the planner-program artifact envelope."""

    source = extract_python_source(raw_text)
    validate_planner_program_source(source)
    metadata: dict[str, Any] = {
        "generated_by": generated_by,
        "mode": "vlm_authored_python",
    }
    if model:
        metadata["model"] = model
    if prompt_id:
        metadata["prompt_id"] = prompt_id
    return {
        "schema_version": "v0.1",
        "program_id": program_id,
        "task_instruction": task_instruction,
        "language": "python",
        "entrypoint": ENTRYPOINT,
        "allowed_api_version": API_VERSION,
        "program_text": source,
        "declared_symbol_table": declared_symbol_table or {},
        "safety_contract": {
            "ast_checked": True,
            "restricted_builtins": True,
            "imports_allowed": False,
            "io_allowed": False,
            "network_allowed": False,
        },
        "generation_metadata": metadata,
    }


def execute_planner_program(
    *,
    program_text: str,
    context: ProgramContext,
) -> dict[str, Any]:
    """Execute a checked program and return the collected ProgramIR."""

    validate_planner_program_source(program_text)
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
    }
    exec(compile(program_text, "<parttoolgen_planner_program>", "exec"), namespace, namespace)
    entrypoint = namespace.get(ENTRYPOINT)
    if not callable(entrypoint):
        raise PlannerProgramError(f"{ENTRYPOINT} was not defined as a callable")
    result = entrypoint(context)
    if result is None:
        return context.program()
    if isinstance(result, dict):
        collected = context.program()
        if result == collected:
            return collected
        raise PlannerProgramError(f"{ENTRYPOINT} returned a dict not produced by ctx.program()")
    if result is context:
        return context.program()
    raise PlannerProgramError(f"{ENTRYPOINT} must return None, ctx, or ctx.program()")


def execute_planner_program_file(
    *,
    program_path: str | Path,
    context: ProgramContext,
) -> dict[str, Any]:
    return execute_planner_program(
        program_text=Path(program_path).read_text(encoding="utf-8"),
        context=context,
    )
