#!/usr/bin/env python3
"""Restricted runtime for whole-tool pose objective functions."""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Dict

import numpy as np

from parttoolgen.compiler.reference_resolver import ReferenceResolver, ReferenceResolutionError
from parttoolgen.compiler.residuals import normalize_quat, pose_axis, pose_point
from parttoolgen.program.sandbox import SAFE_BUILTINS, PlannerProgramError, validate_planner_program_source


class ObjectiveRuntimeError(ValueError):
    """Raised when an objective function cannot be loaded or evaluated."""


ObjectiveFunction = Callable[[Dict[str, Any], "ObjectivePoseRefs"], Any]


class _SafeLinalg:
    @staticmethod
    def norm(value: Any, ord: int | float | None = None, axis: int | None = None) -> Any:
        return np.linalg.norm(value, ord=ord, axis=axis)


class _SafeNP:
    """Small numeric facade for objective code; intentionally not full NumPy."""

    linalg = _SafeLinalg()
    pi = math.pi

    @staticmethod
    def array(value: Any, dtype: Any = float) -> np.ndarray:
        return np.array(value, dtype=dtype)

    @staticmethod
    def asarray(value: Any, dtype: Any = float) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def abs(value: Any) -> Any:
        return np.abs(value)

    @staticmethod
    def clip(value: Any, low: Any, high: Any) -> Any:
        return np.clip(value, low, high)

    @staticmethod
    def cross(a: Any, b: Any) -> np.ndarray:
        return np.cross(a, b)

    @staticmethod
    def dot(a: Any, b: Any) -> Any:
        return np.dot(a, b)

    @staticmethod
    def maximum(a: Any, b: Any) -> Any:
        return np.maximum(a, b)

    @staticmethod
    def minimum(a: Any, b: Any) -> Any:
        return np.minimum(a, b)

    @staticmethod
    def sqrt(value: Any) -> Any:
        return np.sqrt(value)

    @staticmethod
    def sum(value: Any, axis: int | None = None) -> Any:
        return np.sum(value, axis=axis)


SAFE_NUMERIC = _SafeNP()
SAFE_NUMERIC_CALLS = {
    "abs",
    "array",
    "asarray",
    "clip",
    "cross",
    "dot",
    "maximum",
    "minimum",
    "sqrt",
    "sum",
}
SAFE_REFS_CALLS = {
    "axis",
    "distance",
    "orientation_xyzw",
    "point",
    "position",
    "tool_axis",
    "tool_point",
    "vector_between",
    "world_axis",
    "world_point",
}


def _program_source(planner_program: dict[str, Any] | str) -> str:
    if isinstance(planner_program, str):
        source = planner_program
    elif isinstance(planner_program, dict):
        source = planner_program.get("program_text") or planner_program.get("source")
    else:
        source = None
    if not isinstance(source, str) or not source.strip():
        raise ObjectiveRuntimeError("planner_program did not include Python source")
    return source


def _objective_names(program_ir: dict[str, Any]) -> set[str]:
    return {
        str(objective.get("function_name"))
        for objective in program_ir.get("objectives", [])
        if objective.get("function_name")
    }


def _validate_objective_function(node: ast.FunctionDef) -> None:
    args = node.args
    if (
        len(args.args) != 2
        or args.args[0].arg != "pose"
        or args.args[1].arg != "refs"
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
    ):
        raise ObjectiveRuntimeError(
            f"objective function {node.name} must have signature {node.name}(pose, refs)"
        )
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            raise ObjectiveRuntimeError(f"objective function {node.name} contains nested code")
        if isinstance(child, ast.Attribute):
            if not _is_allowed_objective_attribute_access(child):
                raise ObjectiveRuntimeError(
                    f"objective function {node.name} accesses unsupported attribute {child.attr}"
                )


def _is_allowed_objective_attribute_access(attribute: ast.Attribute) -> bool:
    if isinstance(attribute.value, ast.Name):
        if attribute.value.id == "refs":
            return attribute.attr in SAFE_REFS_CALLS
        if attribute.value.id == "np":
            return attribute.attr in SAFE_NUMERIC_CALLS or attribute.attr == "linalg"
        if attribute.value.id == "math":
            return not attribute.attr.startswith("_")
    if (
        isinstance(attribute.value, ast.Attribute)
        and attribute.value.attr == "linalg"
        and isinstance(attribute.value.value, ast.Name)
        and attribute.value.value.id == "np"
    ):
        return attribute.attr == "norm"
    return False


def load_objective_functions(
    *,
    planner_program: dict[str, Any] | str,
    program_ir: dict[str, Any],
) -> dict[str, ObjectiveFunction]:
    """Load only the objective functions named by ProgramIR from checked source."""

    names = _objective_names(program_ir)
    if not names:
        return {}
    source = _program_source(planner_program)
    try:
        tree = validate_planner_program_source(source)
    except PlannerProgramError as exc:
        raise ObjectiveRuntimeError(str(exc)) from exc

    functions: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            _validate_objective_function(node)
            functions.append(node)
    found = {node.name for node in functions}
    missing = sorted(names - found)
    if missing:
        raise ObjectiveRuntimeError(f"missing objective function(s): {', '.join(missing)}")

    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    globals_dict: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "np": SAFE_NUMERIC,
    }
    locals_dict: dict[str, Any] = {}
    exec(compile(module, "<parttoolgen_objectives>", "exec"), globals_dict, locals_dict)
    loaded: dict[str, ObjectiveFunction] = {}
    for name in names:
        function = locals_dict.get(name) or globals_dict.get(name)
        if not callable(function):
            raise ObjectiveRuntimeError(f"objective {name} did not compile to a callable")
        loaded[name] = function
    return loaded


def _pose_dict(pose: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pose, dict):
        raise ObjectiveRuntimeError("pose must be a dict")
    position = pose.get("position")
    orientation = pose.get("orientation_xyzw")
    if not isinstance(position, list) or len(position) != 3:
        raise ObjectiveRuntimeError("pose.position must be a 3-vector")
    if not isinstance(orientation, list) or len(orientation) != 4:
        raise ObjectiveRuntimeError("pose.orientation_xyzw must be a quaternion")
    return {
        "position": [float(x) for x in position],
        "orientation_xyzw": normalize_quat([float(x) for x in orientation]),
    }


def _array3(value: list[float]) -> np.ndarray:
    return np.asarray([float(x) for x in value], dtype=float)


class ObjectivePoseRefs:
    """Restricted geometry accessor passed to objective functions."""

    def __init__(
        self,
        *,
        resolver: ReferenceResolver,
        allowed_refs: set[str],
        objective_id: str,
    ) -> None:
        self._resolver = resolver
        self._allowed_refs = set(allowed_refs)
        self._objective_id = objective_id

    def _check(self, ref: str) -> str:
        if not isinstance(ref, str) or not ref:
            raise ObjectiveRuntimeError(f"{self._objective_id} requested an invalid ref")
        if ref not in self._allowed_refs:
            raise ObjectiveRuntimeError(
                f"{self._objective_id} requested ref outside objective whitelist: {ref}"
            )
        return ref

    def tool_point(self, pose: dict[str, Any], ref: str) -> np.ndarray:
        """Return a tool-frame point ref transformed by candidate `pose`."""

        ref = self._check(ref)
        try:
            point = self._resolver.tool_point(ref).point_tool
        except ReferenceResolutionError as exc:
            raise ObjectiveRuntimeError(str(exc)) from exc
        return _array3(pose_point(_pose_dict(pose), point or []))

    def world_point(self, ref: str) -> np.ndarray:
        """Return a world-frame point ref."""

        ref = self._check(ref)
        try:
            point = self._resolver.world_point(ref).point_world
        except ReferenceResolutionError as exc:
            raise ObjectiveRuntimeError(str(exc)) from exc
        return _array3(point or [])

    def point(self, pose: dict[str, Any], ref: str) -> np.ndarray:
        """Return a point ref in world coordinates, transforming tool refs by pose."""

        ref = self._check(ref)
        try:
            resolved = self._resolver.point(ref)
        except ReferenceResolutionError as exc:
            raise ObjectiveRuntimeError(str(exc)) from exc
        if resolved.point_tool is not None:
            return _array3(pose_point(_pose_dict(pose), resolved.point_tool))
        return _array3(resolved.point_world or [])

    def tool_axis(self, pose: dict[str, Any], ref: str) -> np.ndarray:
        """Return a tool-frame axis/vector ref transformed by candidate `pose`."""

        ref = self._check(ref)
        try:
            axis = self._resolver.tool_vector(ref).vector_tool
        except ReferenceResolutionError as exc:
            raise ObjectiveRuntimeError(str(exc)) from exc
        return _array3(pose_axis(_pose_dict(pose), axis or []))

    def world_axis(self, ref: str) -> np.ndarray:
        """Return a world-frame axis/vector ref."""

        ref = self._check(ref)
        try:
            axis = self._resolver.world_vector(ref).vector_world
        except ReferenceResolutionError as exc:
            raise ObjectiveRuntimeError(str(exc)) from exc
        return _array3(axis or [])

    def axis(self, pose: dict[str, Any], ref: str) -> np.ndarray:
        """Return an axis/vector ref in world coordinates."""

        ref = self._check(ref)
        try:
            resolved = self._resolver.vector(ref)
        except ReferenceResolutionError as exc:
            raise ObjectiveRuntimeError(str(exc)) from exc
        if resolved.vector_tool is not None:
            return _array3(pose_axis(_pose_dict(pose), resolved.vector_tool))
        return _array3(resolved.vector_world or [])

    def position(self, pose: dict[str, Any]) -> np.ndarray:
        """Return the candidate tool origin position."""

        return _array3(_pose_dict(pose)["position"])

    def orientation_xyzw(self, pose: dict[str, Any]) -> np.ndarray:
        """Return the candidate tool orientation quaternion in xyzw order."""

        return np.asarray(_pose_dict(pose)["orientation_xyzw"], dtype=float)

    def distance(self, a: Any, b: Any) -> float:
        """Return Euclidean distance between two numeric 3-vectors."""

        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    def vector_between(self, pose: dict[str, Any], start_ref: str, end_ref: str) -> np.ndarray:
        """Return the world vector from `start_ref` to `end_ref`."""

        return self.point(pose, end_ref) - self.point(pose, start_ref)


def _objective_allowed_refs(objective: dict[str, Any]) -> set[str]:
    refs = objective.get("refs") or []
    if not isinstance(refs, list):
        raise ObjectiveRuntimeError(f"{objective.get('id', '<objective>')}.refs must be a list")
    return {str(ref) for ref in refs if isinstance(ref, str) and ref}


def _coerce_scalar(value: Any, objective_id: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape not in {(), (1,)}:
        raise ObjectiveRuntimeError(f"{objective_id} returned a non-scalar value")
    number = float(array.reshape(-1)[0])
    if not math.isfinite(number):
        raise ObjectiveRuntimeError(f"{objective_id} returned a non-finite value")
    return number


def pose_objective_residuals(
    *,
    pose: dict[str, Any],
    objectives: list[dict[str, Any]],
    functions: dict[str, ObjectiveFunction],
    resolver: ReferenceResolver,
) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for objective in objectives:
        objective_id = str(objective.get("id") or objective.get("function_name"))
        function_name = str(objective.get("function_name"))
        function = functions.get(function_name)
        if function is None:
            raise ObjectiveRuntimeError(f"objective function not loaded: {function_name}")
        refs = ObjectivePoseRefs(
            resolver=resolver,
            allowed_refs=_objective_allowed_refs(objective),
            objective_id=objective_id,
        )
        try:
            raw_value = function(pose, refs)
        except ObjectiveRuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert VLM code failures into compiler errors.
            raise ObjectiveRuntimeError(f"{objective_id} failed: {exc}") from exc
        residuals[f"{objective_id}.objective_residual"] = _coerce_scalar(raw_value, objective_id)
    return residuals


def pose_objective_cost(
    *,
    pose: dict[str, Any],
    objectives: list[dict[str, Any]],
    functions: dict[str, ObjectiveFunction],
    resolver: ReferenceResolver,
) -> float:
    cost = 0.0
    residuals = pose_objective_residuals(
        pose=pose,
        objectives=objectives,
        functions=functions,
        resolver=resolver,
    )
    for objective in objectives:
        objective_id = str(objective.get("id") or objective.get("function_name"))
        residual = residuals[f"{objective_id}.objective_residual"]
        penalty = max(float(residual), 0.0)
        cost += float(objective.get("weight", 1.0)) * penalty * penalty
    return float(cost)
