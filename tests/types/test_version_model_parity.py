"""Pin the committed surface packages against the spec oracles.

The packages under ``src/mcp/types/v*`` are generated-then-hand-validated
source; the oracle modules under ``tests/spec_oracles`` are regenerated
verbatim from the pinned schemas. Two comparisons live here:

1. Per-surface parity: each surface package matches its own version's oracle
   — definition sets in both directions, and per model the wire aliases,
   requiredness, and a normalized form of every field annotation — so a hand
   edit that drifts from the pinned schema fails.
2. The additive-coverage proof: the ``v2025_11_25`` surface serves every
   protocol version through 2025-11-25, which is sound only if those four
   schemas evolve strictly additively. For each of the three older oracles,
   every definition must resolve on the surface and every field must carry a
   compatible-or-wider annotation there; an absence fails the suite — it
   would falsify the premise the two-surface layout stands on.

The deliberate scaffold-pass deltas are the annotated tolerance tables below;
everything else must match exactly. Two deltas need no tolerance entry:
inheritance flattening (pydantic's ``model_fields`` already includes
inherited fields on the oracle side, so flattened package classes compare
equal), and the deterministic synthetic class names (derived from the oracle
in ``_synthetic_renames``, mirroring the scaffold pass).
"""

from __future__ import annotations

import functools
import importlib
import operator
import re
from types import ModuleType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import AnyUrl, BaseModel, FileUrl, create_model
from typing_extensions import TypeAliasType

VERSIONS = (
    "v2025_11_25",
    "v2026_07_28",
)

# Tolerance: value-transforming pydantic types are downgraded to plain ``str``
# in the packages — URL normalization and base64 re-encoding would change wire
# bytes on a validate -> re-dump round trip. (``Base64Str`` needs no entry: it
# is ``Annotated[str, ...]``, so both sides already compare as ``str``.)
_VALUE_DOWNGRADES = {"AnyUrl": "str", "FileUrl": "str"}

# Tolerance: the packages widen ``structuredContent`` from ``dict[str, Any]``
# to ``Any`` — the newest schema types the field ``Any``, and the wire models
# never narrow a value the SDK models accept.
_WIDENED_FIELDS = frozenset({"structured_content"})

# Tolerance: the pinned 2026-07-28 schema.json renders JSONValue's primitive
# branch as ["string", "integer", "boolean"], but its schema.ts source defines
# all six JSON types (string | number | boolean | null | object | array). The
# oracle reproduces the render verbatim; the package follows the schema.ts
# definition so fractional numbers and nulls validate. The package alias is
# pinned here verbatim, so any further drift still fails.
_ALIAS_OVERRIDES: dict[tuple[str, str], str] = {
    ("v2026_07_28", "JSONValue"): "JSONObject | list[JSONValue] | str | int | float | bool | None",
}

# Tolerance: the pinned schema.json renderings type a few schema.ts `number`
# positions as "integer" — the same render artifact fixed for the JSONValue
# alias. The packages keep the int arm and gain a float arm at exactly these
# positions, so fractional elicitation answers and number-schema bounds have a
# wire form; the generated oracles keep the render verbatim, and the package
# annotation is pinned here verbatim so any further drift still fails.
# Position -> the exact package annotation.
_RENDER_ARTIFACT_WIDENED: dict[tuple[str, str, str], Any] = {
    ("v2025_11_25", "ElicitResult", "content"): dict[str, list[str] | str | int | float | bool] | None,
    ("v2025_11_25", "NumberSchema", "default"): int | float | None,
    ("v2025_11_25", "NumberSchema", "maximum"): int | float | None,
    ("v2025_11_25", "NumberSchema", "minimum"): int | float | None,
    ("v2026_07_28", "ElicitResult", "content"): dict[str, list[str] | str | int | float | bool] | None,
}

# The closure for every OTHER package field that admits int but not float:
# the integer rendering is the intended type, justified per field name by the
# spec fact in plain words. A field name absent here whose annotation is
# int-without-float fails test_int_only_number_positions_are_classified, so a
# future render artifact cannot land unexamined.
_INTENDED_INTEGER_FIELDS: dict[str, str] = {
    "id": "JSON-RPC ids: every schema rendering pins the numeric kind to integer; fractional ids are not interoperable",
    "request_id": "a cancelled/targeted request id mirrors the JSON-RPC id type (string or integer)",
    "progress_token": "progress tokens mirror the JSON-RPC id type (string or integer)",
    "code": "JSON-RPC 2.0 defines error codes as integers",
    "total": "a count of completion values; the superset model agrees (int)",
    "max_tokens": "a token count; the superset model agrees (int)",
    "size": "a resource size in bytes; the superset model agrees (int)",
    "ttl": "a task time-to-live in milliseconds; the superset model agrees (int)",
    "ttl_ms": "a cache time-to-live in milliseconds; the superset model agrees (int)",
    "poll_interval": "a polling interval in milliseconds; the superset model agrees (int)",
    "max_length": "JSON Schema maxLength is a non-negative integer keyword",
    "min_length": "JSON Schema minLength is a non-negative integer keyword",
    "max_items": "JSON Schema maxItems is a non-negative integer keyword",
    "min_items": "JSON Schema minItems is a non-negative integer keyword",
}

# The expected-open class policy: every package class is ``extra="ignore"``
# except the ``_meta`` carriers (unknown ``_meta`` keys must survive
# revalidation), the tool input/output schema interiors (schema keywords
# beyond the declared properties ride extra fields), and the subscription
# filter (extensible on the wire). The first two groups are derived from the
# package's own field references in test_extra_policy.
_OPEN_INTERIOR_ALIASES = frozenset({"inputSchema", "outputSchema"})

# Names a module's import block binds; everything else public is a definition.
_IMPORTED_NAMES = frozenset(
    {
        "annotations",
        "Annotated",
        "Any",
        "Literal",
        "TypeAlias",
        "TypeAliasType",
        "AnyUrl",
        "Base64Str",
        "ConfigDict",
        "Field",
        "OracleModel",
        "WireModel",
        "OpenWireModel",
        "SPEC_DEFS",
    }
)

Sig = tuple[Any, ...]


def _package(version: str) -> ModuleType:
    return importlib.import_module(f"mcp.types.{version}")


def _oracle(version: str) -> ModuleType:
    return importlib.import_module(f"tests.spec_oracles.{version}")


def _module_defs(mod: ModuleType) -> dict[str, Any]:
    """Public names a module defines (classes and type aliases)."""
    return {name: obj for name, obj in vars(mod).items() if not name.startswith("_") and name not in _IMPORTED_NAMES}


def _module_classes(mod: ModuleType) -> dict[str, type[BaseModel]]:
    """Model classes a module defines, keyed by class name (aliases excluded)."""
    return {
        name: obj
        for name, obj in vars(mod).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == mod.__name__
        and obj.__name__ == name
    }


def _model_names_in(annotation: Any) -> frozenset[str]:
    """Names of model classes appearing anywhere in an annotation."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return frozenset({annotation.__name__})
    names: set[str] = set()
    for arg in get_args(annotation):
        names.update(_model_names_in(arg))
    return frozenset(names)


def _synthetic_renames(oracle: ModuleType) -> dict[str, str]:
    """Oracle synthetic class name -> package class name.

    Mirrors the scaffold's deterministic-naming pass: a generated
    ``Params<n>``/``Meta<n>`` class referenced by exactly one field of one
    owner is named ``<Owner>Params``/``<Owner>Meta`` in the package; shared
    synthetics keep their generated names.
    """
    classes = _module_classes(oracle)
    synthetic = {name for name in classes if re.fullmatch(r"(?:Params|Meta)\d*", name)}
    references: dict[str, list[tuple[str, str]]] = {name: [] for name in synthetic}
    for owner_name, cls in classes.items():
        for field_name, info in cls.model_fields.items():
            for name in synthetic & _model_names_in(info.annotation):
                references[name].append((owner_name, field_name))
    renames: dict[str, str] = {}
    taken = set(classes)
    for name in sorted(synthetic):
        refs = references[name]
        if len(refs) != 1:
            continue
        owner, field_name = refs[0]
        suffix = {"params": "Params", "meta": "Meta"}.get(field_name)
        if suffix is None:
            continue
        target = f"{owner}{suffix}"
        if target in taken:
            continue
        renames[name] = target
        taken.add(target)
    return renames


def _sig(annotation: Any, *, rename: dict[str, str], widen_dicts: bool = False) -> Sig:
    """Canonicalize an annotation into a comparable signature tuple.

    Classes appear by name (mapped through ``rename``), unions as unordered
    member sets; ``widen_dicts`` collapses ``dict[str, Any]`` to ``Any`` for
    the widened-field tolerance.
    """
    if annotation is None or annotation is type(None):
        return ("none",)
    if annotation is Any:
        return ("any",)
    if isinstance(annotation, TypeAliasType):
        return ("aliasref", annotation.__name__)
    origin = get_origin(annotation)
    if origin is Annotated:
        return _sig(get_args(annotation)[0], rename=rename, widen_dicts=widen_dicts)
    if origin is Literal:
        return ("literal", frozenset(get_args(annotation)))
    if origin is Union or origin is UnionType:
        return ("union", frozenset(_sig(arg, rename=rename, widen_dicts=widen_dicts) for arg in get_args(annotation)))
    if origin is dict:
        key, value = (_sig(arg, rename=rename, widen_dicts=widen_dicts) for arg in get_args(annotation))
        if widen_dicts and key == ("cls", "str") and value == ("any",):
            return ("any",)
        return ("dict", key, value)
    if origin is not None:
        args = tuple(_sig(arg, rename=rename, widen_dicts=widen_dicts) for arg in get_args(annotation))
        return ("generic", origin.__name__, args)
    if isinstance(annotation, type):
        return ("cls", rename.get(annotation.__name__, annotation.__name__))
    return ("opaque", repr(annotation))


def _assert_classes_match(
    oracle_cls: type[BaseModel],
    package_cls: type[BaseModel],
    *,
    rename: dict[str, str],
    context: str,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Field-level comparison: names, wire aliases, requiredness, annotations.

    ``overrides`` maps a field name to the verbatim package annotation pinned
    for it (the render-artifact widenings); for those fields the package is
    compared against the pin instead of the oracle's rendered annotation.
    """
    oracle_fields = oracle_cls.model_fields
    package_fields = package_cls.model_fields
    assert set(package_fields) == set(oracle_fields), f"{context}: field set differs"
    for field_name, oracle_info in oracle_fields.items():
        package_info = package_fields[field_name]
        assert package_info.alias == oracle_info.alias, f"{context}.{field_name}: wire alias differs"
        assert package_info.is_required() == oracle_info.is_required(), f"{context}.{field_name}: requiredness differs"
        widen = field_name in _WIDENED_FIELDS
        package_sig = _sig(package_info.annotation, rename={}, widen_dicts=widen)
        if overrides is not None and field_name in overrides:
            pinned_sig = _sig(overrides[field_name], rename={})
            assert package_sig == pinned_sig, f"{context}.{field_name}: annotation differs from its pinned widening"
            continue
        oracle_sig = _sig(oracle_info.annotation, rename=rename, widen_dicts=widen)
        assert package_sig == oracle_sig, f"{context}.{field_name}: annotation differs"


@pytest.mark.parametrize("version", VERSIONS)
def test_definition_sets_match(version: str) -> None:
    """Both directions: every oracle definition is in the package and vice versa."""
    oracle_defs = _module_defs(_oracle(version))
    package_defs = _module_defs(_package(version))
    rename = _synthetic_renames(_oracle(version))
    expected = {rename.get(name, name) for name in oracle_defs}
    missing = expected - set(package_defs)
    assert not missing, f"{version}: oracle definitions missing from the package: {sorted(missing)}"
    extra = set(package_defs) - expected
    assert not extra, f"{version}: package definitions with no oracle counterpart: {sorted(extra)}"


@pytest.mark.parametrize("version", VERSIONS)
def test_model_fields_match(version: str) -> None:
    """Every shared model class matches its oracle field for field."""
    oracle = _oracle(version)
    package = _package(version)
    rename = {**_synthetic_renames(oracle), **_VALUE_DOWNGRADES}
    package_classes = _module_classes(package)
    for oracle_name, oracle_cls in _module_classes(oracle).items():
        package_cls = package_classes[rename.get(oracle_name, oracle_name)]
        overrides = {
            field_name: annotation
            for (widened_version, class_name, field_name), annotation in _RENDER_ARTIFACT_WIDENED.items()
            if widened_version == version and class_name == package_cls.__name__
        }
        _assert_classes_match(
            oracle_cls,
            package_cls,
            rename=rename,
            context=f"{version}.{package_cls.__name__}",
            overrides=overrides,
        )


@pytest.mark.parametrize("version", VERSIONS)
def test_alias_definitions_match(version: str) -> None:
    """Every non-class definition (type alias) matches its oracle form."""
    oracle = _oracle(version)
    package = _package(version)
    rename = {**_synthetic_renames(oracle), **_VALUE_DOWNGRADES}
    oracle_classes = _module_classes(oracle)
    package_defs = _module_defs(package)
    for name, oracle_obj in _module_defs(oracle).items():
        if name in oracle_classes:
            continue
        package_obj = package_defs[rename.get(name, name)]
        override = _ALIAS_OVERRIDES.get((version, name))
        if override is not None:
            assert isinstance(package_obj, TypeAliasType), f"{version}.{name}: overridden alias must stay lazy"
            assert package_obj.__value__ == override, f"{version}.{name}: alias value differs from its override"
            continue
        if isinstance(oracle_obj, TypeAliasType):
            assert isinstance(package_obj, TypeAliasType), f"{version}.{name}: oracle is a lazy alias"
            oracle_sig = _sig(oracle_obj.__value__, rename=rename)
            package_sig = _sig(package_obj.__value__, rename={})
        else:
            oracle_sig = _sig(oracle_obj, rename=rename)
            package_sig = _sig(package_obj, rename={})
        assert package_sig == oracle_sig, f"{version}.{name}: alias value differs"


@pytest.mark.parametrize("version", VERSIONS)
def test_extra_policy(version: str) -> None:
    """Package classes are closed except the enumerated open classes.

    Open by design: the ``_meta`` carriers (unknown ``_meta`` keys survive
    revalidation), the tool input/output schema interiors, and the
    subscription filter. Everything else is ``extra="ignore"`` so a field the
    target version never defined registers as a loss on revalidation.
    """
    classes = _module_classes(_package(version))
    expected_open: set[str] = {"SubscriptionFilter"} & set(classes)
    for cls in classes.values():
        for field_name, info in cls.model_fields.items():
            alias = info.alias or field_name
            if alias == "_meta" or alias in _OPEN_INTERIOR_ALIASES:
                expected_open.update(name for name in _model_names_in(info.annotation) if name in classes)
    for name, cls in classes.items():
        expected = "allow" if name in expected_open else "ignore"
        assert cls.model_config.get("extra") == expected, f"{version}.{name}: extra={cls.model_config.get('extra')!r}"
        assert cls.model_config.get("populate_by_name") is True, f"{version}.{name}: populate_by_name is not set"


# --- number-render closure sweep -------------------------------------------------


def _admits_int_without_float(annotation: Any, seen: frozenset[int] = frozenset()) -> bool:
    """True when ``annotation`` admits int somewhere without a float sibling.

    Walks unions, containers, Annotated metadata, and lazy aliases (cycle-safe
    via ``seen``); Literal values are exact constants, never an int admission.
    """
    if isinstance(annotation, TypeAliasType):
        if id(annotation) in seen:
            return False
        return _admits_int_without_float(annotation.__value__, seen | {id(annotation)})
    origin = get_origin(annotation)
    if origin is Annotated:
        return _admits_int_without_float(get_args(annotation)[0], seen)
    if origin is Literal:
        return False
    if origin is Union or origin is UnionType:
        members = get_args(annotation)
        if int in members and float not in members:
            return True
        return any(_admits_int_without_float(member, seen) for member in members if member is not int)
    if origin is not None:
        return any(_admits_int_without_float(arg, seen) for arg in get_args(annotation))
    return annotation is int


@pytest.mark.parametrize("version", VERSIONS)
def test_int_only_number_positions_are_classified(version: str) -> None:
    """Every package field position that admits int but not float is claimed
    by exactly one pinned table: the render-artifact widenings (the package
    must carry the float arm the schema.json rendering lost — schema.ts types
    those positions number) or the intended-integer field closure (the spec
    fact really is integral). An unclassified position fails, so a new
    integer rendering cannot land unexamined; a widened position that loses
    its float arm fails too."""
    for class_name, cls in _module_classes(_package(version)).items():
        for field_name, info in cls.model_fields.items():
            int_only = _admits_int_without_float(info.annotation)
            if (version, class_name, field_name) in _RENDER_ARTIFACT_WIDENED:
                assert not int_only, f"{version}.{class_name}.{field_name}: pinned widening lost its float arm"
            elif int_only:
                assert field_name in _INTENDED_INTEGER_FIELDS, (
                    f"{version}.{class_name}.{field_name}: int-without-float position is neither a pinned "
                    "render-artifact widening nor a pinned intended-integer field"
                )


# --- additive-coverage proof ------------------------------------------------
#
# The v2025_11_25 surface serves every protocol version through 2025-11-25.
# The proof: every definition the 2024-11-05, 2025-03-26, and 2025-06-18
# schemas make must resolve on the surface, and every field must carry a
# compatible-or-wider annotation there (the surface admits every value the
# older schema admits). Definitions are taken from each oracle's SPEC_DEFS
# manifest — the schema's own definition list — and anonymous sub-objects
# (generator-synthesized classes) are verified positionally by the recursive
# walk, never by their generated names.

_PRE_2025_11_25_ORACLES = ("v2024_11_05", "v2025_03_26", "v2025_06_18")

# Definitions the later schemas renamed; same wire shape under a new name.
_CROSS_VERSION_RENAMES: dict[str, str] = {
    # 2025-11-25 renamed the success envelope and recycled "JSONRPCResponse"
    # for the success|error union.
    "JSONRPCResponse": "JSONRPCResultResponse",
    "JSONRPCError": "JSONRPCErrorResponse",
    # 2025-06-18 renamed the completion reference type (same tag and fields).
    "ResourceReference": "ResourceTemplateReference",
}

# Older-schema definitions deliberately without a surface counterpart. Each
# entry is a reviewed decision; test_coverage_exclusions_hold pins the facts
# the justifications rest on so an exclusion cannot rot.
_COVERAGE_DEF_EXCLUSIONS: dict[tuple[str, str], str] = {
    (
        "v2024_11_05",
        "Annotated",
    ): "structural base interface with one optional annotations field; 2025-03-26 dropped the standalone "
    "definition and every consuming class declares the field directly, where the class walk verifies it",
    (
        "v2025_03_26",
        "JSONRPCBatchRequest",
    ): "JSON-RPC batch arrays existed only in the 2025-03-26 schema (removed in 2025-06-18); batching is an "
    "envelope/transport concern owned by the session layer, not a payload type, and the SDK never emits batch frames",
    (
        "v2025_03_26",
        "JSONRPCBatchResponse",
    ): "JSON-RPC batch arrays existed only in the 2025-03-26 schema (removed in 2025-06-18); batching is an "
    "envelope/transport concern owned by the session layer, not a payload type, and the SDK never emits batch frames",
}

# Union definitions whose batch-array members are dropped before comparison,
# under the same batch justification as above (the members are inlined
# list[...] forms of the excluded definitions).
_BATCH_UNION_DEFS = frozenset({("v2025_03_26", "JSONRPCMessage")})

# Positions where 2025-11-25 gave a previously-open capability object a typed
# shape. Mechanical coverage cannot call a typed class wider than an open
# object, but every value an older peer can actually send (an empty or
# vendor-keyed object) still validates: the typed class has only optional
# fields and ignores unknown keys, and the keys it types did not exist in the
# older schemas. The walk verifies that all-optional premise per pin.
_TYPED_INTERIOR_POSITIONS: dict[tuple[str, str, str], str] = {
    ("v2024_11_05", "ClientCapabilities", "sampling"): "sampling was an open object before 2025-11-25 typed it",
    ("v2025_03_26", "ClientCapabilities", "sampling"): "sampling was an open object before 2025-11-25 typed it",
    ("v2025_06_18", "ClientCapabilities", "sampling"): "sampling was an open object before 2025-11-25 typed it",
    ("v2025_06_18", "ClientCapabilities", "elicitation"): "elicitation was an open object before 2025-11-25 typed it",
}

# The committed surface downgrades value-transforming URL types to plain str
# (see _VALUE_DOWNGRADES); the leaves compare as str.
_URL_LEAF_DOWNGRADES: dict[Any, Any] = {AnyUrl: str, FileUrl: str}


def _unwrap_alias(obj: Any) -> Any:
    """A lazy alias definition's value; any other definition unchanged."""
    return obj.__value__ if isinstance(obj, TypeAliasType) else obj


def _unwrap_annotation(annotation: Any) -> Any:
    """Resolve lazy aliases and strip Annotated metadata."""
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
            continue
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
            continue
        return annotation


def _annotation_members(annotation: Any) -> tuple[Any, ...]:
    """An annotation's union members, or the annotation itself."""
    if get_origin(annotation) is Union or isinstance(annotation, UnionType):
        return get_args(annotation)
    return (annotation,)


def _covers(surface: Any, oracle: Any, older: str, memo: set[tuple[int, int]]) -> bool:
    """True when ``surface`` admits every value ``oracle`` admits.

    Classes compare structurally (by wire field name, requiredness, and
    recursive coverage), never by class name — the generator names anonymous
    sub-objects differently per version. ``memo`` holds class pairs already
    being verified, so repeated and recursive references resolve.
    """
    surface = _unwrap_annotation(surface)
    oracle = _unwrap_annotation(oracle)
    if surface is Any:
        return True
    if oracle is Any:
        return False
    surface_members = _annotation_members(surface)
    oracle_members = _annotation_members(oracle)
    if len(surface_members) > 1 or len(oracle_members) > 1:
        return all(any(_covers(s, o, older, memo) for s in surface_members) for o in oracle_members)
    if oracle is type(None):
        return surface is type(None)
    surface_origin = get_origin(surface)
    oracle_origin = get_origin(oracle)
    if oracle_origin is Literal:
        return surface_origin is Literal and set(get_args(oracle)) <= set(get_args(surface))
    if isinstance(oracle, type) and issubclass(oracle, BaseModel):
        if isinstance(surface, type) and issubclass(surface, BaseModel):
            key = (id(surface), id(oracle))
            if key in memo:
                return True
            memo.add(key)
            return not _field_coverage_failures(surface, oracle, older, memo, context=oracle.__name__)
        # An open-object annotation admits anything a model class admits.
        return surface_origin is dict and _unwrap_annotation(get_args(surface)[1]) is Any
    if oracle_origin is dict:
        if surface_origin is not dict:
            return False
        surface_key, surface_value = get_args(surface)
        oracle_key, oracle_value = get_args(oracle)
        return _covers(surface_key, oracle_key, older, memo) and _covers(surface_value, oracle_value, older, memo)
    if oracle_origin is not None:
        if surface_origin is not oracle_origin:
            return False
        surface_args, oracle_args = get_args(surface), get_args(oracle)
        if len(surface_args) != len(oracle_args):
            return False
        return all(_covers(s, o, older, memo) for s, o in zip(surface_args, oracle_args, strict=True))
    return _URL_LEAF_DOWNGRADES.get(surface, surface) is _URL_LEAF_DOWNGRADES.get(oracle, oracle)


def _model_arm_fields_all_optional(annotation: Any) -> bool:
    """True when the annotation has a model-class arm with no required field
    (the pinned typed-interior premise: an empty object always validates)."""
    for member in _annotation_members(_unwrap_annotation(annotation)):
        member = _unwrap_annotation(member)
        if isinstance(member, type) and issubclass(member, BaseModel):
            return not any(info.is_required() for info in member.model_fields.values())
    return False


def _field_coverage_failures(
    surface_cls: type[BaseModel],
    oracle_cls: type[BaseModel],
    older: str,
    memo: set[tuple[int, int]],
    *,
    context: str,
) -> list[str]:
    """Coverage failures of ``surface_cls`` against ``oracle_cls``, one per field."""
    failures: list[str] = []
    surface_by_wire = {info.alias or name: info for name, info in surface_cls.model_fields.items()}
    for field_name, oracle_info in oracle_cls.model_fields.items():
        wire_name = oracle_info.alias or field_name
        surface_info = surface_by_wire.get(wire_name)
        if surface_info is None:
            failures.append(f"{context}.{field_name}: the surface class has no field named {wire_name!r}")
            continue
        if surface_info.is_required() and not oracle_info.is_required():
            failures.append(f"{context}.{field_name}: the surface newly requires the field")
            continue
        if (older, oracle_cls.__name__, field_name) in _TYPED_INTERIOR_POSITIONS:
            # The pin holds only while the surface arm stays all-optional
            # (an older peer's empty object must keep validating).
            if not _model_arm_fields_all_optional(surface_info.annotation):
                failures.append(f"{context}.{field_name}: pinned typed-interior premise broken")
            continue
        if not _covers(surface_info.annotation, oracle_info.annotation, older, memo):
            failures.append(f"{context}.{field_name}: the surface annotation is not compatible-or-wider")
    return failures


@pytest.mark.parametrize("older", _PRE_2025_11_25_ORACLES)
def test_older_schema_definitions_all_resolve_on_the_surface(older: str) -> None:
    """Every definition an older schema makes resolves on the 2025-11-25
    surface, by name or recorded rename. An absence falsifies the
    strictly-additive premise the two-surface layout stands on, so it fails
    outright — there is no tolerance list for absences."""
    oracle = _oracle(older)
    surface = _package("v2025_11_25")
    spec_defs: tuple[str, ...] = oracle.SPEC_DEFS
    unresolved = [
        def_name
        for def_name in spec_defs
        if (older, def_name) not in _COVERAGE_DEF_EXCLUSIONS
        and not hasattr(surface, _CROSS_VERSION_RENAMES.get(def_name, def_name))
    ]
    assert not unresolved, (
        f"{older}: schema definitions with no 2025-11-25 surface counterpart "
        f"(the additive premise is falsified): {sorted(unresolved)}"
    )


def _definition_coverage_failures(older: str, oracle: ModuleType, surface: ModuleType) -> list[str]:
    """Coverage failures of the surface against every definition in
    ``oracle.SPEC_DEFS``, skipping the pinned exclusions."""
    failures: list[str] = []
    memo: set[tuple[int, int]] = set()
    spec_defs: tuple[str, ...] = oracle.SPEC_DEFS
    for def_name in spec_defs:
        if (older, def_name) in _COVERAGE_DEF_EXCLUSIONS:
            continue
        oracle_obj: Any = getattr(oracle, def_name)
        surface_obj: Any = getattr(surface, _CROSS_VERSION_RENAMES.get(def_name, def_name))
        if (
            isinstance(oracle_obj, type)
            and issubclass(oracle_obj, BaseModel)
            and isinstance(surface_obj, type)
            and issubclass(surface_obj, BaseModel)
        ):
            memo.add((id(surface_obj), id(oracle_obj)))
            failures += _field_coverage_failures(surface_obj, oracle_obj, older, memo, context=f"{older}.{def_name}")
            continue
        oracle_value: Any = _unwrap_alias(oracle_obj)
        surface_value: Any = _unwrap_alias(surface_obj)
        if (older, def_name) in _BATCH_UNION_DEFS:
            # Drop the inlined batch-array members (the excluded batch
            # definitions); every remaining member must still be covered.
            kept = tuple(m for m in _annotation_members(oracle_value) if get_origin(m) is not list)
            oracle_value = functools.reduce(operator.or_, kept)
        if not _covers(surface_value, oracle_value, older, memo):
            failures.append(f"{older}.{def_name}: the surface definition is not compatible-or-wider")
    return failures


@pytest.mark.parametrize("older", _PRE_2025_11_25_ORACLES)
def test_surface_covers_every_older_schema_definition(older: str) -> None:
    """The field-level half of the proof: per shared definition, every older
    field exists on the surface under the same wire name, is not newly
    required, and carries a compatible-or-wider annotation; alias
    definitions (unions and scalars) are covered member-wise."""
    failures = _definition_coverage_failures(older, _oracle(older), _package("v2025_11_25"))
    assert not failures, "surface does not cover these older-schema positions:\n" + "\n".join(failures)


def test_coverage_exclusions_hold() -> None:
    """Pin the facts each coverage exclusion rests on, so a regenerated
    oracle that changes them re-opens the decision instead of passing."""
    # The generator names the def's class AnnotatedModel (the bare name
    # would collide with the typing import).
    annotated = _oracle("v2024_11_05").AnnotatedModel
    assert set(annotated.model_fields) == {"annotations"}
    assert not annotated.model_fields["annotations"].is_required()
    oracle_2025_03_26 = _oracle("v2025_03_26")
    for batch_name in ("JSONRPCBatchRequest", "JSONRPCBatchResponse"):
        assert get_origin(getattr(oracle_2025_03_26, batch_name)) is list
    # The typed-interior pins name real positions: each older oracle types
    # the position as an open object, and the surface arm is all-optional.
    surface_classes = _module_classes(_package("v2025_11_25"))
    for older, class_name, field_name in _TYPED_INTERIOR_POSITIONS:
        oracle_info = _module_classes(_oracle(older))[class_name].model_fields[field_name]
        oracle_arms = _annotation_members(_unwrap_annotation(oracle_info.annotation))
        assert any(get_origin(arm) is dict for arm in oracle_arms), (older, class_name, field_name)
        surface_info = surface_classes[class_name].model_fields[field_name]
        assert _model_arm_fields_all_optional(surface_info.annotation), (older, class_name, field_name)


# --- coverage-walker unit tests (branches the real comparison cannot reach) ---


def test_covers_rejects_an_oracle_any_against_a_narrower_surface() -> None:
    assert not _covers(int, Any, "v2024_11_05", set())


def test_covers_rejects_a_lost_union_member() -> None:
    assert not _covers(int | str, float | int, "v2024_11_05", set())


def test_covers_requires_a_literal_superset() -> None:
    assert _covers(Literal["a", "b"], Literal["a"], "v2024_11_05", set())
    assert not _covers(Literal["a"], Literal["a", "b"], "v2024_11_05", set())
    assert not _covers(str, Literal["a"], "v2024_11_05", set())


def test_covers_accepts_a_class_only_under_a_fully_open_dict() -> None:
    cls = create_model("SomeModel")
    assert _covers(dict[str, Any], cls, "v2024_11_05", set())
    assert not _covers(dict[str, str], cls, "v2024_11_05", set())


def test_covers_rejects_a_dict_against_a_non_dict_surface() -> None:
    assert not _covers(list[str], dict[str, str], "v2024_11_05", set())


def test_covers_rejects_mismatched_generic_origins_and_arity() -> None:
    assert not _covers(list[str], tuple[str], "v2024_11_05", set())
    assert not _covers(tuple[str], tuple[str, int], "v2024_11_05", set())


def test_covers_rejects_a_plain_leaf_mismatch() -> None:
    assert not _covers(str, int, "v2024_11_05", set())


def test_covers_reports_a_newly_required_field() -> None:
    oracle_cls = create_model("Payload", value=(str | None, None))
    surface_cls = create_model("Payload", value=(str, ...))
    assert not _covers(surface_cls, oracle_cls, "v2024_11_05", set())


def test_covers_resolves_mutually_recursive_classes() -> None:
    # Two classes referencing each other: the in-progress memo entry breaks
    # the cycle and the pair verifies.
    oracle_node = create_model("Node", child=("Node | None", None))
    oracle_node.model_rebuild(_types_namespace={"Node": oracle_node})
    assert _covers(oracle_node, oracle_node, "v2024_11_05", set())


def test_covers_unwraps_aliases_and_annotated_metadata() -> None:
    # IntAlias is the module-level alias test datum shared with the
    # int-admission walker tests below.
    assert _covers(IntAlias, Annotated[int, "wire metadata"], "v2024_11_05", set())


def test_typed_interior_pin_fails_when_the_surface_arm_requires_a_field() -> None:
    # A pinned typed-interior position holds only while every field of the
    # surface's class arm is optional; a required field breaks the premise
    # (an older peer's empty object would no longer validate) and must fail.
    oracle_cls = create_model("ClientCapabilities", sampling=(dict[str, Any] | None, None))
    strict_arm = create_model("StrictArm", required_field=(str, ...))
    surface_cls = create_model("ClientCapabilities", sampling=(strict_arm | None, None))
    failures = _field_coverage_failures(surface_cls, oracle_cls, "v2024_11_05", set(), context="probe")
    assert failures == ["probe.sampling: pinned typed-interior premise broken"]


def test_definition_walk_reports_an_uncovered_definition() -> None:
    # A fake oracle whose definition the surface no longer covers: the walk
    # must localize the failure to the definition name.
    oracle = ModuleType("fake_oracle")
    setattr(oracle, "SPEC_DEFS", ("Widened",))
    setattr(oracle, "Widened", int | str)
    surface = ModuleType("fake_surface")
    setattr(surface, "Widened", int)
    assert _definition_coverage_failures("v2024_11_05", oracle, surface) == [
        "v2024_11_05.Widened: the surface definition is not compatible-or-wider"
    ]


def test_model_arm_check_needs_a_model_arm() -> None:
    optional_only = create_model("OptionalOnly", value=(str | None, None))
    required = create_model("Required", value=(str, ...))
    assert _model_arm_fields_all_optional(optional_only | None)
    assert not _model_arm_fields_all_optional(required | None)
    assert not _model_arm_fields_all_optional(dict[str, Any])


# --- helper unit tests (synthetic data; the canonicalizers must not go vacuous) ---


def _fake_oracle(**classes: type[BaseModel]) -> ModuleType:
    module = ModuleType("fake_oracle")
    for name, cls in classes.items():
        setattr(module, name, cls)
    return module


# Alias object as test data for the canonicalizer below.
SomeAlias = TypeAliasType("SomeAlias", str)


def test_synthetic_rename_requires_a_params_or_meta_field_reference() -> None:
    # A synthetic class referenced from a field that is not `params`/`meta`
    # derives no owner-based name and keeps its generated one.
    params = create_model("Params1", __module__="fake_oracle")
    owner = create_model("Owner", __module__="fake_oracle", payload=(params | None, None))
    assert _synthetic_renames(_fake_oracle(Params1=params, Owner=owner)) == {}


def test_synthetic_rename_never_collides_with_an_existing_class_name() -> None:
    params = create_model("Params1", __module__="fake_oracle")
    owner = create_model("Owner", __module__="fake_oracle", params=(params | None, None))
    taken = create_model("OwnerParams", __module__="fake_oracle")
    assert _synthetic_renames(_fake_oracle(Params1=params, Owner=owner, OwnerParams=taken)) == {}


def test_sig_keeps_alias_references_by_name() -> None:
    assert _sig(SomeAlias, rename={}) == ("aliasref", "SomeAlias")


def test_sig_unwraps_annotated_metadata() -> None:
    assert _sig(Annotated[str, "wire metadata"], rename={}) == ("cls", "str")


# Alias objects as test data for the int-admission walker below.
IntAlias = TypeAliasType("IntAlias", int)


def test_int_admission_walker_resolves_aliases() -> None:
    assert _admits_int_without_float(IntAlias)
    assert not _admits_int_without_float(SomeAlias)


def test_int_admission_walker_stops_on_alias_cycles() -> None:
    # A self-referential alias is walked once; revisiting it resolves False.
    assert not _admits_int_without_float(IntAlias, frozenset({id(IntAlias)}))


def test_int_admission_walker_treats_literal_values_as_constants() -> None:
    assert not _admits_int_without_float(Literal[0, 1])


def test_int_admission_walker_unwraps_annotated_metadata() -> None:
    # Annotated survives only nested inside other annotations (pydantic strips
    # it from the top level of model_fields), so the branch is pinned here.
    assert _admits_int_without_float(Annotated[int, "wire metadata"])


def test_int_admission_walker_sees_a_float_sibling() -> None:
    assert not _admits_int_without_float(int | float | None)
    assert _admits_int_without_float(dict[str, str | int | bool] | None)
