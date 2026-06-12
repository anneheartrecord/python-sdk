"""Version-aware wire boundary for MCP types.

Serialize a monolith model for, or parse wire data under, a specific
negotiated protocol version. A "monolith" model is one of the version-superset
models in ``mcp.types._types``: one class per protocol construct, carrying
every supported version's fields, in contrast to the wire-shape surface
packages (``mcp.types.v2025_11_25``, which serves every protocol version
through 2025-11-25, and ``mcp.types.v2026_07_28``). The unique key for every
behavior in this module is (monolith type, negotiated version). Versions are
opaque strings ordered by ``KNOWN_PROTOCOL_VERSIONS``; nothing here
negotiates, dispatches, or holds session state.

Emission is additive-only: nothing is ever removed from a caller's payload at
any version. ``serialize_for`` dumps the monolith model and, for 2025-11-25
and earlier, emits that dump unchanged — newer fields flow to older peers,
which every deployed SDK tolerates (unknown object keys are ignored or
preserved, never rejected). For 2026-07-28 it injects the protocol-required
fields the caller left unset, validates the result through the 2026-07-28
surface models (imported lazily, so this module stays cheap until first use)
as a check — the emitted bytes are always the injected monolith dump, never a
surface model's re-dump — and emits. Parsing is one lenient superset parse at
every version plus the few documented 2026-07-28 inbound mandates.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import cache
from types import ModuleType
from typing import Any, Final, TypeVar, cast, get_args, overload

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

from mcp.shared.version import KNOWN_PROTOCOL_VERSIONS, is_version_at_least
from mcp.types._spec_names import SDK_TO_SCHEMA_RENAMES
from mcp.types._types import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    AudioContent,
    CacheableResult,
    ElicitResult,
    EmbeddedResource,
    EmptyResult,
    ImageContent,
    InputRequiredResult,
    Notification,
    Request,
    RequestParams,
    ResourceLink,
    Result,
    ResultType,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from mcp.types._versions import (
    CLIENT_NOTIFICATION_METHODS,
    CLIENT_REQUEST_METHODS,
    SERVER_NOTIFICATION_METHODS,
    SERVER_REQUEST_METHODS,
)
from mcp.types.jsonrpc import JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse

__all__ = [
    "CLIENT_NOTIFICATION_METHODS",
    "CLIENT_REQUEST_METHODS",
    "KNOWN_PROTOCOL_VERSIONS",
    "SERVER_NOTIFICATION_METHODS",
    "SERVER_REQUEST_METHODS",
    "UnknownProtocolVersionError",
    "UnsupportedAtVersionError",
    "parse_as",
    "serialize_for",
]

# KNOWN_PROTOCOL_VERSIONS (the ordered version registry) lives in
# mcp.shared.version; the four per-version method tables are plain data in
# mcp.types._versions. Both are re-exported here as the boundary's public
# surface. The tables record which methods exist at each version; acting on
# that (rejecting a version-invalid request, dropping or logging a
# version-invalid notification) is session-layer behavior, as are the other
# capability-keyed halves of version shaping: the sampling tools capability
# gate for tool content on 2025-11-25 sessions, the elicitation url capability
# gate, and the per-request logLevel send condition.

_T = TypeVar("_T", bound=BaseModel)


class UnknownProtocolVersionError(ValueError):
    """``version`` is not a known protocol version (raised on emission only).

    Inbound parsing never raises this: an unknown version is most plausibly
    newer than this SDK, and lenient parsing cannot misrepresent data. On
    emission the type layer must never guess a wire shape, so
    ``serialize_for`` refuses instead.
    """

    def __init__(self, version: str) -> None:
        super().__init__(f"unknown protocol version {version!r}; known versions: {', '.join(KNOWN_PROTOCOL_VERSIONS)}")
        self.version: str = version
        self.known: tuple[str, ...] = KNOWN_PROTOCOL_VERSIONS


class UnsupportedAtVersionError(ValueError):
    """The value cannot be legally represented on the 2026-07-28 wire.

    Raised by ``serialize_for`` on the 2026-07-28 path only — emission for
    2025-11-25 and earlier is the plain monolith dump and never raises this.
    It arises from a model whose type the 2026-07-28 schema does not define
    (the removed lifecycle, subscription, server-request, and tasks
    constructs), an ``InputRequiredResult`` with neither ``input_requests``
    nor ``request_state`` set, or a strict-check failure — the surface model
    rejected the injected dump because a required field is missing (a client
    request whose ``params._meta`` lacks the caller-supplied
    ``clientInfo``/``clientCapabilities`` entries) or a present value has a
    shape the schema does not admit (the chained ``ValidationError`` carries
    the detail). Unknown keys never fail the check.
    """

    def __init__(self, message: str, *, version: str) -> None:
        super().__init__(message)
        self.version: str = version


_SURFACE_MODULES: Final[Mapping[str, str]] = {
    "2024-11-05": "mcp.types.v2025_11_25",
    "2025-03-26": "mcp.types.v2025_11_25",
    "2025-06-18": "mcp.types.v2025_11_25",
    "2025-11-25": "mcp.types.v2025_11_25",
    "2026-07-28": "mcp.types.v2026_07_28",
}
"""The wire-shape surface package serving each known protocol version.

Two surfaces cover the registry: every version through 2025-11-25 is served
by the ``v2025_11_25`` package (the schemas through that revision evolve
strictly additively, so the newest of them describes them all — pinned by
``tests/types/test_version_model_parity.py``), and 2026-07-28 by the
``v2026_07_28`` package. ``serialize_for`` consults a surface only on its
2026-07-28 validation check; for earlier versions the emitted wire form is
the monolith dump itself and no model lookup happens.
"""

# serialize_for-only name aliases for SDK classes whose wire shape the schemas
# publish under a different export name and that the spec-name divergence map
# cannot carry (the map requires schema counterparts; these are SDK-only
# names). The SDK splits the wide-content sampling result into its own class,
# while the 2025-11-25 and 2026-07-28 schemas type the class they name
# CreateMessageResult wide.
_WIRE_NAME_ALIASES: Final[Mapping[str, str]] = {"CreateMessageResultWithTools": "CreateMessageResult"}

_ENVELOPE_MODELS: Final = (JSONRPCRequest, JSONRPCNotification, JSONRPCResponse, JSONRPCError)
_BODY_MODELS: Final = (Request, Notification, Result)

# The 2026-07-28 schema defines each request and notification as a complete
# JSON-RPC frame, so the generated surface classes declare the envelope
# fields (requests: jsonrpc + id; notifications: jsonrpc) as required.
# serialize_for emits message BODIES; the validation check is handed these
# constants alongside the dump, and the emitted dump never carries them.
_ENVELOPE_FIELD_STUBS: Final[Mapping[str, Any]] = {"jsonrpc": "2.0", "id": 0}

# The 2026-07-28 schema requires the reserved _meta entries on every client
# request, so the injection/validation rule is keyed on that revision's client
# request methods.
_REQUIRED_META_METHODS: Final[frozenset[str]] = CLIENT_REQUEST_METHODS["2026-07-28"]
_REQUIRED_META_KEYS: Final[tuple[str, ...]] = (
    PROTOCOL_VERSION_META_KEY,
    CLIENT_INFO_META_KEY,
    CLIENT_CAPABILITIES_META_KEY,
)

_RECOGNIZED_RESULT_TYPES: Final[frozenset[str]] = frozenset(
    literal for arm in get_args(ResultType) for literal in get_args(arm)
)
"""The ``resultType`` values the spec names, read from the monolith
``ResultType`` alias so the recognized set has a single source."""

_CONTENT_BLOCK_TAGS: Final[Mapping[str, Any]] = {
    block.__name__: block.model_fields["type"].default
    for block in (
        TextContent,
        ImageContent,
        AudioContent,
        ResourceLink,
        EmbeddedResource,
        ToolUseContent,
        ToolResultContent,
    )
}
"""Every monolith content-block class and its wire ``type`` tag."""


def serialize_for(model: BaseModel, version: str) -> dict[str, Any]:
    """Dump ``model`` as its wire JSON for a session negotiated at ``version``.

    ``model`` is a top-level message body (a concrete request, notification,
    or result model) or a ``mcp.types.jsonrpc`` envelope model; any other
    monolith model (a bare fragment: content blocks, ``SamplingMessage``,
    capabilities objects, params classes, ...) raises ``TypeError`` —
    fragments are shaped only in situ, inside the body that carries them.

    Returns the message body (requests/notifications/results) or the full
    frame when given an envelope model. Emission is additive-only — nothing
    is ever removed from the dump, at any version:

    - For 2025-11-25 and earlier the output is the plain
      ``model_dump(by_alias=True, mode="json", exclude_none=True)`` of the
      model, byte for byte, and this path never raises beyond the two
      argument checks above. Newer fields and types flow to older peers
      unchanged (deployed SDKs ignore unknown object keys); holding back a
      construct a peer cannot understand — a content-block type or result
      shape its version predates — is the session layer's capability and
      version gating, not this function's.
    - For 2026-07-28 the boundary injects the protocol-required fields the
      caller left unset (``resultType`` on the results the 2026-07-28 schema
      defines it on, the ``ttlMs``/``cacheScope`` don't-cache pair on
      cacheable results, the reserved ``protocolVersion`` ``_meta`` entry on
      client requests — merged, never overwriting a caller-set value), then
      validates the dump through the 2026-07-28 surface models as a check:
      the emitted bytes are always the injected monolith dump, never a
      surface model's re-dump. Caller-set fields the 2026-07-28 schema does
      not define never fail the check and emit unchanged.

    On 2026-07-28 sessions, client requests must already carry the
    caller-supplied ``clientInfo`` and ``clientCapabilities`` entries in
    ``params._meta``; sourcing session identity is the session layer's job,
    and the boundary injects only ``protocolVersion``.

    Null-valued elicitation content entries — constructible for v1.x
    compatibility, typed by no schema version — are caller data and pass
    through verbatim at every version that models elicitation.

    A 2026-07-28 validation failure is reported as
    ``UnsupportedAtVersionError`` and cannot distinguish a value that
    revision truly cannot express from a defect in its surface package; when
    a raise surprises you, read the chained validation error and check
    ``mcp.types.v2026_07_28`` first.

    Raises:
        UnknownProtocolVersionError: ``version`` is not a known protocol
            version.
        UnsupportedAtVersionError: only for ``version`` 2026-07-28 —
            ``model``'s type does not exist in that revision's schema, an
            ``InputRequiredResult`` sets neither ``input_requests`` nor
            ``request_state``, or the surface-model check rejects the
            injected dump: a client request missing the caller-supplied
            identity entries above, or any value shape the 2026-07-28
            schema does not admit.
    """
    if not _is_serializable_payload(model):
        raise TypeError("serialize_for expects a message body or an envelope model")
    if version not in KNOWN_PROTOCOL_VERSIONS:
        raise UnknownProtocolVersionError(version)
    dump = model.model_dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(model, _ENVELOPE_MODELS):
        # Envelope frames are version-independent, and the untyped
        # params/result interior of a generic envelope passes through opaque:
        # payload shaping happens when the typed payload model itself is
        # serialized, never by inspecting an untyped dict.
        return dump
    if not is_version_at_least(version, "2026-07-28"):
        # Alternative considered: narrowing outbound values to the target revision's declared shapes on older versions.
        return dump
    wire_cls = _surface_class(type(model), version)
    _inject_required_fields(model, wire_cls, dump, version)
    _check_wire_form(model, wire_cls, dump, version)
    return dump


def _is_serializable_payload(model: BaseModel) -> bool:
    """True when ``model`` is in ``serialize_for``'s payload domain."""
    return isinstance(model, _BODY_MODELS) or isinstance(model, _ENVELOPE_MODELS)


def _inject_required_fields(model: BaseModel, wire_cls: type[BaseModel], dump: dict[str, Any], version: str) -> None:
    """Inject the 2026-07-28 protocol-required fields the caller left unset.

    Injections touch the top-level body only — embedded request/response
    payloads (the ``inputRequests``/``inputResponses`` map values) are never
    recursed into; embedded-payload hygiene is the caller's responsibility.
    """
    if isinstance(model, Result) and "resultType" in _wire_field_names(wire_cls):
        # resultType is required on the results the 2026-07-28 schema defines
        # it on: the Result base and every server result. The revision
        # removed the server -> client request channel, so the client results
        # (CreateMessageResult, ElicitResult, ListRootsResult) survive only
        # as embedded payload values, carry no resultType, and get none
        # injected. Absent means "complete", an input-required result must
        # say so, and a caller-set value is never overwritten.
        dump.setdefault("resultType", "input_required" if isinstance(model, InputRequiredResult) else "complete")
    if isinstance(model, CacheableResult):
        # ttlMs/cacheScope are required on these results from 2026-07-28;
        # when the handler leaves them unset the boundary fills the
        # don't-cache pair: immediately stale, single-user scope.
        # Alternative considered: injecting nothing and requiring handlers to set both fields.
        dump.setdefault("ttlMs", 0)
        dump.setdefault("cacheScope", "private")
    if isinstance(model, Request) and dump.get("method") in _REQUIRED_META_METHODS:
        # 2026-07-28 client requests carry the reserved _meta entries.
        # protocolVersion is the one entry derivable here and is merged
        # without overwriting a caller-set value; clientInfo and
        # clientCapabilities are session identity, never synthesized — when
        # absent, the validation check refuses loudly.
        params: dict[str, Any] = dump.setdefault("params", {})
        meta: dict[str, Any] = params.setdefault("_meta", {})
        meta.setdefault(PROTOCOL_VERSION_META_KEY, version)


def _check_wire_form(model: BaseModel, wire_cls: type[BaseModel], dump: dict[str, Any], version: str) -> None:
    """Validate the injected dump through its 2026-07-28 surface model.

    A check, never a transformation: the caller emits the injected dump
    whether or not this function consulted the surface model's parse of it.
    The surface models ignore unknown fields, so a caller-set key the
    2026-07-28 schema does not define can never fail here; what fails is a
    missing required field (the reserved ``_meta`` identity entries) or a
    value shape the schema does not admit.
    """
    if isinstance(model, InputRequiredResult) and model.input_requests is None and model.request_state is None:
        # The 2026-07-28 schema requires at least one of
        # inputRequests/requestState on the wire; the requirement is spec
        # prose (both fields are optional in the schema's type), so model
        # validation cannot enforce it.
        raise UnsupportedAtVersionError(
            "InputRequiredResult with neither input_requests nor request_state set "
            f"has no legal wire form at protocol version {version}",
            version=version,
        )
    stubs = {key: value for key, value in _ENVELOPE_FIELD_STUBS.items() if key in wire_cls.model_fields}
    try:
        wire_cls.model_validate({**stubs, **_check_view(model, dump)})
    except ValidationError as err:
        raise UnsupportedAtVersionError(
            f"{type(model).__name__} has no legal wire form at protocol version {version}: {_summarize(err)}",
            version=version,
        ) from err


def _check_view(model: BaseModel, dump: dict[str, Any]) -> dict[str, Any]:
    """The dump as the validation check sees it; the emitted dump is untouched.

    Null-valued elicitation content entries are withheld from the dict handed
    to the surface model: no schema version types a null form answer (the
    monolith admits ``None`` values for v1.x constructor compatibility), and
    emitted values are caller data the boundary passes through verbatim
    rather than narrowing or refusing — python v1.x itself constructs,
    accepts, and emits the same body. Every other value reaches the surface
    model unchanged: a value the 2026-07-28 schema truly cannot express still
    refuses loudly.
    """
    if not isinstance(model, ElicitResult):
        return dump
    content = dump.get("content")
    if not isinstance(content, dict):
        return dump
    entries = cast("dict[str, Any]", content)
    if all(value is not None for value in entries.values()):
        return dump
    view = dict(dump)
    view["content"] = {key: value for key, value in entries.items() if value is not None}
    return view


def _surface_module(version: str) -> ModuleType:
    """Import (on first use) and return the surface models serving ``version``.

    Loaded lazily so importing ``mcp.types`` (or this module) never pays for
    the surface packages; ``sys.modules`` caches after the first load.
    """
    return importlib.import_module(_SURFACE_MODULES[version])


def _surface_class(cls: type[BaseModel], version: str) -> type[BaseModel]:
    """Return the monolith ``cls``'s model class on the surface serving ``version``.

    Called on the 2026-07-28 emission path only (no other path consults a
    surface). Lookup is by name: the monolith name itself, then the
    schema-side name where the SDK deliberately diverges. A miss means the
    version's schema does not define the type — it has no wire form there.
    """
    module = _surface_module(version)
    for name in (cls.__name__, SDK_TO_SCHEMA_RENAMES.get(cls.__name__), _WIRE_NAME_ALIASES.get(cls.__name__)):
        if name is not None:
            found = getattr(module, name, None)
            if found is not None:
                return found
    raise UnsupportedAtVersionError(f"{cls.__name__} has no wire form at protocol version {version}", version=version)


def _summarize(err: ValidationError) -> str:
    """One line for the first validation error (the full chain is preserved)."""
    first = err.errors()[0]
    where = ".".join(str(segment) for segment in first["loc"]) or "<body>"
    remainder = f" (+{err.error_count() - 1} more)" if err.error_count() > 1 else ""
    return f"{where}: {first['msg']}{remainder}"


@overload
def parse_as(type_: type[_T], data: Mapping[str, Any], version: str) -> _T: ...
@overload
def parse_as(type_: Any, data: Mapping[str, Any], version: str) -> Any: ...
def parse_as(type_: Any, data: Mapping[str, Any], version: str) -> Any:
    """Validate inbound wire ``data`` as ``type_`` under ``version`` semantics.

    ``type_`` is a monolith model class or a public union alias
    (``ClientRequest``, ``ServerResult``, ``ContentBlock``,
    ``JSONRPCMessage``, ...). Parsing is one lenient superset parse at every
    version — unknown fields are never rejected — plus the 2026-07-28
    inbound mandates: a result carrying an unrecognized ``resultType`` value
    is rejected, a client request must carry all three reserved ``_meta``
    entries, and embedded input-request entries must each carry ``method``.
    Result-bearing unions resolve their member structurally — the arms
    matching the payload's keys are tried best match first and the first that
    validates wins — so the open-shaped ``EmptyResult`` arm cannot mask a
    better-matching member's validation failures, and a body is rejected only
    when every matching arm rejects it (with the best-matching arm's errors);
    unknown-shaped result bodies still parse (as the ``EmptyResult`` arm).
    Unknown ``version`` strings parse leniently with NO version-keyed mandates
    applied, and never raise for the version string itself.

    Raises:
        pydantic.ValidationError: ``data`` is not valid for ``type_`` at
            ``version``.
    """
    apply_mandates = is_version_at_least(version, "2026-07-28")
    result_arms = _result_union_arms(type_)
    if result_arms is None:
        parsed = _validate_refined(type_, data)
    elif apply_mandates and InputRequiredResult in result_arms and data.get("resultType") == "input_required":
        # 2026-07-28 response bodies discriminate by resultType: an
        # input-required body must resolve to InputRequiredResult even when it
        # also carries fields of another member. Union targets only — a
        # concrete `type_` always parses as the requested class.
        parsed = _validate_refined(InputRequiredResult, data)
    else:
        parsed = _parse_result_union(result_arms, data)
    if apply_mandates:
        _reject_unrecognized_result_type(type_, data)
        _reject_input_request_entries_without_method(parsed)
        _reject_missing_required_meta(parsed)
    return parsed


@cache
def _adapter(type_: Any) -> TypeAdapter[Any]:
    """One ``TypeAdapter`` per parse target, cached on the type object."""
    return TypeAdapter[Any](type_)


@cache
def _result_union_arms(type_: Any) -> tuple[type[Result], ...] | None:
    """The member tuple when ``type_`` is a union of ``Result`` subclasses
    (``ServerResult``, ``ClientResult``); ``None`` for anything else."""
    arms = get_args(type_)
    if arms and all(isinstance(arm, type) and issubclass(arm, Result) for arm in arms):
        return cast("tuple[type[Result], ...]", arms)
    return None


@cache
def _wire_field_names(cls: type[BaseModel]) -> frozenset[str]:
    """A model's wire-facing key set: each field's alias when it has one."""
    return frozenset(field.alias or name for name, field in cls.model_fields.items())


def _ranked_result_candidates(arms: tuple[type[Result], ...], data: Mapping[str, Any]) -> list[type[Result]]:
    """Rank the result-union members that structurally match the payload.

    A plain smart-union parse cannot do this job: ``EmptyResult`` declares no
    required fields, so it validates EVERY JSON object and would swallow the
    validation failures of a better-matching member — a discover result
    missing its required ``supportedVersions`` and a tool result whose content
    carries an unknown ``type`` must reject, not quietly fall back to
    ``EmptyResult``. Every arm recognizing strictly more of the payload's
    top-level keys than the base ``Result`` fields is a candidate, ranked by
    recognized-key count, ties keeping union declaration order (key counting
    cannot tell apart sibling arms with identical key sets, such as the
    single-content and array-content sampling results, so candidates are
    later validated in rank order rather than committing to one). The ranking
    is version-free and ignores unknown fields, so inbound leniency is
    untouched.
    """
    payload_keys = frozenset(data)
    base = len(payload_keys & _wire_field_names(Result))
    matched = [(len(payload_keys & _wire_field_names(arm)), arm) for arm in arms]
    return [arm for _, arm in sorted((m for m in matched if m[0] > base), key=lambda m: -m[0])]


def _parse_result_union(arms: tuple[type[Result], ...], data: Mapping[str, Any]) -> Any:
    """Validate ``data`` against the ranked candidate arms; first success wins.

    When every candidate fails, the reject surfaces the best-ranked
    candidate's errors (through the unknown-content-tag refinement), exactly
    as if that arm had been validated alone; the all-fail case never falls
    back to ``EmptyResult``. With no candidate at all — no arm beats the base
    ``Result`` key set — the body parses as the ``EmptyResult`` arm.
    """
    candidates = _ranked_result_candidates(arms, data)
    if not candidates:
        return _validate_refined(EmptyResult, data)
    try:
        return _adapter(candidates[0]).validate_python(data)
    except ValidationError as first_error:
        for candidate in candidates[1:]:
            try:
                return _adapter(candidate).validate_python(data)
            except ValidationError:
                continue
        refined = _refine_unknown_content_type(first_error, data)
        if refined is None:
            raise
        raise refined from first_error


def _validate_refined(type_: Any, data: Mapping[str, Any]) -> Any:
    """Superset-parse ``data`` as ``type_``, refining unknown content tags."""
    try:
        return _adapter(type_).validate_python(data)
    except ValidationError as err:
        refined = _refine_unknown_content_type(err, data)
        if refined is None:
            raise
        raise refined from err


def _refine_unknown_content_type(err: ValidationError, data: Mapping[str, Any]) -> ValidationError | None:
    """Convert per-arm failures on an unknown content ``type`` tag into a
    single unknown-tag error at the failing location.

    The monolith content unions are plain unions (their pre-2026 shape), so an
    unknown ``type`` value fails every arm with per-arm structural errors
    rather than one tag error — but an unknown content type is an unknown
    union member to every deployed SDK, including when it fails nested inside
    a parsed result's ``content`` list. Only a dict whose ``type`` value is a
    string outside the failing arms' tag set is converted: a recognized tag
    with bad fields, and a tag-less entry (e.g. an input-request entry with no
    ``method``), keep their structural errors.

    A plain-union error location is ``(..., "<ArmClassName>", "<field>")``
    (verified against pydantic 2.12); the arm-name segment is how
    content-union failures are recognized here.
    """
    failing_locations: dict[tuple[str | int, ...], set[str]] = {}
    for line in err.errors():
        location = line["loc"]
        for index, segment in enumerate(location):
            if isinstance(segment, str) and segment in _CONTENT_BLOCK_TAGS:
                failing_locations.setdefault(location[:index], set()).add(segment)
                break
    line_errors: list[InitErrorDetails] = []
    for location, arm_names in failing_locations.items():
        fragment: Any = data
        for segment in location:
            fragment = fragment[segment]
        if not isinstance(fragment, dict):
            continue
        tag = cast("dict[str, Any]", fragment).get("type")
        expected_tags = sorted(str(_CONTENT_BLOCK_TAGS[name]) for name in arm_names)
        if isinstance(tag, str) and tag not in expected_tags:
            line_errors.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        "union_tag_invalid",
                        "Input tag '{tag}' found using {discriminator} does not match any of the "
                        "expected tags: {expected_tags}",
                        {
                            "discriminator": "'type'",
                            "tag": tag,
                            "expected_tags": ", ".join(repr(expected) for expected in expected_tags),
                        },
                    ),
                    loc=location,
                    input=fragment,
                )
            )
    if not line_errors:
        return None
    return ValidationError.from_exception_data(err.title, line_errors)


def _reject_unrecognized_result_type(type_: Any, data: Mapping[str, Any]) -> None:
    """2026-07-28 inbound mandate: an unrecognized ``resultType`` rejects.

    Applies when the parse target is a ``Result`` class or a result-bearing
    union — a stray ``resultType`` key on a request or any other type is an
    ordinary unknown field and stays accepted. An absent field is accepted
    (the spec defines absence as "complete") and a recognized value is
    retained; only a present-and-unrecognized string value rejects, with the
    pinned error type ``result_type_invalid``.
    """
    if not _is_result_parse_target(type_):
        return
    value = data.get("resultType")
    if isinstance(value, str) and value not in _RECOGNIZED_RESULT_TYPES:
        raise ValidationError.from_exception_data(
            getattr(type_, "__name__", "Result"),
            [
                InitErrorDetails(
                    type=PydanticCustomError(
                        "result_type_invalid",
                        "unrecognized resultType {result_type}; this protocol version defines "
                        "'complete' and 'input_required'",
                        {"result_type": value},
                    ),
                    loc=("resultType",),
                    input=value,
                )
            ],
        )


def _reject_input_request_entries_without_method(parsed: Any) -> None:
    """2026-07-28 inbound mandate: embedded input-request entries carry
    ``method``.

    The values of an input-required result's ``inputRequests`` map are full
    request payloads, and the schema requires ``method`` on every request. The
    monolith request models default their method literal (so handler code can
    construct them without boilerplate), which would let a method-less entry
    quietly validate as the one member whose remaining fields are all
    optional; the mandate instead checks that every entry actually supplied
    the field. A missing ``method`` is a structural failure (plain ``missing``
    error), not an unknown union member.
    """
    if not isinstance(parsed, InputRequiredResult) or parsed.input_requests is None:
        return
    missing = [key for key, entry in parsed.input_requests.items() if "method" not in entry.model_fields_set]
    if missing:
        raise ValidationError.from_exception_data(
            type(parsed).__name__,
            [InitErrorDetails(type="missing", loc=("inputRequests", key, "method"), input=None) for key in missing],
        )


def _is_result_parse_target(type_: Any) -> bool:
    """True when ``type_`` is a ``Result`` class or a result-bearing union."""
    if isinstance(type_, type):
        return issubclass(cast("type[object]", type_), Result)
    return _result_union_arms(type_) is not None


def _reject_missing_required_meta(parsed: Any) -> None:
    """2026-07-28 inbound mandate: client requests carry the reserved
    ``_meta`` triple.

    Every 2026-07-28 client request requires the
    ``io.modelcontextprotocol/{protocolVersion,clientInfo,clientCapabilities}``
    entries in ``params._meta``, each independently; a missing ``params``, a
    missing ``_meta``, or any missing entry rejects with the pinned error type
    ``missing_required_meta`` (one error per missing entry).
    """
    if not isinstance(parsed, Request):
        return
    request = cast("Request[Any, Any]", parsed)
    if request.method not in _REQUIRED_META_METHODS:
        return
    params = request.params
    meta: Mapping[str, Any] = params.meta if isinstance(params, RequestParams) and params.meta is not None else {}
    missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValidationError.from_exception_data(
            type(request).__name__,
            [
                InitErrorDetails(
                    type=PydanticCustomError(
                        "missing_required_meta",
                        "required reserved _meta entry {meta_key} is missing",
                        {"meta_key": key},
                    ),
                    loc=("params", "_meta", key),
                    input=dict(meta),
                )
                for key in missing
            ],
        )
