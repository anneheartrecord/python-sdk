"""Emission facts of the wire boundary.

Emission is additive-only: for 2025-11-25 and earlier the output is the plain
monolith dump byte for byte and nothing ever raises; for 2026-07-28 the
boundary injects the protocol-required fields, validates through the
2026-07-28 surface models as a check, and still emits the injected dump.
Each test names the spec fact it pins in plain words, with its provenance
class: spec-mandated (the published schema or spec prose requires it) or
deployed-peer-mandated (a behavior real deployed SDKs enforce on the wire).
Byte-identity assertions compare against the plain monolith dump via
``json.dumps`` so key order is part of the guarantee.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import BaseModel, FileUrl, ValidationError

from mcp.types import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    LOG_LEVEL_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    AudioContent,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    CancelledNotification,
    CancelledNotificationParams,
    CancelTaskRequest,
    CancelTaskRequestParams,
    ClientCapabilities,
    CompleteResult,
    Completion,
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    CreateMessageResultWithTools,
    DiscoverResult,
    ElicitCompleteNotification,
    ElicitCompleteNotificationParams,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    ElicitResult,
    EmptyResult,
    ErrorData,
    GetPromptResult,
    Icon,
    Implementation,
    InitializedNotification,
    InitializeRequest,
    InitializeRequestParams,
    InputRequiredResult,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    ListRootsRequest,
    ListRootsResult,
    ListToolsRequest,
    ListToolsResult,
    LoggingMessageNotification,
    LoggingMessageNotificationParams,
    PaginatedRequestParams,
    PingRequest,
    ProgressNotification,
    ProgressNotificationParams,
    PromptMessage,
    RequestParamsMeta,
    ResourceLink,
    Root,
    RootsCapability,
    SamplingMessage,
    ServerCapabilities,
    SubscribeRequest,
    SubscribeRequestParams,
    SubscriptionFilter,
    SubscriptionsListenRequest,
    SubscriptionsListenRequestParams,
    TaskMetadata,
    TextContent,
    Tool,
    ToolUseContent,
)
from mcp.types.wire import (
    UnknownProtocolVersionError,
    UnsupportedAtVersionError,
    serialize_for,
)

V1 = "2024-11-05"
V2 = "2025-03-26"
V3 = "2025-06-18"
V4 = "2025-11-25"
D = "2026-07-28"
RELEASED = (V1, V2, V3, V4)


def monolith_dump(model: BaseModel) -> dict[str, Any]:
    """The plain user dump — the byte-identity reference for released versions."""
    return model.model_dump(by_alias=True, mode="json", exclude_none=True)


def as_bytes(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=False)


def identity_meta() -> dict[str, Any]:
    """The caller-supplied session identity a 2026-07-28 client request needs."""
    return {
        CLIENT_INFO_META_KEY: {"name": "example-client", "version": "1.0.0"},
        CLIENT_CAPABILITIES_META_KEY: {},
    }


def as_meta(entries: dict[str, Any]) -> RequestParamsMeta:
    """Build a params _meta value from plain entries (the open-map wire form)."""
    return cast("RequestParamsMeta", entries)


# --- payload domain and version registry ---------------------------------


@pytest.mark.parametrize(
    "fragment",
    [
        TextContent(text="x"),
        ClientCapabilities(),
        SamplingMessage(role="user", content=TextContent(text="x")),
        PaginatedRequestParams(),
    ],
    ids=lambda fragment: type(fragment).__name__,
)
def test_serialize_for_refuses_bare_fragments(fragment: BaseModel) -> None:
    """Fragments are shaped only inside the body that carries them; a bare
    fragment is a programming error, refused identically at every version."""
    with pytest.raises(TypeError, match="message body or an envelope model"):
        serialize_for(fragment, V4)


def test_bare_fragment_refused_before_the_version_check() -> None:
    """Argument validation precedes version lookup: (bare fragment, unknown
    version) deterministically raises TypeError."""
    with pytest.raises(TypeError):
        serialize_for(TextContent(text="x"), "not-a-version")


def test_serialize_for_unknown_version() -> None:
    """Emission never guesses a wire shape for a version it does not know."""
    with pytest.raises(UnknownProtocolVersionError) as exc_info:
        serialize_for(EmptyResult(), "2030-01-01")
    assert exc_info.value.version == "2030-01-01"
    assert exc_info.value.known == (V1, V2, V3, V4, D)


# --- envelope frames (version-independent) --------------------------------


@pytest.mark.parametrize("version", [V1, V4, D])
def test_envelope_request_emits_verbatim(version: str) -> None:
    """JSON-RPC envelope frames are identical in every protocol version; a
    bodyless request emits without a params key (spec-mandated)."""
    frame = JSONRPCRequest(jsonrpc="2.0", id=1, method="ping")
    assert serialize_for(frame, version) == {"jsonrpc": "2.0", "id": 1, "method": "ping"}


def test_envelope_notification_emits_verbatim() -> None:
    frame = JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized")
    assert serialize_for(frame, V1) == {"jsonrpc": "2.0", "method": "notifications/initialized"}


def test_envelope_error_frame_emits_verbatim() -> None:
    """Error frames carry id and the error object unchanged (spec-mandated)."""
    frame = JSONRPCError(jsonrpc="2.0", id=5, error=ErrorData(code=-32601, message="Method not found"))
    assert serialize_for(frame, V1) == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {"code": -32601, "message": "Method not found"},
    }


def test_generic_envelope_interiors_are_opaque() -> None:
    """The untyped result interior of a generic envelope passes through with
    no injection and no strip — payload shaping applies only to typed payload
    models, never by guessing what an untyped dict holds."""
    frame = JSONRPCResponse(jsonrpc="2.0", id=2, result={"resultType": "complete", "ttlMs": 9})
    assert serialize_for(frame, V1)["result"] == {"resultType": "complete", "ttlMs": 9}


# --- identity: released-version dumps are byte-identical ------------------

_IDENTITY_CASES: list[BaseModel] = [
    PingRequest(),
    InitializeRequest(
        params=InitializeRequestParams(
            protocol_version=V2,
            capabilities=ClientCapabilities(roots=RootsCapability(list_changed=True)),
            client_info=Implementation(name="example-client", version="1.0.0"),
        )
    ),
    # A request whose params._meta carries user-set reserved keys and a vendor
    # key: _meta entries are retained verbatim on emission at every version
    # (deployed-peer-mandated: open _meta maps in all deployed SDKs).
    CallToolRequest(
        params=CallToolRequestParams(
            name="echo",
            arguments={"text": "hi"},
            _meta={
                PROTOCOL_VERSION_META_KEY: D,
                CLIENT_INFO_META_KEY: {"name": "example-client", "version": "1.0.0"},
                CLIENT_CAPABILITIES_META_KEY: {},
                LOG_LEVEL_META_KEY: "info",
                "vendor-trace": "trace-9001",
            },
        )
    ),
    SubscribeRequest(params=SubscribeRequestParams(uri="file:///r")),
    ListRootsRequest(),
    CreateMessageRequest(
        params=CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=TextContent(text="q"))], max_tokens=10
        )
    ),
    InitializedNotification(),
    CancelledNotification(params=CancelledNotificationParams(request_id=7)),
    ProgressNotification(params=ProgressNotificationParams(progress_token="t", progress=0.5)),
    LoggingMessageNotification(params=LoggingMessageNotificationParams(level="info", data="x")),
    EmptyResult(),
    CallToolResult(content=[TextContent(text="hello")]),
    ListToolsResult(tools=[Tool(name="t", input_schema={"type": "object"})]),
    GetPromptResult(messages=[PromptMessage(role="user", content=TextContent(text="hi"))]),
    CompleteResult(completion=Completion(values=["a"])),
    ListRootsResult(roots=[Root(uri=FileUrl("file:///workspace"))]),
    CreateMessageResult(role="assistant", content=TextContent(text="ok"), model="m"),
]


@pytest.mark.parametrize("version", RELEASED)
@pytest.mark.parametrize("model", _IDENTITY_CASES, ids=lambda model: type(model).__name__)
def test_released_version_emission_is_byte_identical(model: BaseModel, version: str) -> None:
    """For values valid at the target released version, emission is the plain
    monolith dump byte for byte — same keys, same order, same values."""
    assert as_bytes(serialize_for(model, version)) == as_bytes(monolith_dump(model))


# Values carrying constructs the target versions' schemas do not define:
# emission is additive-only, so these dump identically everywhere too.
# Deployed-peer-mandated pass-through: no deployed SDK rejects unknown object
# fields on non-empty bodies, and gating whole newer constructs (a result
# shape or content kind a peer's version predates) is the session layer's
# capability and version gating, not the type layer's.
_NEWER_CONSTRUCT_CASES: list[BaseModel] = [
    CallToolResult(content=[TextContent(text="hello")], result_type="complete"),
    ListToolsResult(tools=[], ttl_ms=5000, cache_scope="public"),
    CallToolRequest(params=CallToolRequestParams(name="t", task=TaskMetadata(ttl=60_000))),
    InputRequiredResult(request_state="opaque-state"),
    ElicitResult(action="accept", content={"choices": ["a", "b"]}),
]


@pytest.mark.parametrize("version", RELEASED)
@pytest.mark.parametrize("model", _NEWER_CONSTRUCT_CASES, ids=lambda model: type(model).__name__)
def test_newer_constructs_dump_unchanged_on_released_versions(model: BaseModel, version: str) -> None:
    """A field or type a released version's schema does not define still
    emits as the plain monolith dump — nothing is ever removed from a
    caller's payload, and emission on these versions never raises."""
    assert as_bytes(serialize_for(model, version)) == as_bytes(monolith_dump(model))


# --- resultType ------------------------------------------------------------


def test_result_type_injected_on_2026_07_28_emission() -> None:
    """resultType is required on 2026-07-28 results; an unset field emits as
    "complete" (spec-mandated)."""
    out = serialize_for(CallToolResult(content=[TextContent(text="hello")]), D)
    assert out == {"content": [{"type": "text", "text": "hello"}], "isError": False, "resultType": "complete"}


def test_result_type_user_value_never_clobbered() -> None:
    out = serialize_for(CallToolResult(content=[], result_type="complete"), D)
    assert out["resultType"] == "complete"


def test_input_required_result_announces_itself() -> None:
    """An input-required result emits resultType "input_required" with its
    embedded requests intact (spec-mandated)."""
    result = InputRequiredResult(request_state="opaque-state")
    out = serialize_for(result, D)
    assert out == {"requestState": "opaque-state", "resultType": "input_required"}


@pytest.mark.parametrize("version", [V1, V4])
def test_user_set_result_type_emits_on_released_versions(version: str) -> None:
    """A user-set resultType survives emission at every version: extra keys
    on non-empty results pass every deployed peer, and the boundary never
    removes caller data (deployed-peer-mandated pass-through). The one
    deployed-peer hazard — an otherwise-empty result body — is documented on
    the field rather than guarded by a strip."""
    result = CallToolResult(content=[TextContent(text="hello")], result_type="complete")
    out = serialize_for(result, version)
    assert out["resultType"] == "complete"
    assert as_bytes(out) == as_bytes(monolith_dump(result))


@pytest.mark.parametrize("version", [V1, V4])
def test_empty_result_dumps_exactly_empty(version: str) -> None:
    """An empty result is exactly {} on released versions by construction:
    every post-2025-06-18 field defaults to None and the dump excludes None.
    That matters because deployed TypeScript and Rust clients hard-reject an
    empty result carrying any extra key (deployed-peer-mandated; recorded on
    the result_type field's docstring)."""
    assert as_bytes(serialize_for(EmptyResult(), version)) == "{}"


def test_empty_result_carries_result_type_at_2026_07_28() -> None:
    assert serialize_for(EmptyResult(), D) == {"resultType": "complete"}


@pytest.mark.parametrize(
    "result",
    [
        ElicitResult(action="accept"),
        CreateMessageResult(role="assistant", content=TextContent(text="ok"), model="m"),
        ListRootsResult(roots=[]),
    ],
    ids=lambda result: type(result).__name__,
)
def test_no_result_type_injected_on_client_results_at_2026_07_28(result: BaseModel) -> None:
    """The 2026-07-28 schema defines resultType on the Result base and every
    server result, but the revision removed the server -> client request
    channel: the client results survive only as embedded payload values and
    carry no resultType, so none is injected (spec-mandated scope)."""
    assert "resultType" not in serialize_for(result, D)


# --- caching directives ----------------------------------------------------


def test_caching_defaults_injected_on_2026_07_28() -> None:
    """ttlMs/cacheScope are required on cacheable results from 2026-07-28;
    unset fields get the don't-cache pair (spec-mandated requiredness, SDK
    default choice)."""
    out = serialize_for(ListToolsResult(tools=[]), D)
    assert out["ttlMs"] == 0
    assert out["cacheScope"] == "private"


def test_caching_user_values_pass_unclobbered() -> None:
    out = serialize_for(ListToolsResult(tools=[], ttl_ms=5000, cache_scope="public"), D)
    assert out["ttlMs"] == 5000
    assert out["cacheScope"] == "public"


def test_user_set_caching_fields_emit_on_released_versions() -> None:
    """User-set caching directives survive emission on versions that predate
    them: peers ignore unknown keys on non-empty bodies, and nothing is ever
    removed from a caller's payload (deployed-peer-mandated pass-through)."""
    result = ListToolsResult(tools=[], ttl_ms=5000, cache_scope="public")
    out = serialize_for(result, V4)
    assert out["ttlMs"] == 5000
    assert out["cacheScope"] == "public"
    assert as_bytes(out) == as_bytes(monolith_dump(result))


def test_discover_result_emits_with_policy_defaults_at_2026_07_28() -> None:
    result = DiscoverResult(
        supported_versions=[V4, D],
        capabilities=ServerCapabilities(),
        server_info=Implementation(name="fixture-server", version="1.0.0"),
    )
    out = serialize_for(result, D)
    assert out["supportedVersions"] == [V4, D]
    assert out["ttlMs"] == 0
    assert out["cacheScope"] == "private"
    assert out["resultType"] == "complete"
    assert "instructions" not in out


def test_discover_result_dumps_on_released_versions() -> None:
    """server/discover and its result exist only in the 2026-07-28 schema,
    but a released-version emission still dumps it unchanged: the type layer
    never blocks a payload, and keeping a 2026-07-28-only method off a
    2025-11-25 session is the session layer's dispatch gating (the method
    tables carry the data)."""
    result = DiscoverResult(
        supported_versions=[D], capabilities=ServerCapabilities(), server_info=Implementation(name="s", version="1")
    )
    assert as_bytes(serialize_for(result, V4)) == as_bytes(monolith_dump(result))


# --- reserved _meta entries on client requests ----------------------------


def test_protocol_version_injected_into_request_meta_at_2026_07_28() -> None:
    """2026-07-28 client requests carry the reserved _meta entries; the
    boundary derives and merges protocolVersion, materializing params and
    _meta when the handler left them unset (spec-mandated)."""
    request = CallToolRequest(
        params=CallToolRequestParams(name="get-weather", arguments={"city": "Berlin"}, _meta=as_meta(identity_meta()))
    )
    out = serialize_for(request, D)
    assert out["params"]["_meta"][PROTOCOL_VERSION_META_KEY] == D
    assert out["params"]["_meta"][CLIENT_INFO_META_KEY] == {"name": "example-client", "version": "1.0.0"}
    assert out["params"]["_meta"][CLIENT_CAPABILITIES_META_KEY] == {}
    assert out["params"]["name"] == "get-weather"
    assert out["params"]["arguments"] == {"city": "Berlin"}


def test_protocol_version_merge_never_overwrites_a_caller_value() -> None:
    meta = identity_meta() | {PROTOCOL_VERSION_META_KEY: "caller-pinned"}
    request = ListToolsRequest(params=PaginatedRequestParams(_meta=as_meta(meta)))
    out = serialize_for(request, D)
    assert out["params"]["_meta"][PROTOCOL_VERSION_META_KEY] == "caller-pinned"


def test_progress_token_coexists_with_the_reserved_entries() -> None:
    meta = identity_meta() | {"progressToken": "tok-1"}
    out = serialize_for(ListToolsRequest(params=PaginatedRequestParams(_meta=as_meta(meta))), D)
    assert out["params"]["_meta"]["progressToken"] == "tok-1"
    assert out["params"]["_meta"][PROTOCOL_VERSION_META_KEY] == D


def test_missing_session_identity_refuses_at_2026_07_28() -> None:
    """The boundary never synthesizes clientInfo/clientCapabilities — they
    are session identity. A bare request has no legal 2026-07-28 wire form
    (the schema requires all three reserved entries)."""
    with pytest.raises(UnsupportedAtVersionError) as exc_info:
        serialize_for(ListToolsRequest(), D)
    assert exc_info.value.version == D
    assert isinstance(exc_info.value.__cause__, ValidationError)
    # Two entries are missing; the message carries one and counts the rest.
    assert "more" in str(exc_info.value)


def test_partially_missing_session_identity_also_refuses() -> None:
    request = ListToolsRequest(
        params=PaginatedRequestParams(_meta={CLIENT_INFO_META_KEY: {"name": "c", "version": "1"}})
    )
    with pytest.raises(UnsupportedAtVersionError) as exc_info:
        serialize_for(request, D)
    assert "clientCapabilities" in str(exc_info.value)


def test_nothing_injected_below_2026_07_28() -> None:
    """On earlier versions an unset params stays omitted — the dump is the
    plain monolith dump (deployed-peer-mandated byte identity)."""
    assert as_bytes(serialize_for(ListToolsRequest(), V4)) == as_bytes(monolith_dump(ListToolsRequest()))


# --- capabilities ----------------------------------------------------------


def test_roots_list_changed_survives_at_2026_07_28() -> None:
    """The 2026-07-28 schema's roots capability is an empty object — its
    listChanged flag exists only through 2025-11-25 — but a caller who sets
    the flag keeps it: the boundary never removes caller data, the surface
    models ignore unknown keys, and 2026-07-28 peers do the same
    (deployed-peer-mandated pass-through)."""
    request = ListToolsRequest(
        params=PaginatedRequestParams(
            _meta={
                CLIENT_INFO_META_KEY: Implementation(name="ExampleClient", version="1.0.0"),
                CLIENT_CAPABILITIES_META_KEY: ClientCapabilities(roots=RootsCapability(list_changed=True)),
            }
        )
    )
    out = serialize_for(request, D)
    assert out == {
        "method": "tools/list",
        "params": {
            "_meta": {
                CLIENT_INFO_META_KEY: {"name": "ExampleClient", "version": "1.0.0"},
                CLIENT_CAPABILITIES_META_KEY: {"roots": {"listChanged": True}},
                PROTOCOL_VERSION_META_KEY: D,
            }
        },
    }


def test_capabilities_extensions_emit_below_2026_07_28() -> None:
    """The extensions field is new in 2026-07-28, but a caller who sets it on
    an earlier-version session keeps it: unknown capability keys are
    wire-safe against every deployed peer, and the emission is the plain
    monolith dump (deployed-peer-mandated pass-through)."""
    capabilities = ClientCapabilities(
        roots=RootsCapability(list_changed=True),
        extensions={"io.modelcontextprotocol/oauth-client-credentials": {}},
    )
    request = InitializeRequest(
        params=InitializeRequestParams(
            protocol_version=V4, capabilities=capabilities, client_info=Implementation(name="c", version="1")
        )
    )
    out = serialize_for(request, V4)
    assert out["params"]["capabilities"]["extensions"] == {"io.modelcontextprotocol/oauth-client-credentials": {}}
    assert as_bytes(out) == as_bytes(monolith_dump(request))


def test_capabilities_extensions_emitted_at_2026_07_28() -> None:
    """Client extensions ride the _meta clientCapabilities projection at
    2026-07-28 (spec-mandated: the field exists there)."""
    meta = {
        CLIENT_INFO_META_KEY: {"name": "c", "version": "1"},
        CLIENT_CAPABILITIES_META_KEY: ClientCapabilities(extensions={"io.modelcontextprotocol/x": {}}),
    }
    out = serialize_for(ListToolsRequest(params=PaginatedRequestParams(_meta=as_meta(meta))), D)
    assert out["params"]["_meta"][CLIENT_CAPABILITIES_META_KEY] == {"extensions": {"io.modelcontextprotocol/x": {}}}


def test_server_extensions_emitted_in_discover_result() -> None:
    result = DiscoverResult(
        supported_versions=[D],
        capabilities=ServerCapabilities(extensions={"io.modelcontextprotocol/y": {}}),
        server_info=Implementation(name="s", version="1"),
    )
    assert serialize_for(result, D)["capabilities"] == {"extensions": {"io.modelcontextprotocol/y": {}}}


def test_capability_extension_values_admit_every_json_type_at_2026_07_28() -> None:
    """Extension values are arbitrary JSON, so fractional numbers and nulls at
    any depth survive the 2026-07-28 revalidation (spec-mandated: the schema
    source types extension values as any JSON value)."""
    extension_value = {"ratio": 0.5, "experimental": None, "steps": [1.5, None, "done"]}
    meta = {
        CLIENT_INFO_META_KEY: {"name": "c", "version": "1"},
        CLIENT_CAPABILITIES_META_KEY: ClientCapabilities(extensions={"io.modelcontextprotocol/x": extension_value}),
    }
    out = serialize_for(ListToolsRequest(params=PaginatedRequestParams(_meta=as_meta(meta))), D)
    emitted = out["params"]["_meta"][CLIENT_CAPABILITIES_META_KEY]["extensions"]["io.modelcontextprotocol/x"]
    assert emitted == extension_value


# --- tasks (a 2025-11-25 schema fact; emission never gates them) -----------


def test_task_metadata_emitted_at_2025_11_25() -> None:
    """The task field on augmentable params is a 2025-11-25 schema fact
    (spec-mandated availability; the emission itself is the plain dump)."""
    request = CallToolRequest(params=CallToolRequestParams(name="t", task=TaskMetadata(ttl=60_000)))
    assert serialize_for(request, V4)["params"]["task"] == {"ttl": 60000}


@pytest.mark.parametrize("version", [V3, D])
def test_task_metadata_emits_outside_2025_11_25_too(version: str) -> None:
    """Only the 2025-11-25 schema defines the task field, but a caller who
    sets it elsewhere keeps it — on earlier versions the emission is the
    plain dump, and at 2026-07-28 the field is an unknown key the surface
    models ignore rather than reject. Targeting task metadata at the one
    version where peers act on it is the session layer's job
    (deployed-peer-mandated pass-through: unknown keys are wire-safe)."""
    meta = as_meta(identity_meta()) if version == D else None
    request = CallToolRequest(params=CallToolRequestParams(name="t", task=TaskMetadata(ttl=60_000), _meta=meta))
    out = serialize_for(request, version)
    assert out["params"]["task"] == {"ttl": 60000}
    if version == V3:
        assert as_bytes(out) == as_bytes(monolith_dump(request))


def _initialize_with_tasks(protocol_version: str, tasks: dict[str, Any] | None) -> InitializeRequest:
    capabilities = ClientCapabilities() if tasks is None else ClientCapabilities.model_validate({"tasks": tasks})
    return InitializeRequest(
        params=InitializeRequestParams(
            protocol_version=protocol_version,
            capabilities=capabilities,
            client_info=Implementation(name="c", version="1"),
        )
    )


def test_tasks_capability_subtree_emitted_at_2025_11_25() -> None:
    request = _initialize_with_tasks(V4, {"requests": {"sampling": {"createMessage": {}}}})
    out = serialize_for(request, V4)
    assert out["params"]["capabilities"]["tasks"] == {"requests": {"sampling": {"createMessage": {}}}}


def test_tasks_capability_subtree_emits_below_2025_11_25_too() -> None:
    """The tasks capability subtree exists only in the 2025-11-25 schema, but
    a caller who sets it on an older session keeps it: unknown capability
    keys are wire-safe against every deployed peer (deployed-peer-mandated
    pass-through)."""
    request = _initialize_with_tasks(V3, {"requests": {"sampling": {"createMessage": {}}}})
    out = serialize_for(request, V3)
    assert out["params"]["capabilities"]["tasks"] == {"requests": {"sampling": {"createMessage": {}}}}
    assert as_bytes(out) == as_bytes(monolith_dump(request))


# --- newer optional fields pass through (no narrowing) ---------------------


def test_icons_and_title_pass_through_on_older_versions() -> None:
    """New optional fields on known types are wire-safe against every
    deployed peer; emission never strips them on versions that predate them
    (deployed-peer-mandated: no gating needed)."""
    result = ListToolsResult(
        tools=[Tool(name="t", title="T", input_schema={"type": "object"}, icons=[Icon(src="https://e/i.png")])]
    )
    for version in (V1, V3):
        out = serialize_for(result, version)
        assert out["tools"][0]["title"] == "T"
        assert out["tools"][0]["icons"] == [{"src": "https://e/i.png"}]
        assert as_bytes(out) == as_bytes(monolith_dump(result))


@pytest.mark.parametrize("version", [V1, V3, D])
def test_scalar_structured_content_passes_at_every_version(version: str) -> None:
    """Values are never narrowed on emission: a non-object structuredContent
    passes through unchanged everywhere."""
    out = serialize_for(CallToolResult(content=[], structured_content=5), version)
    assert out["structuredContent"] == 5


def test_unset_structured_content_is_absent() -> None:
    out = serialize_for(CallToolResult(content=[]), D)
    assert "structuredContent" not in out


def test_object_structured_content_emitted_at_2026_07_28() -> None:
    out = serialize_for(
        CallToolResult(content=[TextContent(text="22.5 C")], structured_content={"temperature": 22.5}), D
    )
    assert out["structuredContent"] == {"temperature": 22.5}
    assert out["resultType"] == "complete"


def test_opened_tool_schemas_pass_through_unchanged() -> None:
    """Tool input schemas accept the full JSON Schema vocabulary; every
    keyword — including $ref/$defs and conditionals — survives emission
    verbatim (spec-mandated: the schemas leave these objects open)."""
    schema = {
        "type": "object",
        "properties": {"query": {"$ref": "#/$defs/nonEmptyString"}},
        "required": ["query"],
        "if": {"required": ["mode"]},
        "then": {"required": ["filters"]},
        "$defs": {"nonEmptyString": {"type": "string", "minLength": 1}},
    }
    result = ListToolsResult(tools=[Tool(name="search", input_schema=schema)], ttl_ms=0, cache_scope="private")
    assert serialize_for(result, D)["tools"][0]["inputSchema"] == schema


# --- content blocks ---------------------------------------------------------


def test_audio_content_passes_through_at_2024_11_05() -> None:
    """audio content entered the schema in 2025-03-26 but is deliberately not
    gated on emission to older peers (sibling parity; peers reject unknown
    blocks at request level — accepted risk)."""
    result = CallToolResult(content=[AudioContent(data="QQ==", mime_type="audio/wav")])
    out = serialize_for(result, V1)
    assert out["content"][0] == {"type": "audio", "data": "QQ==", "mimeType": "audio/wav"}


@pytest.mark.parametrize("version", [V1, V2])
def test_resource_link_passes_through_before_2025_06_18(version: str) -> None:
    result = CallToolResult(content=[ResourceLink(name="r", uri="https://example.com/r")])
    out = serialize_for(result, version)
    assert out["content"][0]["type"] == "resource_link"
    assert out["content"][0]["uri"] == "https://example.com/r"


# --- sampling and tool content bounds ---------------------------------------


def test_tool_content_dumps_at_2025_06_18_and_earlier() -> None:
    """tool_use/tool_result sampling content entered the schema in
    2025-11-25; older emissions still dump it unchanged — every deployed SDK
    rejects an unknown content tag on receipt, so sending it to a peer that
    cannot understand it is gated by the session layer's sampling-tools
    capability check, never reshaped or blocked by the type layer
    (deployed-peer evidence carried at the session gate)."""
    request = CreateMessageRequest(
        params=CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=ToolUseContent(name="t", id="1", input={}))],
            max_tokens=5,
        )
    )
    assert as_bytes(serialize_for(request, V3)) == as_bytes(monolith_dump(request))
    assert serialize_for(request, V4)["params"]["messages"][0]["content"]["type"] == "tool_use"


def test_array_sampling_content_dumps_at_2025_06_18_and_earlier() -> None:
    """Multi-block sampling messages entered the schema in 2025-11-25; older
    emissions dump them unchanged (single-block peers fail on receipt, so
    the session layer gates the construct by negotiated version — the type
    layer neither collapses nor refuses it)."""
    request = CreateMessageRequest(
        params=CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=[TextContent(text="a"), TextContent(text="b")])],
            max_tokens=5,
        )
    )
    assert as_bytes(serialize_for(request, V3)) == as_bytes(monolith_dump(request))
    out = serialize_for(request, V4)
    assert out["params"]["messages"][0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_wide_sampling_result_dumps_at_2025_06_18_and_earlier() -> None:
    """The wide-content sampling result is typed wide by the schemas from
    2025-11-25; on earlier versions it still dumps unchanged (the session
    layer keeps wide results off sessions whose peers parse single-block
    content)."""
    result = CreateMessageResultWithTools(role="assistant", content=[TextContent(text="x")], model="m")
    assert as_bytes(serialize_for(result, V3)) == as_bytes(monolith_dump(result))
    assert serialize_for(result, V4)["content"] == [{"type": "text", "text": "x"}]


# --- multi-round-trip results ------------------------------------------------


def test_input_required_result_dumps_below_2026_07_28() -> None:
    """InputRequiredResult exists only in the 2026-07-28 schema; earlier
    emissions still dump it unchanged — keeping the new result shape off a
    pre-2026-07-28 session is the session layer's version gating."""
    result = InputRequiredResult(request_state="s")
    assert as_bytes(serialize_for(result, V4)) == as_bytes(monolith_dump(result))


def test_empty_input_required_result_refused() -> None:
    """The 2026-07-28 schema requires at least one of inputRequests /
    requestState on the wire; the constraint is spec prose, checked
    explicitly (spec-mandated)."""
    with pytest.raises(UnsupportedAtVersionError, match="neither input_requests nor request_state"):
        serialize_for(InputRequiredResult(), D)


def test_embedded_input_responses_pass_through_verbatim() -> None:
    """The boundary never reshapes embedded request/response payloads:
    caller-set _meta and resultType on inputResponses values survive
    2026-07-28 emission untouched (embedded hygiene is the caller's job)."""
    embedded = CreateMessageResult(
        role="assistant", content=TextContent(text="ok"), model="m", result_type="complete", _meta={"k": "v"}
    )
    request = CallToolRequest(
        params=CallToolRequestParams(name="retry-me", _meta=as_meta(identity_meta()), input_responses={"r1": embedded})
    )
    out = serialize_for(request, D)
    entry = out["params"]["inputResponses"]["r1"]
    assert entry["resultType"] == "complete"
    assert entry["_meta"] == {"k": "v"}


# --- elicitation and cancellation bounds -------------------------------------


def test_url_mode_elicitation_dumps_at_2025_06_18() -> None:
    """URL-mode elicitation entered the schema in 2025-11-25; an earlier
    emission still dumps it unchanged — the url capability gate that keeps
    it off form-only peers is the session layer's."""
    request = ElicitRequest(
        params=ElicitRequestURLParams(message="auth needed", url="https://example.com/auth", elicitation_id="e-1")
    )
    assert as_bytes(serialize_for(request, V3)) == as_bytes(monolith_dump(request))
    assert serialize_for(request, V4)["params"]["mode"] == "url"


def test_list_string_elicit_content_dumps_below_2025_11_25() -> None:
    """Multi-select (list-of-strings) elicitation values entered the schema
    in 2025-11-25; an earlier emission still dumps them unchanged."""
    result = ElicitResult(action="accept", content={"choices": ["a", "b"]})
    assert as_bytes(serialize_for(result, V3)) == as_bytes(monolith_dump(result))
    assert serialize_for(result, V4)["content"] == {"choices": ["a", "b"]}


@pytest.mark.parametrize("version", [V3, V4, D])
def test_fractional_elicit_content_emits_at_every_modeled_version(version: str) -> None:
    """Form answers are string | number | boolean (string arrays from
    2025-11-25), so a fractional number is a legal elicitation answer at
    every version that models elicitation; the value keeps its exact JSON
    rendering (spec-mandated; the pinned schema renderings say "integer"
    only as a render artifact the surface packages deliberately widen).
    Byte identity holds at 2026-07-28 too: ElicitResult is a client result,
    so no resultType is injected there."""
    result = ElicitResult(action="accept", content={"ratio": 0.5})
    out = serialize_for(result, version)
    assert out["content"] == {"ratio": 0.5}
    assert as_bytes(out) == as_bytes(monolith_dump(result))


@pytest.mark.parametrize("version", [V3, V4, D])
def test_null_elicit_content_values_pass_through_at_every_modeled_version(version: str) -> None:
    """No schema version types a null elicitation answer — the monolith's
    None value arm exists for v1.x constructor compatibility — but emitted
    values are caller data and travel verbatim at every version that models
    elicitation, exactly as python v1.x itself constructs, accepts, and
    emits the same body (deployed-peer-mandated pass-through; the
    2026-07-28 validation check skips the null entries rather than refusing
    a body the boundary will emit from the dump anyway)."""
    result = ElicitResult(action="accept", content={"x": None, "y": "ok"})
    out = serialize_for(result, version)
    assert out["content"] == {"x": None, "y": "ok"}
    assert as_bytes(out) == as_bytes(monolith_dump(result))


def test_elicit_result_dumps_before_2025_06_18() -> None:
    """Elicitation entered the schema in 2025-06-18; an emission at an older
    version still dumps the result unchanged (whether the construct belongs
    on the session is the session layer's gate, not the type layer's)."""
    result = ElicitResult(action="accept", content={"ratio": 0.5})
    assert as_bytes(serialize_for(result, V2)) == as_bytes(monolith_dump(result))


@pytest.mark.parametrize("version", [V3, V4])
def test_fractional_schema_bounds_emit_byte_identically(version: str) -> None:
    """JSON Schema number bounds are numbers: a fractional minimum/maximum in
    a requested schema is legal at every version with elicitation and emits
    byte-identically (spec-mandated; integer-only bounds in the pinned
    schema renderings are the same render artifact)."""
    request = ElicitRequest(
        params=ElicitRequestFormParams(
            message="Rate this answer",
            requested_schema={
                "type": "object",
                "properties": {"score": {"type": "number", "minimum": 0.5, "maximum": 9.5}},
            },
        )
    )
    assert as_bytes(serialize_for(request, version)) == as_bytes(monolith_dump(request))


@pytest.mark.parametrize("version", [V3, V4, D])
def test_form_elicitation_schema_bounds_emit_byte_identically(version: str) -> None:
    """The requested-schema interior is caller data and travels verbatim:
    the emitted bytes are always the monolith dump — the 2026-07-28
    validation check never substitutes a value — so a fractional bound keeps
    its exact JSON rendering (1.0 stays 1.0) and an integral one is never
    re-rendered (120 stays 120) (deployed-peer-mandated byte identity)."""
    request = ElicitRequest(
        params=ElicitRequestFormParams(
            message="How old are you?",
            requested_schema={
                "type": "object",
                "properties": {"age": {"type": "number", "minimum": 1.0, "maximum": 120}},
            },
        )
    )
    assert as_bytes(serialize_for(request, version)) == as_bytes(monolith_dump(request))


def test_id_less_cancellation_dumps_at_every_released_version() -> None:
    """requestId on a cancellation is required on the wire through 2025-06-18
    and optional from 2025-11-25 (spec-mandated), but emission never blocks
    an id-less notification: whether one belongs on an old session is the
    session layer's call, and the dump is emitted unchanged either way."""
    without_id = CancelledNotification(params=CancelledNotificationParams(reason="bored"))
    assert serialize_for(without_id, V3) == {"method": "notifications/cancelled", "params": {"reason": "bored"}}
    assert serialize_for(without_id, V4) == {"method": "notifications/cancelled", "params": {"reason": "bored"}}
    with_id = CancelledNotification(params=CancelledNotificationParams(request_id=7))
    assert serialize_for(with_id, V3) == {"method": "notifications/cancelled", "params": {"requestId": 7}}


# --- subscriptions -----------------------------------------------------------


def test_subscription_filter_extras_survive_emission() -> None:
    """Extensions merge extra keys into the subscription filter on the wire;
    they survive 2026-07-28 emission (spec-mandated open object)."""
    filter_ = SubscriptionFilter.model_validate({"toolsListChanged": True, "taskIds": ["task-1"]})
    request = SubscriptionsListenRequest(
        params=SubscriptionsListenRequestParams(notifications=filter_, _meta=as_meta(identity_meta()))
    )
    out = serialize_for(request, D)
    assert out["params"]["notifications"] == {"toolsListChanged": True, "taskIds": ["task-1"]}


def test_legacy_subscribe_has_no_wire_form_at_2026_07_28() -> None:
    """2026-07-28 removed resources/subscribe (spec-mandated)."""
    with pytest.raises(UnsupportedAtVersionError):
        serialize_for(SubscribeRequest(params=SubscribeRequestParams(uri="file:///r")), D)


def test_listen_request_dumps_below_2026_07_28() -> None:
    """subscriptions/listen exists only in the 2026-07-28 schema; an earlier
    emission still dumps it unchanged (the method tables tell the session
    layer the method is absent there — dispatch data, not an emission
    block)."""
    request = SubscriptionsListenRequest(params=SubscriptionsListenRequestParams(notifications=SubscriptionFilter()))
    assert as_bytes(serialize_for(request, V4)) == as_bytes(monolith_dump(request))


def test_initialize_has_no_wire_form_at_2026_07_28() -> None:
    """2026-07-28 removed the initialize handshake (server/discover replaces
    it), so the 2026-07-28 surface defines no such type and strict emission
    refuses (spec-mandated)."""
    request = InitializeRequest(
        params=InitializeRequestParams(
            protocol_version=V4, capabilities=ClientCapabilities(), client_info=Implementation(name="c", version="1")
        )
    )
    with pytest.raises(UnsupportedAtVersionError) as exc_info:
        serialize_for(request, D)
    assert exc_info.value.version == D


def test_ping_has_no_wire_form_at_2026_07_28() -> None:
    with pytest.raises(UnsupportedAtVersionError):
        serialize_for(PingRequest(), D)


def test_initialized_notification_has_no_wire_form_at_2026_07_28() -> None:
    """2026-07-28 removed the lifecycle notifications along with the
    handshake (spec-mandated)."""
    with pytest.raises(UnsupportedAtVersionError):
        serialize_for(InitializedNotification(), D)


def test_tasks_legacy_types_have_no_wire_form_at_2026_07_28() -> None:
    """The 2025-11-25 tasks methods continue as an extension rather than core
    protocol; the 2026-07-28 schema defines none of their types
    (spec-mandated)."""
    request = CancelTaskRequest(params=CancelTaskRequestParams(task_id="t-1"))
    with pytest.raises(UnsupportedAtVersionError):
        serialize_for(request, D)


# --- spec-name divergences ----------------------------------------------------


@pytest.mark.parametrize("version", [V4, D])
def test_elicit_complete_notification_emits_under_its_schema_name(version: str) -> None:
    """The SDK keeps its v1 class name; the schema spells the definition
    'ElicitationCompleteNotification'. The 2026-07-28 validation check
    resolves the surface class through the recorded rename, and the wire
    shape is identical at both versions."""
    notification = ElicitCompleteNotification(params=ElicitCompleteNotificationParams(elicitation_id="e-1"))
    assert serialize_for(notification, version) == {
        "method": "notifications/elicitation/complete",
        "params": {"elicitationId": "e-1"},
    }
