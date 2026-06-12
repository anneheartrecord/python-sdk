"""Shared pydantic bases for the wire-shape surface packages.

Every model in the ``mcp.types.v*`` packages builds on one of the two bases
here, so the two surface packages cannot silently diverge in model
configuration. There is deliberately no alias generator: each wire name in a
surface package is an explicit ``Field(alias=...)``, so the package file shows
exactly what goes on the wire and cannot inherit serialization behavior from
elsewhere.
"""

from pydantic import BaseModel, ConfigDict


class WireModel(BaseModel):
    """Base for surface-package models: unknown fields validate and are ignored.

    ``extra="ignore"`` is a deliberate divergence from the schemas, which
    declare most wire objects open to extra fields. The boundary only ever
    parses a surface model as a check — the emitted bytes are always the
    monolith dump, never a surface model's re-dump — so the policy's one
    wire-visible effect is that a caller-set key the target revision never
    defined can never fail that check. Keeping the classes closed (rather
    than ``extra="allow"``) pins them to exactly the schema-declared fields,
    the shape the parity suite compares against the generated oracles.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class OpenWireModel(BaseModel):
    """Base for ``_meta`` carrier models: unknown fields are retained.

    Unknown ``_meta`` keys must survive a validate -> re-dump round trip at
    every protocol revision, so the classes a ``_meta`` field references stay
    open.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")
