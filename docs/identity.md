# Identity and Content Hashing

Implemented in `src/experionyx/hashing.py` and `Entity.id` in `domain.py`.

## Canonical serialization
UTF-8 JSON, keys sorted, no whitespace, ASCII-escaped, NaN/Infinity rejected. Python `int`
and `float` serialize differently (`1` vs `1.0`) and are therefore different content. The float
text form is Python's shortest round-trip repr; canonical form is defined by this
implementation, not an external standard (e.g. RFC 8785).

`content_hash(payload)` = `sha256:` + hex SHA-256 of the canonical JSON.

## Entity IDs
`<prefix>_<first 32 hex chars of content_hash({"kind", "identity"})>` (128 bits), prefixes:
`inv cfg env exp run obs art clm evd`. IDs are derived, not stored on the object, so an object
can never disagree with its ID.

**Included in identity** — the "identity fields" column of [domain-model.md](domain-model.md).
**Excluded** — `created_at`, lifecycle `status`, and fields that are content rather than identity
(an observation's value, an artifact's size, a note). No timestamps or randomness enter an ID.

Consequences: the same logical experiment/config/environment always gets the same ID; re-adding
it is a `DuplicateError`; two runs that are deliberate repeats need distinct `attempt` values.

## What is content-addressed
- `ConfigurationRef`, `EnvironmentSnapshot`: ID = hash of the full content.
- `Artifact`: ID includes the file's sha256 digest.
- Everything else: ID is deterministic from its identity fields, but is not a hash of the full
  record.

## Record hash
`Entity.content_hash()` hashes the *entire* serialized record (all fields). The registry stores
it beside each record and re-verifies both the ID and this hash on every read
(`CorruptRecordError` on mismatch). This detects accidental or out-of-band modification; it is
not tamper-proofing (whoever can edit the database can edit the hash).

## Not covered
Hashing of artifact file contents, dataset contents and model weights is recorded by callers as
`digest` fields; nothing computes or verifies them yet.
