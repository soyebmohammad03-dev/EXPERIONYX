"""Exception hierarchy. Every failure the registry or domain can raise is one of these."""


class ExperionyxError(Exception):
    """Base class for all EXPERIONYX errors."""


class ValidationError(ExperionyxError, ValueError):
    """A domain object or serialized payload violates an invariant."""


class NotFoundError(ExperionyxError, LookupError):
    """A requested record does not exist."""


class DuplicateError(ExperionyxError):
    """A record with the same ID already exists (records are immutable)."""


class MissingReferenceError(ExperionyxError):
    """A record references another record that does not exist."""


class SchemaVersionError(ExperionyxError):
    """A database or payload has an unsupported schema version."""


class CorruptRecordError(ExperionyxError):
    """A stored record does not match its recorded ID or content hash."""
