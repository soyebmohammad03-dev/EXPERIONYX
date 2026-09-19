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


class ExecutionError(ExperionyxError):
    """Base class for execution-engine failures."""


class PreparationError(ExecutionError):
    """A run could not be prepared. Raised (not recorded) when no Run identity exists yet."""


class ReplayError(ExecutionError):
    """A replay request is invalid (unknown, non-terminal or unreplayable run)."""


class ArtifactError(ExperionyxError):
    """An artifact could not be registered."""


class ArtifactIntegrityError(ArtifactError):
    """An artifact is missing or its bytes no longer match the registered digest/size."""


class ConcurrentModificationError(ExperionyxError):
    """A record changed between read and write."""
