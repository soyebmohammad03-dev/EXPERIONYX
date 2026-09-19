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


class AdapterError(ExperionyxError):
    """Base class for model/dataset adapter failures. Framework exceptions are chained as causes."""


class AdapterNotFoundError(AdapterError, LookupError):
    """No adapter is registered under the requested name."""


class DuplicateAdapterError(AdapterError):
    """An adapter with this name is already registered."""


class AdapterUnavailableError(AdapterError):
    """The adapter is registered but its framework (an optional extra) is not installed."""


class UnsupportedCapabilityError(AdapterError):
    """The requested operation is not among the adapter's capabilities."""


class DeviceUnavailableError(AdapterError):
    """A specifically requested device is not available; there is no silent fallback."""


class ModelLoadError(AdapterError):
    """A model artifact could not be loaded."""


class DatasetLoadError(AdapterError):
    """A dataset could not be loaded."""


class InvalidModelError(AdapterError):
    """The loaded object is not a model this adapter can drive."""


class InvalidDatasetError(AdapterError):
    """The dataset is malformed or unsupported by this adapter."""


class ModelFingerprintError(AdapterError):
    """A model fingerprint could not be computed or no longer matches its registration."""


class DatasetFingerprintError(AdapterError):
    """A dataset fingerprint could not be computed or no longer matches its registration."""


class InferenceError(AdapterError):
    """Inference failed or was given invalid inputs."""


class EvaluationError(ExperionyxError):
    """An evaluation request is invalid or cannot be carried out as configured."""


class FaultError(ExperionyxError):
    """Base class for fault-injection failures."""


class FaultNotFoundError(FaultError, LookupError):
    """No fault type (or version) is registered under the requested name."""


class FaultCompatibilityError(FaultError):
    """A fault cannot be applied to the given inputs, labels or model."""


class FaultLimitError(FaultError):
    """A fault experiment exceeds a configured safety limit; nothing was run."""


class FailureError(ExperionyxError):
    """Failure discovery, registration or reproduction problem."""


class FailureLimitError(FailureError):
    """A configured discovery bound was exceeded and the work cannot be done honestly."""


class InteractionError(ExperionyxError):
    """Base class for fault-interaction analysis problems."""


class DesignRefusal(InteractionError):
    """The experimental design cannot support an interaction analysis; nothing was analyzed.
    `issues` lists every problem found (what was required, what was found, why it matters)."""

    def __init__(self, issues: "tuple[object, ...]") -> None:
        self.issues = issues
        super().__init__("interaction design refused: " + "; ".join(str(i) for i in issues))


class ReliabilityError(ExperionyxError):
    """Base class for reliability-profile problems."""


class ProfileRefusal(ReliabilityError):
    """The requested sources cannot form one compatible profile; nothing was built."""

    def __init__(self, issues: "tuple[object, ...]") -> None:
        self.issues = issues
        super().__init__("reliability profile refused: " + "; ".join(str(i) for i in issues))


class BenchmarkError(ExperionyxError):
    """Base class for robustness-benchmark problems."""


class BenchmarkRefusal(BenchmarkError):
    """The benchmark definition is invalid or unsupported; nothing was executed or recorded."""

    def __init__(self, issues: "tuple[object, ...]") -> None:
        self.issues = issues
        super().__init__("benchmark refused: " + "; ".join(str(i) for i in issues))
