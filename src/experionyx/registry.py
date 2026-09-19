"""Storage-agnostic registry interface. Domain code depends on this, never on SQLite."""

from typing import Protocol, TypeVar

from experionyx.domain import Entity, Experiment, Run

E = TypeVar("E", bound=Entity)


class Registry(Protocol):
    """Append-only store of entities. Records are immutable once added; the only permitted
    mutation is a lifecycle status transition of an Experiment or Run (`update_status`)."""

    def add(self, entity: Entity) -> None:
        """Store `entity`. Raises DuplicateError if its ID exists, MissingReferenceError if it
        references a record that does not."""
        ...

    def get(self, cls: type[E], entity_id: str) -> E:
        """Retrieve by ID. Raises NotFoundError."""
        ...

    def exists(self, cls: type[Entity], entity_id: str) -> bool: ...

    def find(self, cls: type[E], **filters: str) -> list[E]:
        """All records of `cls` whose indexed fields equal `filters`, ordered by ID."""
        ...

    def update_status(self, entity: Experiment | Run) -> None:
        """Persist a legal status transition. Everything except `status` must be unchanged."""
        ...

    def close(self) -> None: ...
