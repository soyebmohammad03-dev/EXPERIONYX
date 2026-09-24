"""Read-only FastAPI surface over the EXPERIONYX registry (see docs/api.md).

Every route here only ever calls `Registry.get`/`find`/`exists` or a subsystem's dedicated
read-only registry wrapper (`search`/`compare`/`document`) -- never `add`/`update_status`. This is
a viewer, not a second implementation of the analysis engines."""
