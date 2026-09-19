# Model and Dataset Identity

## Registration
`RegisteredModel` / `RegisteredDataset` (registry tables `models`, `datasets`) bind a name and
version to metadata (which includes the adapter, adapter version and **fingerprint**), a `source`
locator and load `options`. An `Experiment` refers to them through its `ModelRef`/`DatasetRef`
(`name`, `version`, `digest == fingerprint`).

- **Identity** (record ID): name, version, adapter, adapter version, fingerprint, options.
  `source` is *not* identity: the same content at another path is the same model.
- A `(name, version)` cannot be re-registered with a different fingerprint under the same adapter
  version; use a new version for changed content.
- `source` is stored as given. Prefer workspace-relative paths (resolved against the executor's
  `inputs_root`) or `builtin:` names; absolute paths can leak private directory names.
- Registered records are immutable and retrievable (`find` by name, version, adapter, fingerprint).

## Execution
Before a run starts (`Executor(adapters=…)`), each ref that carries a digest and has a
registered record is resolved to its adapter, **loaded, re-fingerprinted and compared** with the
registered fingerprint. A mismatch, a load failure, or a stale adapter version fails the run in
the PREPARATION stage (the run and the reason stay visible). A ref with no registered record is
treated as an unbound external reference (a warning is logged and provenance shows no binding for
it). Provenance records, per bound input, the record ID, adapter, adapter version and the
fingerprint measured at load time, plus the requested and resolved device; these take part in
the provenance fingerprint. Inference configuration lives in `ExecutionParameters.metadata`.

## Model fingerprints
Never `str()`/`repr()` of a model. Fingerprints are identity mechanisms, not proof that two models
behave the same in every environment.

| Adapter | Fingerprint | Guarantees | Does not guarantee |
|---|---|---|---|
| sklearn | SHA-256 of the persisted file's bytes (streamed) | that exact artifact is unchanged | that a re-saved copy of an equal estimator matches (pickle embeds versions/layout); nothing about library versions used to load it |
| torch | SHA-256 over module tree (submodule and class names) and each state-dict entry's name, dtype, shape and raw bytes | identical weights and architecture, independent of file format | train/eval mode, non-persistent buffers, custom Python `forward` logic of non-TorchScript modules |

## Dataset fingerprints
They describe the representation the experiment actually uses. Nothing is sorted or normalized:
row order, feature names, dtypes, targets and split definitions all change the fingerprint.

| Adapter | Fingerprint |
|---|---|
| sklearn (tabular) | SHA-256 over a canonical header (X/y dtype+shape, feature names, split names+sizes), then little-endian X bytes, y bytes and split index bytes, streamed in row chunks. Generated (`builtin:`) datasets are fingerprinted by their data; generation config is recorded in `source` metadata |
| torch (map-style) | SHA-256 over a header (length, labeled, splits) and every sample's component dtype, shape and bytes in order. Reads the whole dataset once, one sample at a time (bounded memory); cached |

Limits: the fingerprint is computed once per adapter instance and cached, so mutating the
underlying arrays afterwards is not detected until a fresh load; float bytes are hashed as-is
(NaN payloads and signed zeros are distinct); object-dtype data is rejected.
