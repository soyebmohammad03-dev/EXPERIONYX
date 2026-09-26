# Security Policy

Report vulnerabilities privately via this repository's GitHub "Report a vulnerability" (private
security advisory) feature; do not open public issues for them.

EXPERIONYX is a pre-1.0, single-maintainer research tool (no published PyPI release yet). Only the
`main` branch is supported; there is no versioned security-fix policy beyond keeping `main` current.

The API/UI (`experionyx viz serve`) has no authentication layer — it is meant to run at the same
trust boundary as the CLI against a workspace you already control. Do not expose it on an
untrusted network without a reverse proxy in front of it (see
[docs/limitations.md](docs/limitations.md)).

Never commit secrets, credentials, or real model/dataset artifacts.
