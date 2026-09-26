---
name: github-repository-presentation
description: Professional practices for presenting a research/software GitHub repository — README structure, metadata, badges, diagrams, screenshots, community health files, citation, release presentation, accessibility, and truthful claims. Use when polishing a repository's public presentation, not when adding features.
---

# GitHub repository presentation

Presentation work only. Never invent capabilities, metrics, benchmarks, screenshots, users, or
results to make a repository look more finished than it is. Every visual and claim must be
traceable to something that actually exists in the repository.

## Before changing anything

Inspect the full repository first: README, `docs/`, `.github/`, package metadata, existing assets,
and (if `gh` is authenticated) the live repository metadata. Do not duplicate content that already
exists — link to it instead.

## README structure (adapt, don't force every section)

1. Hero: project name, one-line subtitle, 1-2 sentence description.
2. A small number of truthful badges (CI status, supported language versions, license). Never a
   badge asserting quality ("production ready", "best-in-class") — badges state facts, not opinions.
3. Why the project exists (the problem, in a few sentences).
4. What it actually does (concrete capabilities, not aspirations).
5. Architecture / lifecycle diagram, if one materially helps a newcomer understand the system in
   under a minute.
6. Quick start with commands verified to actually run.
7. A real example/workflow, using the project's own real output — never fabricated.
8. Links out to `docs/` for anything detailed. Keep the README itself scannable.
9. Limitations, stated plainly, not buried or softened.
10. Citation / contributing / license.

## Visual assets

- SVG for diagrams (scales cleanly, stays small, is diffable in git).
- Screenshots must come from actually running the software. Never mock up a UI in an image editor
  and present it as a screenshot.
- A repository's visual identity should reflect what the project *is* — restrained and technical
  for a research/engineering tool, not generic "AI" stock art, gradients-for-their-own-sake, or
  gaming/marketing aesthetics.
- Every image needs descriptive alt text and a filename that says what it shows.
- Keep a source/editable form of diagrams (SVG markup itself, since it's already text) rather than
  only a rendered raster.

## Repository metadata

- Description and topics (`gh repo edit --description ... --add-topic ...`) should match what the
  project currently does, not what it's planned to do.
- If a social preview image is wanted: GitHub has no API for uploading it — it's set manually under
  repository Settings → General → Social preview. Produce the asset and say so explicitly; don't
  claim it was uploaded if it wasn't.

## Community health files

- `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`: only add what fits the
  project's actual state. A security policy for a pre-release solo project should say so, not
  describe a support matrix that doesn't exist.
- `CITATION.cff`: only include fields that are true. Never fabricate a DOI, ORCID, journal,
  institution, funding source, or citation count. Omit fields rather than invent them.
- Issue/PR templates should ask for what the project actually needs to triage well (for a
  research tool: reproducibility info, run/experiment IDs, evidence) — keep them short enough that
  people actually fill them in.

## Research-credibility framing

For a research or evidence-driven project, be explicit and consistent about three buckets:
**implemented**, **limitations**, **future work**. Never let prose imply something is implemented
when it's aspirational. Preserve whatever the project's actual scientific-integrity invariants are
(e.g. no fabricated evidence, no silent negative-evidence inference, uncertainty preserved) in how
you describe it — presentation work must not soften or obscure these.

## Final QA pass

Before calling this done: open every generated image, check every internal doc link resolves,
check code blocks/commands actually work, check the project name is spelled consistently, and
re-read for any placeholder text or claim that isn't backed by something real in the repo.
