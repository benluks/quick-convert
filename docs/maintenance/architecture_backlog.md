# Architecture maintenance

This page contains remaining work, not an alternative description of the
current architecture. Use [Core concepts](../getting-started/concepts.md) for
ownership, [Design philosophy](../design_philosophy.md) for rationale, and the
[extension guide](../extending/index.md) for contracts.

## Established boundaries

Systems own inference; training modules own objectives and optimization;
pipelines own workflows and persistence. Saved inference artifacts are separate
from resumable Lightning checkpoints. The supported task model is configured at
`system`, and trained module weights retain their `system.*` prefix. Historical
imports and checkpoint keys have explicit compatibility paths.

Resource loading supports tensors, tokens, and audio. Reconstruction and VQ-ASR
share content-adaptation mechanics while retaining task-specific outputs.
Source metadata is checked without pretending it describes transformed values.
See [data and resources](../data/index.md), [content adaptation](../extending/index.md#adapt-content-representations),
and [inference loading](../guides/inference.md) for usage.

The earlier training and CosyVoice migration checklists have been retired:
completed steps and obsolete descriptions remain available in Git history.

## Remaining work

- Review residual task implementations under `pipelines` against the existing
  system boundary before moving them; do not introduce a universal pipeline
  base without shared lifecycle behavior.
- Review vendored CosyVoice and Matcha scope using a transitive import graph.
  Preserve required files, licenses, source revisions, and local modifications.
- Review the global `cosyvoice`/`matcha` package aliases and coexistence with
  separately installed packages before changing vendored import behavior.
- Review the default long-recording speaker-embedding policy for repeatability;
  keep stochastic inference opt-in.
- Verify pretrained Whisper, emotion2vec, and MPM inference in environments with
  the required checkpoints. Local synthetic tests do not establish that result.
- Review or retire the legacy web interface.
- Consolidate tests when stronger behavior coverage subsumes a weaker check.
  Retain distinct data, gradient, checkpoint, compatibility, and model-backed
  integration boundaries.

Backend-independent decoder protocols, alternate mel ownership, and moving the
vocoder outside the spectrogram generator remain design options, not committed
migrations. Revisit them only when an actual caller needs a different boundary.

## Operational notes

[Downstream notification](downstream-notification.md) documents the optional
private pipeline trigger. It does not establish downstream integration success.
