# Architecture cleanup backlog

This page records architectural questions that should survive individual cleanup
branches. Items here are investigation prompts, not decisions or permission to
perform broad refactors without reviewing the current behavior first.

## Training systems and inference-ready models

The current implementation and proposed boundary are mapped in
[`training_architecture.md`](training_architecture.md).

The supported training path now separates inference artifacts from resumable
Lightning checkpoints and configures the task model at top-level `system`.
Remaining work should:

- inventory task-level implementations that still live under `pipelines` and
  decide which belong under `systems`;
- keep pipelines responsible for orchestration rather than model architecture;
- preserve resumable training checkpoints separately from portable inference
  artifacts; and
- verify the design against the supported VQ-ASR and SSL-reconstruction paths
  before generalizing it.

Lightning modules now wrap plain systems. Framework adapters and runners live
under `quick_convert.training`, while the training workflow remains a pipeline.
Former import paths are retained as compatibility aliases.
