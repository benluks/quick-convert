# Architecture cleanup backlog

This page records architectural questions that should survive individual cleanup
branches. Items here are investigation prompts, not decisions or permission to
perform broad refactors without reviewing the current behavior first.

## Training systems and inference-ready models

The current implementation and proposed boundary are mapped in
[`training_architecture.md`](training_architecture.md).

The supported training path now separates inference artifacts from resumable
Lightning checkpoints and configures the task model at top-level `system`.
Future architecture work should:

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

## LLM cleanup completion gate

The stacked cleanup branches are complete when all of the following are true:

- each branch is strictly based on its immediate predecessor and has no
  unresolved merge conflicts;
- the reference manifest, W2V-BERT precompute, VQ-ASR, SSL-reconstruction, and
  tokenizer workflows compose from their checked-in Hydra configurations;
- core CI and the applicable isolated integration workflows pass at the tip of
  the stack;
- public documentation describes the current package paths and the
  pipeline/system/component boundary;
- compatibility behavior is covered where configs, checkpoints, or public
  imports moved; and
- remaining experimental, integration, and legacy profiles are explicitly
  classified rather than mistaken for supported reference workflows.

Open research features and explicitly classified optional integrations do not
block this gate. Merging the stack into `main` is a separate repository action:
the code can be ready before those merges are performed, but the repository is
not the published finished state until the stack lands.
