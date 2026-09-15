# Architecture cleanup backlog

This page records architectural questions that should survive individual cleanup
branches. Items here are investigation prompts, not decisions or permission to
perform broad refactors without reviewing the current behavior first.

## Training systems and inference-ready models

The training path currently makes it difficult to load a trained model for
inference without also reconstructing Lightning-specific training state. A
future training-focused pass should:

- inventory task-level implementations that still live under `pipelines` and
  decide which belong under `systems`;
- keep pipelines responsible for orchestration rather than model architecture;
- distinguish an inference-ready model or system from its Lightning training
  wrapper;
- define an explicit checkpoint boundary so users can load model weights for
  inference without optimizer, scheduler, logger, or trainer state;
- preserve resumable training checkpoints separately from portable inference
  artifacts; and
- verify the design against the supported VQ-ASR and SSL-reconstruction paths
  before generalizing it.

Open questions include whether Lightning modules should wrap plain systems,
whether systems should expose their own export/load API, and which configuration
layer owns checkpoint selection. These should be resolved with the training
configurations and actual checkpoint formats in view.
