# Design philosophy

Quick Convert separates reusable task behavior from workflow orchestration and training infrastructure. The distinction is about responsibility, not directory ceremony.

## Components

Components are focused building blocks: encoders, decoders, feature extractors, losses, quantizers, and signal-processing utilities. They should have explicit input/output contracts and should not own an end-to-end task workflow.

## Systems

Systems provide complete task-level capabilities through inference-oriented Python APIs. Examples include ASR, reconstruction, anonymization, and speech emotion recognition. A system can compose components or another task-level system, but it should not depend on a pipeline or training framework.

## Pipelines

Pipelines orchestrate work over data: preparing outputs, iterating datasets, calling systems or components, and persisting results. Not every workflow needs all three layers. Feature precomputation can apply a component directly; evaluation can call a system; training wraps a system in a framework-specific module.

## Training

Training concerns live outside systems. A trainer owns the loop and a training module adds objectives, logging, and optimization behavior around a system. This keeps inference state distinct from optimizer, scheduler, logger, and checkpoint-resumption state. A completed run can be exported into the versioned inference artifact format and loaded without Lightning or training-only state.

## Data and resources

Datasets discover samples and load audio. Resources attach named annotations or sidecar features—such as transcripts, token IDs, speaker embeddings, or content features—without expanding the core sample type for every experiment. Resource providers describe where values come from; loading and collation remain separate.

## Configuration

Hydra configurations live under `quick_convert/configs/` and are installed with the package. A run config selects a pipeline and the systems, components, datasets, resources, and trainer needed for a workflow.

The preferred top-level task model key is `system`. Avoid parallel vocabulary such as `architecture.system`; it obscures the boundary between the inference object and its training wrapper.

## Support boundary

The supported public spine is under `quick_convert.data`, `quick_convert.components`, `quick_convert.systems`, `quick_convert.training`, `quick_convert.pipelines`, and `quick_convert.inference`. Code under `quick_convert.external` is vendored implementation detail reached through first-party adapters.
