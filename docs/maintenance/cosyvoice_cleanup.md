# CosyVoice integration cleanup

## Status

Deferred maintenance item. The current CosyVoice-based SSL reconstruction path is the supported replacement for the removed Chatterbox backend. This document records known integration and library-API issues so they can be addressed deliberately later.

Reference architecture:

- `configs/run/train_sslr_w2vbert_cmdiff.yaml`
- `configs/architecture/sslr/w2vbert_cmdiff_cosyvoice.yaml`

Current path:

```text
W2V-BERT
-> layer fusion
-> CosyVoiceSpectrogramGenerator
-> CausalMaskedDiffWithXvec
-> CausalConditionalCFM
-> CausalConditionalDecoder
-> CosyVoiceHiFTDecoder
```

## Goals

- Keep the working CosyVoice model behavior intact.
- Make the first-party adapters predictable and convenient as a Python library.
- Avoid loading models or downloading checkpoints unless the requested operation needs them.
- Make supported arguments and return values truthful and documented.
- Isolate the supported CosyVoice flow path from unrelated vendored CLI and LLM code.
- Establish focused tests before pruning vendored files.

## First-party boundary

The intended Quick Convert-facing adapters are:

- `quick_convert/components/decoders/cosyvoice.py`
- `quick_convert/components/decoders/hift_generator.py`
- `quick_convert/components/speaker/speaker_encoders/cosyvoice_campplus.py`

The implementation they wrap currently lives under:

- `quick_convert/external/cosyvoice/`
- `quick_convert/external/matcha/`

Matcha is still a live transitive dependency of the supported CosyVoice flow. Its conditional flow matcher, decoder blocks, transformer blocks, and mel-spectrogram implementation are currently used.

## Checklist

### 1. Avoid unconditional vocoder loading

Current behavior:

`CosyVoiceSpectrogramGenerator.__init__()` immediately calls `CosyVoiceHiFTDecoder.from_pretrained()`. This may download and load `hift.pt` even when only training loss or mel generation is needed.

- [ ] Decide whether the vocoder should be constructor-injected, lazy-loaded, or both.
- [ ] Permit construction and `compute_loss()` without downloading a vocoder checkpoint.
- [ ] Load the pretrained vocoder only when waveform generation is requested.
- [ ] Define behavior when `run_vocoder=True` but no vocoder is configured.
- [ ] Keep the loaded vocoder frozen and in evaluation mode.
- [ ] Add a test proving loss computation does not initialize or download the vocoder.

Acceptance criteria:

- A decoder can be instantiated and trained offline when all required flow components are local.
- Mel-only inference does not load HiFT.
- Waveform inference loads or uses HiFT exactly once.

### 2. Stop mutating caller-owned lengths

Current behavior:

`CosyVoiceSpectrogramGenerator.forward()` assigns into the supplied `length` tensor while reconciling feature and length mismatches.

- [ ] Replace in-place assignment with a cloned or newly computed tensor.
- [ ] Define the expected response when a declared length exceeds the available feature sequence.
- [ ] Add a test asserting that the input length tensor is unchanged after inference.
- [ ] Test mixed-length batches, including a maximum length that exceeds the padded feature dimension.

Acceptance criteria:

- Calling the decoder never changes caller-owned feature or length tensors.
- Length reconciliation is explicit and documented.

### 3. Make the public inference API truthful

Current concerns:

- `forward()` accepts `n_timesteps`, `max_len`, and `cond`, but they do not currently affect the flow call.
- A separate `inference()` method calls the flow but does not return its result.
- `project_speaker()` references `self.speaker_proj`, which is not defined by the adapter.

- [ ] Determine which inference controls the supported CosyVoice backend can actually honor.
- [ ] Either implement or remove each unused argument.
- [ ] Consolidate or clearly distinguish `forward()` and `inference()`.
- [ ] Remove or repair `project_speaker()`; speaker projection currently belongs to the flow.
- [ ] Define and type the mel-only and mel-plus-waveform return values.
- [ ] Document tensor shapes, devices, dtypes, and length semantics.
- [ ] Add tests that fail if accepted arguments are silently ignored.

Acceptance criteria:

- Every public argument affects behavior or is removed.
- Every public inference method returns a documented value.
- Unsupported conditioning modes fail clearly.

### 4. Isolate donor batch translation

Current behavior:

The first-party adapter constructs a donor-style dictionary with keys such as `speech_token`, `speech_feat`, and `embedding`.

- [ ] Move donor batch construction into one private, typed conversion method.
- [ ] Validate feature, mel, speaker-embedding, and length shapes before invoking vendored code.
- [ ] Keep donor-specific key names out of higher-level training modules.
- [ ] Consider a small first-party request/output dataclass if it materially improves clarity.
- [ ] Test the conversion independently with synthetic tensors.

Acceptance criteria:

- CosyVoice dictionary conventions are isolated in the adapter.
- Invalid shapes fail at the boundary with actionable messages.

### 5. Narrow vendored import coupling

Current behavior:

`quick_convert.external.cosyvoice.utils.class_utils` provides low-level activation, positional-embedding, subsampling, and attention registries. Importing it also eagerly imports CosyVoice LLM classes, Qwen/Transformers, high-level CLI models, flows, and HiFT models.

The supported upsample encoder imports these low-level registries and therefore inherits the unrelated high-level import graph.

- [ ] Identify the minimal registries required by the supported flow and upsample encoder.
- [ ] Split low-level registries from high-level model discovery, or make the first-party integration import the required low-level classes directly.
- [ ] Ensure importing the supported flow does not import CosyVoice CLI, dataset, tokenizer, LLM, or vLLM modules.
- [ ] Check whether `transformers` remains a real dependency of the supported path after isolation.
- [ ] Add an import test in an environment containing only the declared CosyVoice dependencies.

Acceptance criteria:

- The supported decoder/encoder path imports without loading unrelated TTS application layers.
- The `cosyvoice` extra explicitly declares every dependency required by that path.

### 6. Review global package aliases

Current behavior:

`quick_convert.external.cosyvoice.__init__` registers vendored packages under the global names `cosyvoice` and `matcha` in `sys.modules`. This preserves unchanged upstream absolute imports but can conflict with separately installed packages.

- [ ] Document why the aliases exist.
- [ ] Test behavior when external `cosyvoice` or `matcha` packages are already installed.
- [ ] Decide whether to retain the aliases, convert supported imports to package-relative paths, or isolate alias setup behind the adapter.
- [ ] Avoid partially initialized aliases during circular imports.

Acceptance criteria:

- Import behavior is deterministic and does not silently substitute an unintended package.

### 7. Make speaker embedding inference deterministic

Current behavior:

The vendored ONNX `EmbeddingExtractor` randomly selects a ten-second crop for longer waveforms. Repeated calls can therefore yield different embeddings for the same input.

- [ ] Decide the desired policy for long audio: deterministic crop, configurable crop, full-audio segmentation and pooling, or explicit stochastic mode.
- [ ] Keep stochastic behavior opt-in for an inference-facing `SpeakerEncoder`.
- [ ] Expose the policy through the first-party wrapper rather than relying on hidden vendored behavior.
- [ ] Add repeatability tests for the default inference mode.
- [ ] Test short, exactly ten-second, and longer inputs.

Acceptance criteria:

- Default speaker embedding inference is repeatable.
- Any stochastic crop is explicit and seedable.

### 8. Audit and eventually prune vendored scope

Do this only after the supported import graph and adapter tests are established.

Likely unrelated areas include vendored training binaries, dataset pipelines, CLI/front-end code, tokenizer code, LLM code, and vLLM integration. Some may currently be imported indirectly through utility registries even though the supported Quick Convert path does not use their behavior.

- [ ] Generate the transitive import graph for the supported Xvec flow, DiT flow, HiFT vocoder, and CAMPPlus wrapper.
- [ ] Mark vendored files as required, indirectly required, or unused.
- [ ] Preserve licenses, copyright headers, source repository, source revision, and a record of local modifications.
- [ ] Remove only files proven unused by supported paths.
- [ ] Confirm that both causal masked diffusion and DiT configurations still compose after pruning.
- [ ] Run adapter tests after each pruning group.

Acceptance criteria:

- The vendored tree contains the supported implementation and its documented dependencies.
- Provenance remains traceable.
- No supported configuration refers to removed files.

### 9. Verify dependency extras and import surfaces

- [ ] Audit the `cosyvoice` optional dependency group against actual imports.
- [ ] Avoid relying on packages installed accidentally through another extra.
- [ ] Test importing base interfaces without the CosyVoice extra.
- [ ] Test importing CosyVoice adapters with the CosyVoice extra.
- [ ] Avoid eager imports that make unrelated optional backends mandatory.
- [ ] Document checkpoint downloads and cache behavior.

Acceptance criteria:

- A minimal Quick Convert installation can import core data and interface modules.
- Installing the CosyVoice extra is sufficient for the supported CosyVoice adapter path.

## Suggested test layers

### Fast unit tests

Use synthetic tensors and injected fake flow/vocoder objects.

- adapter construction without downloads;
- shape and length validation;
- no input mutation;
- donor-batch conversion;
- return contracts;
- lazy vocoder initialization.

### Optional integration tests

Mark separately from the default suite.

- construct the real causal masked diffusion configuration;
- construct the real DiT configuration;
- load CAMPPlus ONNX;
- load HiFT;
- run a very short mel-only inference;
- run a very short waveform inference.

These tests may require model downloads and should not run in the default fast suite.

## Deferred design questions

- Should `CosyVoiceSpectrogramGenerator` remain a concrete backend-named class, or implement a small general spectrogram-decoder protocol?
- Should mel extraction be a separately injected component?
- Should the vocoder remain part of the spectrogram generator or be composed one level higher?
- Is the DiT configuration a supported workflow or only an experiment?
- Should pretrained checkpoint loading use separate factory methods exclusively?
- Which local modifications to vendored CosyVoice and Matcha need to be recorded explicitly?

Resolve these questions when this maintenance item becomes active; they are not settled by this document.
