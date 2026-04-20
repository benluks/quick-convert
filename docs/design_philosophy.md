# 🧩 Quick Convert — Architecture & Design Philosophy

Quick Convert is a **modular research framework** for building, composing, and evaluating speech anonymization systems from reusable components.

This document explains the philosophy behind the architecture so you can confidently extend the codebase without fighting it.

---

# 🧠 Core Idea

Most voice conversion/anonymization repos are built as **monoliths**:

* one model
* one pipeline
* tightly coupled assumptions

That works for reproducing results.

It does *not* work for research.

Quick Convert flips this:

> **An anonymizer is not a model.
> It is a composition of components.**

---

# 🏗️ The Three Layers

The architecture is intentionally split into three conceptual layers:

## 1. Components — “Lego Blocks”

Components are the smallest reusable building blocks.

They represent **functional roles**, not specific implementations.

Examples:

* Content encoders (ASR / SSL)
* Speaker encoders
* Acoustic / prosody encoders
* Tokenizers (VQ / RVQ / KMeans)
* Decoders / generators
* Vocoders
* Duration models
* Conditioning / fusion modules

Each component:

* has a **clear interface**
* exposes **metadata about its behavior**
* can be swapped for another implementation of the same role

👉 Think: *What does this module do?*, not *where did it come from?*

---

## 2. Systems — “Model Assemblies”

A system is a **composition of components** into a working model.

Examples:

* ASR-BN anonymization pipeline
* Conditional RVQ model
* kNN-VC (as a monolithic system)
* NAC (as a monolithic system)

A system defines:

* how components are connected
* how data flows between them
* what conditioning is applied
* what assumptions exist (frame rate, dimensions, etc.)

👉 Think: *How do these pieces work together?*

---

## 3. Pipelines — “Workflows”

Pipelines are **not models**.

They are workflows that operate on data using systems.

Examples:

* anonymize a dataset
* evaluate ASV performance
* extract embeddings
* preprocess audio
* run experiments

👉 Think: *What are we doing with the model?*

---

# 🧱 Functional Roles (Not Backends)

The most important design choice:

> **We organize by role, not by repo.**

Instead of:

* “this is a Whisper model”
* “this is a SpeechBrain model”

We think in terms of:

* `ContentEncoder`
* `SpeakerEncoder`
* `Tokenizer`
* `Decoder`
* `Vocoder`

Each role can have many implementations:

| Role           | Examples                                  |
| -------------- | ----------------------------------------- |
| ContentEncoder | Whisper, WavLM, HuBERT                    |
| SpeakerEncoder | ECAPA, x-vector, ResNet                   |
| Tokenizer      | RVQ, VQ, KMeans                           |
| Decoder        | Conditional decoder, autoregressive model |
| Vocoder        | HiFi-GAN, WaveGlow                        |

👉 This is what makes composition possible.

---

# 🔌 Interfaces Over Implementations

Every component must define a **contract**.

Example (conceptual):

```python
class SpeakerEncoder(ABC):
    @abstractmethod
    def encode(self, wav, sr) -> SpeakerEmbedding:
        ...
```

This ensures:

* components are interchangeable
* systems stay clean
* backend-specific hacks stay isolated

---

# 📦 Typed Data (Avoid Tensor Chaos)

Raw tensors are not enough.

We explicitly represent what data *means*:

```python
@dataclass
class FrameFeatures:
    values: torch.Tensor
    frame_hz: float
    feature_dim: int
    kind: str
```

```python
@dataclass
class SpeakerEmbedding:
    values: torch.Tensor
    embedding_dim: int
```

```python
@dataclass
class TokenSequence:
    tokens: torch.Tensor
    discrete: bool
    frame_hz: float
```

Why this matters:

* prevents silent mismatches
* makes debugging easier
* enables validation of component compatibility

---

# 🔄 Composition via Configuration

Systems are defined using Hydra configs.

Example:

```yaml
system:
  _target_: quick_convert.systems.anonymization.ConditionalRVQSystem

  content_encoder:
    _target_: quick_convert.components.content_encoders.WhisperEncoder
    model_name: medium

  speaker_encoder:
    _target_: quick_convert.components.speaker_encoders.ECAPAEncoder

  tokenizer:
    _target_: quick_convert.components.tokenizers.RVQTokenizer
    num_quantizers: 8

  decoder:
    _target_: quick_convert.components.decoders.ConditionalDecoder
```

This allows:

* swapping components without code changes
* rapid experimentation
* reproducibility

---

# 🧠 Two Types of Backends

Not all models are equal.

We support both:

## 1. Atomic Components

Reusable parts:

* ECAPA speaker encoder
* Whisper encoder
* RVQ tokenizer

## 2. Monolithic Systems

Hard-to-decompose models:

* kNN-VC
* NAC

These are treated as **systems**, not components.

👉 Do not force decomposition where it doesn’t make sense.

---

# 📦 Vendor vs Dependency

We use a pragmatic approach:

## Use as dependency when:

* it installs cleanly
* it behaves like a library
* minimal patching is needed

## Vendor (copy) when:

* repo is not package-friendly
* inference requires heavy modification
* reproducibility is critical

Vendored code lives in:

```
quick_convert/vendor/
```

Each vendored module should include:

* source repo link
* commit hash
* explanation of modifications

---

# 🧩 Adapters, Not Hacks

When components don’t align:

* different frame rates
* different dimensions
* different representations

👉 We add **adapters**, not hacks.

Examples:

* temporal resampling
* projection layers
* pooling modules
* conditioning transforms

Adapters live in:

```
components/common/
```

---

# ⚠️ Design Principles

## 1. Separate concerns aggressively

* components ≠ systems ≠ pipelines

## 2. Prefer composition over inheritance

* build systems from pieces
* avoid deep class hierarchies

## 3. Keep backend logic isolated

* no Whisper assumptions in pipelines
* no ECAPA assumptions in decoders

## 4. Explicit > implicit

* annotate frame rates
* annotate embedding types
* annotate assumptions

## 5. Accept imperfection

* not all models are cleanly modular
* some duplication is okay
* clarity beats cleverness

---

# 🚀 What This Enables

This architecture lets you:

* swap speaker encoders in seconds
* test new conditioning strategies
* mix discrete and continuous representations
* prototype new anonymization ideas quickly
* evaluate systems consistently

Example experiments:

* RVQ + speaker embedding vs RVQ alone
* different duration modeling strategies
* acoustic vs semantic conditioning
* hybrid anonymization pipelines

---

# 🧭 Mental Model

Think of Quick Convert as:

> 🧩 A toolbox of components
> 🏗️ A system builder
> 🔁 A pipeline runner

Not:

* a single model
* a single method
* a fixed architecture

---

# ✅ TL;DR

* Components = reusable building blocks
* Systems = composed models
* Pipelines = workflows
* Roles > repos
* Interfaces > implementations
* Configs drive composition
* Adapt, don’t hack

---

If you’re adding something new, ask:

> “Is this a component, a system, or a pipeline?”

That question alone will keep the architecture clean.
