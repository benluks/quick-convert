# Hydra-Forward Configuration Structure

## Overview

This project uses a **Hydra-forward configuration structure**, meaning:

> Configs are composed declaratively, then *forwarded* into instantiated Python objects with minimal glue code.

Instead of manually wiring components together in Python, **Hydra composes a full object graph**, and your code simply executes it.

---

## Core Idea

Each config layer answers a different question:

- **run/** → *What are we doing right now?*
- **pipeline/** → *What is the high-level process?*
- **dataset/** → *What data are we using?*
- **anonymizer/** → *What model/algorithm are we using?*
- **global/** → *What are shared defaults?*

Hydra merges these into a single config, then you pass it forward.

---

## Example Entry Point

```yaml
# configs/run/anonymize_knnvc_clac.yaml

defaults:
  - /global: default
  - /pipeline: anonymization
  - /anonymizer: knnvc
  - /dataset: clac
  - _self_
```

This is the **composition root**.

Think of it as:

> “Take these building blocks and assemble a runnable system.”

---

## Mental Model: Layered Assembly

Hydra builds your config in layers:

### 1. Global Defaults

```yaml
out_root: anonymized
out_dir_suffix: ""
```

These are **shared variables** used everywhere.

---

### 2. Pipeline Definition

```yaml
_target_: quick_convert.pipelines.anonymization.AnonymizationPipeline

anonymizer: null
dataset: null
target_speaker: null
out_dir: ${out_root}/${hydra:runtime.choices.dataset}/${hydra:runtime.choices.anonymizer}_${target_id}${out_dir_suffix}
```

This defines the **top-level object**.

Important:
- `_target_` tells Hydra what to instantiate
- fields (`anonymizer`, `dataset`) will be **filled in later**

---

### 3. Dataset Config

```yaml
_target_: quick_convert.data.clac.ClacDataset
root: ...
splits: ...
file_format: wav
return_spkid: true
```

Defines how to **instantiate the dataset**.

---

### 4. Anonymizer Config

```yaml
_target_: quick_convert.pipelines.anonymization.KNNVCAnonymizer
```

Defines the **model/algorithm**.

---

### 5. Run-Specific Overrides

```yaml
target_id: "1069"

pipeline:
  target_speaker: ${target_id}
```

This is where **execution-specific logic lives**.

---

## How It All Comes Together

After composition, Hydra produces something conceptually like:

```yaml
pipeline:
  _target_: AnonymizationPipeline
  anonymizer:
    _target_: KNNVCAnonymizer
  dataset:
    _target_: ClacDataset
    ...
  target_speaker: "1069"
  out_dir: ...
```

Then in code:

```python
pipeline = instantiate(cfg.pipeline)
pipeline.run(**cfg.get("run", {}))
```

That’s the **forward step**:
- Config → Object graph → Execution

---

## Why This Design Works

### 1. Decoupling

Each component is defined independently:
- dataset doesn’t know about anonymizer
- anonymizer doesn’t know about pipeline

---

### 2. Composability

You can swap components without changing code:

```bash
anonymize dataset=other_dataset anonymizer=nac
```

---

### 3. Minimal Python Glue

Your Python code becomes:

```python
pipeline = instantiate(cfg.pipeline)
pipeline.run()
```

No manual wiring.

---

## Key Hydra Concepts in This Project

### `_target_`

Defines what Python class to instantiate.

---

### `defaults`

Controls composition order.

Order matters:
- earlier configs provide structure
- later ones override

---

### `_self_`

Ensures the current file is applied **after** defaults.

Without it, overrides may not behave as expected.

---

### `hydra:runtime.choices`

Used for dynamic naming:

```yaml
${hydra:runtime.choices.dataset}
```

This reflects CLI/config selections.

---

## Navigating the Structure

When debugging or extending:

### Step 1: Start from `run/`

This tells you:
- which dataset
- which pipeline
- which anonymizer

---

### Step 2: Inspect Each Group

Follow the defaults:
- `/pipeline/...`
- `/dataset/...`
- `/anonymizer/...`

---

### Step 3: Find `_target_`

This tells you:
- which Python class is actually used

---

### Step 4: Trace Interpolations

Look for:

```yaml
${...}
```

These connect pieces across configs.

---

## Recommended Improvements

### 1. Make Pipeline Inputs Explicit

Instead of `null`, consider:

```yaml
anonymizer: ???
dataset: ???
```

Hydra will enforce that they must be provided.

---

### 2. Standardize Run Arguments

Keep all runtime arguments under:

```yaml
run:
  ...
```

This keeps execution separate from configuration.

---

### 3. Avoid Hidden Coupling

Prefer:

```yaml
pipeline:
  target_speaker: ${target_id}
```

over embedding logic inside classes.

---

## Summary

This structure is:

> **Declarative composition → automatic instantiation → minimal execution code**

Or more simply:

> “Configs define everything. Python just runs it.”
