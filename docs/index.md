Quick Convert is a toolkit for speech anonymization and ASV (automatic speaker verification) workflows.

It provides a flexible, config-driven pipeline built around Hydra, allowing you to run anonymization, training, and evaluation with minimal friction.

---

## Getting Started

If you're new, start here:

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Design Philisophy] (design_philosophy.md)

---

## Core Concepts

These explain how the system is structured under the hood.

### Configuration (Hydra)

- [Hydra Structure](config/hydra_structure.md)

Learn how configurations are composed, overridden, and used to drive pipelines.

### The 3 module types: Piplines, Systems, Components (for the future)

- [Systems] Where the task-specific logic is implemented. Examples of systems are:
- - Anonymization
- - Automatic Speaker Verification (ASV)
- - Automatic Speech Recognition (ASR)
- [Components](components/index.md) are the puzzle pieces that make up a system. For example, the same speaker encoder may extract a speaker embedding as part of an [anonymization system] as well as an [ASV system].
- - [Donors](components/donors.md)
- [Pipelines] Pipelines string the . This is the top-level wrapper, although I put it last in this list, because I feel it's easier to understand once **Systems** abd **Components** have been explained.

Note: As of yet, you'll notice that there is non `systems` module. The implemented anonymization systems currently exist under `pipelines`, and need to be refactored.

---

### Data Handling

- [Base Dataset](data/base_dataset.md)

Understand how datasets are represented, loaded, and iterated over.

---

## What This Project Covers

- Speech anonymization pipelines  
- ASV (speaker verification) training and evaluation  
- Dataset preparation and transformation  
- Hydra-based experiment management  

---

## Suggested Reading Paths

### 👶 First-time user
1. Installation  
2. Quickstart  
3. Hydra Structure  

### Working on experiments
1. Hydra Structure  
2. Base Dataset  

### Extending the codebase
1. Base Dataset  
2. Hydra Structure  

---

## Notes

This documentation is intentionally lightweight and close to the codebase.  
If something is unclear, it's usually best to check the corresponding module directly.

---