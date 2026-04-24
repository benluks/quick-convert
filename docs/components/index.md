Components are the broadest and richest of the 3 module types, and where I expect most of the engineering will have to be done. There is no base component. Instead, each module within components has its own base. So far, the existing component modules are:

- [`content_encoders`] encode frame-wise content from an utterance. This is usually linguistic content such as the VQ-BN extractor output in `ASR-BN` (B5), the E2E ASR system in `STTTS`, (B3) the `HuBERT` encodings in `NAC` (B5) or `SSL`.
- [`donors`](components/donors.md): one-off copies of code from other directories, built to make a higher-level component or system run with minimal modifications.
- [`feature_extractors`] are high-level wrappers for all types of feature extractors.
- [`speaker_encoders`] are models that extract information about the speaker. Think of an x-vector/speaker embedding extractor used in ASV or for anonymization systems.

Components are the puzzle pieces that make up a system. For example, the same speaker encoder may extract a speaker embedding as part of an [anonymization system] as well as an [ASV system]. An anonymizer might swap out a WavLM feature extractor for a WhisperEncoder one. The components directory currently 




