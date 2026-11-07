# Vendored code

This directory contains third-party implementations retained behind
first-party adapters in `quick_convert.components`. They are implementation
details rather than Quick Convert's public API.

- `cosyvoice/` derives from
  [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice). Files retain
  upstream copyright and Apache-2.0 notices where present.
- `matcha/` derives from
  [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
  Its HiFi-GAN subtree includes its upstream license.
- `psst1/` derives from the PSST1 research code (emotion-compensation, local
  checkout, uncommitted working copy as of 2026-10-22). `flow.py` adapts
  `flow_matching/flow_matcher.py` (itself from
  [facebookresearch/DiT](https://github.com/facebookresearch/DiT), CC BY-NC
  4.0) with the emotion-conditioning branches removed. `hifigan.py` adapts
  `adapted_from_facebookresearch/models_test.py` (from
  [facebookresearch/speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis),
  CC BY-NC 4.0, as modified by Xiaoxiao Miao) with weight normalization folded
  into the exported weights and the discriminators removed. The research code
  is itself CC BY-NC 4.0, so this subtree is for non-commercial use only.

The trees contain local import-isolation and adapter changes and are not
verbatim upstream checkouts. Exact source revisions were not recorded when the
code was introduced; future updates should record the revision and local
changes here.

These notices do not establish a license for Quick Convert itself. The
repository needs an explicit project-level license before redistribution terms
are clear.
