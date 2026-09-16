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

The trees contain local import-isolation and adapter changes and are not
verbatim upstream checkouts. Exact source revisions were not recorded when the
code was introduced; future updates should record the revision and local
changes here.

These notices do not establish a license for Quick Convert itself. The
repository needs an explicit project-level license before redistribution terms
are clear.
