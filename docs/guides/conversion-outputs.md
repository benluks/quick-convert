# Conversion outputs and resuming

Speech conversion workflows save trimmed WAV files under
`<out_dir>/<split>/<utterance_id>.wav`. IDs and splits must be plain names.
Duplicate destinations are rejected. WAV publication is atomic.

`run.yaml` records run identity, status and counts. `conversions.jsonl` is the
durable output journal; `targets.jsonl` records prepared-target provenance.
`config.yaml` and `configs/` retain configuration snapshots. `targets.yaml`,
`target_assignments.jsonl`, and `speechscope.yaml` are rebuilt from the journals.
The SpeechScope manifest contains source/variant paths, lengths, sample rates and
target references without serializing conditioning tensors.

A rerun skips an output only when its journal entry, source file identity, target
assignment and output checksum agree. Changed model or target identity requires
a new output directory or explicit `overwrite=True`. Unrecorded existing files
are rejected; use a fresh directory when migrating old outputs. File identity
uses resolved paths, size and modification time, so it is not a content hash of
model weights or reference audio. Output WAVs are hashed.

An interrupted final journal append is repaired; corruption in earlier rows is
an error. Failed runs retain completed outputs for resuming. Use one writer per
output directory. Records contain local paths and supplied metadata; review them
before sharing results.
