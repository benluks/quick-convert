# Feature precomputation

Precompute features to reuse them in training, fitting, or evaluation. Choose a
run from the [workflow catalog](workflows.md) and check its
[optional dependencies](../config/dependencies.md).

```bash
uv run quick-convert precompute_content_w2vbert_librispeech data_root=/data
```

The pipeline writes feature files and a manifest in the configured
`pipeline.out_dir`. See [cached content representations](../data/index.md#cached-content-representations)
for the SSL payload and [attaching a sidecar resource](../library/python-api.md#attach-a-sidecar-resource)
for loading it into a dataset.

## Resume feature precomputation

With `pipeline.skip_existing=true`, the precompute workflow loads and extracts
only recordings whose feature file is missing. It publishes complete feature
files individually and replaces `manifest.jsonl` only after the run succeeds.
The manifest includes both retained and newly computed features.

Keep the same dataset rows, extractor configuration, and input revision when
resuming. The output directory's `provenance.json` records the resolved dataset
and extractor configuration plus a digest of ordered sample IDs, splits, and
paths. Batch size and worker count can change between runs. Use
`+pipeline.cache_key=my-input-revision` to identify changes that the config and
paths do not capture, such as replacing weights or audio at the same path.
Python callers enabling resume must pass `cache_key="my-input-revision"`.

For an older output directory without an identity, or when changing the
extractor or input revision, set `pipeline.out_dir` to a fresh directory. Feature files retain the
`<split>/<utt_id>.pt` layout.
