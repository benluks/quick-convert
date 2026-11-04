# Upsample Conformer encoder

`quick_convert.components.encoders.upsample_conformer.UpsampleConformerEncoder`
is a first-party adapter around vendored CosyVoice attention, convolution, and
positional-encoding blocks. It accepts `[batch, time, input_size]` features and
an integer valid-length vector. It returns features and a `[batch, 1, time]`
valid-frame mask. `output_lengths(lengths)` reports lengths after upsampling.

For a compact custom encoder:

```python
import torch
from quick_convert.components.encoders.upsample_conformer import UpsampleConformerEncoder

encoder = UpsampleConformerEncoder(
    input_size=1024,
    output_size=256,
    attention_heads=4,
    num_blocks=3,
    upsample_factor=2,
)
features, valid = encoder(torch.randn(2, 10, 1024), torch.tensor([10, 7]))
assert features.shape == (2, 20, 256)
assert valid.sum((1, 2)).tolist() == [20, 14]
```

Python defaults use one six-block stack, without a second stack, second
projection, lookahead, or resampling. `num_blocks` counts the first stack;
`post_upsample_blocks` explicitly adds blocks after resampling.
`upsample_factor=1` omits the resampler. `second_projection=True` adds another
input projection before the second stack.

`lookahead_frames=None` omits the lookahead module; zero still creates a learned
causal residual convolution. Nonempty `context` requires lookahead and evaluation
mode. `causal` controls the convolution branches, while `streaming=True` selects
fixed chunk attention with unlimited left context. These are distinct controls.

The decoder preset deliberately selects the donor-equivalent second stack,
projection and lookahead. For equivalent architectures, parameter names and
shapes match the donor and support strict checkpoint loading. A different width,
depth or resampling topology requires a matching checkpoint. Existing saved
configs referencing the vendored class remain loadable; these underlying
vendored classes are implementation details rather than supported extension APIs.

For new configs, replace `do_upsample` and `upsample_stride` with
`upsample_factor`: use `1` to disable resampling or the desired integer factor
to enable it. Old donor constructor fields are rejected by the first-party
adapter. Preserving the donor topology also requires `post_upsample_blocks=4`,
`second_projection=true`, and `lookahead_frames=3`, as the decoder preset does.
