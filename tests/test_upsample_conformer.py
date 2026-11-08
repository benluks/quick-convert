import pytest
import torch

from quick_convert.components.sequence.upsample_conformer import UpsampleConformerEncoder
from quick_convert.external.cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder as Original


@pytest.mark.parametrize("target_lengths", [torch.tensor([4]), torch.tensor([4, 0]), torch.tensor([4.0, 2.0])])
def test_invalid_requested_lengths(target_lengths):
    model = UpsampleConformerEncoder(input_size=4, output_size=8, attention_heads=2, num_blocks=0)
    with pytest.raises(ValueError, match="target_lengths"):
        model(torch.ones(2, 3, 4), torch.tensor([3, 2]), target_lengths=target_lengths)


def test_requested_lengths_mask_padding_and_keep_input_owned():
    model = UpsampleConformerEncoder(
        input_size=4,
        output_size=8,
        attention_heads=2,
        num_blocks=0,
        dropout_rate=0,
        positional_dropout_rate=0,
    ).eval()
    values = torch.randn(2, 3, 4, requires_grad=True)
    original = values.detach().clone()
    requested = torch.tensor([2, 5])
    output, mask = model(values, torch.tensor([3, 2]), target_lengths=requested)
    assert output.shape == (2, 5, 8)
    assert mask.sum(-1).flatten().tolist() == requested.tolist()
    assert output[0, 2:].count_nonzero() == 0
    output.square().sum().backward()
    assert values.grad[1, 2:].count_nonzero() == 0
    torch.testing.assert_close(values, original)
    with pytest.raises(ValueError, match="offline"):
        model(values, torch.tensor([3, 2]), target_lengths=requested, streaming=True)


@pytest.mark.parametrize("upsample", [False, True])
@pytest.mark.parametrize("cnn", [False, True])
def test_original_checkpoint_output_and_gradient_parity(upsample, cnn):
    settings = dict(
        input_size=16,
        output_size=512,
        attention_heads=8,
        linear_units=32,
        num_blocks=1,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        use_cnn_module=cnn,
        macaron_style=cnn,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        static_chunk_size=4,
    )
    original = Original(**settings, do_upsample=upsample, upsample_stride=2 if upsample else 1).eval()
    custom = UpsampleConformerEncoder(
        **settings,
        post_upsample_blocks=4,
        second_projection=True,
        lookahead_frames=3,
        upsample_factor=2 if upsample else 1,
    ).eval()
    custom.load_state_dict(original.state_dict(), strict=True)
    original.train()
    custom.train()
    train_x = torch.randn(2, 7, 16)
    for module in (original, custom):
        for child in module.modules():
            if isinstance(child, torch.nn.Dropout):
                child.p = 0.1
    torch.manual_seed(123)
    expected_train, _ = original(train_x, torch.tensor([7, 5]))
    torch.manual_seed(123)
    actual_train, _ = custom(train_x, torch.tensor([7, 5]))
    torch.testing.assert_close(actual_train, expected_train, rtol=0, atol=0)
    original.eval()
    custom.eval()
    lengths = torch.tensor([7, 5])
    for streaming in (False, True):
        x = torch.randn(2, 7, 16, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected, expected_mask = original(x, lengths, streaming=streaming)
        actual, mask = custom(y, lengths, streaming=streaming)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(mask, expected_mask)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
    context = torch.randn(2, 3, 16)
    with torch.no_grad():
        expected, _ = original(x, lengths, context=context, streaming=True)
        actual, _ = custom(y, lengths, context=context, streaming=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_custom_width_depth_and_lengths(factor):
    model = UpsampleConformerEncoder(
        input_size=12, output_size=16, attention_heads=4, linear_units=24, num_blocks=2, upsample_factor=factor
    ).eval()
    assert len(model.encoders) == 2
    assert len(model.up_encoders) == 0
    assert model.up_embed is None
    lengths = torch.tensor([7, 4])
    output, mask = model(torch.randn(2, 7, 12), lengths)
    assert output.shape == (2, 7 * factor, 16)
    assert torch.equal(mask.sum((1, 2)), model.output_lengths(lengths))


def test_second_stack_without_projection():
    model = UpsampleConformerEncoder(
        input_size=12,
        output_size=16,
        attention_heads=4,
        num_blocks=0,
        post_upsample_blocks=2,
        upsample_factor=3,
        lookahead_frames=0,
    )
    output, _ = model(torch.randn(1, 5, 12), torch.tensor([5]))
    assert output.shape == (1, 15, 16)


@pytest.mark.parametrize(
    "settings",
    [
        dict(num_blocks=-1),
        dict(post_upsample_blocks=-1),
        dict(upsample_factor=0),
        dict(lookahead_frames=-1),
        dict(input_layer="conv2d"),
        dict(output_size=15),
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        UpsampleConformerEncoder(input_size=12, **settings)


def test_decoder_config_uses_the_first_party_encoder():
    from importlib.resources import files

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(
        str(files("quick_convert.configs").joinpath("components/generators/acoustic/causal_masked_diff.yaml"))
    )
    encoder = config.flow.encoder
    encoder.static_chunk_size = 4
    model = instantiate(encoder)
    assert isinstance(model, UpsampleConformerEncoder)
    assert len(model.encoders) == 6
    assert len(model.up_encoders) == 4
    assert model.pre_lookahead_layer.pre_lookahead_len == 3
    assert model.up_embed is not None
