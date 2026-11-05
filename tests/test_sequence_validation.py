import pytest
import torch

from quick_convert.utils.sequence_validation import validate_padded_lengths


def test_valid_padded_lengths_allow_empty_sequences():
    validate_padded_lengths(torch.zeros(2, 4, 3), torch.tensor([0, 4]))


@pytest.mark.parametrize(
    ("lengths", "message"),
    [
        (torch.tensor([1]), "shape"),
        (torch.tensor([-1, 2]), "non-negative"),
        (torch.tensor([1, 5]), "padded time"),
    ],
)
def test_invalid_padded_lengths(lengths, message):
    with pytest.raises(ValueError, match=message):
        validate_padded_lengths(torch.zeros(2, 4, 3), lengths)


def test_time_dimension_is_explicit():
    validate_padded_lengths(torch.zeros(2, 3, 4), torch.tensor([4, 2]), time_dim=2)
