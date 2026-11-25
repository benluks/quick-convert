import pytest
import torch

from quick_convert.components.conditioning.nearest_neighbour import CosineTopK, NearestNeighbourMatching


def test_matching_uses_cosine_and_unnormalized_values():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    keys = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 100.0]])
    matcher = NearestNeighbourMatching(CosineTopK(k=2, chunk_size=1))
    torch.testing.assert_close(matcher(query, keys), torch.tensor([[1.5, 0.5], [0.5, 50.5]]))
    values = torch.tensor([[2.0], [4.0], [8.0]])
    torch.testing.assert_close(matcher(query, keys, values), torch.tensor([[3.0], [6.0]]))


def test_chunked_matching_matches_dense():
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(19, 8, generator=generator)
    keys = torch.randn(23, 8, generator=generator)
    expected = CosineTopK(chunk_size=100)(query, keys)
    torch.testing.assert_close(CosineTopK(chunk_size=3)(query, keys), expected)


def test_invalid_pool():
    matcher = NearestNeighbourMatching()
    with pytest.raises(ValueError, match="fewer frames"):
        matcher(torch.ones(2, 3), torch.ones(3, 3))
    with pytest.raises(ValueError, match="aligned frames"):
        matcher(torch.ones(2, 3), torch.ones(4, 3), torch.ones(3, 2))


def test_empty_query():
    result = NearestNeighbourMatching()(torch.empty(0, 3), torch.ones(4, 3))
    assert result.shape == (0, 3)
