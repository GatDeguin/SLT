import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from slt.utils import masked_mean, set_seed


def test_set_seed_reproducibility():
    set_seed(123, deterministic=True)
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(3)

    set_seed(123, deterministic=True)
    assert random.random() == pytest.approx(r1)
    assert np.random.rand() == pytest.approx(n1)
    t2 = torch.rand(3)
    assert torch.allclose(t1, t2)


def test_set_seed_toggles_cudnn_flags():
    set_seed(321, deterministic=True)

    if not hasattr(torch.backends, "cudnn"):
        pytest.skip("cuDNN backend not available")

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_seed(321, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_masked_mean_basic():
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    result = masked_mean(tensor, mask, dim=1)
    expected = torch.tensor([1.5, 4.0])
    assert torch.allclose(result, expected)


def test_masked_mean_broadcast_and_keepdim():
    tensor = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    mask = torch.tensor([1, 0, 1], dtype=torch.float32)

    result = masked_mean(tensor, mask, dim=-1)
    expected = torch.tensor([[1.0, 4.0], [7.0, 10.0]])
    assert torch.allclose(result, expected)

    keepdim_result = masked_mean(tensor, torch.ones_like(tensor), dim=-1, keepdim=True)
    assert torch.allclose(keepdim_result, tensor.mean(dim=-1, keepdim=True))
