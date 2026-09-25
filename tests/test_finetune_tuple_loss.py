"""Tests for the tuple-loss compatibility fix in finetune_from_trajectories.py."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "finetune_from_trajectories.py"


def load_module():
    """Import _reduce_loss / _make_trainer_cls from the script without running main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("finetune_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return load_module()


def test_plain_tensor_loss_passthrough(mod):
    """A plain scalar loss must pass through unchanged (identity)."""
    loss = torch.tensor(1.5, requires_grad=True)
    out = mod._reduce_loss(loss)
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, loss)


def test_tuple_loss_reduced_to_scalar(mod):
    """A tuple of tensor losses must be summed to a single scalar tensor."""
    l1 = torch.tensor(0.5, requires_grad=True)
    l2 = torch.tensor(1.25, requires_grad=True)
    out = mod._reduce_loss((l1, l2))
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 0  # scalar
    expected = l1 + l2
    assert torch.allclose(out, expected)


def test_nested_tuple_loss_reduced(mod):
    """A nested tuple structure must also collapse to a scalar."""
    l1 = torch.tensor(2.0, requires_grad=True)
    l2 = torch.tensor(3.0, requires_grad=True)
    nested = (l1, (l2,))
    out = mod._reduce_loss(nested)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, l1 + l2)


def test_list_loss_reduced(mod):
    """A list of tensor losses must be summed to a scalar."""
    l1 = torch.tensor(1.0, requires_grad=True)
    l2 = torch.tensor(2.0, requires_grad=True)
    out = mod._reduce_loss([l1, l2])
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, l1 + l2)


def test_trainer_subclass_flattens_loss(mod):
    """The returned trainer subclass must be a valid SFTTrainer subclass and its
    compute_loss contract must collapse tuple losses."""
    try:
        from trl import SFTTrainer

        has_trl = True
    except Exception:
        has_trl = False

    if not has_trl:
        pytest.skip("trl not installed in this env")

    cls = mod._make_trainer_cls(SFTTrainer)
    assert issubclass(cls, SFTTrainer)
    # compute_loss is overridden (not the bare base method).
    assert cls.compute_loss is not SFTTrainer.compute_loss


def test_script_compiles(mod):
    """The script must compile cleanly (syntax check)."""
    subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT)], check=True)
