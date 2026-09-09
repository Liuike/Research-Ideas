import torch

from optimizer_resurrection.models import SigmoidMLP
from optimizer_resurrection.train import seed_everything


def test_model_initialization_repeats_for_same_seed():
    seed_everything(19)
    first = SigmoidMLP(12, 8, 3, 2)
    state = {name: value.clone() for name, value in first.state_dict().items()}
    seed_everything(19)
    second = SigmoidMLP(12, 8, 3, 2)
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in state.items())

