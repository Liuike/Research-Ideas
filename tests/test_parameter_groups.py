from optimizer_resurrection.models import SigmoidMLP, VanillaRNN
from optimizer_resurrection.optim import ManifoldMuon, Muon, build_optimizer, split_parameters


def _optimizer_ids(optimizer):
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def test_mlp_primary_assignment_is_w2_through_wl_only():
    model = SigmoidMLP(20, 8, 4, 3, init="orthogonal")
    compared, auxiliary = split_parameters(model)
    assert compared == [layer.weight for layer in model.hidden[1:]]
    assert id(model.hidden[0].weight) in {id(p) for p in auxiliary}
    assert id(model.output.weight) in {id(p) for p in auxiliary}
    bundle = build_optimizer(model, "muon_o", 0.01, 0.001)
    assert isinstance(bundle.primary, Muon)
    assert _optimizer_ids(bundle.primary) == {id(p) for p in compared}


def test_rnn_primary_assignment_is_recurrent_matrix_only():
    model = VanillaRNN(2, 8, init="orthogonal")
    compared, _ = split_parameters(model)
    bundle = build_optimizer(model, "mm", 0.01, 0.001)
    assert compared == [model.W_hh]
    assert isinstance(bundle.primary, ManifoldMuon)
    assert _optimizer_ids(bundle.primary) == {id(model.W_hh)}


def test_standard_assignment_is_explicitly_broader():
    mlp = SigmoidMLP(20, 8, 3, 2)
    compared, _ = split_parameters(mlp, "standard")
    assert compared == [layer.weight for layer in mlp.hidden]
    rnn = VanillaRNN(2, 8)
    compared, _ = split_parameters(rnn, "standard")
    assert compared == [rnn.W_hh, rnn.W_xh]


def test_baselines_do_not_use_adam_auxiliaries():
    model = VanillaRNN(2, 8)
    sgd = build_optimizer(model, "bptt_sgd", 0.01, 0.001)
    assert sgd.auxiliary is None
    assert _optimizer_ids(sgd.primary) == {id(parameter) for parameter in model.parameters()}
    adam = build_optimizer(model, "adam_x", 0.01, 0.001)
    assert adam.auxiliary is None
    assert _optimizer_ids(adam.primary) == {id(parameter) for parameter in model.parameters()}
