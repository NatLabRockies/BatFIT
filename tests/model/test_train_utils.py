import torch

from batfit.model.param_utils.train_utils import (
    _unpack_batch,
    learning_rate_schedule,
)
from batfit.model.paramNN import ProbParamFM, ProbProtParamFM
from batfit.model.surrogate_utils.train_utils import (
    learning_rate_schedule as learning_rate_schedule_surr,
)


def test__unpack_batch():
    batch_size, n_points = 4, 32
    device = torch.device("cpu")
    x = torch.rand(batch_size, 2, n_points)
    v = torch.rand(batch_size, 1, n_points)
    p = torch.rand(batch_size, 3)
    t_end = torch.rand(batch_size, 1)
    y = torch.rand(batch_size, 6)

    # (X, Y): no protocol, no end time
    model = ProbParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
    )
    x_out, p_out, t_out, y_out = _unpack_batch(model, [x, y], device)
    assert x_out is x and p_out is None and t_out is None
    assert torch.equal(y_out, y)

    # (X, T, Y): end time second to last
    model_td = ProbParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        signal_scaling="time_dependent_zscore",
    )
    _, p_out, t_out, y_out = _unpack_batch(model_td, [v, t_end, y], device)
    assert p_out is None
    assert torch.equal(t_out, t_end)
    assert torch.equal(y_out, y)

    # (X, P, Y) and (X, P, T, Y): protocol parameters second
    model_p = ProbProtParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[8],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_chirp.yaml",
    )
    _, p_out, t_out, y_out = _unpack_batch(model_p, [x, p, y], device)
    assert torch.equal(p_out, p) and t_out is None
    assert torch.equal(y_out, y)

    model_pt = ProbProtParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[8],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_chirp.yaml",
        signal_scaling="time_dependent_zscore",
    )
    _, p_out, t_out, y_out = _unpack_batch(model_pt, [v, p, t_end, y], device)
    assert torch.equal(p_out, p)
    assert torch.equal(t_out, t_end)
    assert torch.equal(y_out, y)


def test_learning_rate_schedule():
    lr_beg, lr_end = 1e-3, 1e-5
    # constant for the first epoch_end // 10 epochs
    assert learning_rate_schedule(0, 100, lr_beg, lr_end) == lr_beg
    assert learning_rate_schedule(9, 100, lr_beg, lr_end) == lr_beg
    # geometric decay, reaching lr_end and staying there
    lr_mid = learning_rate_schedule(60, 100, lr_beg, lr_end)
    assert abs(lr_mid - lr_beg * (lr_end / lr_beg) ** 0.5) < 1e-12
    assert (
        abs(learning_rate_schedule(200, 100, lr_beg, lr_end) - lr_end) < 1e-12
    )
    # a 1-epoch run (epoch_end = 1 * 3 // 4 = 0) must not divide by zero
    assert learning_rate_schedule(0, 0, lr_beg, lr_end) == lr_beg


def test_learning_rate_schedule_surr():
    lr_beg, lr_end = 1e-3, 1e-5
    # same schedule as the NPE one
    for epoch in (0, 9, 60, 200):
        assert learning_rate_schedule_surr(
            epoch, 100, lr_beg, lr_end
        ) == learning_rate_schedule(epoch, 100, lr_beg, lr_end)
    # a 1-epoch run (epoch_end = 1 * 3 // 4 = 0) must not divide by zero
    assert learning_rate_schedule_surr(0, 0, lr_beg, lr_end) == lr_beg
