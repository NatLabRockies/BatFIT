import numpy as np

from batfit.calibration.cal_utils import get_nchan, make_data_in


def test_get_nchan():
    assert get_nchan("phi") == 2
    assert get_nchan("dvdq") == 2
    assert get_nchan("dqdv") == 2
    assert get_nchan("phi-dvdq") == 3
    assert get_nchan("phi-dqdv") == 3
    assert get_nchan("dvdq-dqdv") == 3
    assert get_nchan("phi-dvdq-dqdv") == 4


def test_make_data_in():
    n_points = 20
    t = np.linspace(0.0, 1.0, n_points).astype("float32")
    phi = np.sin(t).astype("float32")

    data_t = {"discharge": t}
    data_phis_c = {"discharge": phi}

    data_in = make_data_in(
        target_mode="phi",
        cyc_mode="discharge",
        n_points=n_points,
        data_t_dV_dQ=None,
        data_dV_dQ_x=None,
        data_t=data_t,
        data_phis_c=data_phis_c,
        data_dV_dQ_y=None,
        data_dQ_dV_y=None,
    )

    assert data_in.shape == (1, 2, n_points)
    assert np.allclose(data_in[0, 0, :], t)
    assert np.allclose(data_in[0, 1, :], phi)
