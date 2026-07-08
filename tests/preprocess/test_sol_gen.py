import os
import pickle
import tempfile
from unittest import mock

from batfit import BATFIT_EXP
from batfit.preprocess.sim_setup import make_params
from batfit.preprocess.sol_gen import multi_run, multi_run_ser


def _fake_single_run(
    deg_param_sample,
    sim_params,
    count=None,
    nsim=None,
    parallel_env=None,
    run_mode=None,
    prot_param_sample=None,
):
    """Deterministic stand-in for single_run: no simulation, echo params."""
    params_list = [
        deg_param_sample[key] for key in sim_params["deg_param_names"]
    ]
    prot_params_list = None
    if prot_param_sample is not None:
        prot_params_list = [
            prot_param_sample[key] for key in sim_params["prot_param_names"]
        ]
    return params_list, prot_params_list, "fake_rootsol"


def test_multi_run():
    # multi_run with parallel_env=None must produce the same sols.pkl as
    # multi_run_ser, and must forward the save_* flags to save_datapoint
    sim_params = make_params(os.path.join(BATFIT_EXP, "spm_chirp.yaml"))

    # 3 sims, values within the spm_chirp.yaml bounds
    # deg names: i0_a, ds_c, x0_a, x0_c, i0_c, eps_s_c_am
    deg_lines = [
        "1 1 1 0.9 1 0.8",
        "0.5 2 0.5 0.8 0.5 0.75",
        "2 5 1.5 0.95 1.2 0.9",
    ]
    # prot names: time_start, amplitude, length
    prot_lines = [
        "100 0.1 20",
        "500 0.2 50",
        "1500 0.05 80",
    ]

    results = {}
    seen_kwargs = []

    def _fake_save_datapoint(
        params_list, rootsol, db=None, prot_params_list=None, **kwargs
    ):
        seen_kwargs.append(kwargs)
        db.append(
            {
                "sim_id": int(db.n_data + 1),
                "params": params_list,
                "prot_params": prot_params_list,
                "sol": {"fake": True},
            }
        )

    for label in ["ser", "multi"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "parameter_list.txt"), "w") as f:
                f.write("\n".join(deg_lines) + "\n")
            with open(
                os.path.join(tmp_dir, "protocol_parameter_list.txt"), "w"
            ) as f:
                f.write("\n".join(prot_lines) + "\n")

            with mock.patch(
                "batfit.preprocess.sol_gen.single_run", _fake_single_run
            ), mock.patch(
                "batfit.preprocess.sol_gen.save_datapoint",
                _fake_save_datapoint,
            ):
                if label == "ser":
                    multi_run_ser(
                        sim_params=sim_params,
                        folder_save=tmp_dir,
                        save_separate_sols=True,
                        save_combined_sols=True,
                    )
                else:
                    multi_run(
                        sim_params=sim_params,
                        folder_save=tmp_dir,
                        parallel_env=None,
                        save_separate_sols=True,
                        save_combined_sols=True,
                    )

            with open(os.path.join(tmp_dir, "sols.pkl"), "rb") as f:
                results[label] = pickle.load(f)

    # identical combined databases from both entry points
    assert results["ser"].keys() == results["multi"].keys()
    for key in results["ser"]:
        assert results["ser"][key] == results["multi"][key]

    # per-sim protocol params must line up with the file order
    prot_expected = [[float(v) for v in line.split()] for line in prot_lines]
    deg_expected = [[float(v) for v in line.split()] for line in deg_lines]
    sorted_records = [results["ser"][k] for k in sorted(results["ser"])]
    assert [r["prot_params"] for r in sorted_records] == prot_expected
    assert [r["params"] for r in sorted_records] == deg_expected

    # multi_run must forward the save_* flags to save_datapoint
    assert all(kw["save_separate_sols"] is True for kw in seen_kwargs)
    assert all(kw["save_combined_sols"] is True for kw in seen_kwargs)
