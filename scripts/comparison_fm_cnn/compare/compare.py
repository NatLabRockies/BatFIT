"""Compare four models on the shared SPM discharge test split.

Models compared
---------------
- CNN NPE (constrain_output=True, physical Y)            — blue
- CNN NPE z-scored (constrain_output=False, scale_y=True) — purple
- FM with prior matching (empirical prior, scale_y=True)  — red
- FM without prior matching (N(0,I) base, scale_y=True)   — green

Metrics computed on the shared test split:
- Relative accuracy (1 - mean_rel_error) per parameter
- Coverage at 1σ, 2σ, 3σ using the sample distribution

Outputs
-------
- comparison_results.npz
- figures/corner_obs<N>.{png,pdf}
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import sys

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import get_device_type

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_best_model_file(models_dir: str) -> str:
    """Return the checkpoint path with the lowest recorded test loss.

    :param models_dir: directory produced by a training run
    :return: absolute path to the best .pt file
    """
    loss_file = os.path.join(models_dir, "test_loss.csv")
    vals = np.loadtxt(loss_file, delimiter=";", skiprows=1)
    if vals.ndim == 1:
        vals = vals[np.newaxis, :]
    best_idx = int(np.argmin(vals[:, 1]))
    final_path = os.path.join(models_dir, "model_final.pt")
    if best_idx == vals.shape[0] - 1 and os.path.isfile(final_path):
        return final_path
    filenames = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("model_") and f.endswith(".pt") and "final" not in f
    ]
    iterations = [int(f[6:-3]) for f in filenames]
    best_step = int(vals[best_idx, 0])
    closest = min(iterations, key=lambda x: abs(x - best_step))
    return os.path.join(models_dir, f"model_{closest}.pt")


def _load_model(models_dir: str) -> torch.nn.Module:
    """Load the best checkpoint from a training directory.

    :param models_dir: training output directory
    :return: model in eval mode on CPU
    """
    model_pkl = os.path.join(models_dir, "model.pkl")
    best_pt = _find_best_model_file(models_dir)
    logger.info(f"Loading {best_pt}")
    model = create_model_from_log(model_pkl, best_pt)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _make_test_loader(
    data_path: str, batch_size: int = 256
) -> "torch.utils.data.DataLoader":
    """Build a DataLoader from the saved test split (pre-scaled signals).

    Y labels are always physical (from data_split.npz); individual result
    collectors apply their own inverse transforms as needed.

    :param data_path: folder containing ``data_split.npz`` and ``scaler_X.pkl``
    :param batch_size: DataLoader batch size
    :return: DataLoader yielding (X_scaled, Y_physical) batches
    """
    split = np.load(os.path.join(data_path, "data_split.npz"))
    X_scaled = scale_input_from_scaler(
        split["X_test"], os.path.join(data_path, "scaler_X.pkl")
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(split["Y_test"], dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


def _collect_cnn_results(
    model: torch.nn.Module,
    test_loader: "torch.utils.data.DataLoader",
    scaler_X,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CNN (constrain_output=True) forward pass and draw Gaussian samples.

    Applies inv_transform_output to map the Sigmoid output to physical space
    before sampling and returning.

    :return: (mu_all, sigma_all, samples_all, truth_all) — all in physical space
    """
    model = model.to(device)
    model.eval()
    mu_list, sigma_list, truth_list = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler_X,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            mu, gamma = model(batch_in.to(device))
            mu, gamma = model.inv_transform_output(
                mu,
                gamma,
                model.min_par.to(device),
                model.amp_par.to(device),
            )
            mu_list.append(mu.cpu().numpy())
            sigma_list.append(gamma.cpu().numpy())
            truth_list.append(batch[1].numpy())

    mu_all = np.vstack(mu_list)
    sigma_all = np.vstack(sigma_list)
    truth_all = np.vstack(truth_list)

    rng = np.random.default_rng(42)
    samples_all = rng.normal(
        loc=mu_all[:, np.newaxis, :],
        scale=np.clip(sigma_all[:, np.newaxis, :], a_min=1e-8, a_max=None),
        size=(mu_all.shape[0], n_samples, mu_all.shape[1]),
    ).astype(np.float32)

    return mu_all, sigma_all, samples_all, truth_all


def _collect_cnn_scaled_results(
    model: torch.nn.Module,
    test_loader: "torch.utils.data.DataLoader",
    scaler_X,
    scaler_Y,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CNN (constrain_output=False, scale_y=True) and inv-transform samples.

    The model outputs (mu_z, sigma_z) in z-scored parameter space.  Gaussian
    samples are drawn in that space and then inverse-transformed to physical
    space via scaler_Y before returning.

    :param scaler_Y: StandardScaler fitted on Y_train; used to invert z-scoring
    :return: (mu_all, sigma_all, samples_all, truth_all) — all in physical space
    """
    model = model.to(device)
    model.eval()
    mu_list, sigma_list, truth_list = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler_X,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            mu, gamma = model(batch_in.to(device))
            mu_list.append(mu.cpu().numpy())
            sigma_list.append(gamma.cpu().numpy())
            truth_list.append(batch[1].numpy())

    mu_z = np.vstack(mu_list)  # z-scored
    sigma_z = np.vstack(sigma_list)  # z-scored
    truth_all = np.vstack(truth_list)

    rng = np.random.default_rng(42)
    samples_z = rng.normal(
        loc=mu_z[:, np.newaxis, :],
        scale=np.clip(sigma_z[:, np.newaxis, :], a_min=1e-8, a_max=None),
        size=(mu_z.shape[0], n_samples, mu_z.shape[1]),
    ).astype(np.float32)

    # Inverse-transform from z-scored → physical space
    n_test, n_samp, n_params = samples_z.shape
    samples_all = (
        scaler_Y.inverse_transform(samples_z.reshape(-1, n_params))
        .reshape(n_test, n_samp, n_params)
        .astype(np.float32)
    )

    mu_all = scaler_Y.inverse_transform(mu_z).astype(np.float32)
    # Approximate physical sigma: sigma_z * per-parameter training std
    sigma_all = (sigma_z * scaler_Y.scale_).astype(np.float32)

    return mu_all, sigma_all, samples_all, truth_all


def _collect_fm_results(
    model: torch.nn.Module,
    test_loader: "torch.utils.data.DataLoader",
    scaler_X,
    scaler_Y,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_samples: int,
    n_ode_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw posterior samples from a ProbParamFM model.

    The FM model was trained with z-scored Y (scale_y=True), so model.sample()
    returns samples in z-scored space.  scaler_Y is used to inverse-transform
    them to physical space before returning.

    :return: (samples_all, truth_all) — both in physical space
    """
    model = model.to(device)
    model.eval()
    samples_list, truth_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler_X,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            x_signal = batch_in.to(device)
            samps = model.sample(
                x_signal, n_samples=n_samples, n_steps=n_ode_steps
            )
            samples_list.append(samps.cpu().numpy())
            truth_list.append(batch[1].numpy())

    samples_z = np.vstack(
        samples_list
    )  # (n_test, n_samples, n_params) z-scored
    truth_all = np.vstack(truth_list)  # (n_test, n_params) physical

    n_test, n_samp, n_params = samples_z.shape
    samples_all = (
        scaler_Y.inverse_transform(samples_z.reshape(-1, n_params))
        .reshape(n_test, n_samp, n_params)
        .astype(np.float32)
    )

    return samples_all, truth_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_rel_accuracy(mu_est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-parameter relative accuracy: 1 - mean(|mu - truth| / range).

    :param mu_est: (n_test, n_params)
    :param truth: (n_test, n_params)
    :return: (n_params,)
    """
    amp = truth.max(axis=0) - truth.min(axis=0) + 1e-12
    return 1.0 - np.mean(np.abs(mu_est - truth) / amp, axis=0)


def _compute_coverage(
    samples: np.ndarray, truth: np.ndarray
) -> tuple[float, float, float]:
    """Coverage at 1σ, 2σ, 3σ using sample mean and std.

    :param samples: (n_test, n_samples, n_params)
    :param truth: (n_test, n_params)
    :return: (cov1, cov2, cov3)
    """
    mu = samples.mean(axis=1)
    std = samples.std(axis=1)
    diff = np.abs(truth - mu)
    cov1 = float(np.mean(diff <= 1.0 * std))
    cov2 = float(np.mean(diff <= 2.0 * std))
    cov3 = float(np.mean(diff <= 3.0 * std))
    return cov1, cov2, cov3


# ---------------------------------------------------------------------------
# Corner plots
# ---------------------------------------------------------------------------


def _plot_four_corner(
    samples_cnn: np.ndarray,
    samples_cnn_scaled: np.ndarray,
    samples_fm_pm: np.ndarray,
    samples_fm_no_pm: np.ndarray,
    truth: np.ndarray | None = None,
    labels: list[str] | None = None,
    extent: list[tuple[float, float]] | None = None,
    out_path: str | None = None,
    obs_idx: int | None = None,
    fontsize: int = 14,
) -> plt.Figure:
    """Overlay posterior contours from four models in a single corner plot.

    Colors: CNN=blue, CNN z-scored=purple, FM prior-match=red, FM N(0,I)=green.
    Ground truth shown as black lines when provided.

    :param samples_cnn: (n_samples, n_params) — CNN physical
    :param samples_cnn_scaled: (n_samples, n_params) — CNN z-scored
    :param samples_fm_pm: (n_samples, n_params) — FM prior-match
    :param samples_fm_no_pm: (n_samples, n_params) — FM N(0,I)
    :param truth: (n_params,) — ground-truth values
    :param labels: parameter axis labels
    :param extent: list of (min, max) per parameter for axis limits
    :param out_path: directory for saved figures; skipped if None
    :param obs_idx: index appended to the filename
    :param fontsize: label font size
    :return: matplotlib Figure
    """

    def _clean(s: np.ndarray) -> np.ndarray:
        return s[np.all(np.isfinite(s), axis=1)]

    s_cnn = _clean(samples_cnn)
    s_cnn_sc = _clean(samples_cnn_scaled)
    s_pm = _clean(samples_fm_pm)
    s_no_pm = _clean(samples_fm_no_pm)

    common_kw = dict(
        labels=labels,
        range=extent,
        label_kwargs={"fontsize": fontsize},
        show_titles=False,
        plot_datapoints=False,
        fill_contours=False,
        smooth=1.0,
        plot_density=True,
        bins=20,
        hist_kwargs={"density": True},
    )

    common_kw["hist_kwargs"]["color"] = "blue"
    fig = corner.corner(s_cnn, color="blue", **common_kw)
    common_kw["hist_kwargs"]["color"] = "purple"
    corner.corner(s_cnn_sc, color="purple", fig=fig, **common_kw)
    common_kw["hist_kwargs"]["color"] = "red"
    corner.corner(s_pm, color="red", fig=fig, **common_kw)
    common_kw["hist_kwargs"]["color"] = "green"
    corner.corner(s_no_pm, color="green", fig=fig, **common_kw)

    if truth is not None:
        ndim = s_cnn.shape[1]
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            axes[i, i].axvline(truth[i], color="black", lw=2)
            for j in range(i):
                axes[i, j].axvline(truth[j], color="black", lw=2)
                axes[i, j].axhline(truth[i], color="black", lw=2)
                axes[i, j].plot(truth[j], truth[i], "ko", markersize=3)

    legend_handles = [
        plt.Line2D([0], [0], color="blue", lw=2, label="CNN (physical)"),
        plt.Line2D([0], [0], color="purple", lw=2, label="CNN (z-scored)"),
        plt.Line2D([0], [0], color="red", lw=2, label="FM — prior match"),
        plt.Line2D([0], [0], color="green", lw=2, label="FM — N(0,I)"),
        plt.Line2D([0], [0], color="black", lw=2, label="Truth"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=fontsize - 2
    )
    plt.tight_layout()

    if out_path is not None:
        tag = f"_obs{obs_idx}" if obs_idx is not None else ""
        fig.savefig(os.path.join(out_path, f"corner{tag}.png"), dpi=150)
        fig.savefig(os.path.join(out_path, f"corner{tag}.pdf"))

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])

    device = torch.device(get_device_type(enable_cuda=True, enable_mps=True))

    with open(os.path.join(inp.data_path, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(inp.data_path, "scaler_Y.pkl"), "rb") as f:
        scaler_Y = pickle.load(f)

    noise_levels, a_min, a_max = make_noise_levels(
        target_mode=inp.target_mode,
        noise_levels=[
            0,
            0.001444 * 2 * inp.noise_factor,
            0.001786 * 2,
            2.01 * 2,
        ],
        cyc_mode=inp.cyc_mode,
    )

    test_loader = _make_test_loader(inp.data_path)

    logger.info("Loading CNN model ...")
    cnn_model = _load_model(inp.cnn_models_dir)

    logger.info("Loading CNN z-scored model ...")
    cnn_scaled_model = _load_model(inp.cnn_scaled_models_dir)

    logger.info("Loading FM (prior match) model ...")
    fm_pm_model = _load_model(inp.fm_pm_models_dir)

    logger.info("Loading FM (no prior match) model ...")
    fm_no_pm_model = _load_model(inp.fm_no_pm_models_dir)

    logger.info("Running CNN inference ...")
    mu_cnn, sigma_cnn, samples_cnn, truth = _collect_cnn_results(
        cnn_model,
        test_loader,
        scaler_X,
        noise_levels,
        a_min,
        a_max,
        n_samples=inp.n_samples,
        device=device,
    )

    logger.info("Running CNN z-scored inference ...")
    mu_cnn_sc, sigma_cnn_sc, samples_cnn_sc, _ = _collect_cnn_scaled_results(
        cnn_scaled_model,
        test_loader,
        scaler_X,
        scaler_Y,
        noise_levels,
        a_min,
        a_max,
        n_samples=inp.n_samples,
        device=device,
    )

    logger.info("Running FM (prior match) inference ...")
    samples_fm_pm, _ = _collect_fm_results(
        fm_pm_model,
        test_loader,
        scaler_X,
        scaler_Y,
        noise_levels,
        a_min,
        a_max,
        n_samples=inp.n_samples,
        n_ode_steps=inp.n_ode_steps,
        device=device,
    )

    logger.info("Running FM (no prior match) inference ...")
    samples_fm_no_pm, _ = _collect_fm_results(
        fm_no_pm_model,
        test_loader,
        scaler_X,
        scaler_Y,
        noise_levels,
        a_min,
        a_max,
        n_samples=inp.n_samples,
        n_ode_steps=inp.n_ode_steps,
        device=device,
    )

    # Accuracy
    mu_fm_pm = samples_fm_pm.mean(axis=1)
    mu_fm_no_pm = samples_fm_no_pm.mean(axis=1)

    acc_cnn = _compute_rel_accuracy(mu_cnn, truth)
    acc_cnn_sc = _compute_rel_accuracy(mu_cnn_sc, truth)
    acc_fm_pm = _compute_rel_accuracy(mu_fm_pm, truth)
    acc_fm_no_pm = _compute_rel_accuracy(mu_fm_no_pm, truth)

    # Coverage
    cov_cnn = _compute_coverage(samples_cnn, truth)
    cov_cnn_sc = _compute_coverage(samples_cnn_sc, truth)
    cov_fm_pm = _compute_coverage(samples_fm_pm, truth)
    cov_fm_no_pm = _compute_coverage(samples_fm_no_pm, truth)

    param_names = list(cnn_model.sim_params["deg_param_names"])

    sep = "=" * 92
    logger.info(f"\n{sep}")
    logger.info("Relative accuracy  (higher is better)")
    logger.info(sep)
    logger.info(
        f"{'Parameter':<12} {'CNN':<18} {'CNN z-scored':<18} "
        f"{'FM prior match':<18} {'FM N(0,I)':<18}"
    )
    logger.info("-" * 84)
    for i, name in enumerate(param_names):
        logger.info(
            f"{name:<12} {acc_cnn[i]:<18.4f} {acc_cnn_sc[i]:<18.4f} "
            f"{acc_fm_pm[i]:<18.4f} {acc_fm_no_pm[i]:<18.4f}"
        )
    logger.info("-" * 84)
    logger.info(
        f"{'Mean':<12} {acc_cnn.mean():<18.4f} {acc_cnn_sc.mean():<18.4f} "
        f"{acc_fm_pm.mean():<18.4f} {acc_fm_no_pm.mean():<18.4f}"
    )
    logger.info(sep)

    from scipy.stats import norm as _norm

    true_cov = [_norm.cdf(n) - _norm.cdf(-n) for n in [1, 2, 3]]

    logger.info(
        f"\n{'Coverage (fraction of (test × param) pairs within ±nσ)'}"
    )
    logger.info(sep)
    logger.info(
        f"{'nσ':<6} {'Expected':<12} {'CNN':<18} {'CNN z-scored':<18} "
        f"{'FM prior match':<18} {'FM N(0,I)':<18}"
    )
    logger.info("-" * 88)
    for n_idx, (tc, c_cnn, c_sc, c_pm, c_no_pm) in enumerate(
        zip(true_cov, cov_cnn, cov_cnn_sc, cov_fm_pm, cov_fm_no_pm), start=1
    ):
        logger.info(
            f"{n_idx}σ     {tc:<12.4f} {c_cnn:<18.4f} {c_sc:<18.4f} "
            f"{c_pm:<18.4f} {c_no_pm:<18.4f}"
        )
    logger.info(sep)

    np.savez(
        "comparison_results.npz",
        param_names=np.array(param_names),
        truth=truth,
        mu_cnn=mu_cnn,
        sigma_cnn=sigma_cnn,
        samples_cnn=samples_cnn,
        mu_cnn_scaled=mu_cnn_sc,
        sigma_cnn_scaled=sigma_cnn_sc,
        samples_cnn_scaled=samples_cnn_sc,
        samples_fm_pm=samples_fm_pm,
        samples_fm_no_pm=samples_fm_no_pm,
        acc_cnn=acc_cnn,
        acc_cnn_scaled=acc_cnn_sc,
        acc_fm_pm=acc_fm_pm,
        acc_fm_no_pm=acc_fm_no_pm,
        coverage_cnn=np.array(cov_cnn),
        coverage_cnn_scaled=np.array(cov_cnn_sc),
        coverage_fm_pm=np.array(cov_fm_pm),
        coverage_fm_no_pm=np.array(cov_fm_no_pm),
    )
    logger.info("Saved comparison_results.npz")

    figure_dir = "figures"
    os.makedirs(figure_dir, exist_ok=True)

    n_corner = min(inp.n_corner_obs, truth.shape[0])
    obs_indices = np.linspace(0, truth.shape[0] - 1, n_corner, dtype=int)

    min_par = cnn_model.min_par.numpy()
    max_par = (cnn_model.min_par + cnn_model.amp_par).numpy()
    extent = [(float(mn), float(mx)) for mn, mx in zip(min_par, max_par)]

    for obs_idx in obs_indices:
        logger.info(f"Corner plot for test observation {obs_idx} ...")
        _plot_four_corner(
            samples_cnn=samples_cnn[obs_idx],
            samples_cnn_scaled=samples_cnn_sc[obs_idx],
            samples_fm_pm=samples_fm_pm[obs_idx],
            samples_fm_no_pm=samples_fm_no_pm[obs_idx],
            truth=truth[obs_idx],
            labels=param_names,
            extent=extent,
            out_path=figure_dir,
            obs_idx=int(obs_idx),
        )
        plt.close("all")

    logger.info(f"Corner plots saved to ./{figure_dir}/")
    logger.info("Done.")
