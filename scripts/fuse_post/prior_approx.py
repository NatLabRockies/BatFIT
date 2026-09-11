import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d


def fit_gaussian_copula(prior_samples):
    """
    Fits a Gaussian Copula using robust Empirical CDFs and strictly enforces
    a Positive-Definite Correlation Matrix.
    """
    N, D = prior_samples.shape
    marginal_cdfs = []
    marginal_icdfs = []
    normal_transformed_samples = np.zeros_like(prior_samples)

    for d in range(D):
        col_data = prior_samples[:, d].copy()
        sorted_data = np.sort(col_data)

        # 1. Bulletproof Forward CDF (np.searchsorted handles duplicates perfectly)
        # We use default args (sd=sorted_data, n=N) to avoid Python's late-binding loop closures
        cdf = lambda x, sd=sorted_data, n=N: np.clip(
            np.searchsorted(sd, x, side="right") / n, 1e-5, 1 - 1e-5
        )

        # 2. Bulletproof Inverse CDF (np.quantile is mathematically exact for empirical data)
        icdf = lambda u, cd=col_data: np.quantile(cd, u)

        marginal_cdfs.append(cdf)
        marginal_icdfs.append(icdf)

        # Transform to standard normal space
        u_data = cdf(col_data)
        normal_transformed_samples[:, d] = stats.norm.ppf(u_data)

    # Calculate initial correlation matrix
    sigma_c = np.corrcoef(normal_transformed_samples, rowvar=False)

    # 3. Bulletproof Matrix Projection (Eigenvalue Clipping)
    # This guarantees the matrix is mathematically valid for sampling
    eigvals, eigvecs = np.linalg.eigh(sigma_c)

    # Force any negative or near-zero eigenvalues to be strictly positive
    eigvals[eigvals < 1e-6] = 1e-6

    # Reconstruct the matrix
    sigma_c_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalize to ensure the diagonal is exactly 1 (since it must be a correlation matrix)
    diag_std = np.sqrt(np.diag(sigma_c_psd))
    sigma_c_psd = sigma_c_psd / np.outer(diag_std, diag_std)

    return {
        "sigma_c": sigma_c_psd,
        "marginal_cdfs": marginal_cdfs,
        "marginal_icdfs": marginal_icdfs,
        "D": D,
    }


def sample_gaussian_copula(copula_model, num_samples):
    """
    Samples from the fitted Copula using the modern NumPy Generator.
    """
    sigma_c = copula_model["sigma_c"]
    marginal_icdfs = copula_model["marginal_icdfs"]
    D = copula_model["D"]

    rng = np.random.default_rng()

    # Because we mathematically forced sigma_c to be Positive-Definite,
    # this will no longer generate NaNs.
    latent_normal_samples = rng.multivariate_normal(
        mean=np.zeros(D), cov=sigma_c, size=num_samples, method="eigh"
    )

    u_samples = stats.norm.cdf(latent_normal_samples)
    generated_samples = np.zeros_like(latent_normal_samples)

    # Vectorized transformation back to parameter space
    for d in range(D):
        generated_samples[:, d] = marginal_icdfs[d](u_samples[:, d])

    return generated_samples


def plot_copula_verification(raw_samples, copula_samples, dims_to_plot=None):
    """
    Plots an overlapping corner plot to compare raw samples with copula-generated samples.
    """
    # Default to plotting the first 4 dimensions if not specified (to keep it readable)
    if dims_to_plot is None:
        plot_indices = list(range(min(4, raw_samples.shape[1])))
    else:
        plot_indices = dims_to_plot

    raw_sub = raw_samples[:, plot_indices]
    copula_sub = copula_samples[:, plot_indices]
    labels = [f"$\\theta_{{{i}}}$" for i in plot_indices]

    # Plot the raw prior samples (Blue)
    fig = corner.corner(
        raw_sub,
        color="blue",
        labels=labels,
        hist_kwargs={"density": True, "alpha": 0.5, "linewidth": 2},
        plot_datapoints=False,
        plot_density=True,
        levels=(0.68, 0.95),  # Show 1-sigma and 2-sigma contours
    )

    breakpoint()
    # Overplot the Copula samples (Red)
    corner.corner(
        copula_sub,
        fig=fig,
        color="red",
        hist_kwargs={"density": True, "alpha": 0.5, "linewidth": 2},
        plot_datapoints=False,
        plot_density=True,
        levels=(0.68, 0.95),
    )

    # Add a custom legend to the top right
    blue_line = mlines.Line2D(
        [], [], color="blue", linewidth=3, label="Raw Prior $p(\\theta)$"
    )
    red_line = mlines.Line2D(
        [], [], color="red", linewidth=3, label="Copula Approx"
    )
    fig.axes[0].legend(
        handles=[blue_line, red_line],
        loc="upper right",
        bbox_to_anchor=(len(plot_indices), 1),
        fontsize=12,
    )

    plt.suptitle("Gaussian Copula Verification", fontsize=16)
    plt.show()


# --- Demonstration Script ---
if __name__ == "__main__":
    D = 31

    # Generate tricky "uniform-ish" mock data with strong correlations
    # Dim 0 is uniform, Dim 1 is highly correlated to Dim 0
    raw_prior = np.load("data/assembled_data_diffcap.npz")["Y_data"]

    print("Fitting Copula...")
    fitted_model = fit_gaussian_copula(raw_prior)

    print("Generating Samples from Copula...")
    sampled_prior = sample_gaussian_copula(fitted_model, num_samples=10000)

    print("Plotting Verification (Dims 0, 1, 2, 3)...")
    plot_copula_verification(
        raw_prior, sampled_prior, dims_to_plot=[0, 4, 5, 7, 8, 11, 12, 21, 23]
    )
