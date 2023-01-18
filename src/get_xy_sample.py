import numpy as np

from src.EDistribution import EDistribution
from src.config import RNG


def get_xy_sample(
    distribution_type: EDistribution, beta: np.array, rho: float, sample_size: int
) -> tuple[np.array, np.array]:

    """Generates sample of X and y such that y = X*beta + eps, eps ~ N(0, 1)

    Args:
        beta (np.array): Linear transformation vector.
        rho (float): Covariance between covariates.

    Returns:
        tuple: X-sample, y-sample.
    """

    N = beta.shape[0]
    cov = rho * np.ones([N, N])
    cov = cov + np.diag(1 - rho * np.ones(N))

    N = beta.shape[0]
    X = EDistribution.get_sample(N=N, distribution_type=distribution_type, sample_size=sample_size, rho=rho)

    error = RNG.normal(loc=0, scale=1, size=X.shape[0])
    y = np.matmul(X, beta) + error

    return X, y
