from enum import Enum
import numpy as np
from src.config import RNG


class EDistribution(str, Enum):
    normal = "normal"

    @staticmethod
    def get_sample(
        K: int,
        distribution_type: str,
        sample_size: int,
        rho: float,
        mean: float = 0,
    ):

        if distribution_type == EDistribution.normal:
            cov = rho * np.ones([K, K])
            cov = cov + np.diag(1 - rho * np.ones(K))
            X = RNG.multivariate_normal(
                mean=np.zeros(K) * mean,
                cov=cov,
                size=sample_size,
            )

        return X
