import sys

sys.path.append(".")

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import product
import numpy as np

from src.EModel import EModel
from src.EDistribution import EDistribution
from src.get_xy_sample import get_xy_sample
from src.config import (
    RNG,
    RANDOM_STATE,
    GLOBAL_PATH_TO_REPO,
)
from src.EDistribution import EDistribution


save_results_config = {
    "Ks": [200, 100, 50, 10, 5],
    "N_K_ratios": [10, 5, 3, 2, 1],
    "n_K_ratios": [0.01, 0.1, 0.2, 0.5, 0.8],
    "rhos": [0.01, 0.1, 0.25, 0.5, 0.9, 0.99],
    "sample_size": 100_000,
}


def save_results(
    distribution_type: EDistribution,
    csv_name: str | None = None,
    config: dict = save_results_config,
):

    if csv_name is None:
        csv_name = f"{distribution_type}"

    save_path = f"{GLOBAL_PATH_TO_REPO}/data/{csv_name}.csv"

    index_columns = [
        "K",
        "n_K_ratio",
        "n",
        "N_K_ratio",
        "M",
        "rho",
    ]
    columns = ["model", "R2", "MSE"]
    df = pd.DataFrame(None, columns=index_columns + columns)

    Ks = config["Ks"]
    N_K_ratios = config["N_K_ratios"]
    n_K_ratios = config["n_K_ratios"]
    rhos = config["rhos"]
    sample_size = config["sample_size"]

    loader = tqdm(
        product(Ks, rhos), total=len(Ks) * len(rhos)
    )

    for K, rho in loader:

        beta = RNG.normal(loc=0, scale=5, size=K)
        X, y = get_xy_sample(
            distribution_type=distribution_type,
            beta=beta,
            rho=rho,
            sample_size=sample_size,
        )

        for N_K_ratio in N_K_ratios:

            M = K * N_K_ratio

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = train_test_split(
                X,
                y,
                random_state=RANDOM_STATE,
                train_size=M,
            )

            for n_K_ratio in n_K_ratios:

                n = int(np.ceil(K * n_K_ratio))

                score_dict = (
                    EModel.train_and_evaluate_all_models(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        n_components=n,
                    )
                )
                score_df = pd.DataFrame(score_dict)
                score_df.index.name = "model"
                score_df = score_df.reset_index()
                for col_name, value in zip(
                    index_columns,
                    [K, n_K_ratio, n, N_K_ratio, M, rho],
                ):
                    score_df[col_name] = value
                df = pd.concat([df, score_df])
                loader.set_postfix(
                    **score_dict["R2"]
                    | {"rho": rho, "K": K, "n": n}
                )

        df.to_csv(save_path, index=False)

    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    save_results(
        distribution_type=EDistribution.normal,
        csv_name="hoplic",
    )
