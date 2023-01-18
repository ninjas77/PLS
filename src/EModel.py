from enum import Enum

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from src.config import RANDOM_STATE


class EModel(str, Enum):
    linreg = "linreg"
    pcr = "pcr"
    pls = "pls"

    @staticmethod
    def train(model_name: str, X: np.array, y: np.array, n_components: int | None = None):

        if model_name not in list(EModel):
            raise ValueError(f"No such model. Available models are {list(EModel)}")

        if model_name == EModel.linreg:
            model = LinearRegression()

        elif model_name == EModel.pcr:
            model = make_pipeline(PCA(n_components=n_components, random_state=RANDOM_STATE), LinearRegression())

        elif model_name == EModel.pls:
            model = PLSRegression(n_components=n_components)

        model.fit(X, y)

        return model

    @staticmethod
    def train_and_evaluate_all_models(
        X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, n_components: int
    ):

        """Trains LS, PLS and PCR models on X_train and y_test and evaluates them on X_test and y_test.

        Args:
            n_components (int): Number of relevant components to use for predicition in PCR and PLS.

        Returns:
            dict: Contains model names with their respective R2 values.
        """

        score_dict = {"R2": {}, "MSE": {}}

        for model_name in list(EModel):
            model = EModel.train(model_name=model_name, X=X_train, y=y_train, n_components=n_components)
            score_dict["R2"][model_name.value] = model.score(X_test, y_test)
            preds_test = model.predict(X_test)
            score_dict["MSE"][model_name.value] = mean_squared_error(preds_test, y_test)

        return score_dict
