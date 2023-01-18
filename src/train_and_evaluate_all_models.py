import numpy as np
from sklearn.metrics import mean_squared_error

from src.EModel import EModel

def train_and_evaluate_all_models(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, n_components: int):
    
    """Trains LS, PLS and PCR models on X_train and y_test and evaluates them on X_test and y_test.

    Args:
        n_components (int): Number of relevant components to use for predicition in PCR and PLS.

    Returns:
        dict: Contains model names with their respective R2 values.
    """

    score_dict = {'R2': {}, 'MSE': {}}

    for model_name in list(EModel):
        model = EModel.train(model_name=model_name, X=X_train, y=y_train, n_components=n_components)
        score_dict['R2'][model_name.value] = model.score(X_test, y_test)
        preds_test = model.predict(X_test)
        score_dict['MSE'][model_name.value] = mean_squared_error(preds_test, y_test)

    return score_dict