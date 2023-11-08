import logging
import mlflow
import numpy as np
import pandas as pd
import catboost
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from sklearn.metrics import accuracy_score, r2_score
from typing import Tuple

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model:catboost.core.CatBoostClassifier,
    X_test:np.ndarray,
    y_test:np.ndarray
) -> Tuple[Annotated[float,"accuracy"],Annotated[float,"r2"]]:
    """
    Args:
        model:catboost.core.CatBoostClassifier,
        X_test:np.ndarray,
        y_test:np.ndarray
    Returns:
        r2:float,
        accuracy:float
    """
    try:
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test,prediction)
        mlflow.log_metric("accuracy",accuracy)
        
        r2 = r2_score(y_test,prediction)
        mlflow.log_metric("r2",r2)
        
        return accuracy, r2
    except Exception as e:
        logging.error(e)
        raise e