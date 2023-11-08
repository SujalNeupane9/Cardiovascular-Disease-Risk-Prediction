import logging
import mlflow
import numpy as np
import pandas as pd
from model.model_develop import CatBoostModel
import catboost
from zenml import step
from zenml.client import Client

from .config import ModelConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train:np.ndarray,
    X_test:np.ndarray,
    y_train:np.ndarray,
    y_test: np.ndarray,
    config:ModelConfig
) -> catboost.core.CatBoostClassifier:
    """
    Args:
        X_train:np.ndarray,
        X_test:np.ndarray,
        y_train:np.ndarray,
        y_test: np.ndarray
        
    Returns:
        model: catboost.core.CatBoostClassifier
    """
    try:
        model = CatBoostModel()
        trained_model = model.train(X_train,y_train,config.params)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e