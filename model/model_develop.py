import pandas as pd
from abc import ABC, abstractclassmethod
from catboost import CatBoostClassifier

class Model(ABC):
    """
    Abstract base class for initializing any model
    Args:
        X_train:Training features
        y_train: Target labels
    """
    pass

class CatBoostModel(Model):
    """
    CatBoostModel whch implements the catboost interface
    """
    def train(self,X_train,y_train,config):
        clf = CatBoostClassifier(**config)
        clf.fit(X_train,y_train)
        return clf
    
