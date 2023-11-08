import logging
import pandas as pd
from model.data_cleaning import DataPreProcessing
from typing_extensions import Annotated
from zenml import step
from typing import Tuple
import numpy as np

@step
def clean_data(data:pd.DataFrame) -> Tuple[
    Annotated[np.ndarray,"X_train"],
    Annotated[np.ndarray,"X_test"],
    Annotated[np.ndarray,"y_train"],
    Annotated[np.ndarray,"y_test"],
]:
    """
    Data cleaning which preprocesses the data and divides it into train and test sets
    """
    try:
        data_cleaning = DataPreProcessing()
        X_train,X_test,y_train,y_test = data_cleaning.handle_data(data)
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(e)
        raise e
    
