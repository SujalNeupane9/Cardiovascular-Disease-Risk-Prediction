import logging
import pandas as pd
from zenml import step

@step
def ingest_data():
    """
    Args:
        None
    Returns:
        df.pd.DataFrame
    """
    try:
        df = pd.read_csv('E:\dl\Cardiovascular-Disease-Risk-Prediction\data\CVD_cleaned.csv')
        return df
    except Exception as e:
        logging.error(e)
        raise e
    
