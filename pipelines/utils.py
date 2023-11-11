import logging

import pandas as pd
import numpy as np
from model.data_cleaning import DataPreProcessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

def get_data_for_test():
    try:
        df = pd.read_csv("data/CVD_cleaned.csv")
        df = df.sample(n=200)
        return df
    except Exception as e:
        logging.error(e)
        raise e
    
def preprocessfortest(data:pd.DataFrame) -> np.ndarray:
        num_cols = [col for col in data.columns if data[col].dtype in ['int64','float64']]
        cat_cols = [col for col in data.columns if data[col].dtype=='object' and col != 'Heart_Disease']
        
        numerical_preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ("ordinal", OrdinalEncoder())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ("numerical", numerical_preprocessor, num_cols),
            ("categorical", categorical_preprocessor, cat_cols)
        ])
        processed_data = preprocessor.fit_transform(data)
        return processed_data
        