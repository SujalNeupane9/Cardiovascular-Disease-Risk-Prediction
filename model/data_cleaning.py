import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> np.ndarray:
        pass
    
class DataPreProcessing(DataStrategy):
    """
    Data Preprocessing and splitting of data into train and test set
    """
    def __init__(self):
        pass
    
    def preprocessing(self, data: pd.DataFrame):
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
        return preprocessor
    
    def handle_data(self, data: pd.DataFrame) -> np.ndarray:
        try:
            X = data.drop("Heart_Disease", axis=1)
            y = data.Heart_Disease
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            preprocessor = self.preprocessing(data)
            
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
