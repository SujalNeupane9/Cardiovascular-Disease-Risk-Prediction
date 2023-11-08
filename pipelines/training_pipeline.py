from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True,settings={"docker":docker_settings})
def train_pipeline():
    """

    Args:
        ingest_data : DataClass 
        clean_data : DataClass
        model_train : DataClass
        evaluation : DataClass
        
    Returns:
        accuracy: float
        r2: float
    """
    df = ingest_data()
    X_train,X_test,y_train,y_test = clean_data(df)
    model = train_model(X_train,X_test,y_train,y_test)
    accuracy, r2 = evaluation(model,X_test,y_test)
    print(accuracy)
    print(r2)
    

    