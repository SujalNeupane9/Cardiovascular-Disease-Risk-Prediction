import pandas as pd
import json
import numpy as np
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from .utils import get_data_for_test
from .utils import preprocessfortest
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentConfig(BaseParameters):
    """
    Parameters that are used to trigger the deployment
    """
    min_accuracy:float = 0.8
    
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name:str,
    pipeline_step_name:str,
    running:bool =True,
    model_name:str='model'
) -> MLFlowDeploymentService:
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name=model_name
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction deployed by the"
            f"{pipeline_step_name} step in the {pipeline_name}"
            f"pipeline for the '{model_name}'"
            f" model is currently running"
        )
        
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def deployment_trigger(
    accuracy:float,
    min_acc:float=0.8) -> bool:
    return accuracy > min_acc


@step
def predictor(
    service: MLFlowDeploymentService,
    data
) -> np.ndarray:
    """Run an inference request against a prediction service"""
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    df = pd.DataFrame(data["data"])
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    proc_data = preprocessfortest(data)
    proc_data = json.loads(proc_data)
    prediction = service.predict(proc_data)
    return prediction
    

@pipeline(enable_cache=True,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float = 0.8,
    workers:int=1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_data()
    X_train,X_test,y_train,y_test = clean_data(df)
    model = train_model(X_train,X_test,y_train,y_test)
    accuracy, r2 = evaluation(model,X_test,y_test)
    deployment_decision = deployment_trigger(accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision = deployment_decision,
        workers=workers,
        timeout=timeout
    ) 
    
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str,
                       pipeline_step_name: str):
    batch_data = get_data_for_test()
    processed_data = preprocessfortest(batch_data)
    processed_data_json = processed_data.tolist()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False
    )
    predictor(service=model_deployment_service, data=processed_data_json)
