import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.utils import preprocessfortest
from pipelines.deployment import prediction_service_loader
from run_deployment import main

def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    high_level_image = Image.open("_assets/pipeline.png")
    st.image(high_level_image, caption="High Level Pipeline")

    #whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the Cardiovascular Disease Risk for a given order based on features like Weight_(kg),BMI,Smoking_History etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    #st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )
    General_Health = st.selectbox("General Health",['Poor', 'Very Good', 'Good', 'Fair', 'Excellent'])
    Checkup = st.selectbox('Checkup',['Within the past 2 years', 'Within the past year',
       '5 or more years ago', 'Within the past 5 years', 'Never'])
    Exercise = st.selectbox('Exercise',['No', 'Yes'])
    Heart_Disease = st.selectbox('Heart_Disease',['No', 'Yes'])
    Skin_Cancer = st.selectbox('Skin_Cancer',['No', 'Yes'])
    Other_Cancer = st.selectbox('Other_Cancer',['No', 'Yes'])
    Depression = st.selectbox('Depression',['No', 'Yes'])
    Diabetes = st.selectbox('Diabetes',['No', 'Yes', 'No, pre-diabetes or borderline diabetes',
       'Yes, but female told only during pregnancy'])
    Arthritis = st.selectbox('Arithritis',['No', 'Yes'])
    Sex = st.selectbox('Sex',['Male', 'Female'])
    Age_Category = st.selectbox('Age_Category',['70-74', '60-64', '75-79', '80+', '65-69', '50-54', '45-49',
       '18-24', '30-34', '55-59', '35-39', '40-44', '25-29'])
    Height = st.number_input("Height_(cm)")
    Weight = st.number_input("Weight_(kg)")
    BMI = st.number_input("BMI")
    Smoking_History = st.selectbox('Smoking_History',['No', 'Yes'])
    Alcohol_Consumption = st.number_input("Alcohol_Consumption")
    Fruit_Consumption = st.number_input("Fruit_Consumption")
    Green_Vegetables_Consumption = st.number_input("Green_Vegetables_Consumption")
    FriedPotato_Consumption = st.number_input("FriedPotato_Consumption")
    
    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            main()
            
        df = pd.DataFrame(
            {
                "General_Health":[General_Health],                   
                "Checkup":[Checkup],                        
                "Exercise":[Exercise],                       
                "Heart_Disease":[Heart_Disease],            
                "Skin_Cancer":[Skin_Cancer],               
                "Other_Cancer":[Other_Cancer],              
                "Depression":[Depression],               
                "Diabetes":[Diabetes],                 
                "Arthritis":[Arthritis],                
                "Sex":[Sex],                      
                "Age_Category":[Age_Category],              
                "Height_(cm)":[Height],                   
                "Weight_(kg)":[Weight],
                "BMI":[BMI],    
                "Smoking_History":[Smoking_History],  
                "Alcohol_Consumption":[Alcohol_Consumption],
                "Fruit_Consumption":[Fruit_Consumption],  
                "Green_Vegetables_Consumption":[Green_Vegetables_Consumption], 
                "FriedPotato_Consumption": [FriedPotato_Consumption]      
            }
        ) 
        data = json.loads(data)
        data.pop("columns")
        data.pop("index")
        df = pd.DataFrame(data["data"])
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        proc_data = preprocessfortest(data)
        proc_data = json.loads(proc_data)
        prediction = service.predict(proc_data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                prediction
            )
        )
            
            
if __name__ == "__main__":
    main()