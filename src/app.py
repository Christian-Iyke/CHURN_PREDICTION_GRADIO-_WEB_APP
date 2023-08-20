import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier



# Our Model to be used
rf = RandomForestClassifier()

#train_data = pd.read_csv(r'cleaned_data.csv')

with  gr.Blocks( ) as demo:
    gr.Markdown(
    """
    # Welcome to Telsco Churn 👋 !
    ## Customer Churn Classification App
    Start predicting customer churn.
    """, css= "h1 {color: red}")
    
    with gr.Row():
        gender = gr.Dropdown(label='Gender', choices=['Female', 'Male'])
        Contract  = gr.Dropdown(label='Contract', choices=['Month-to-month', 'One year', 'Two year'])
        InternetService = gr.Dropdown(label='Internet Service', choices=['DSL', 'Fiber optic', 'No'])

    with gr.Accordion('Yes or no'):

        with gr.Row():
            OnlineSecurity = gr.Radio(label="Online Security", choices=["Yes", "No", "No internet service"])
            OnlineBackup = gr.Radio(label="Online Backup", choices=["Yes", "No", "No internet service"])
            DeviceProtection = gr.Radio(label="Device Protection", choices=["Yes", "No", "No internet service"])
            TechSupport = gr.Radio(label="Tech Support", choices=["Yes", "No", "No internet service"])
            StreamingTV = gr.Radio(label="TV Streaming", choices=["Yes", "No", "No internet service"])
            StreamingMovies = gr.Radio(label="Movie Streaming", choices=["Yes", "No", "No internet service"]) 


#def create_new_columns(train_data):
    #train_data['Monthly Variations'] = (train_data.loc[:, 'TotalCharges']) -((train_data.loc[:, 'tenure'] * train_data.loc[:, 'MonthlyCharges']))
    #labels =['{0}-{1}'.format(i, i+2) for i in range(0, 73, 3)]
    #train_data['tenure_group'] = pd.cut(train_data['tenure'], bins=(range(0, 78, 3)), right=False, labels=labels)
    #train_data.drop(columns=['tenure'], inplace=True)
    
    
# Useful Functions
def load_ml_components(fp):
    'load the ml components to re-use in app'
    with open(fp, 'rb') as f:
        object = pickle.load(f)
    return object


# Variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'ML_Assets','ml.pkl')

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp) 
print(f'\n[Info] ML components loaded: {list(ml_components_dict.keys())}')


pipeline = ml_components_dict['pipeline']

model = rf


# function to make predictions

def predict_churn(gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                  OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                  Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    
    # collects data into a list
    data = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                   OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                   Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    
    # convert data into a numpy array
    x = np.array([data])
    
    # creates a dataframe 
    dataframe = pd.DataFrame(x, columns = 'train_features')
    
    dataframe = dataframe.astype({'MonthlyCharges': 'float', 'TotalCharges': 'float', 'tenure': 'float'})
    
    # creates the new features 
    create_new_columns(dataframe)
    
    # preprocess the data using pipeline
    processed_data = pipeline.transform(dataframe)
    processed_dataframe = 'create_processed_dataframe'(processed_data, dataframe)
    
    # the model make predictions
    
    predictions = model.predict_proba(processed_dataframe)
    
    return round(predictions[0][0], 3), round(predictions[0][1], 3)

demo = gr.interface(fn=predict_churn, inputs=[gender, 'SeniorCitizen', 'Partner', 'Dependents', 
                                              'Tenure', 'PhoneService', 'MultipleLines',     
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 
                                              'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
                                              'TotalCharges'], outputs = ['output1', 'output2'])
if __name__ == "__main__":
    demo.launch()