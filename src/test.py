import gradio as gr
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


# Our Model to be used
rf = RandomForestClassifier()

Cat_Encoder = OrdinalEncoder(categories='auto')


# Useful Functions
def load_ml_components(fp):
    'load the ml components to re-use in app'
    with open(fp, 'rb') as f:
        object = pickle.load(f)
    return object


def Receive_Input_Process_And_Predict ( gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, 
                  OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,StreamingTV, StreamingMovies, 
                  Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    
      
    '''receive inputs, process them and predict using the ML Model'''
    
    df = pd.DataFrame({'gender':[gender], 'SeniorCitizen':[SeniorCitizen], 'Partner':[Partner], 'Dependents':[Dependents], 'tenure':[tenure],
       'PhoneService':[PhoneService], 'MultipleLines':[MultipleLines], 'InternetService':[InternetService], 'OnlineBackup':[OnlineBackup],
       'DeviceProtection':[DeviceProtection], 'TechSupport':[TechSupport], 'StreamingTV':[StreamingTV], 'StreamingMovies':[StreamingMovies],
       'Contract':[Contract], 'PaperlessBilling':[PaperlessBilling], 'PaymentMethod':[PaymentMethod], 'MonthlyCharges':[MonthlyCharges],
       'TotalCharges':[TotalCharges]})
    
    df.replace('', np.nan , inplace = True)
    
    print(f'Inputs as DataFrame : {df.to_markdown()}')
    
    x_for_pred = df
    
    x_for_pred_ok = pd.concat([scaler.transform(num_imputer.transform(x_for_pred[num_cols])) if len(num_cols)> 0 else None, 
                                                encoder.transform(cat_imputer.transform(x_for_pred[cat_cols])) if len(cat_cols) > 0 else None],
                               axis = 1)
    
    y_pred = model.predict(x_for_pred_ok)
    
    print(f'[Info] Prediction as been made and the output is this:{y_pred}')
    
    return y_pred
    
# Variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'ML_Assets','ml.pkl')

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp) 
print(f'\n[Info] ML components loaded: {list(ml_components_dict.keys())}')


num_imputer = ml_components_dict['num_imputer']
cat_imputer = ml_components_dict['cat_imputer']
encoder = Cat_Encoder
model = rf
scaler = ml_components_dict['scaler']
num_cols = ml_components_dict['num_cols']
cat_cols = ml_components_dict['cat_cols']
label = ml_components_dict['label']

print(f"\[Info] Categorical columns: {',' .join(num_cols)}")

print(f"\[Info] Numeric columns: {',' .join(cat_cols)}\n")

#cat_n_uniques = {cat_cols[i]: opt_arr.tolist()
    
                #for (i, opt_arr)  in enumerate(encoder.categories)
#}


# Interface
inputs = [gr.Dropdown( elem_id = i) for i, choices in enumerate(encoder.categories)] + [gr.Number(elem_id =i) for i in range(4)]


demo = gr.interface( Receive_Input_Process_And_Predict, 
                    inputs, 
                    
                    "number",
                    
                    example = [],
                    
)

if __name__ == "__main__":
    
    demo.launch(debug =True)