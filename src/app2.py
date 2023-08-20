import gradio as gr
import pandas as pd
import os
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier

# Useful Functions
rf = RandomForestClassifier()

def load_ml_components(fp):
    'load the ml components to re-use in app'
    with open(fp, 'rb') as f:
        object = pickle.load(f)
    return object


def interface_function(*args):
    ''''''
    ''''''
    
    
# Variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'ML_Assets','ml.pkl')

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp) 


num_imputer = ml_components_dict['num_imputer']
cat_imputer = ml_components_dict['cat_imputer']
encoder = ml_components_dict['OrdinalEncoder']
model = rf
scaler = ml_components_dict['StandardScaler']



# Interface
demo = gr.interface(interface_function, ['text'], 'number', examples = [],)
if __name__ == '__main__':
    demo.launch(debug = True)

