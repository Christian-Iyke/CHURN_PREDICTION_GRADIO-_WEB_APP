import gradio as gr
import pandas as pd
import pickle
import os


# Useful Functions
def load_ml_components(fp):
    'load the ml components to re-use in app'
    with open(fp, 'rb') as f:
        object = pickle.load(f)
    return object

# Variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'ml_Assets','ml.pkl')

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp) 

labels = ml_components_dict['label']
idx_to_labels = {i: l for (i, l) in enumerate(labels)}

