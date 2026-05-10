
import shap
import pandas as pd

def generate_shap(model, df):

    sample = df.sample(50)

    explainer = shap.Explainer(model)

    shap_values = explainer(sample)

    shap.summary_plot(shap_values, sample)

    return shap_values