import gradio as gr
from source.model_implementation import heart_prediction_scratch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def is_heart_disease(value):
        if value == 0:
            return "Person is not having Heart Disease"
        else:
            return "Person is having Heart Disease"
        
def heart_prediction(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope, model, Index):
    sex = ["M", "F"][sex]
    resting_bp = float(resting_bp)
    cholesterol = float(cholesterol)
    exercise_angina = ["Y", "N"][exercise_angina]
    oldpeak = float(oldpeak)

    x,y = heart_prediction_scratch(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,model,Index)
    return is_heart_disease(x),is_heart_disease(y)

# Create a GUI for the heart_prediction function using gr.Interface
heart_prediction_gui = gr.Interface(
    fn=heart_prediction,

    inputs=[gr.Slider(0, 100, value=0, label="Age", info="Choose between 1 and 100"),
            gr.Radio(["Male", "Female"], type="index", label="Sex", info="Select your gender"),
            gr.Radio(["TA", "ATA", "NAP", "ASY"], label="Chest Pain Type", info="TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic"),
            gr.Textbox(label="Resting blood pressure [mm Hg]"),
            gr.Textbox(label="Serum cholesterol [mm/dl]"),
            gr.Radio(["< 120 mg/dl", "> 120 mg/dl"], type="index", label="Fasting blood sugar"),
            gr.Radio(["Normal", "ST", "LVH"], label="Resting electrocardiogram results", info="ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria"),
            gr.Slider(60, 202, label="Maximum heart rate achieved", info="Numeric value between 60 and 202"),
            gr.Radio(["Yes", "No"], value=60, type="index", label="Exercise-induced angina"),
            gr.Textbox(label="Oldpeak"),
            gr.Radio(["Up", "Flat", "Down"], label="Slope of the peak exercise ST segment", info="Up: upsloping, Flat: flat, Down: downsloping"),
            gr.Dropdown(["Decision Tree Classifier", "Logistic Regression Classifier", "Random Forest Classifier", "SVM Classifier"], label="Model", type="index", info="Will add more models later!"),
            gr.Slider(1, 811, value=1, label="Index to Test", info="Choose between 1 and 100")
            ],

    outputs=[
        gr.Textbox(label="Model Prediction"), 
        gr.Textbox(label="Actual Value")
    ],
    examples=[
        [40,"Male","ATA",140,289.0,"< 120 mg/dl","Normal",172,"No",0.0,"Up","Decision Tree Classifier",0],
        [49,"Female","NAP",160,180.0,"< 120 mg/dl","Normal",156,"No",1.0,"Flat","Logistic Regression Classifier",1],
        [37,"Male","ATA",130,283.0,"< 120 mg/dl","ST",98,"No",0.0,"Up","Random Forest Classifier",2],
        [48,"Female","ASY",138,214.0,"< 120 mg/dl","Normal",108,"Yes",1.5,"Flat","SVM Classifier",3],
    ],
    title="Heart Disease Prediction System"
)

heart_prediction_gui.launch()