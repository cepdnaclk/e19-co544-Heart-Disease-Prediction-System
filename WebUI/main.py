import gradio as gr

def heart_prediction(age, sec, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope, model):
    sex = ["M", "F"][sex]
    resting_bp = float(resting_bp)
    cholesterol = float(cholesterol)
    exercise_angina = ["Y", "N"][exercise_angina]
    oldpeak = float(oldpeak)

    # TODO: Include necessary code to make a prediction using the provided parameters

    # For now, the function simply returns "Hello"
    return "Hello"

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
            ],

    outputs=["text"],
)

heart_prediction_gui.launch()