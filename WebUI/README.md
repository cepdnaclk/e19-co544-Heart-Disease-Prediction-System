## Installation

Before running the script, you need to install the `gradio` library. You can install it using pip:

```bash
pip install gradio
```

# Heart Prediction Function

This Python script uses the `gradio` library to create a GUI for a heart prediction function. The function `heart_prediction` takes in various parameters related to a person's health and a machine learning model, and it is expected to return a prediction about the person's heart health.

## Function Parameters

-   `age`: Age of the person (integer between 0 and 100).
-   `sex`: Sex of the person (Male or Female).
-   `chest_pain_type`: Type of chest pain experienced (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic).
-   `resting_bp`: Resting blood pressure in mm Hg (float).
-   `cholesterol`: Serum cholesterol in mm/dl (float).
-   `fasting_bs`: Fasting blood sugar level (< 120 mg/dl or > 120 mg/dl).
-   `resting_ecg`: Resting electrocardiogram results (Normal, ST, LVH).
-   `max_hr`: Maximum heart rate achieved (integer between 60 and 202).
-   `exercise_angina`: Whether exercise-induced angina is present (Yes or No).
-   `oldpeak`: Oldpeak (float).
-   `st_slope`: Slope of the peak exercise ST segment (Up, Flat, Down).
-   `model`: The machine learning model to use for prediction (Decision Tree Classifier, Logistic Regression Classifier, Random Forest Classifier, SVM Classifier).

## GUI

The GUI for this function is created using `gr.Interface`. It includes sliders, radio buttons, textboxes, and a dropdown to input the parameters for the `heart_prediction` function.
