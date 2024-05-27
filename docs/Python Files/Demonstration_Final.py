import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


numeric_columns = ['Age', 'RestingBP', 'Cholesterol','MaxHR', 'Oldpeak']

features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
df = pd.read_csv('heart_done.csv')

# Function to get user inputs for each feature
def get_inputs():
    global df
    # Get user inputs for each feature
    # age = float(input("Enter Age: "))
    # sex = input("Enter Sex (M/F): ")
    # chest_pain_type = input("Enter Chest Pain Type (TA/ATA/NAP/ASY): ")
    # resting_bp = float(input("Enter Resting Blood Pressure (mm Hg): "))
    # cholesterol = float(input("Enter Cholesterol (mm/dl): "))
    # fasting_bs = int(input("Enter Fasting Blood Sugar (1 if > 120 mg/dl, 0 otherwise): "))
    # resting_ecg = input("Enter Resting ECG (Normal/ST/LVH): ")
    # max_hr = float(input("Enter Maximum Heart Rate Achieved: "))
    # exercise_angina = input("Enter Exercise Angina (Y/N): ")
    # oldpeak = float(input("Enter Oldpeak (ST measured in depression): "))
    # st_slope = input("Enter ST Slope (Up/Flat/Down): ")
    # Hard coded values for testing
    age = 49
    sex = 'F'
    chest_pain_type = 'NAP'
    resting_bp = 160
    cholesterol = 180
    fasting_bs = 283
    resting_ecg = 'Normal'
    max_hr = 150
    exercise_angina = 'N'
    oldpeak = 1.5
    st_slope = 'Up'

    # Create a DataFrame with the user inputs
    input_row = pd.DataFrame([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]], columns=features)
    
    # Append the input row to the input_data DataFrame
    df = df._append(input_row, ignore_index=True)
    
    


# Get user inputs
get_inputs()
# print(df)


# Feature Engineering
def categorize_blood_pressure(RestingBP):
    if RestingBP < 90:
        return "Low"
    elif 90 <= RestingBP <= 120:
        return "Normal"
    else:
        return "High"
    
df['blood_pressure_group'] = df['RestingBP'].apply(categorize_blood_pressure)

def categorize_cholesterol(Cholesterol):
    if Cholesterol < 195:
        return "Desirable"
    elif 195 <= Cholesterol <= 240:
        return "High Cholesterol"
    else:
        return "Excessive Cholestrol"
    
df['Cholestoral_group'] = df['Cholesterol'].apply(categorize_cholesterol)

def Max_Heart_Rate(MaxHR):
    if 60 <MaxHR < 100:
        return "Normal"
    elif 100 <= MaxHR <= 200:
        return "High MaxHR"
    else:
        return "Excessive MaxHR"
    
df['HR_Groups'] = df['MaxHR'].apply(Max_Heart_Rate)

def assign_age_group(age):

    age_groups = {
    (0, 40): 'Young',
    (41, 60): 'Middle-aged',
    (61, 80): 'Elderly',
    (81, float('inf')): 'Very Elderly'
    }

    for age_range, group in age_groups.items():
        if age_range[0] <= age <= age_range[1]:
            return group
df['age_group'] = df['Age'].apply(assign_age_group)

# One Hot Encoding
Categorical_coulmns_with_multiple_uniques = ['ChestPainType','RestingECG','ST_Slope']
OneHotEncoding = OneHotEncoder(sparse_output=False, drop='first')
Encoded_df = pd.DataFrame(OneHotEncoding.fit_transform(df[Categorical_coulmns_with_multiple_uniques]), columns = OneHotEncoding.get_feature_names_out())

# Mapping the categorical columns
df_Mapped = pd.DataFrame()
df_Mapped['FastingBS'] = df['FastingBS']
df_Mapped['age_group'] = df['age_group'].map({"Young": 0, "Middle-aged": 1, "Elderly": 2, "Very Elderly":3})
df_Mapped['Cholestoral_group'] = df['Cholestoral_group'].map({"Desirable": 0, "High Cholesterol": 1, "Excessive Cholestrol": 2})
df_Mapped['HR_Groups'] = df['HR_Groups'].map({"Normal": 0, "High MaxHR": 1, "Excessive MaxHR": 2})
df_Mapped['blood_pressure_group'] = df['blood_pressure_group'].map({"Low": 0, "Normal": 1, "High": 2})
df_Mapped['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df_Mapped['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})

# Scaling the numeric columns
numeric_columns = ['Age', 'RestingBP', 'Cholesterol','MaxHR', 'Oldpeak']
df_numeric = df[numeric_columns]
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df_numeric), columns = scaler.get_feature_names_out())

# Resetting the index of the dataframes
scaled_df = scaled_df.reset_index(drop=True)
Encoded_df = Encoded_df.reset_index(drop=True)
df_Mapped = df_Mapped.reset_index(drop=True)

# Concatenating the dataframes
final_df = pd.concat([scaled_df, Encoded_df, scaled_df], axis=1)

# Adding the target column
final_df['HeartDisease'] = df['HeartDisease'].reset_index(drop=True)

# Copy the last row to a new DataFrame
input_data = final_df.iloc[[-1]].copy()

# Drop the target column from the input data
input_data = input_data.drop('HeartDisease', axis=1)

# Remove the last row from the original DataFrame
final_df = final_df.iloc[:-1]

# Check for missing values
# missing_values = final_df.isnull().sum()
# print(input_data)
# print(final_df)
# print(missing_values)

# Split the data into input and target

from sklearn.model_selection import train_test_split\

X_input = input_data


# Load the model
from joblib import load

# Creating The Model
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split\
    
X = final_df.drop('HeartDisease', axis=1)
y = final_df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y, test_size=.3, random_state=42)

# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)

RandomForestModel = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       max_features='log2', min_samples_leaf=2,
                       min_samples_split=5)


acc_score = []
roc_score = []
f1 = []
name_model = []


# Training the Model
RandomForestModel.fit(X_train,y_train)

#Setting the Prediction value form the RandomForest
y_pred =  RandomForestModel.predict(X_test)

#Testing the accuracy of the Model
acc_score.append(accuracy_score(y_test, y_pred))
#Testing the ROC
roc_score.append(roc_auc_score(y_test, y_pred))
#Tesing the F1 score
f1.append(f1_score(y_test, y_pred, average='weighted'))
name_model.append(type(RandomForestModel).__name__)

result = pd.DataFrame(
    {'Model Name' : name_model,
     'accuracy': acc_score,
     'roc auc' : roc_score,
     'f1-weighted' : f1}
)
# print(result.sort_values('f1-weighted',ascending=False))

# Load the model from the file
# model = load('random_forest_model.joblib')

y_pred =  RandomForestModel.predict(X_input)

print("Model Prediction : ", y_pred[0])

actual_value = int(input("Enter the Index: "))

print(final_df.iloc[actual_value]['HeartDisease'])



