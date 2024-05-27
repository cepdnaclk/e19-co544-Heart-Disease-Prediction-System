import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# IQR Method for TRemoving Outliers
import pandas as pd
numeric_columns = ['Age', 'RestingBP', 'Cholesterol','MaxHR', 'Oldpeak']

def remove_outliers_iqr(dataset, numerical_variables):
    cleaned_dataset = dataset.copy()  # Make a copy of the dataset to avoid modifying the original dataset

    for variable in numerical_variables:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = cleaned_dataset[variable].quantile(0.25)
        Q3 = cleaned_dataset[variable].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = cleaned_dataset[(cleaned_dataset[variable] < lower_bound) | (cleaned_dataset[variable] > upper_bound)]
        print(variable, outliers)
        # Remove outliers
        cleaned_dataset = cleaned_dataset[(cleaned_dataset[variable] >= lower_bound) & (cleaned_dataset[variable] <= upper_bound)]

        # Print information about the removed outliers
        if not outliers.empty:
            print(f"Removed {len(outliers)} outliers from variable '{variable}'.")

    return cleaned_dataset

# Define the feature names
features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
df = pd.read_csv('heart.csv')

df = df[~df.duplicated()]
# Filter the DataFrame for rows where 'Cholesterol' is between 0 and 1
filtered_rows = df[(df['Cholesterol'] >= 0) & (df['Cholesterol'] <= 150)]

# Generate random values between 0 and 150 for replacement
random_values = np.random.uniform(0, 150, size=len(filtered_rows))

# Replace the values in the DataFrame with the generated random values
df.loc[(df['Cholesterol'] >= 0) & (df['Cholesterol'] <= 150), 'Cholesterol'] = random_values

df = remove_outliers_iqr(df,numeric_columns)
df.to_csv('heart_done.csv', index=False)
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

missing_values = final_df.isnull().sum()

print(missing_values)

# # Getting the input data
# print(final_df.iloc[-1])

# input_data = pd.DataFrame(final_df.iloc[-1], columns=df.columns)

# from sklearn.model_selection import train_test_split\

# X_input = input_data.drop('HeartDisease', axis=1)
# y_input = input_data['HeartDisease']

# # Load the model
# from joblib import load
# # Load the model from the file
# model = load('random_forest_model.joblib')

# y_pred =  model.predict(X_input)
# actual_value = int(input("Enter the Index: "))
# print(final_df.iloc[actual_value])


