# -*- coding: utf-8 -*-
"""Heart-disease-detection-data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SMlEouRkfI2GasWG8vNJuy0V2yhKPnWd

#Heart Disease Detection

##Importing essential Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Reading The Dtaset"""

df=pd.read_csv('./drive/MyDrive/Heart_Disease_Dataset/heart.csv')

"""Exploring the data"""

df.head(15)

df.tail(10)

"""#Features


1.   Age: age of the patient [years]
2. Sex: sex of the patient [M: Male, F: Female]
3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. RestingBP: resting blood pressure [mm Hg]
5. Cholesterol: serum cholesterol [mm/dl]
6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

# Target
* HeartDisease: output class [1: heart disease, 0: Normal]

Size and the shape of the data
"""

df.shape

"""Essential Values of the dataset"""

df.describe()

"""Check for missing Values"""

missing_count = df.isnull().sum()
print(missing_count)

"""##So there is no missing Values in this dataset

Checking duplicate rows
"""

df1 = df[~df.duplicated()]
print(df.shape)
print(df1.shape)

"""As we can see here,there are some duplicate rows in the dataset

> Removing Duplicate rows and form a unique row dataset
"""

df = df[~df.duplicated()]
print(df.shape)

"""Checking Unique values of each variable"""

df.nunique()

"""##So we can see that

* Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope, HeartDisease are **Categorical variables**
* Age, RestingBP, Cholesterol, MaxHR, Oldpeak are **Numerical Variables**

Counts of categorical variables
"""

categorical_variables = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for t in categorical_variables:
  print(df[t].value_counts().reset_index(name='Count'))
  print()

"""##Here are some plots of the data counts for Categorical Variables"""

df.Sex.value_counts().plot(kind='bar')

df.ChestPainType.value_counts().plot(kind='bar')

df.FastingBS.value_counts().plot(kind='bar')

"""###As we can see in the data of the dataset,Some data have String values for data.We have to encode them in order to identify them to the model."""



df.head(15)

# Filter the DataFrame for rows where 'Cholesterol' is between 0 and 1
filtered_rows = df[(df['Cholesterol'] >= 0) & (df['Cholesterol'] <= 150)]

# Generate random values between 0 and 150 for replacement
random_values = np.random.uniform(0, 150, size=len(filtered_rows))

# Replace the values in the DataFrame with the generated random values
df.loc[(df['Cholesterol'] >= 0) & (df['Cholesterol'] <= 150), 'Cholesterol'] = random_values

"""#Data visualization for Numerical Variables"""

numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

for numeric_Val in numeric_columns:
    plt.figure()
    df[numeric_Val].hist(bins=50)
    plt.title("Histogram of " + numeric_Val)
    plt.xlabel(numeric_Val)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

"""##By seeing those plot we can see that there are some outliers in some varibles.
These outliers may lead our model to bad predictions,So its better to get rid of them before continuing to modelling

we wish to use Techniques like IQR method to remove outliers from the variables.

> ## Removing Outliers By the IQR method
"""

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
df = remove_outliers_iqr(df,numeric_columns)
df.shape
# Example usage:
# cleaned_data = remove_outliers_iqr(your_dataset, ['numerical_var1', 'numerical_var2', ...])

"""Analysis How each Numerical Variable is affected

12 outliers from variable 'Oldpeak

---

70 outliers from variable 'Cholesterol'

---
28 outliers from variable 'RestingBP'

---
"""

count = (df[(df['Age'] > 50) & (df['HeartDisease'] == 1)])

chestPainTypes = len(df[(df['ChestPainType']=='TA') & (df['HeartDisease']==1)])
chestPainTypes*100/46
BPcheckNormal = len(df[(df['RestingBP'] > 140)])
print(BPcheckNormal)

"""Analysis about how each numerical Feature is affected

Age

I. Lowest age recorded was 28, Highest age recorded was 77.

Taking Criterias as Age < 50, 80 > Age > 50

Age <= 50 and Heart Disease == 1

Entries = 125

Age <= 50 and Heart Disease == 0

Entries = 191

80 > Age > 50 and Heart Disease == 1

Entries = 383

80 > Age > 50 and Heart Disease == 0

Entries = 219

By this we can see that Elderly people are more tends to be having a Heart Disease
"""

Women = len(df[(df['Sex']=='F')])

WomenCount55 = len(df[(df['Sex']=='F') & (df['Age'] >= 55)])
print(Women)
print(WomenCount55)

dfHeartDisease = df[(df['HeartDisease']==1)]
Perc=len(dfHeartDisease)
# print(Perc)
MaleCount = len(dfHeartDisease[(dfHeartDisease['Sex']=='M')])
WomenCount = len(dfHeartDisease[(dfHeartDisease['Sex']=='F') & (dfHeartDisease['Age'] >= 55)])

# print(WomenCount)
# Chest Pain Counts
ChestPainCount_TA = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='TA')])
ChestPainCount_ATA = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='ATA')])
ChestPainCount_NAP = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='NAP')])
ChestPainCount_ASY = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='ASY')])

ChestPainCountNormal_TA = len(df[(df['ChestPainType'] == 'TA')])
ChestPainCountNormal_ATA = len(df[(df['ChestPainType'] == 'ATA')])
ChestPainCountNormal_NAP = len(df[(df['ChestPainType'] == 'NAP')])
ChestPainCountNormal_ASY = len(df[(df['ChestPainType'] == 'ASY')])

print(ChestPainCount_TA*100/ChestPainCountNormal_TA)
print(ChestPainCount_ATA*100/ChestPainCountNormal_ATA)
print(ChestPainCount_NAP*100/ChestPainCountNormal_NAP)
print(ChestPainCount_ASY*100/ChestPainCountNormal_ASY)

#Age and Sex
AgeCount = len(dfHeartDisease[(dfHeartDisease['Age'] > 50)])
# MenWith45 = len(dfHeartDisease[(dfHeartDisease['Age'] > 45) & (dfHeartDisease['Sex']=='M')])
WomenWith55 = len(dfHeartDisease[(dfHeartDisease['Age'] > 55) & (dfHeartDisease['Sex']=='F')])

Men = len(df[(df['Age'] >= 45) & (df['Sex']=='M')])
MenWith45 = len(dfHeartDisease[(dfHeartDisease['Age'] >= 45) & (dfHeartDisease['Sex']=='M')])

# print((MenWith45*100/Men))


# #Resting BP check
# BPcheckNormal = len(df[(df['RestingBP'] > 140)])
# print(BPcheckNormal)
# BPcheck = len(dfHeartDisease[(dfHeartDisease['RestingBP'] > 140)])
# print(BPcheck*100/BPcheckNormal)

# # Chrolestrol
# Chloestrols = len(df[(df['Cholesterol'] < 100)])
# print(Chloestrols)
# ChloestrolsHD = len(dfHeartDisease[(dfHeartDisease['Cholesterol'] < 100)])
# print(ChloestrolsHD)
# print(ChloestrolsHD*100/Chloestrols)

# # FastingBS
# FastingBS1 = len(df[(df['FastingBS'] == 1)])
# print(FastingBS1)
# FastingBS1HD = len(dfHeartDisease[(dfHeartDisease['FastingBS'] == 1)])
# print(FastingBS1HD)
# print(FastingBS1HD*100/FastingBS1)

# # RestingECG
# RestingECG1 = len(df[(df['RestingECG'] == 'LVH')])
# print(RestingECG1)
# RestingECG_HD = len(dfHeartDisease[(dfHeartDisease['RestingECG'] == 'LVH')])
# print(RestingECG_HD)
# print(RestingECG_HD*100/RestingECG1)

# # ExerciseAngina
# ExerciseAngina1 = len(df[(df['ExerciseAngina'] == 'Y')])
# print(ExerciseAngina1)
# ExerciseAngina_HD = len(dfHeartDisease[(dfHeartDisease['ExerciseAngina'] == 'Y')])
# print(ExerciseAngina_HD)
# print(ExerciseAngina_HD*100/ExerciseAngina1)

# #Oldpeak
# Oldpeak1 = len(df[(df['Oldpeak'] == 'Y')])
# print(Oldpeak1)
# Oldpeak_HD = len(dfHeartDisease[(dfHeartDisease['Oldpeak'] == 'Y')])
# print(Oldpeak_HD)
# print(Oldpeak_HD*100/Oldpeak1)

# #ST_Slope
# ST_Slope1 = len(df[(df['ST_Slope'] == 'Up')])
# print(ST_Slope1)
# ST_Slope_HD = len(dfHeartDisease[(dfHeartDisease['ST_Slope'] == 'Up')])
# print(ST_Slope_HD)
# print(ST_Slope_HD*100/ST_Slope1)

selected_df = df.loc[df['RestingECG'] == 'ST', ['RestingECG', 'HeartDisease']]
selected_df

df.iloc[2]

df1 = len(df[(35 <= df['Age']) & (df['Age'] <= 44) & (df['Sex'] == 'M') & (df['HeartDisease']==1)])
df1

ChloestrolsCheck = len(df[(df['Cholesterol'] > -1) & (df['Cholesterol']<2)])
# ChloestrolsCheck
checkingValuesforCT = df[(df['Cholesterol'] > -1) & (df['Cholesterol'] < 2)]['Cholesterol']
checkingValuesforCT

df1 = df

"""> ## Feature Generation

## Numerical Columns

numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

---

## Age and Sex
Male

(0-30] Male as youngMale,

(30-45] Male as middleMale

(45-60] Male as oldMale

(60-80] Male as ElderlyMale

(80-inf) Male as veryOldMale

Female

(0-30] Female as youngFemale

(30-45) Female as middleFemale

(45-60] Female as oldFemale

(60-80] Female as ElderlyFemale

(80-inf) Female as veryOldFemale
"""

# Define the age bins and labels
age_bins = [0, 30, 45, 60, 80, float('inf')]
age_labels = ['young', 'middle', 'old', 'Elderly', 'veryOld']

# Bin the 'Age' column into the specified age groups
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Map the 'Sex' column to the specified categories
sex_mapping = {'M': 'Male', 'F': 'Female'}
df['sex_category'] = df['Sex'].map(sex_mapping)

df['sexCrossAge'] = df['age_group'].str.cat(df['sex_category'], sep=' ')

df.drop(columns=['age_group', 'sex_category'], inplace=True)

print(df)

df.head(20)

"""Resting_BP Categorization"""

def categorize_blood_pressure(RestingBP):
    if RestingBP < 90:
        return "Low"
    elif 90 <= RestingBP <= 120:
        return "Normal"
    else:
        return "High"

df['blood_pressure_group'] = df['RestingBP'].apply(categorize_blood_pressure)
df.head()

"""> By reasearching Some Papers and Getting medical Knowledge From the Doctor categogorize the Cholestrol values as follows"""

def categorize_cholesterol(Cholesterol):
    if Cholesterol < 195:
        return "Desirable"
    elif 195 <= Cholesterol <= 240:
        return "High Cholesterol"
    else:
        return "Excessive Cholestrol"

df['Cholestoral_group'] = df['Cholesterol'].apply(categorize_cholesterol)
df

"""normal value - 60 - 120
less than 60
more than 120
"""

def Max_Heart_Rate(MaxHR):
    if 60 <MaxHR < 100:
        return "Normal"
    elif 100 <= MaxHR <= 200:
        return "High MaxHR"
    else:
        return "Excessive MaxHR"

df['HR_Groups'] = df['MaxHR'].apply(Max_Heart_Rate)
df

df[(df['ST_Slope'] == 'Down')]

"""Oldpeak values are in between 0 and 6"""

df.drop(columns=['AgeCrossSex'], inplace=True)
# Define the age bins and labels
age_bins = [0, 30, 45, 60, 80, float('inf')]
age_labels = ['young', 'middle', 'old', 'Elderly', 'veryOld']

# Bin the 'Age' column into the specified age groups
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Map the 'Sex' column to the specified categories
sex_mapping = {'M': 'Male', 'F': 'Female'}
df['sex_category'] = df['Sex'].map(sex_mapping)

df['AgeCrossSex'] = df['age_group'].str.cat(df['sex_category'], sep=' ')

df.drop(columns=['age_group', 'sex_category'], inplace=True)

df



"""One hot encoding for all the values"""

from sklearn.preprocessing import OneHotEncoder

Categorical_coulmns_with_multiple_uniques = ['ChestPainType','RestingECG','ST_Slope']
OneHotEncoding = OneHotEncoder(sparse_output=False, drop='first')

Encoded_df = pd.DataFrame(OneHotEncoding.fit_transform(df[Categorical_coulmns_with_multiple_uniques]), columns = OneHotEncoding.get_feature_names_out())
Encoded_df

df=pd.read_csv('./updated_heart.csv')

"""youngMale
(30-45] Male as middleMale
(45-60] Male as oldMale
(60-inf] Male as ElderlyMale

(0-30] Female as youngFemale
(30-45) Female as middleFemale
(45-60] Female as oldFemale
(60-inf] Female as ElderlyFemale

> Encoding String values to Integer Values
"""

df_Mapped = pd.DataFrame()
df_Mapped['AgeCrossSex'] = df['AgeCrossSex'].map({'young Male': 0, 'middle Male': 1, 'old Male': 2, 'Elderly Male': 3, 'young Female': 4, 'middle Female': 5, 'old Female': 6, 'Elderly Female': 7})
df_Mapped['Cholestoral_group'] = df['Cholestoral_group'].map({"Desirable": 0, "High Cholesterol": 1, "Excessive Cholestrol": 2})
df_Mapped['HR_Groups'] = df['HR_Groups'].map({"Normal": 0, "High MaxHR": 1, "Excessive MaxHR": 2})
df_Mapped['blood_pressure_group'] = df['blood_pressure_group'].map({"Low": 0, "Normal": 1, "High": 2})
df_Mapped['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df_Mapped['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
df_Mapped.head()

df.drop(columns=['Unnamed: 0'], inplace=True)

df.head()

numeric_columns = ['Age', 'RestingBP', 'Cholesterol','MaxHR', 'Oldpeak']
df_numeric = df[numeric_columns]
df_numeric

for t in categorical_variables:
  print(df[t].value_counts().reset_index(name='Count'))
  print()

"""##Scaling Features
Scaling is important because it helps to bring all features to the same scale or range.
So any feature have no unwanted weight over another.
"""

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df_numeric), columns = scaler.get_feature_names_out())

"""> # Now we have scaled dataframe,Encoded Dataframe and mapped dataframe,We have to concatenate all the dataframes"""

Final_df = pd.concat([scaled_df, Encoded_df, df_numeric], axis=1)
Final_df.shape

"""Saving the dataframe as a csv file"""

Final_df.to_csv('final_heart.csv', index=False)

"""##Correlation Matrix"""

import matplotlib.pyplot as plt
import seaborn as sns

# Define the numeric columns
numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Assuming your DataFrame representing heart disease data is named heart_disease_df
correlation_matrix_heart = df[numeric_columns].corr()

# Plotting the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_heart, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix - Heart Disease Data')
plt.savefig('heart_disease_correlation_mat.png')
plt.show()

"""As we can see in the corrrelation matrix there is no significant correlation among any feature in the dataset, **So it is essential to use all the Features for training.**"""

