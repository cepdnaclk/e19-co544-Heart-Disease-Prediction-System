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

df=pd.read_csv('./heart.csv')

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

"""###As we can see in the data of the dataset,Some data have String values for data.We have to encode them in order to identify them to the model.

> Encoding String values to Integer Values
"""

CATEGORICAL_COLUMNS = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERIC_COLUMNS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

df['ChestPainType'] = df['ChestPainType'].map({'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3})

df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'LVH': 1, 'ST': 2})

df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})

df['ST_Slope'] = df['ST_Slope'].map({'Flat': 0, 'Up': 1, 'Down': 2})

print(df.head())

for t in categorical_variables:
  print(df[t].value_counts().reset_index(name='Count'))
  print()

df.head(15)

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

Analysis How each Numerical Variable is affected
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
print(Perc)
MaleCount = len(dfHeartDisease[(dfHeartDisease['Sex']=='M')])
WomenCount = len(dfHeartDisease[(dfHeartDisease['Sex']=='F') & (dfHeartDisease['Age'] >= 55)])

# print(WomenCount)
# Chest Pain Counts
ChestPainCount_TA = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='TA')]) #Heart Disease Percentage is 100%
ChestPainCount_ATA = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='ATA')])
ChestPainCount_NAP = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='NAP')])
ChestPainCount_ASY = len(dfHeartDisease[(dfHeartDisease['ChestPainType']=='ASY')])


#Age and Sex
AgeCount = len(dfHeartDisease[(dfHeartDisease['Age'] > 50)])
# MenWith45 = len(dfHeartDisease[(dfHeartDisease['Age'] > 45) & (dfHeartDisease['Sex']=='M')])
WomenWith55 = len(dfHeartDisease[(dfHeartDisease['Age'] > 55) & (dfHeartDisease['Sex']=='F')])

Men = len(df[(df['Age'] >= 45) & (df['Sex']=='M')])
MenWith45 = len(dfHeartDisease[(dfHeartDisease['Age'] >= 45) & (dfHeartDisease['Sex']=='M')])

#Checking the vLIDITI OF THE DATA WITH REAL LIFE PROOF FROM A DOCTOR
print((MenWith45*100/Men))
# #Resting BP check
# BPcheckNormal = len(df[(df['RestingBP'] > 140)])
# print(BPcheckNormal)
# BPcheck = len(dfHeartDisease[(dfHeartDisease['RestingBP'] > 140)])
# print(BPcheck*100/BPcheckNormal)

# # Chrolestrol
# Chloestrols = len(df[(df['Cholesterol'] > 200)])
# print(Chloestrols)
# ChloestrolsHD = len(dfHeartDisease[(dfHeartDisease['Cholesterol'] > 200)])
# print(ChloestrolsHD)
# print(ChloestrolsHD*100/Chloestrols)

# # FastingBS
# FastingBS1 = len(df[(df['FastingBS'] == 1)])
# print(FastingBS1)
# FastingBS1HD = len(dfHeartDisease[(dfHeartDisease['FastingBS'] == 1)])
# print(FastingBS1HD)
# print(FastingBS1HD*100/FastingBS1)

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

"""##Scaling Features
Scaling is important because it helps to bring all features to the same scale or range.
So any feature have no unwanted weight over another.
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numericaldf = df[numeric_columns]
scaled_df = pd.DataFrame(scaler.fit_transform(numericaldf), columns = scaler.get_feature_names_out())

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