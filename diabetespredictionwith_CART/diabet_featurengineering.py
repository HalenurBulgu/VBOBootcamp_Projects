import numpy as np
import pandas as pd

import pickle
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load():
    data = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/diabetes.csv")
    return data

df = load()
df.head()

#DEĞİŞKENLERİ İNCELEYELİM

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
if len(cat_cols) == 0:
    print("There is not Categorical Column",",","Number of Numerical Columns: ", len(num_cols), "\n", num_cols)
elif len(num_cols) == 0:
    print("There is not Numerical Column",",","Number of Categorical Column: ", len(cat_cols), "\n", cat_cols)
else:
    print("")


num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "Outcome"]
num_cols

###Pregnancies is num but cat column.


#DEĞİŞKENLERİN DEĞERLERİNİ İNCELEYELİM

df.describe([0.05,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.isnull().sum()

#pregnancies hariç diğer değişkenlerin min değerinin 0 gelmesi imkansızdır.
#bu sebeple bu değerleri nan ile dolduralım.

nan_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in nan_cols:
    df[col].replace(0,np.NaN,inplace=True)

df.isnull().sum()

#NAN DEĞERLERİ İNCELEYELİM
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    dtypes = dataframe.dtypes
    dtypesna = dtypes.loc[(np.sum(dataframe.isnull()) != 0)]
    missing_df = pd.concat([n_miss, np.round(ratio, 2), dtypesna], axis=1, keys=['n_miss', 'ratio', 'type'])
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("no missing value")
missing_values_table(df)

#NAN DEĞERLERİ MEDYAN İLE DOLDURALIM

for col in df.columns:
    df.loc[(df["Outcome"]==0) & (df[col].isnull()),col] = df[df["Outcome"]==0][col].median()
    df.loc[(df["Outcome"]==1) & (df[col].isnull()),col] = df[df["Outcome"]==1][col].median()

missing_values_table(df)

#DEĞİŞKENLER ARASI KORELASYONU İNCELEYELİM
df.corr()

#SAYISAL DEĞİŞKENLERDE OUTLIER VAR MI İNCELEYELİM

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df, col))
#skinthickness ve insulin değişkenlerinde outlier var bunları threshold yapalım.

from helpers.data_prep import replace_with_thresholds

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

from helpers.eda import check_df
check_df(df)


#DEĞİŞKEN TÜRETELİM
""" 
#1.Glucose Tolerance Test
It is a blood test that involves taking multiple blood samples over time, usually 2 hours.It used to diagnose diabetes. The results can be classified as normal, impaired, or abnormal.

Normal Results for Diabetes -> Two-hour glucose level less than 140 mg/dL

Impaired Results for Diabetes -> Two-hour glucose level 140 to 200 mg/dL

Abnormal (Diagnostic) Results for Diabetes -> Two-hour glucose level greater than 200 mg/dL

#2.BloodPressure
The diastolic reading, or the bottom number, is the pressure in the arteries when the heart rests between beats. This is the time when the heart fills with blood and gets oxygen. A normal diastolic blood pressure is lower than 80. A reading of 90 or higher means you have high blood pressure.

Normal: Systolic below 120 and diastolic below 80
Elevated: Systolic 120–129 and diastolic under 80
Hypertension stage 1: Systolic 130–139 and diastolic 80–89
Hypertension stage 2: Systolic 140-plus and diastolic 90 or more
Hypertensive crisis: Systolic higher than 180 and diastolic above 120.

#3.BMI
The standard weight status categories associated with BMI ranges for adults are shown in the following table.

Below 18.5 -> Underweight
18.5 – 24.9 -> Normal or Healthy Weight
25.0 – 29.9 -> Overweight
30.0 and Above -> Obese

#4.Triceps Skinfolds
For adults, the standard normal values for triceps skinfolds are:

2.5mm (men)
18.0mm (women) """

df.sort_values(by="Glucose")
df.sort_values(by="BloodPressure")

#Glikoz değerleri 0 ile 200 arasında ve bu sebeple değişken türetirken 200 üzerinde olan sınıf görülmeyebilir. Onehot kullanılacaksa dikkat edilmeli.

#Glucose_range oluşturulduğunda labellardan ötürü değişken tipi kategorik olarak gözükmekte ve bu sebeple eğer iki sınıf kullanılacaksa yani label encoding uygulanacaksa object koşuluna uymamakta o yüzden astype object yapılmalı.

df['Glucose_Range'] = pd.cut(x=df['Glucose'], bins = [0,140,200], labels = ["Normal","Prediabetes"]).astype('O')
df['BMI_Range'] = pd.cut(x=df['BMI'], bins=[0,18.5,24.9,29.9,100],labels = ["Underweight","Healty","Overweight","Obese"])
df['BloodPressure_Range'] = pd.cut(x=df['BloodPressure'], bins=[0,79,89,123],labels = ["Normal","HS1","HS2"])
df['SkinThickness_Range'] = df['SkinThickness'].apply(lambda x: 1 if x <= 18.0 else 0)

def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"
df["Insulin_range"] = df.apply(set_insulin, axis=1)

df.head()
df.dtypes

pd.set_option ("display.max_columns", None)
pd.set_option ('display.expand_frame_repr', False)
pd.set_option ('display.width', 170)
df.head()


#LABEL ENCODING
binary_cols = [col for col in df.columns if len(df[col].unique()) ==2 and df[col].dtypes == 'O']


from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

#ONE HOT ENCODING
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


#####################################################################
#RARE ENCODİNG
#onehot encoding oncesi yapmak önemli. Eğer değişkenler içerisinde sınıf sayısı az olan değerler olsaydı:
from helpers.data_prep import rare_analyser

rare_analyser(df, "Outcome", 0.05)

from helpers.data_prep import rare_encoder
df = rare_encoder(df, 0.01)
rare_analyser(df, "Outcome", 0.01)

df.head()
######################################################################
