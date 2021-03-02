
#!pip install skompiler
#!pip install astor
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile
import pickle

from helpers.eda import *
from helpers.data_prep import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load():
    data = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/diabetes.csv")
    return data

df = load()
df.head()

###########################
#DEĞİŞKENLERİ İNCELEYELİM
##########################

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

######################################
#DEĞİŞKENLERİN DEĞERLERİNİ İNCELEYELİM
######################################

df.describe([0.05,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.isnull().sum()

#pregnancies hariç diğer değişkenlerin min değerinin 0 gelmesi imkansızdır.
#bu sebeple bu değerleri nan ile dolduralım.

nan_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in nan_cols:
    df[col].replace(0,np.NaN,inplace=True)

df.isnull().sum()

##########################
#NAN DEĞERLERİ İNCELEYELİM
##########################

missing_values_table(df)

####################################
#NAN DEĞERLERİ MEDYAN İLE DOLDURALIM
####################################

for col in df.columns:
    df.loc[(df["Outcome"]==0) & (df[col].isnull()),col] = df[df["Outcome"]==0][col].median()
    df.loc[(df["Outcome"]==1) & (df[col].isnull()),col] = df[df["Outcome"]==1][col].median()

missing_values_table(df)


##########################################
#DEĞİŞKENLER ARASI KORELASYONU İNCELEYELİM
##########################################

df.corr()

#################################################
#SAYISAL DEĞİŞKENLERDE OUTLIER VAR MI İNCELEYELİM
#################################################

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df, col))

#skinthickness ve insulin değişkenlerinde outlier var bunları threshold yapabiliriz fakat CART kullanacağız o yüzden önemli değil.

############################################################################
#VERİ SETİ BİLGİSİ
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
##############################################################################


###################
#DEĞİŞKEN TÜRETELİM
###################

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


###############
#LABEL ENCODING
###############

binary_cols = [col for col in df.columns if len(df[col].unique()) ==2 and df[col].dtypes == 'O']


from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

df.head()


#################
#ONE HOT ENCODING
#################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


############MACHINE LEARNING##################


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)

#overfittinge düştük.


#####################################
# HOLDOUT YÖNTEMİ İLE MODEL DOĞRULAMA
#####################################

#Nasıl  1 olduğunu doğrulayalım. Train test olarak modeli ayıralım.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#1 geldi

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
#0.8111 geldi.


############################
# KARAR KURALLARINI ÇIKARMAK
############################

tree_rules = export_text(cart_model, feature_names=list(X_train.columns))
print(tree_rules)

##############################################
# KARAR KURALLARININ PYTHON KODLARINI ÇIKARMAK
##############################################

print(skompile(cart_model.predict).to('python/code'))


######################################
# KARAR KURALLARINA GÖRE TAHMİN YAPMAK
######################################

def predict_with_rules(x):
    return (((((((((0 if x[5] <= 36.19999885559082 else (0 if x[1] <= 86.5 else 1) if
    x[5] <= 40.400001525878906 else 0) if x[0] <= 0.5 else ((0 if x[5] <=
    22.59999942779541 else 1) if x[5] <= 23.050000190734863 else 0) if x[4] <=
    87.5 else 1 if x[1] <= 83.5 else 0) if x[1] <= 128.0 else 0 if x[5] <=
    27.300000190734863 else 1) if x[4] <= 97.0 else 1) if x[4] <= 99.5 else
    0) if x[6] <= 1.2800000309944153 else 1 if x[7] <= 39.5 else 0) if x[1] <=
    151.5 else 1 if x[3] <= 21.5 else 0 if x[6] <= 0.7055000066757202 else
    1 if x[4] <= 101.25 else 0) if x[4] <= 109.0 else (((1 if x[4] <= 123.0
     else 0) if x[6] <= 0.13450000435113907 else 0) if x[0] <= 7.5 else 1) if
    x[1] <= 124.5 else 0 if x[5] <= 25.449999809265137 else (((1 if x[7] <=
    49.0 else 0) if x[5] <= 30.399999618530273 else 0) if x[3] <= 31.5 else
    1) if x[4] <= 137.5 else 0) if x[4] <= 143.0 else (1 if x[10] <= 0.5 else
    (0 if x[1] <= 158.0 else 1) if x[7] <= 30.0 else 0 if x[3] <= 19.5 else
    1) if x[4] <= 169.75 else (((0 if x[6] <= 1.2335000038146973 else 0 if
    x[5] <= 29.100000381469727 else 1) if x[3] <= 41.5 else 1 if x[1] <=
    135.5 else 0) if x[1] <= 169.0 else 1 if x[9] <= 0.5 else 0) if x[7] <=
    27.5 else 0 if x[5] <= 26.300000190734863 else (((((1 if x[7] <= 29.5 else
    0) if x[2] <= 75.0 else (1 if x[4] <= 175.5 else 0) if x[4] <= 182.0 else
    1) if x[6] <= 0.8039999902248383 else 1) if x[1] <= 147.5 else 1 if x[6
    ] <= 1.6019999980926514 else 0) if x[5] <= 37.85000038146973 else 0) if
    x[5] <= 39.05000114440918 else 1 if x[5] <= 46.35000038146973 else 0)

x = [12, 13, 20, 23, 25, 26, 28, 29]

a = [2, 6, 100, 70, 35, 30, 62, 80]

predict_with_rules(x) #diabet
predict_with_rules(a) #diabet değil

#####################################
# DEĞİŞKEN ÖNEM DÜZEYLERİNİ İNCELEMEK
#####################################
import matplotlib.pyplot as plt
import seaborn as sns

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_model, X_train)


################################
# HİPERPARAMETRE OPTİMİZASYONU
################################

cart_model
cart_model = DecisionTreeClassifier(random_state=17)

###########İlk parametre denemesi

# arama yapılacak hiperparametre setleri

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4]}
cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)

# TRAIN
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#0.96 geldi.

#TEST
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
###  0.91 geldi. ###


##############ikinci parametre denemesi

# arama yapılacak hiperparametre setleri
cart_params = {'max_depth': range(1, 4, 11),
               "min_samples_split": [2, 3, 4, 6]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)

# TRAIN
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
#0.84 geldi

#TEST
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
### 0.83 geldi. ###


################################
# KARAR AGACINI GORSELLESTIRMEK
################################
from sklearn.tree import export_graphviz, export_text
def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

tree_graph_to_png(tree=cart_model, feature_names=X_train.columns, png_file_to_save='cart.png')


