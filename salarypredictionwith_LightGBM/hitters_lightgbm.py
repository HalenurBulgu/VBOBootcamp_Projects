"""
*AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş
sayısı

*Hits: 1986-1987 sezonundaki isabet sayısı

*HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı

*Runs: 1986-1987 sezonunda takımına kazandırdığı sayı

*RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı

*Walks: Karşı oyuncuya yaptırılan hata sayısı

*Years: Oyuncunun major liginde oynama süresi (sene)

*CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı

*CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı

*CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı

*CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı

*CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı

*CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı

*League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N
seviyelerine sahip bir faktör

*Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve
W seviyelerine sahip bir faktör

*PutOuts: Oyun icinde takım arkadaşınla yardımlaşma

*Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı

*Errors: 1986-1987 sezonundaki oyuncunun hata sayısı

*Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)

*NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve
N seviyelerine sahip bir faktör

"""
import numpy as np
import pandas as pd

import pickle
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option("display.float_format",lambda x: "%.5f" % x)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option('display.width', 170)


def load():
    data = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/hitters.csv")
    return data

df = load()
df.head()

#DEĞİŞKENLERİ İNCELEYELİM

#nan değer var mı inceleyelim
missing_values_table(df)


#DDEĞİŞKEN TÜRLERİNİ İNCELEYELİM
grab_col_names(df,cat_th=10, car_th=20)
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df,cat_th=10, car_th=20)


#DEĞİŞKENLERİN DEĞERLERİNİ İNCELEYELİM
df.describe([0.05,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.isnull().sum()


#SAYISAL DEĞİŞKENLERDE OUTLIER VAR MI İNCELEYELİM

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df, col))

#outlier yok.


#OUTLIER THRESHOLD
from helpers.data_prep import replace_with_thresholds

for col in num_cols:
    replace_with_thresholds(df, col)



#VERİ SETİNİ İNCELEYELİM
from helpers.eda import check_df
check_df(df)


#KORELASYONU İNCELEYELİM
df.corr()


#DEĞİŞKEN TÜRETELİM

df.head()
#df=pd.get_dummies(df,columns = ['League', 'Division', 'NewLeague'], drop_first = True)
#df=df.drop(['League', 'Division', 'NewLeague'], axis=1)

df["comp_1986_bat"]=df['AtBat']/df['CAtBat']
df["hits_ratio"] = df["Hits"] / df["CHits"]
df["hitsratiosuccess"] = df["Hits"] / df["AtBat"]
df["hmrun_ratio"] = df["HmRun"] / df["CHmRun"]
df["averagenumber_of_hits_perhit"]=df['CHits']/df['CAtBat']
df["walks_ratio"] = df["Walks"] / df["CWalks"]
df['RBI_ratio'] = df['RBI'] / df['CRBI']
df['Year_range'] = pd.qcut(x=df['Years'], q=3, labels = ["new","experienced","oldhand"])
df['CAtBat_average'] = df['CAtBat'] / df['Years']
df['CHits_average'] = df['CHits'] / df['Years']
df['CHmRun_average'] = df['CHmRun'] / df['Years']
df['CRun_average'] = df['CRuns'] / df['Years']
df['CRBI_average'] = df['CRBI'] / df['Years']
df['CWalks_average'] = df['CWalks'] / df['Years']

df.dtypes
df.head()

#LABEL ENCODING
binary_cols = [col for col in df.columns if len(df[col].unique()) ==2 ]

from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

#ONEHOT ENCODING:
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()




#STANDARTLAŞTIRMA UYGULAYALIM
#Hangi yöntemin uygulanacağı modele göre değişmekle birlikte en düşük hatayı standartscaler verdi.

from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()

num_cols1=['AtBat','Hits','HmRun','Runs','RBI','Walks','Years','CAtBat','CHits','CHmRun','CRuns','CRBI','CWalks','PutOuts', 'Assists','Errors']

df[num_cols1] = scaler.fit_transform(df[num_cols1])
df.head()

#NAN DEĞERLERİ TEKRAR İNCELEYELİM

#Salary bağımlı değişken olduğu için nan değerleri doldurmayacağız.
#salary değişkenindeki nan değerleri drop edelim.
df.dropna(inplace=True)

# conda install -c conda-forge lightgbm

import warnings
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


y = df["Salary"]
X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)


#######################################
# LightGBM: Model & Tahmin
#######################################

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#234.36

#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1, 0.3, 0.5],
               "n_estimators": [500, 800, 1200, 2000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#rmse:233.09

#######################################
# Feature Importance
#######################################

def plot_importance(model, X, num=len (X)):
    feature_imp = pd.DataFrame ({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure (figsize=(10, 15))
    sns.set (font_scale=1)
    sns.barplot (x="Value", y="Feature", data=feature_imp.sort_values (by="Value",
                                                                       ascending=False)[0:num])
    plt.title ('Feature Importance')
    plt.tight_layout ()
    plt.savefig ('importances-01.png')
    plt.show ()

plot_importance(lgbm_tuned, X_train)




