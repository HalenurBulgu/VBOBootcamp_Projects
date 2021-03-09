"""* SalePrice - mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalışılan hedef değişkendir.
* MSSubClass: İnşaat sınıfı
* MSZoning: Genel imar sınıflandırması
* LotFrontage: Mülkiyetin sokak ile cephe uzunluğu
* LotArea: Parsel büyüklüğü (fit kare cinsinden)
* Street: Sokak/cadde erişiminin tipi
* Alley: Evin arka cephesindeki bağlantı yolu tipi
* LotShape: Mülkün genel şekli/durumu
* LandContour: Mülkün düzlüğü (tepe olup olmaması)
* Utilities: Mevcut elektrik/su/dogalgaz vb hizmet turleri
* LotConfig: Parsel durumu (ic, dış, iki parsel arası ya da kose gibi)
* LandSlope: Mülkün eğimi
* Neighborhood: Ames şehir sınırları içindeki fiziksel konumu
* Condition1: Ana yol veya tren yoluna yakınlık
* Condition2: Ana yola veya demiryoluna yakınlık (eğer ikinci bir yol/demiryolu varsa)
* BldgType: Konut tipi
* HouseStyle: Konut stili
* OverallQual: Genel malzeme ve işçilik kalitesi
* OverallCond: Konutun genel durum değerlendirmesi
* YearBuilt: Orijinal yapım tarihi
* YearRemodAdd: Yenilenme (elden geçirme, renovasyon) tarihi
* RoofStyle: Çatı tipi
* RoofMatl: Çatı malzemesi
* Exterior1st: Evdeki dış kaplama
* Exterior2nd: Evdeki dış kaplama (birden fazla malzeme varsa)
* MasVnrType: Evin ilave dış duvar kaplama türü (Masonry Veneer : Estetik icin distan örülen ek duvar)
* MasVnrArea: Evin ilave dış duvar kaplama alanı
* ExterQual: Dış malzeme kalitesi
* ExterCond: Dış malzemenin mevcut durumu
* Foundation: Konutun temel tipi
* BsmtQual: Bodrum katin yüksekliği
* BsmtCond: Bodrum katının genel durumu
* BsmtExposure: Bodrumdan bahçenin veya bahçe duvarlarının gorunmesi durumu
* BsmtFinType1: Bodrum katındaki yapılı, badana + zemin olarak tam islem görmüş alanın kalitesi
* BsmtFinSF1: Tam islem görmüş, yapılı alanın metre karesi
* BsmtFinType2: Bodrum katındaki yari yapılı alanın kalitesi (varsa)
* BsmtFinSF2: Bodrumdaki yari yapılı alanın metre karesi
* BsmtUnfSF: Bodrumdaki hiç islem görmemiş alanın metre karesi
* TotalBsmtSF: Bodrum katinin toplam metre karesi
* Heating: Isıtma tipi
* HeatingQC: Isıtma kalitesi ve durumu
* CentralAir: Merkezi klima
* Electrical: Elektrik sistemi
* 1stFlrSF: Birinci Kat metre kare alanı
* 2ndFlrSF: İkinci kat metre kare alanı
* LowQualFinSF: Düşük kaliteli islem/iscilik olan toplam alan (tüm katlar)
* GrLivArea: Zemin katin üstündeki oturma alanı metre karesi
* BsmtFullBath: Bodrum katındaki tam banyolar ( lavabo + klozet + dus + küvet)
* BsmtHalfBath: Bodrum katındaki yarım banyolar ( lavabo + klozet)
* FullBath: Üst katlardaki tam banyolar
* HalfBath: Üst katlardaki yarım banyolar
* BedroomAbvGr: Bodrum seviyesinin üstünde yatak odası sayısı
* KitchenAbvGr: Bodrum seviyesinin üstünde mutfak Sayısı
* KitchenQual: Mutfak kalitesi
* TotRmsAbvGrd: Üst katlardaki toplam oda (banyo içermez)
* Functional: Ev işlevselliği değerlendirmesi
* Fireplaces: Şömine sayısı
* FireplaceQu: Şömine kalitesi
* GarageType: Garajin yeri
* GarageYrBlt: Garajın yapım yılı
* GarageFinish: Garajın iç işçilik/yapim kalitesi
* GarageCars: Garajin araç kapasitesi
* GarageArea: Garajın alanı
* GarageQual: Garaj kalitesi
* GarageCond: Garaj durumu
* PavedDrive: Garajla yol arasındaki bağlantı
* WoodDeckSF: Ustu kapalı ahşap veranda alanı
* OpenPorchSF: Kapı önündeki açık veranda alanı
* EnclosedPorch: Kapı önündeki kapalı veranda alani (muhtemelen ince brandalı)
* 3SsPorch: Üç mevsim kullanılabilen veranda alanı (muhtemelen kis hariç kullanıma uygun, camli kisim)
* ScreenPorch: Sadece sineklik tel ile kapatilmis veranda alanı
* PoolArea: Havuzun metre kare alanı
* PoolQC: Havuz kalitesi
* Fence: Çit kalitesi
* MiscFeature: Diğer kategorilerde bahsedilmeyen çeşitli özellikler
* MiscVal: Çeşitli özelliklerin değeri
* MoSold: Satıldığı ay
* YrSold: Satıldığı yıl
* SaleType: Satış Türü
* SaleCondition: Satış Durumu"""


import warnings

from sklearn.exceptions import ConvergenceWarning
from helpers.data_prep import *
from helpers.eda import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("C:/Users/Lenovo/Masaüstü/VBObootcamp/projects/Houseprice_project/train.csv")
test = pd.read_csv("C:/Users/Lenovo/Masaüstü/VBObootcamp/projects/Houseprice_project/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()


df.isnull().any().sum()

######################################
# EDA
######################################
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Kategorik Değişken Sayısı: ', len(cat_cols))

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Id"]
print('Sayısal değişken sayısı: ', len(num_cols))


check_df(df)
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
print('Kategorik değişken sayısı: ', len(cat_cols))
print('Kardinal değişken sayısı: ', len(cat_but_car))
print('Sayısal değişken sayısı: ', len(num_cols))
print('Numerik gözüken fakat Kategorik değişken sayısı: ', len(num_but_cat))

######################################
# KATEGORIK DEGISKEN ANALIZI
######################################

#cat cols
for col in cat_cols:
    cat_summary(df, col)

#cat bur car

for col in cat_but_car:
    cat_summary(df, col)

for col in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
    print(df[col].value_counts())

#num but cat
for col in num_but_cat:
    cat_summary(df, col)


######################################
# SAYISAL DEGISKEN ANALIZI
######################################

for col in num_cols:
    num_summary(df, col, plot=True)

######################################
# TARGET ANALIZI
######################################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])


###############################################################
#Target ile bagımsız degiskenlerin korelasyonlarını inceleyelim
###############################################################

def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

low_corrs
high_corrs


"""
Highcors:
'OverallQual: 0.7909816005838047',
 'TotalBsmtSF: 0.6135805515591944',
 '1stFlrSF: 0.6058521846919166',
 'GrLivArea: 0.7086244776126511',
 'GarageArea: 0.6234314389183598'
 """

####################################################
#OverallQuall, GrLivArea ile Saleprice'ı inceleyelim
####################################################

import matplotlib.pyplot as plt
sns.boxplot(x="OverallQual", y="SalePrice", data=df,
            whis=[0, 100], width=.6, palette="vlag")
plt.show()

import matplotlib.pyplot as plt
sns.boxplot(x="GrLivArea", y="SalePrice", data=df,
            whis=[0, 100], width=.6, palette="vlag")
plt.show()


#########################
#FEATURE ENGİNEERİNG
#########################

df['Yearbuilt_range'] = pd.qcut(x=df['YearBuilt'], q=3, labels = ["new","old","tooold"])
df['1stFlrSF_range'] = pd.qcut(x=df['1stFlrSF'], q=3, labels = ["small","normal","big"])
df['GrLivArea_range'] = pd.qcut(x=df['GrLivArea'], q=3, labels = ["small","normal","big"])
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']+df["GrLivArea"]
df["Restoration_range"]=df["YearRemodAdd"]-df["YearBuilt"]
df["OutArea"]=df["GarageArea"]+df["PoolArea"]+df["WoodDeckSF"]+df["OpenPorchSF"]
df["OverallMultp"] = df["OverallQual"] * df["OverallCond"]


quality = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1}
df["ExterQual"] = df["ExterQual"].map(quality).astype('int')
df["HeatingQC"]= df["HeatingQC"].map(quality).astype('int')

df["Total_bath"] = df["FullBath"] + (df["HalfBath"] * 0.5)
quality = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1}
df["GarageQual"]= df["GarageQual"].map(quality)
df["BsmtQual"]= df["BsmtQual"].map(quality)
df["GarageQual"]=df["GarageQual"].fillna(0)
df["BsmtQual"]=df["BsmtQual"].fillna(0)
df["GarageQual"]=df["GarageQual"].astype('int')
df["BsmtQual"]=df["BsmtQual"].astype('int')


years={'new': 3, 'old': 2, 'tooold': 1}
df['Yearbuilt_range']=df["Yearbuilt_range"].map(years).astype('int')

range={'big': 3, 'normal': 2, 'small': 1}
df['1stFlrSF_range']=df['1stFlrSF_range'].map(range).astype('int')
df['GrLivArea_range']=df['GrLivArea_range'].map(range).astype('int')


df.groupby(by="Foundation").agg({"SalePrice":sum})
tipe={"Wood":1, "Stone":2, "Slab":4, "BrkTil":16, "CBlock":160, "PConc":320}
df['Foundationtipe']=df["Foundation"].map(tipe).astype('int')

list=["Total_bath","ExterQual","GarageQual","GrLivArea_range","1stFlrSF_range","Foundationtipe","BsmtQual","HeatingQC","Yearbuilt_range"]


######################################
# MISSING_VALUES
######################################

# Eksik değerler olmadığı anlamına gelmekte. 0 atayacağım.
#Yukarıda GarageQual ve BsmtQual'i yapmıştım o yüzden bu listeye almıyorum.

df.isnull().any()

nan_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish',  'GarageCond',  'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

for col in nan_cols:
    df[col].replace(np.nan, 0, inplace=True)


#########################################################
#DEĞİŞKENLERİ TEKRAR BELİRTELİM YENİ DEĞİŞKENLER EKLENDİ
#########################################################

check_df(df)
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
print('Kategorik değişken sayısı: ', len(cat_cols))
print('Kardinal değişken sayısı: ', len(cat_but_car))
print('Sayısal değişken sayısı: ', len(num_cols))
print('Numerik gözüken fakat Kategorik değişken sayısı: ', len(num_but_cat))


######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)


drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature"]

cat_cols1 = [col for col in cat_cols if col not in drop_list]
cat_cols=[col for col in cat_cols1 if col not in list]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", 0.01)


######################################
# LABEL ENCODING & ONE-HOT ENCODING
######################################

cat_cols2 = cat_cols + cat_but_car
cat_cols=[col for col in cat_cols2 if col not in list]

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)
df.head()

"""#LABEL ENCODING
binary_cols = [col for col in df.columns if len(df[col].unique()) ==2]

from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

#ONEHOT ENCODING:
ohe_cols1 = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
ohe_cols = [col for col in ohe_cols1 if col not in list]
from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()"""


######################################
# MISSING_VALUES
######################################
#Medyan atanacak boş değerleri dolduralım

missing_values_table(df)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]
df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)


df.isnull().any()

##########
# OUTLIERS
###########

for col in num_cols:
    print(col, check_outlier(df, col))

df["SalePrice"].describe().T

replace_with_thresholds(df, "SalePrice")


######################################
# TRAIN TEST'IN AYRILMASI
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

train_df.to_pickle("C:/Users/Lenovo/Masaüstü/VBObootcamp/projects/Houseprice_project/train_df.pkl")
test_df.to_pickle("C:/Users/Lenovo/Masaüstü/VBObootcamp/projects/Houseprice_project/test_df.pkl")

#######################################
# MODEL: LGBM
#######################################
import warnings
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

# y = train_df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=46)

lgb_model = LGBMRegressor(random_state=46).fit(X_train, y_train)
y_pred = lgb_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y.mean()

y_pred = lgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
#0.1157

#######################################
# Model Tuning
#######################################
lgb_params = {"learning_rate": [0.0007, 0.01, 0.1],
               "n_estimators": [1200, 2000, 2500],
               "max_depth": [5, 8, 10],
               "num_leaves":[8, 15],
               "colsample_bytree": [0.8, 0.5]}

lgb_model = LGBMRegressor(random_state=42)
lgb_cv_model = GridSearchCV(lgb_model, lgb_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
lgb_cv_model.best_params_


#######################################
# Final Model
#######################################

lgb_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X_train, y_train)

y_pred = lgb_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y_pred = lgb_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
#0.1055

#######################################
# Feature Importance
#######################################

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


plot_importance(lgb_tuned, X_train, 50)


#######################################
# SONUCLARIN YUKLENMESI
#######################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

y_pred_sub = lgb_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.head()

submission_df.to_csv('houseprice_lgbm.csv', index=False)
