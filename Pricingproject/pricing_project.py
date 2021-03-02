import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/pricing.csv" ,sep=";")
print("Dataframe columns", df.columns, "\n")
print("Data Shape: ", df.shape, "\n")
print("Head:", "\n", df.head(), "\n")
print("Tail:", "\n", df.tail(), "\n")
print("NA values: ", df.isna().any().sum())
x=df["category_id"].unique()
type(x)
df.category_id.value_counts()
pd.set_option('display.max_columns', None)
df.describe().T
df.info()

############################
# 1.1 Normallik Varsayımı
############################

from scipy.stats import shapiro
for cat in df["category_id"].unique():
    test_statistic , pvalue = shapiro(df.loc[df["category_id"] ==  cat,"price"])
    print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))

#p<0.05 Normallik varsayımı sağlanmamaktadır.


#######################Outlier###########################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
replace_with_thresholds(df, "price")



from scipy.stats import shapiro
for cat in df["category_id"].unique():
    test_statistic , pvalue = shapiro(df.loc[df["category_id"] ==  cat,"price"])
    print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))

############################
###Normallik varsayımı sağlansaydı###
# 1.2 Varyansların Homojenliği Varsayımı
############################
from scipy import stats
import itertools

new = []
for combin in itertools.combinations(df["category_id"].unique(),2):
    new.append(combin)
combin

for combin in new:
    test_statistic,pvalue = stats.levene(df.loc[df["category_id"] ==  combin[0],"price"],df.loc[df["category_id"] ==  combin[1],"price"] )
        print("{0} - {1} -- ".format(combin[0],combin[1]),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))

############################


#Normallik varsayımı sağlanmadığı için direkt olarak Non parametrik test yapabiliriz:
#########################################
#Non-Parametrik İndependet Two Sample Test
#########################################

sample_test_combin= []

for combin in itertools.combinations(df["category_id"].unique(),2):
    sample_test_combin.append(combin)
combin

for combin in sample_test_combin:
    test_statistic,pvalue = stats.stats.mannwhitneyu(df.loc[df["category_id"] ==  combin[0],"price"],df.loc[df["category_id"] ==  combin[1],"price"] )
        print("{0} - {1} -- ".format(combin[0],combin[1]),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))


#489756 - 361254 --  Test statistic = 380129.5000, p-Value = 0.0000 ###Ho rejected
#489756 - 874521 --  Test statistic = 519465.0000, p-Value = 0.0000 ###Ho rejected
#489756 - 326584 --  Test statistic = 70008.0000, p-Value = 0.0000  ###Ho rejected
#489756 - 675201 --  Test statistic = 86744.5000, p-Value = 0.0000  ###Ho rejected
#489756 - 201436 --  Test statistic = 60158.0000, p-Value = 0.0000  ###Ho rejected
#361254 - 874521 --  Test statistic = 218127.0000, p-Value = 0.0241 ###Ho rejected
#361254 - 326584 --  Test statistic = 33158.0000, p-Value = 0.0000  ###Ho rejected
#361254 - 675201 --  Test statistic = 39587.0000, p-Value = 0.3249  ###Ho is not rejected
#361254 - 201436 --  Test statistic = 30006.0000, p-Value = 0.4866  ###Ho is not rejected
#874521 - 326584 --  Test statistic = 38752.0000, p-Value = 0.0000  ###Ho rejected
#874521 - 675201 --  Test statistic = 47530.0000, p-Value = 0.2762  ###Ho is not rejected
#874521 - 201436 --  Test statistic = 34006.0000, p-Value = 0.1478  ###Ho is not rejected
#326584 - 675201 --  Test statistic = 6963.5000, p-Value = 0.0001   ###Ho rejected
#326584 - 201436 --  Test statistic = 5301.0000, p-Value = 0.0005   ###Ho rejected
#675201 - 201436 --  Test statistic = 6121.0000, p-Value = 0.3185   ###Ho is not rejected

####İstatistiksel olarak anlamlı farklılık göstermeyen kategorik gruplar#########

#361254 - 675201 --  Test statistic = 39587.0000, p-Value = 0.3249  ###Ho is not rejected
#361254 - 201436 --  Test statistic = 30006.0000, p-Value = 0.4866  ###Ho is not rejected
#874521 - 675201 --  Test statistic = 47530.0000, p-Value = 0.2762  ###Ho is not rejected
#874521 - 201436 --  Test statistic = 34006.0000, p-Value = 0.1478  ###Ho is not rejected
#675201 - 201436 --  Test statistic = 6121.0000, p-Value = 0.3185   ###Ho is not rejected

#İStatistiksel olarak benzer olan gruplar: 361254,675201,874521,201436

df.groupby("category_id").agg({"price":"mean"})

#İstatistiksel olarak aynı 4 kategorinin ortalamasını belirleyeceğimiz fiyat olarak alacağız.

cats = [361254,874521,675201,201436]
sum = 0
for i in cats:
    sum += df.loc[df["category_id"]== i,"price"].mean()
price = sum/4
print(price)


#Güven aralıkları
import statsmodels.stats.api as sms
prices=[]
for category in cats:
    for i in df.loc[df["category_id"]== category,"price"]:
        prices.append(i)
print(sms.DescrStatsW(prices).tconfint_mean())

#Ürün Satın Alma Simülasyonu
#Güven aralığı ve belirlediğimiz fiyatların minimum, maksimum değerlerinden elde edilebilecek gelirleri hesaplayalım.

freq1 = len(df[df["price"]>=38.94778731600629])
earning_low = freq1 * 38.94778731600629
print(earning_low)

freq2 = len(df[df["price"]>=41.38332006743786])
earning_up = freq2 * 41.38332006743786
print(earning_up)

print((38.94778731600629+41.38332006743786)/2)

