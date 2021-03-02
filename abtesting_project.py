# H0: max bidding ve average bidding arasında anlamlı farklılık yoktur.
# H1: anlamlı farklılık vardır

import numpy as np
from statsmodels.stats.proportion import proportions_ztest
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)

import pandas as pd

control = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/ab_testing_data.xlsx",sheet_name = "Control Group" ,engine='openpyxl')
test = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/ab_testing_data.xlsx",sheet_name = "Test Group" ,engine='openpyxl')


#################################################
#DESCRIBE
#################################################

control.info()
test.info()
control.isnull().any()
test.isnull().any()
def description(dataframe):
    print("Dataframe columns", dataframe.columns, "\n")
    print("Data Shape: ",dataframe.shape,"\n")
    print("Head:","\n",dataframe.head(),"\n")
    print("Tail:", "\n", dataframe.tail(), "\n")
    print("NA values: ",dataframe.isna().any().sum())

description(control)
description(test)

control.describe().T
test.describe().T


AB_test = pd.DataFrame()
###A control grup: Maximum Bidding
###B test grup: Average Bidding

AB_test["A"] = control["Purchase"]
AB_test["B"] = test["Purchase"]
AB_test.describe()
AB_test.head()

AB_test[["A", "B"]].corr()

#############################
#VISUALIZATION
#############################

import seaborn as sns
import matplotlib.pyplot as plt


AB_test.plot.scatter("A", "B")
plt.show()

sns.boxplot(data=control["Purchase"])
plt.show()
sns.boxplot(data=test["Purchase"])
plt.show()



values, names, xs = [], [], []
for i, col in enumerate(AB_test.columns):
    values.append(AB_test[col].values)
    names.append(col)
    xs.append(
        np.random.normal(i + 1, 0.04, AB_test[col].values.shape[0]))


plt.boxplot(values, labels=names)
palette = ['g', 'b']
for x, val, c in zip(xs, values, palette):
    plt.scatter(x, val, alpha=0.8, color=c)
plt.show()

##### AB Testing (Bağımsız İki Örneklem T Testi)

###########################
# Varsayım Kontrolü
# 1.1 Normallik Varsayımı
# 1.2 Varyans Homojenliği
###########################
# Ho: average bidding ve max bidding ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.
# H1: average bidding ve max bidding ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.

# İki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.


############################
# 1.1 Normallik Varsayımı
############################

from scipy.stats import shapiro
test_istatistigi, pvalue = shapiro(control["Purchase"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

from scipy.stats import shapiro
test_istatistigi, pvalue = shapiro(test["Purchase"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


#P>0.05 H0 reddedilemez. Normal dağılır.

############################
# 1.2 Varyansların Homojenliği Varsayımı
############################


from scipy import stats
stats.levene(control["Purchase"],test["Purchase"])


#P>0.05 Varyansların homojenliği hipotezi reddedilemez.

#Varsayımlar sağlandı iki örneklem T testi uygulayacağız.
test_istatistigi, pvalue = stats.ttest_ind(control["Purchase"],test["Purchase"],  equal_var=True)
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

#P>0.05 average bidding ve max bidding ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.

#kontrol grubunun-maksimum teklif verme- satın alınması ile test grubunun -Ortalama Teklif Verme- satın alınması  arasında istatistiksel olarak önemli bir fark yoktur.


control_plt = control["Purchase"].sum()/control["Click"].sum()
test_plt= test["Purchase"].sum() / test["Click"].sum()
df_new=pd.DataFrame({"max bidding":[control_plt], "average bidding": [test_plt]})
df_new.plot.bar()
plt.show()



control_plt = control["Purchase"].sum()/control["Impression"].sum()
test_plt = test["Purchase"].sum() / test["Impression"].sum()
df_plt2=pd.DataFrame({"max bidding":[control_plt], "average bidding": [test_plt]})
df_plt2.plot.bar(rot=0).legend(loc=3) ;
plt.show()
