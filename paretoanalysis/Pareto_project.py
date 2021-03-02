####################################################################################
###Ürün bazında Pareto analizi

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
df_ = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/online_retail_II.xlsx", sheet_name="Year 2010-2011",engine='openpyxl')

df_.head()
df = df_.copy()
df = df[~df["Invoice"].str.contains("C", na=False)]
df=df.dropna()
df
df["Total_gain"]=df["Quantity"]*df["Price"]

df["Total_gain"].sum()

b = df.groupby("StockCode").agg({"Total_gain": "sum"}).sort_values("Total_gain", ascending=False)
b.reset_index(inplace=True)
##3665 ürün var

b["Total_gain"].sum()
##Toplam kazanılan:8911425.904

sayi = 0
j = 0
for i in b["Total_gain"]:
    sayi = sayi + i
    j = j + 1
    if (sayi / 8911425.904) >= 0.8:
        break
print(j)
##Kazanılan ücretler büyükten küçüğe sıralandığında tüm kazanılan miktarın %80'inden fazlasını ilk 777 ürün oluşturmuş.

print(777/3665)
##Bu ürünler tüm ürünlerimin %21'i.

b
df_new_stockcode = b.loc[0:776]

df_new_stockcode.to_csv("pareto_bystockcode.csv")
##################################################################################################

###Müşteri bazında Pareto analizi

x = df.groupby("Customer ID").agg({"Total_gain": "sum"}).sort_values("Total_gain", ascending=False)
x.reset_index(inplace=True)
x
##4339 müşteri var.

x["Total_gain"].sum()
##Toplam kazanılan:8911425.904
sayi = 0
j = 0
for i in x["Total_gain"]:
    sayi = sayi + i
    j = j + 1
    if (sayi / 8911425.904) >= 0.8:
        break
print(j)
##CustomerIDye göre kişilerin harcadıkları miktar büyükten küçüğe sıralandığında tüm kazanılan miktarın %80'ini (belki biraz daha fazlasını) ilk 1133 kişi oluşturmuş.

print(1133/4339)
##Bu kişiler tüm kişilerin %26'sıdır.


x
df_new_customer = b.loc[0:1132]

df_new_customer.to_csv("pareto_bycustomer.csv")
