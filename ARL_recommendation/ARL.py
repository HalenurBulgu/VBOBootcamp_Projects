
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011", engine="openpyxl")


df = df_.copy()
df.info()
df.head()

from helpers.helpers import check_df
check_df(df)

from helpers.helpers import crm_data_prep

df = crm_data_prep(df)
df
check_df(df)

df_ger = df[df['Country'] == "Germany"]
check_df(df_ger)

#stok kodlar tekilleşsin
df_ger.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).head(100)

#satırlarda fatura adı, sütunlarda ürünler ve ilgili verilerde hangi faturadan kaçar tane var?
df_ger.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().iloc[0:30, 0:30]


#DOĞRULAMA YAPALIM
df[(df["StockCode"] == 22908) & (df["Invoice"] == 581578)]


#boş değerleri 1 ve 0'a dönüştürme
df_ger.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

#def create_invoice_product_df(dataframe):
    #return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
        #applymap(lambda x: 1 if x > 0 else 0)

#ger_inv_pro_df = create_invoice_product_df(df_ger)
#ger_inv_pro_df.head()

#FONKSİYONLAŞTIRMA-HELPERS İÇERİSİNDEN
from helpers.helpers import create_invoice_product_df
create_invoice_product_df(df_ger)

ger_inv_pro_df = create_invoice_product_df(df_ger)
ger_inv_pro_df.head()


###########################################
# Birliktelik Kurallarının Çıkarılması
############################################

#Support: x ve ye'nin birlikte görülme olasılığı
frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

#support'a göre belirlediğimiz theshold'a göre kuralları çıkarma
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

#antecendents:önce gelen, 10125 olan ürünün gözlenme olasılığı support:0.013 değeri
#support: antecendents ve consequents  ürünlerinin beraber gözlenme olasılığı,ürünlerden biri alındığında diğerinin alınma olasılığı
#confidence lifti besleyen bir metrik, ilk ürün alındığında ikincisinin alınma olasılığı
#lif: ilk ürün alındığında diğer ürünün alınma olasılığının artışı-kat olarak
#conviction: y olmadan x'in beklenen frekansı
#leverage: supportu yüksek olan değerlere öncelik verir.


rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head()

#ürün kombinasyonları var.





