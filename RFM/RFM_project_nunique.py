import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/online_retail_II.xlsx", sheet_name="Year 2010-2011",engine='openpyxl')

df_.head()
df = df_.copy()
df.head(10)

df.isnull().sum()

# essiz urun sayısı:
df["StockCode"].nunique()

# hangi urunden kacar tane var:
df["StockCode"].value_counts().head()

# en cok siparis edilen urun hangisi? ###WORLD WAR 2 GLIDERS ASSTD DESIGNS###
df.groupby("StockCode").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# toplam kac fatura kesildi-Eşsiz fatura sayısı:
df["Invoice"].nunique()

# fatura basina ortalama kac para kazanilmistir? #iadeleri çıkarmamız gerekir ve iki değişkeni çarparak yeni bir değişken oluşturmak gerekmektedir

# iadeleri çıkararak yeniden df'i oluşturalım:
df = df[~df["Invoice"].str.contains("C", na=False)]

#total price bulmak için iki değişkeni çarpalım
df["TotalPrice"] = df["Quantity"] * df["Price"]

# en pahalı ürünler hangileri?
df.sort_values("Price", ascending=False).head()

# hangi ulkeden kac siparis geldi?
df["Country"].value_counts()

### Data Preparation

df.isnull().sum()
df.dropna(inplace=True)

df["InvoiceDate"].max()

#Belirlenen tarih
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
#rfm sütunlarını oluşturma
rfm.columns = ['Recency', 'Frequency', 'Monetary']

#M sıfırdandan büyükse F sıfırdan küçük olamaz, F sıfırdan büyükse M sıfırdan küçük olamaz
rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)]

#5 recency skoru en yuksek degerdir cunku en az zaman gecmiştir.
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["FrequencyScore"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])


rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))
#rfm'i 5 olan kişiler
rfm[rfm["RFM_SCORE"] == "555"].head()

#rfm'i 1 olan kişiler
rfm[rfm["RFM_SCORE"] == "111"]

#segment olusturma
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}

rfm


#M yok
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

df[["Customer ID"]].nunique()

#rfm segmentlere gore count ve ortalamaları hesaplayalım:
rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count","max"])

#loyal customers
rfm[rfm["Segment"] == "Loyal_Customers"].head()

rfm[rfm["Segment"] == "Loyal_Customers"].index

new_df = pd.DataFrame()

#loyal customersı new_df column bazında atayarak excele aktaralım:
new_df["Loyal_Customers"] = rfm[rfm["Segment"] == "Loyal_Customers"].index

new_df.head()

new_df.to_csv("Loyal_Customers.csv")
