import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from helpers.data_prep import *
from helpers.eda import *
import datetime as dt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_excel("C:/Users/Lenovo/Masaüstü/VBObootcamp/projects/RFM/online_retail_II.xlsx", sheet_name="Year 2010-2011",engine='openpyxl')

pd.set_option("display.max_columns", None)
df.head()


check_df(df)

####
#EDA
####

# fatura basina ortalama kac para kazanilmistir? #iadeleri çıkarmamız gerekir ve iki değişkeni çarparak yeni bir değişken oluşturmak gerekmektedir

# iadeleri çıkararak yeniden df'i oluşturalım:
df = df[~df["Invoice"].str.contains("C", na=False)]

#total price bulmak için iki değişkeni çarpalım
df["TotalPrice"] = df["Quantity"] * df["Price"]


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
rfm.head()


#M sıfırdandan büyükse F sıfırdan küçük olamaz, F sıfırdan büyükse M sıfırdan küçük olamaz
rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)]

#5 recency skoru en yuksek degerdir cunku en az zaman gecmiştir.
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["FrequencyScore"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])


rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))

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

rfm["Segment"].value_counts()

rfm.head()

rfm.reset_index()

######################
##### KNN
######################

knn_df=rfm.iloc[:,:3]
knn_df.head()

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(knn_df)
df[0:5]

################################
# Kümelerin Görselleştirilmesi
################################

k_means = KMeans(n_clusters=10).fit(df)
kumeler = k_means.labels_
type(df)
df = pd.DataFrame(df)

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()


# merkezlerin isaretlenmesi
merkezler = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="black",
            s=200,
            alpha=0.5)
plt.show()


################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(2, 20))
visu.fit(df)
visu.show()

###k=6

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=6).fit(df)
kumeler = kmeans.labels_


rfm["KNN_Segment"] = kumeler
rfm=rfm.reset_index(drop=True)
rfm.head(50)

rfm[["KNN_Segment","Recency","Frequency","Monetary"]].groupby("KNN_Segment").agg(["min","max","mean","count"])


#en çok harcaması olan,en sık alışveriş sayısına sahip olan ve en yakın zamanda alışveriş yapmış olan müşterilerin 5. grupta olduğu gözlemlenmektedir.


#0 grubu grubu About to Sleep, At_Risk, Cant loose, Hibernating, Loyal Customers, Need Attention RFM segmentlerinden müşteriler barındırmaktadır.
rfm[rfm["KNN_Segment"]==0].groupby("Segment").agg(["min","max","mean","count"])


#1 grubu grubu At_Risk, Cant loose, Hibernating RFM segmentlerinden müşteriler barındırmaktadır.
rfm[rfm["KNN_Segment"]==1].groupby("Segment").agg(["min","max","mean","count"])


#2 grubu At_Risk, Cant loose, Hibernating RFM segmentlerinden müşteriler barındırmaktadır.
rfm[rfm["KNN_Segment"]==2].groupby("Segment").agg(["min","max","mean","count"])


#3 grubu At_Risk, About to Sleep, CantLoose, Hibernating, Loyal customers ve Need Attention RFM segmentlerinden müşteriler barındırmaktadır.
rfm[rfm["KNN_Segment"]==3].groupby("Segment").agg(["min","max","mean","count"])

#4 grubu About to Sleep, Champions, Loyal Customers, Need Attention, New Customers, Potential Loyalists, Promising RFM segmentlerinden müşteriler barındırmaktadır.
rfm[rfm["KNN_Segment"]==4].groupby("Segment").agg(["min","max","mean","count"])


#5 grubu Champions, Loyal Customers, Potential_Loyalist RFM segmentlerinden müşteriler barındırmaktadır.
rfm[rfm["KNN_Segment"]==5].groupby("Segment").agg(["min","max","mean","count"])
