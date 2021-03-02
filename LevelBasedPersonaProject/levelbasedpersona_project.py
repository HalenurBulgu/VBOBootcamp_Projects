# PROJE: LEVEL BASED PERSONA TANIMLAMA, BASIT SEGMENTASYON ve KURAL TABANLI SINIFLANDIRMA

#1.adım
import pandas as pd
users = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/veribilimi/users.csv")
purchases = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/veribilimi/purchases.csv")
df = purchases.merge(users, how='inner',on='uid')
df

#2.adım
a=df.groupby(["country", "device", "gender", "age"]).agg({"price":"sum"})
a.head()

#3.adım
agg_df=a.sort_values(by=["price"],ascending=False)
agg_df.head()

#4.adım
agg_df=agg_df.reset_index()
agg_df.head()

#5.adım

agg_df["age_cat"]=pd.cut(agg_df["age"], [0, 18, 23, 40, 76],labels=["0_18","19_23","24_40","41_76"])
agg_df

#6.adım

agg_df["customer_level_based"]=[row[0] + "_" + row[1].upper() + "_" + row[2] + "_" + row[5] for row in agg_df.values]

agg_df.head()

agg_df_new=agg_df[["customer_level_based","price"]]
agg_df_new.head()
agg_df_new=agg_df_new.groupby("customer_level_based").agg({"price":"mean"})
agg_df_new.reset_index(inplace=True)
#7.adım
agg_df_new["segment"]=pd.qcut(agg_df_new["price"], 4, labels=["D", "C", "B", "A"])
agg_df_new

agg_df_new.groupby("segment")[["price"]].mean()
agg_df_new.groupby("segment").agg({"price":"mean"})

#8.adım

new_customer=agg_df_new[agg_df_new["customer_level_based"]=="TUR_IOS_F_41_76"]
new_customer
