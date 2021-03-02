# 1. users ve purchases veri setlerini okutunuz ve veri setlerini "uid" değişkenine göre inner join ile merge ediniz.
import pandas as pd
users = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/veribilimi/users.csv")
purchases = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/veribilimi/purchases.csv")
df = purchases.merge(users, how='inner',on='uid')
df.columns
df

# 2. Kaç unique müşteri vardır?
#tüm kullanıcılar:
users["uid"].nunique()
#satın alma yapan kullanıcılar:
purchases["uid"].nunique()
df[["uid"]].nunique()

# 3. Kaç unique fiyatlandırma var?
df[["price"]].nunique()

# 4. Hangi fiyattan kaçar tane satılmış?
df_price=df.groupby("price").agg({"uid":["count"]})
df_price

# 5. Hangi ülkeden kaçar tane satış olmuş?
df_country=df.groupby("country").agg({"uid":["count"]})
df_country

# 6. Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df["price"].sum()
df_country_price=df.groupby("country").agg({"price":["sum"]})
df_country_price
df_country_price=df.groupby("country")[["price"]].sum()
df_country_price
df_country_price=df[["price","country"]].groupby("country").sum()
df_country_price

# 7. Device türlerine göre göre satış sayıları nedir?
df_device=df.groupby("device").agg({"uid":["count"]})
df_device=df.groupby("device").agg({"price":["count"]})
df_device

# 8. ülkelere göre fiyat ortalamaları nedir?
df_country_price_mean=df.groupby("country").agg({"price":["mean"]})
df_country_price_mean

# 9. Cihazlara göre fiyat ortalamaları nedir?
df_device_mean=df.groupby("device").agg({"price":["mean"]})
df_device_mean

round(df.groupby("device").agg({"price":["mean"]}),3)

# 10. Ülke-Cihaz kırılımında fiyat ortalamaları nedir?
df_device_country=df.groupby(["device","country"]).agg({"price":["mean"]})
df_device_country
