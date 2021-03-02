import pandas as pd

pd.set_option('display.max_columns', 20)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011",engine='openpyxl')

df = df_.copy()
df.head()

df = df[~df["Invoice"].str.contains("C", na=False)]

#alışveriş sayısı 0'dan büyük olanları al
df = df[(df['Quantity'] > 0)]

df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby('Customer ID').agg({'Invoice': lambda x: len(x),
                                         'Quantity': lambda x: x.sum(),
                                         'TotalPrice': lambda x: x.sum()})

## fatura sayısı,toplam işlem, toplam fiyat
cltv_df.columns = ['total_transaction', 'total_unit', 'total_price']
cltv_df.head()

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin(kar marjı)
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin
###CLTV= (((Total_Revenue(toplam gelir) / Total_Number_of_Orders(toplam sipariş sayısı)) *  (Total_Number_of_Orders / Total_Number_of_Customers))/(1 - Repeat_Rate)) X Profit_margin

cltv_df.shape[0]

#Average_Order_Value = Total_Revenue / Total_Number_of_Orders
cltv_df['avg_order_value'] = cltv_df['total_price'] / cltv_df['total_transaction']

# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
cltv_df["purchase_frequency"] = cltv_df['total_transaction'] / cltv_df.shape[0]

# Churn_Rate = 1 - Repeat_Rate
repeat_rate = cltv_df[cltv_df["total_transaction"] > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate

#Kar marjı-Profit_margin
cltv_df['profit_margin'] = cltv_df['total_price'] * 0.05

#Customer_Value
cltv_df['CV'] = (cltv_df['avg_order_value'] * cltv_df["purchase_frequency"]) / churn_rate

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin(kar marjı)
cltv_df['CLTV'] = cltv_df['CV'] * cltv_df['profit_margin']

cltv_df.sort_values("CLTV", ascending=False)

#1-100 arası Transform
scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])

cltv_df.sort_values("CLTV", ascending=False)

cltv_df[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV",
                                                                                               ascending=False).head()
cltv_df.sort_values("total_price", ascending=False)

#Segmentlere ayırma
cltv_df["segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_df[["segment", "total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(
    by="SCALED_CLTV",
    ascending=False).head()
cltv_df[cltv_df["segment"]=="D"]["SCALED_CLTV"].max()

cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})

cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})


