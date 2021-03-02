# Libraries
##########################################
#pip install mysql-connector-python-rf
import datetime as dt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

import lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)


##########################################
# From csv
##########################################

df_ = pd.read_excel("C:/Users/Lenovo/OneDrive/Masaüstü/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011", engine="openpyxl")
df = df_.copy()

# From db
##########################################

# credentials.
creds = {'user': 'synan',
         'passwd': 'haydegidelum',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'group3'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

df_mysql = pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)
type(df_mysql)


##########################################
# Data Preperation
##########################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def crm_data_prep(dataframe):
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    return dataframe


check_df(df)
df_prep = crm_data_prep(df)
check_df(df_prep)
##########################################
# Creating RFM Segments
##########################################

def create_rfm(dataframe):
    # RFM METRIKLERININ HESAPLANMASI
    # Dikkat! RFM için frekanslar nunique.

    today_date = dt.datetime(2011, 12, 11)

    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})

    rfm.columns = ['recency', 'frequency', "monetary"]

    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    # Monetary segment tanımlamada kullanılmadığı için işlemlere alınmadı.

    # SEGMENTLERIN ISIMLENDIRILMESI
    rfm['rfm_segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['rfm_segment'] = rfm['rfm_segment'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "rfm_segment"]]
    return rfm


rfm = create_rfm(df_prep)
rfm.head()

##########################################
# Calculated CLTV
##########################################

def create_cltv_c(dataframe):
    # avg_order_value
    dataframe['avg_order_value'] = dataframe['monetary'] / dataframe['frequency']

    # purchase_frequency
    dataframe["purchase_frequency"] = dataframe['frequency'] / dataframe.shape[0]

    # repeat rate & churn rate
    repeat_rate = dataframe[dataframe.frequency > 1].shape[0] / dataframe.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
    dataframe['profit_margin'] = dataframe['monetary'] * 0.05

    # Customer Value
    dataframe['cv'] = (dataframe['avg_order_value'] * dataframe["purchase_frequency"])

    # Customer Lifetime Value
    dataframe['cltv'] = (dataframe['cv'] / churn_rate) * dataframe['profit_margin']

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(dataframe[["cltv"]])
    dataframe["cltv_c"] = scaler.transform(dataframe[["cltv"]])

    dataframe["cltv_c_segment"] = pd.qcut(dataframe["cltv_c"], 3, labels=["C", "B", "A"])

    dataframe = dataframe[["recency", "frequency", "monetary", "rfm_segment",
                           "cltv_c", "cltv_c_segment"]]

    return dataframe


check_df(rfm)


rfm_cltv = create_cltv_c(rfm)
check_df(rfm_cltv)



##########################################
# Predicted CLTV
##########################################

def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)

    ## recency kullanıcıya özel dinamik.
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)

    ## recency_cltv_p
    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

    ## basitleştirilmiş monetary_avg
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)


    # BGNBD için WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
    ## recency_weekly_cltv_p
    rfm["recency_weekly_cltv_p"] = rfm["recency_cltv_p"] / 7
    rfm["T_weekly"] = rfm["T"] / 7



    # KONTROL
    rfm = rfm[rfm["monetary_avg"] > 0]

    ## recency filtre (daha saglıklı cltvp hesabı için)
    rfm = rfm[(rfm['frequency'] > 1)]

    rfm["frequency"] = rfm["frequency"].astype(int)

    # BGNBD
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly_cltv_p'],
            rfm['T_weekly'])

    # exp_sales_1_month
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])
    # exp_sales_3_month
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])
    # exp_sales_6_month
    rfm["exp_sales_6_month"] = bgf.predict(24,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])
    # exp_sales_12_month
    rfm["exp_sales_12_month"] = bgf.predict(48,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])

    # expected_average_profit
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])

    # 1 aylık cltv_p
    cltv_1 = ggf.customer_lifetime_value(bgf,
                                         rfm['frequency'],
                                         rfm['recency_weekly_cltv_p'],
                                         rfm['T_weekly'],
                                         rfm['monetary_avg'],
                                         time=1,
                                         freq="W",
                                         discount_rate=0.01)
    # 3 aylık cltv_p
    cltv_3 = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=3,
                                       freq="W",
                                       discount_rate=0.01)

    # 6 aylık cltv_p
    cltv_6 = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    # 12 aylık cltv_p
    cltv_12 = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=12,
                                       freq="W",
                                       discount_rate=0.01)
    rfm["cltv_p_1_aylik"] = cltv_1
    rfm["cltv_p_3_aylik"] = cltv_3
    rfm["cltv_p_6_aylik"] = cltv_6
    rfm["cltv_p_12_aylik"] = cltv_12

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p_1_aylik"]])
    rfm["cltv_p_1_aylik"] = scaler.transform(rfm[["cltv_p_1_aylik"]])

    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p_3_aylik"]])
    rfm["cltv_p_3_aylik"] = scaler.transform(rfm[["cltv_p_3_aylik"]])

    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p_6_aylik"]])
    rfm["cltv_p_6_aylik"] = scaler.transform(rfm[["cltv_p_6_aylik"]])

    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p_12_aylik"]])
    rfm["cltv_p_12_aylik"] = scaler.transform(rfm[["cltv_p_12_aylik"]])


    # cltv_p_segment
    rfm["cltv_p_1ay_segment"] = pd.qcut(rfm["cltv_p_1_aylik"], 4, labels=["D","C", "B", "A"])
    rfm["cltv_p_3ay_segment"] = pd.qcut(rfm["cltv_p_3_aylik"], 4, labels=["D","C", "B", "A"])
    rfm["cltv_p_6ay_segment"] = pd.qcut(rfm["cltv_p_6_aylik"], 4, labels=["D","C", "B", "A"])
    rfm["cltv_p_12ay_segment"] = pd.qcut(rfm["cltv_p_12_aylik"], 4, labels=["D","C", "B", "A"])


    rfm = rfm[["monetary_avg", "T", "recency_weekly_cltv_p", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month","exp_sales_6_month","exp_sales_12_month", "expected_average_profit",
               "cltv_p_1_aylik","cltv_p_3_aylik","cltv_p_6_aylik","cltv_p_12_aylik","cltv_p_1ay_segment","cltv_p_3ay_segment","cltv_p_6ay_segment","cltv_p_12ay_segment"]]

    return rfm


rfm_cltv_p = create_cltv_p(df_prep)
check_df(rfm_cltv_p)


crm_final = rfm_cltv.merge(rfm_cltv_p, on="Customer ID", how="left")
check_df(crm_final)
crm_final["expected_avg_profit_and_monateary_corr"]=crm_final["expected_average_profit"].corr(crm_final["monetary"])

crm_final["expected_avg_profit_cltv_12"]=crm_final["expected_average_profit"].corr(crm_final["cltv_p_12_aylik"])

crm_final["expected_avg_profit_cltv_6"]=crm_final["expected_average_profit"].corr(crm_final["cltv_p_6_aylik"])

crm_final["expected_avg_profit_cltv_3"]=crm_final["expected_average_profit"].corr(crm_final["cltv_p_3_aylik"])

crm_final["expected_avg_profit_cltv_1"]=crm_final["expected_average_profit"].corr(crm_final["cltv_p_1_aylik"])

crm_final["cltv_c_and_p"]=crm_final["cltv_p_12_aylik"].corr(crm_final["cltv_c"])

crm_final

crm_final.head(30)
crm_final.tail(10)
crm_final_halenur_bulgu=crm_final
crm_final_halenur_bulgu

crm_final_halenur_bulgu["Frequency_max"]=crm_final_halenur_bulgu.groupby("Customer ID").agg({"frequency":"max"})
crm_final_halenur_bulgu

crm_final_halenur_bulgu["Recency_max"]=crm_final_halenur_bulgu.groupby("Customer ID").agg({"recency":"max"})
crm_final_halenur_bulgu

crm_final_halenur_bulgu["Monetary_sum"]=crm_final_halenur_bulgu.groupby("Customer ID").agg({"monetary":"sum"})
crm_final_halenur_bulgu

crm_final_halenur_bulgu.sort_values(by="monetary_avg", ascending=False).head()

crm_final_halenur_bulgu[(crm_final_halenur_bulgu["cltv_p_segment"]=="C")&(crm_final_halenur_bulgu["cltv_c_segment"]=="C")]

crm_final_halenur_bulgu[crm_final_halenur_bulgu["Customer ID"]==16321]
crm_final_halenur_bulgu[crm_final_halenur_bulgu["Customer ID"]==12349]

crm_final_halenur_bulgu[(crm_final_halenur_bulgu["cltv_p_segment"]=="A")&(crm_final_halenur_bulgu["cltv_c_segment"]=="B")&(crm_final_halenur_bulgu["rfm_segment"]=="potential_loyalists")]
crm_final_halenur_bulgu[crm_final_halenur_bulgu["Customer ID"]==12348]
crm_final_halenur_bulgu[crm_final_halenur_bulgu["Customer ID"]==18123]

crm_final_halenur_bulgu.sort_values(by="cltv_p",ascending=False)

crm_final_halenur_bulgu[crm_final_halenur_bulgu["cltv_p_segment"]=="A"]
crm_final_halenur_bulgu[crm_final_halenur_bulgu["cltv_p_segment"]=="B"]
crm_final_halenur_bulgu[crm_final_halenur_bulgu["cltv_p_segment"]=="C"]








